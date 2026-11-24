"""Tests for the Colab workflow: smoke run, speed tuning, source snapshots and resume handling."""
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from nllb_lora.train import colab_workflow as wf
from nllb_lora.train.sync_checkpoints import synchronize_checkpoint, verify_checkpoint
from nllb_lora.eval.translation_metrics import text_fingerprint


PROJECT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_t4_smoke_uses_two_processes_then_reuses_verified_acceptance(self):
        flow = self.workflow()
        calls = []
        def command(name, args, python=None):
            calls.append(name)
            run = Path(args[args.index("--run-dir") + 1])
            config = wf.read_json(args[args.index("--config") + 1])
            self.assertEqual((config["training"]["save_steps"], config["training"]["eval_steps"]), (7, 7))
            wf.write_json(run / "smoke_health.json", {"passed": True, "optimizer_present": True, "weights_changed": True})
            if name.endswith("initial"):
                self.assertEqual(args[args.index("--stop-after-step") + 1], "3")
            else:
                self.assertIn("--resume-from-drive", args)
                wf.write_json(run / "resume_verification.json", {"status": "RESUME_PROBE_PASSED_REVIEW_REQUIRED", "loaded_state_checks": True})
                wf.write_json(run / "completion.json", {"experiment_complete": True, "reason": "budget_complete", "total_budget": 7, "global_step": 7})
                wf.write_json(run / "validation/step-7/metrics.json", dict(self.metric(), count=16))
        def checkpoint(path):
            return {"step": int(path.name.split("-")[-1]), "files": {"rng_state.pth": {"sha256": "rng-fixture"}}}
        with patch.object(flow, "command", side_effect=command), \
             patch("nllb_lora.train.sync_checkpoints.verify_checkpoint", side_effect=checkpoint):
            flow.verify_t4_resume()
            flow.verify_t4_resume()
        self.assertEqual(calls, ["t4-smoke-resume-initial", "t4-smoke-resume-restore"])
        report = wf.read_json(flow.directory / "artifacts/t4-smoke-resume/report.json")
        self.assertTrue(report["passed"])
        self.assertEqual(report["checkpoint_steps"], [3, 7])
        self.assertIn("fresh Colab runtime not established", report["scope"])

    def test_failed_t4_resume_prevents_full_training(self):
        flow = self.workflow()
        config = wf.read_json(flow.config)
        config["training"]["gpu_profile"] = "t4"
        flow.config = self.base / "t4-config.json"
        wf.write_json(flow.config, config)
        flow.settings.update(mode="full", auto_tune=True)
        with patch.object(flow, "tune_speed"), \
             patch.object(flow, "verify_t4_resume", side_effect=ValueError("resume failed")), \
             patch.object(flow, "train_phase") as train:
            with self.assertRaisesRegex(ValueError, "resume failed"):
                flow.run()
            train.assert_not_called()

    def test_speed_tuning_is_locked_and_reused_without_mutating_source(self):
        flow = self.workflow()
        flow.state["environment_signature"] = {"python": "fixture", "packages": {}}
        original_hash = wf.digest(flow.config)
        def command(name, args, python=None):
            self.assertEqual(name, "speed-tuning")
            context = wf.read_json(args[args.index("--context") + 1])
            selected = wf.read_json(flow.config)
            selected["training"].update(per_device_train_batch_size=16, gradient_accumulation_steps=2,
                fp16=False, bf16=True, gradient_checkpointing=False, optim="adamw_torch_fused",
                attn_implementation="sdpa", dataloader_pin_memory=True)
            selected["generation"]["use_cache"] = True
            wf.write_json(flow.directory / "speed_tuning.json", {"status": "MEASURED", "context": context,
                "selected_config": selected})
        with patch.object(flow, "command", side_effect=command) as commands:
            flow.tune_speed()
            flow.tune_speed()
            commands.assert_called_once()
        self.assertEqual(flow.config.name, "full-config.json")
        self.assertEqual(wf.digest(PROJECT / "configs/nllb_zh_vi.json"), original_hash)
        restored = wf.Workflow(flow.settings, PROJECT, flow.identity["source"])
        self.assertEqual(restored.config, flow.config)
        self.assertEqual(restored.full_budget(), 85605)
        (flow.directory / "full-config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            restored.use_tuned_config()

    def test_speed_tuning_refuses_existing_phases_and_protected_changes(self):
        flow = self.workflow()
        flow.state["phases"]["smoke"] = {"fixture": True}
        with self.assertRaisesRegex(ValueError, "new experiment"):
            flow.tune_speed()
        flow.state["phases"] = {}
        flow.state["environment_signature"] = {}
        def command(name, args, python=None):
            config = wf.read_json(flow.config)
            config["training"]["learning_rate"] *= 2
            config["generation"]["use_cache"] = True
            context = wf.read_json(args[args.index("--context") + 1])
            wf.write_json(flow.directory / "speed_tuning.json", {"status": "MEASURED", "context": context,
                "selected_config": config})
        with patch.object(flow, "command", side_effect=command):
            with self.assertRaisesRegex(ValueError, "protected"):
                flow.tune_speed()
        self.assertNotIn("speed_tuning", flow.state)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def workflow(self):
        settings = {"mode": "pilot_complete", "experiment_id": None,
            "drive_output": str(self.base / "drive"), "local_work": str(self.base / "local"),
            "venv": str(self.base / "venv"), "drive_free_gib": 4600,
            "drive_quota_source": "User estimate: nearly 5 TB", "readiness": str(self.base / "readiness.json")}
        result = wf.Workflow(settings, PROJECT, wf.fingerprint(wf.inventory(PROJECT, wf.CODE_FILES)))
        result.state["cache_key"] = "cache-fixture"
        result.save()
        return result

    def dataset(self, split=True):
        code, bundle = self.base / "code", self.base / "bundle"
        (code / "nllb_lora/train").mkdir(parents=True)
        shutil.copyfile(PROJECT / "nllb_lora/train/prepare_data.py", code / "nllb_lora/train/prepare_data.py")
        files, splits = [], {}
        for name in ("train", "val", "test"):
            splits[name] = {"source": f"final/{name}.zh", "target": f"final/{name}.vi", "expected_pairs": 2}
            for extension, content in (("zh", "甲\n乙\n"), ("vi", "Một\nHai\n")):
                relative = f"final/{name}.{extension}"
                target = bundle / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
                files.append({"path": relative, "bytes": target.stat().st_size, "sha256": wf.digest(target), "lines": 2})
        wf.write_json(bundle / "checksums.json", {"files": files})
        sha = wf.digest(bundle / "checksums.json")
        (bundle / "checksums.json.sha256").write_text(sha + "  checksums.json\n", encoding="utf-8")
        config = {"data": {"manifest": "checksums.json", "manifest_sha256_file": "checksums.json.sha256",
            "expected_manifest_sha256": sha, "splits": splits, "semantic_review": "not_performed"}}
        wf.write_json(code / "configs/nllb_zh_vi.json", config)
        if split:
            path = bundle / "final/train.vi"
            data = path.read_bytes()
            (path.parent / "train.vi.part000").write_bytes(data[:4])
            (path.parent / "train.vi.part001").write_bytes(data[4:])
            path.unlink()
        return code, bundle

    def test_source_requires_configs_and_reuses_only_verified_snapshot(self):
        source = self.base / "source"
        for name in wf.CODE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        config = source / "configs/nllb_zh_vi.json"
        config.rename(source / config.name)
        with self.assertRaisesRegex(FileNotFoundError, "configs/nllb_zh_vi.json"):
            wf.prepare_source(source, self.base / "snapshots")
        (source / config.name).rename(config)
        target, identity = wf.prepare_source(source, self.base / "snapshots")
        self.assertEqual(wf.prepare_source(source, self.base / "snapshots"), (target, identity))
        (target / wf.CODE_FILES[0]).write_text("broken")
        wf.prepare_source(source, self.base / "snapshots")
        self.assertTrue(list(target.parent.glob(".incomplete-*")))
        self.assertEqual(wf.inventory(target, wf.CODE_FILES), wf.inventory(source, wf.CODE_FILES))

    def test_split_bundle_recovery_and_no_drive_mutation(self):
        code, bundle = self.dataset()
        original = wf.inventory(bundle)
        destination = self.base / "frozen"
        destination.mkdir()
        (destination / "partial").write_text("partial")
        report = wf.stage_data(code, bundle, destination)
        self.assertEqual(report["status"], "VERIFIED_INPUTS")
        self.assertEqual(report["splits"]["train"]["pairs"], 2)
        wf.stage_data(code, bundle, destination)
        self.assertEqual(original, wf.inventory(bundle))
        self.assertTrue(list(self.base.glob(".incomplete-frozen-*")))

    def test_missing_reordered_or_corrupt_chunks_never_publish(self):
        code, bundle = self.dataset()
        part = bundle / "final/train.vi.part001"
        value = part.read_bytes()
        part.unlink()
        with self.assertRaisesRegex(ValueError, "exactly"):
            wf.stage_data(code, bundle, self.base / "frozen")
        part.write_bytes(value)
        first = bundle / "final/train.vi.part000"
        original = first.read_bytes()
        first.write_bytes(value)
        part.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "hash/size"):
            wf.stage_data(code, bundle, self.base / "frozen")
        self.assertFalse((self.base / "frozen").exists())

    def test_original_bundle_without_chunks(self):
        code, bundle = self.dataset(split=False)
        self.assertEqual(wf.stage_data(code, bundle, self.base / "frozen")["status"], "VERIFIED_INPUTS")

    def test_partial_publication_is_not_reused(self):
        source, target = self.base / "source", self.base / "target"
        source.mkdir()
        (source / "cache_manifest.json").write_text("manifest")
        (source / "data.arrow").write_bytes(b"full cache")
        with patch.object(wf.shutil, "copytree", side_effect=OSError("interrupted upload")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                wf.publish_tree(source, target)
        self.assertFalse(target.exists())
        wf.publish_tree(source, target)
        (target / "data.arrow").write_bytes(b"corrupt")
        wf.publish_tree(source, target)
        self.assertEqual(wf.inventory(source), wf.inventory(target))
        self.assertTrue(list(self.base.glob(".incomplete-target-*")))

    def test_subprocess_real_stderr_is_logged_with_cwd(self):
        log = self.base / "failure.log"
        with self.assertRaisesRegex(RuntimeError, "exit 7") as error:
            wf.run_command([sys.executable, "-c", "import sys,os; print(os.getcwd()); print('REAL ERROR',file=sys.stderr); sys.exit(7)"], self.base, log)
        self.assertIn("REAL ERROR", str(error.exception))
        self.assertIn("REAL ERROR", log.read_text(encoding="utf-8"))
        self.assertIn(str(self.base), log.read_text(encoding="utf-8"))

    def test_experiment_identity_reused_and_changes_refused(self):
        first = self.workflow()
        second = wf.Workflow(first.settings, PROJECT, first.identity["source"])
        self.assertEqual(first.state["experiment_id"], second.state["experiment_id"])
        with self.assertRaisesRegex(ValueError, "different code"):
            wf.Workflow(first.settings, PROJECT, "changed-code")

    def test_incompatible_install_stops_before_cuda_or_data(self):
        flow = self.workflow()
        with patch.object(wf.venv.EnvBuilder, "create"), patch.object(wf, "run_command", side_effect=RuntimeError("No matching torch wheel for Python 3.13")) as run, patch.object(flow, "command") as probe:
            with self.assertRaisesRegex(RuntimeError, "Python 3.13"):
                flow.environment()
        self.assertEqual(run.call_count, 1)
        self.assertIn("--dry-run", run.call_args.args[0])
        probe.assert_not_called()

    def test_cuda_probe_failure_is_not_accepted_as_environment(self):
        flow = self.workflow()
        flow.python.parent.mkdir(parents=True)
        flow.python.write_bytes(b"fixture executable")
        freeze = flow.directory / "requirements-resolved.txt"
        freeze.write_text("fixture==1\n", encoding="utf-8")
        stamp = flow.python.parent.parent / "workflow_environment.json"
        wf.write_json(stamp, {"identity": {"python": sys.version,
            "lock": wf.digest(PROJECT / "nllb_lora/requirements-colab.lock"), "resolved_lock": wf.digest(freeze)}})
        with patch.object(wf.subprocess, "run") as pip, patch.object(flow, "command", side_effect=RuntimeError("CUDA unavailable")) as probe:
            pip.return_value.stdout = "fixture==1\n"
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                flow.environment()
        self.assertEqual(probe.call_count, 1)
        self.assertNotIn("environment_signature", flow.state)

    def test_cache_spec_matches_cli_and_corrupt_local_copy_restores_from_drive(self):
        flow = self.workflow()
        code, bundle = self.dataset()
        report = wf.stage_data(code, bundle, self.base / "fixture-frozen")
        api = wf.data_api(PROJECT)
        config = wf.read_json(flow.config)
        spec = api.cache_specification(config, report, 256, 256)
        flow.state["environment_signature"] = {"packages": {name: {"actual": importlib.metadata.version(name)} for name in (
            "transformers", "tokenizers", "datasets", "sentencepiece")}}
        flow.settings["drive_bundle"] = str(bundle)
        def command(name, args, python=None):
            if name == "cache":
                if not flow.cache.exists():
                    flow.cache.mkdir()
                    (flow.cache / "dataset.arrow").write_bytes(b"fixture arrow bytes")
                    wf.write_json(flow.cache / "cache_manifest.json", {"cache_key": api.cache_key(spec),
                        "specification": spec, "files": wf.inventory(flow.cache)})
                api.verify_cache(flow.cache, spec)
        with patch.object(wf, "stage_data", return_value=report), patch.object(flow, "command", side_effect=command):
            flow.prepare()
            original = wf.inventory(flow.cache)
            (flow.cache / "dataset.arrow").write_bytes(b"broken local cache")
            flow.prepare()
        self.assertEqual(original, wf.inventory(flow.cache))
        self.assertEqual(flow.state["cache_key"], api.cache_key(spec))
        self.assertTrue(list(flow.cache.parent.glob(".incomplete-cache-*")))

    def simulate_command(self, name, args, python=None):
        args = list(map(str, args))
        if "--run-dir" in args:
            run = Path(args[args.index("--run-dir") + 1])
            steps = int(args[args.index("--max-steps") + 1])
            self.assertNotIn("--stop-after-step", args)
            wf.write_json(run / "completion.json", {"reason": "budget_complete", "global_step": steps, "total_budget": steps, "experiment_complete": True})
            wf.write_json(run / "validation" / ("pilot-best" if name == "pilot" else "step-final") / "metrics.json", self.metric())
            wf.write_json(run / "best_model/config.json", {})
            if name == "benchmark":
                wf.write_json(run / "resource_benchmark.json", {"resource_probe_passed": True, "finite_metrics": True, "optimizer_present_during_eval": True, "validation_count": 5000})
            if name == "sanity":
                wf.write_json(run / "sanity_learning.json", {"finite_loss": True, "learning_diagnostic_improved": True})
            if name == "pilot":
                wf.write_json(run / "pilot_summary.json", {"status": "REQUIRES_MANUAL_QUALITY_REVIEW"})
        else:
            self.assertEqual(args[args.index("--split") + 1], "val")
            output = Path(args[args.index("--output-dir") + 1])
            wf.write_json(output / "metrics.json", dict(self.metric(), split="val"))
            for file in ("predictions.jsonl", "references.jsonl", "sources.jsonl"):
                (output / file).write_text('fixture\n', encoding="utf-8")

    def metric(self):
        return {"bleu": 1.0, "chrf": 2.0, "count": 5000, "reference_sha256": "reference", "generation": {"num_beams": 4}}

    def test_complete_pilot_sequence_and_repeat_skips_verified_phases(self):
        flow = self.workflow()
        with patch.object(flow, "command", side_effect=self.simulate_command) as command:
            result = flow.run()
            self.assertEqual([call.args[0] for call in command.call_args_list], ["smoke", "benchmark", "sanity", "pilot", "pilot-independent-val"])
            self.assertEqual(result["count"], 5000)
            self.assertEqual(result["fresh_runtime_resume_acceptance"], "NOT_ESTABLISHED_BY_RUN_ALL")
        with patch.object(flow, "command") as command:
            flow.run()
            command.assert_not_called()
        # A real corruption is detected, preserved and rerun instead of skipped.
        (flow.directory / "artifacts/smoke/completion.json").write_text("broken")
        with patch.object(flow, "command", side_effect=self.simulate_command) as command:
            flow.run()
            self.assertEqual([call.args[0] for call in command.call_args_list], ["smoke"])

    def test_incomplete_pilot_blocks_eval_and_later_commands(self):
        flow = self.workflow()
        def command(name, args, python=None):
            self.simulate_command(name, args, python)
            if name == "pilot":
                run = Path(args[args.index("--run-dir") + 1])
                wf.write_json(run / "completion.json", {"reason": "interrupted_at_checkpoint", "global_step": 500, "total_budget": 1000, "experiment_complete": False})
        with patch.object(flow, "command", side_effect=command) as commands:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                flow.run()
        self.assertNotIn("pilot-independent-val", [call.args[0] for call in commands.call_args_list])
        self.assertTrue((flow.directory / "failure.json").is_file())

    def test_direct_full_checks_smoke_then_full_and_final_test_without_review(self):
        settings = {"mode": "full", "experiment_id": "full-fixture", "readiness": None,
            "drive_output": str(self.base / "drive"), "local_work": str(self.base / "local"),
            "venv": str(self.base / "venv")}
        flow = wf.Workflow(settings, PROJECT, wf.fingerprint(wf.inventory(PROJECT, wf.CODE_FILES)))
        flow.state["cache_key"] = "cache-fixture"
        flow.save()

        def command(name, args, python=None):
            args = list(map(str, args))
            self.assertNotIn("--readiness", args)
            if name == "smoke":
                self.simulate_command(name, args, python)
            elif name == "full":
                run = Path(args[args.index("--run-dir") + 1])
                budget = int(args[args.index("--max-steps") + 1])
                self.assertEqual(budget, 85605)
                wf.write_json(run / "completion.json", {"reason": "budget_complete", "global_step": budget,
                    "total_budget": budget, "experiment_complete": True})
                wf.write_json(run / "validation/full-best/metrics.json", self.metric())
                wf.write_json(run / "best_model/config.json", {})
                wf.write_json(run / "full_summary.json", {})
            else:
                self.assertEqual(name, "full-test")
                self.assertEqual(args[args.index("--split") + 1], "test")
                output = Path(args[args.index("--output-dir") + 1])
                wf.write_json(output / "metrics.json", dict(self.metric(), split="test"))

        with patch.object(flow, "command", side_effect=command) as commands:
            result = flow.run()
        self.assertEqual([call.args[0] for call in commands.call_args_list], ["smoke", "full", "full-test"])
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["test_count"], 5000)
        with patch.object(flow, "command") as commands:
            flow.run()
            commands.assert_not_called()

    def test_failed_quick_check_prevents_full_training(self):
        flow = self.workflow()
        flow.settings["mode"] = "full"
        with patch.object(flow, "train_phase", side_effect=ValueError("smoke failed")) as phase:
            with self.assertRaisesRegex(ValueError, "smoke failed"):
                flow.run()
        phase.assert_called_once_with("smoke", 7)

    def checkpoint(self, flow, step, early_stop=False):
        config = wf.read_json(flow.config)
        config["training"].update(total_optimizer_steps=1000, warmup_steps=30)
        config["run"].update(phase="pilot", run_id=flow.state["experiment_id"] + "-pilot")
        packages = {name: {"actual": "fixture", "expected": None} for name in (
            "torch", "transformers", "datasets", "accelerate", "sacrebleu", "sentencepiece",
            "numpy", "tokenizers", "safetensors", "huggingface-hub", "psutil")}
        flow.state["environment_signature"] = {"python": sys.version, "packages": packages}
        flow.inputs = {"manifest_sha256": config["data"]["expected_manifest_sha256"]}
        hashes = {Path(name).name: wf.digest(PROJECT / name) for name in (
            "nllb_lora/train/train_nllb.py", "nllb_lora/train/prepare_data.py", "nllb_lora/eval/translation_metrics.py",
            "nllb_lora/train/sync_checkpoints.py", "nllb_lora/train/check_readiness.py", "nllb_lora/requirements-colab.lock")}
        manifest = {"config": config, "environment": {"python": sys.version.split()[0],
            "packages": {name: value["actual"] for name, value in packages.items()}},
            "data": flow.inputs, "code": {"source_sha256": hashes}, "cache_key": flow.state["cache_key"]}
        run = self.base / ("checkpoint-source-" + str(step))
        checkpoint = run / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"):
            wf.write_json(checkpoint / name, {})
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
            (checkpoint / name).write_bytes(b"fixture checkpoint bytes")
        wf.write_json(checkpoint / "trainer_state.json", {"global_step": step, "best_model_checkpoint": str(checkpoint), "best_metric": 2.0, "log_history": [],
            "stateful_callbacks": {"EarlyStoppingCallback": {"attributes": {"early_stopping_patience_counter": 3 if early_stop else 0}}}})
        wf.write_json(run / "resolved_config.json", config)
        wf.write_json(run / "run_manifest.json", manifest)
        ids, predictions = [f"val:{i}" for i in range(1, 5001)], ["translation"] * 5000
        metric = dict(self.metric(), step=step, official_full_validation=True,
            prediction_sha256=text_fingerprint(ids, predictions))
        for directory, value in ((run / "validation" / f"step-{step}", metric),
                                 (run / "validation/step-0", dict(metric, step=0))):
            wf.write_json(directory / "metrics.json", value)
            (directory / "predictions.jsonl").write_text("".join(json.dumps({"id": key, "text": text}) + "\n" for key, text in zip(ids, predictions)), encoding="utf-8")
        drive = flow.directory / "checkpoints/pilot"
        synchronize_checkpoint(checkpoint, drive, run)
        return drive, manifest

    def test_resume_uses_verified_checkpoint_and_fixed_budget(self):
        flow = self.workflow()
        drive, previous = self.checkpoint(flow, 500)
        before = wf.inventory(drive / "checkpoint-500")
        partial = drive / "checkpoint-750"
        partial.mkdir()
        (partial / "partial.txt").write_text("incomplete upload")
        with patch.object(flow, "command", side_effect=self.simulate_command) as commands:
            flow.train_phase("pilot", 1000)
        args = commands.call_args.args[1]
        self.assertIn("--resume-from-drive", args)
        self.assertEqual(args[args.index("--max-steps") + 1], "1000")
        self.assertEqual(before, wf.inventory(drive / "checkpoint-500"))
        self.assertTrue(list(drive.glob(".incomplete-checkpoint-750-*")))
        altered = copy.deepcopy(previous)
        altered["config"]["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "differs"):
            flow.check_resume_context(altered, "pilot", 1000)

    def test_completed_checkpoint_recovers_export_without_training(self):
        flow = self.workflow()
        drive, previous = self.checkpoint(flow, 1000)
        with patch.object(flow, "command") as commands:
            result = flow.train_phase("pilot", 1000)
        commands.assert_not_called()
        self.assertTrue((result / "best_model/export_manifest.json").is_file())
        self.assertTrue(wf.require_complete(result, 1000)["recovered_from_verified_checkpoint"])
        self.assertEqual(verify_checkpoint(drive / "checkpoint-1000")["step"], 1000)

    def test_early_stop_checkpoint_is_recovered_without_extra_updates(self):
        flow = self.workflow()
        self.checkpoint(flow, 500, early_stop=True)
        with patch.object(flow, "command") as commands:
            result = flow.train_phase("pilot", 1000)
        commands.assert_not_called()
        self.assertEqual(wf.require_complete(result, 1000)["reason"], "early_stopping")

if __name__ == "__main__":
    unittest.main()
