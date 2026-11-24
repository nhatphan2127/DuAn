"""Tests for T4 speed-profile tuning, AMP skips and worker paths."""
import json
from pathlib import Path
import tempfile
import unittest

from nllb_lora.train.check_readiness import full_training_budget
from nllb_lora.train.profile_speed import estimate_full, profile_config, select_profile
from nllb_lora.train.profile_speed import choose_fastest, trial_configs, viable, tune, UpdateTimer
from unittest.mock import patch
import subprocess
from types import SimpleNamespace


class SpeedProfileTests(unittest.TestCase):
    def test_t4_tuning_preserves_method_and_does_not_select_unhelpful_grouping(self):
        from nllb_lora.train.prepare_data import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = json.loads(Path("configs/nllb_zh_vi.json").read_text())
            config = profile_config(original, select_profile("Tesla T4", 15 * 1024**3), True)
            write_json(root / "config.json", config)
            tested = []
            def run(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                row = {"status": "OK", "bf16_supported": True, "gpu": "Tesla T4"}
                if job["purpose"] == "train":
                    training = job["config"]["training"]
                    tested.append(training)
                    self.assertFalse(training["bf16"])
                    self.assertFalse(training["tf32"])
                    batch = training["per_device_train_batch_size"]
                    seconds = 2 / batch if training["gradient_checkpointing"] else 0.9
                    if training.get("group_by_length"):
                        seconds = 0.89
                    row = self.measured(seconds, batch * 10)
                    if batch == 4:
                        row = {"status": "FAILED", "error": "OutOfMemoryError"}
                if job["purpose"] == "eval":
                    row["batches"] = [dict(status="OK", seconds=seconds, headroom=0.3,
                        peak_reserved_bytes=peak, batch=batch, count=32) for batch, seconds, peak in ((1,20,10),(2,12,20),(4,11.8,40))]
                write_json(job["output"], row)
            with patch("nllb_lora.train.profile_speed.subprocess.run", side_effect=run):
                report = tune(root / "config.json", root / "cache", root / "speed-tuning", {})
            selected = report["selected_config"]
            self.assertEqual([row["per_device_train_batch_size"] for row in tested[:4]], [1,1,2,4])
            self.assertEqual(selected["training"]["per_device_train_batch_size"], 2)
            self.assertEqual(selected["training"]["per_device_eval_batch_size"], 2)
            self.assertFalse(selected["training"]["group_by_length"])
            self.assertEqual(selected["data"], config["data"])
            self.assertEqual(selected["training"]["eval_steps"], 2000)
            self.assertTrue(any(row.get("eligible") is False for row in report["trials"]))
            self.assertAlmostEqual(report["projected_full_hours"], (85605*0.9+46*5000/32*12)/3600)

    def test_no_stable_t4_configuration_stops_before_evaluation(self):
        from nllb_lora.train.prepare_data import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = profile_config(json.loads(Path("configs/nllb_zh_vi.json").read_text()),
                select_profile("Tesla T4", 15*1024**3), True)
            write_json(root / "config.json", config)
            def run(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                self.assertNotEqual(job["purpose"], "eval")
                row = {"status": "OK", "bf16_supported": False} if job["purpose"] == "hardware" else {
                    "status": "FAILED", "error": "OutOfMemoryError"}
                write_json(job["output"], row)
            with patch("nllb_lora.train.profile_speed.subprocess.run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "No stable"):
                    tune(root / "config.json", root / "cache", root / "speed-tuning", {})
            self.assertEqual(json.loads((root / "speed_tuning.json").read_text())["status"], "NO_STABLE_PROFILE")

    def test_amp_skips_are_not_counted_and_nonfinite_measurement_is_rejected(self):
        timer = UpdateTimer()
        timer.observe(1, True, False)
        for _ in range(3):
            self.assertFalse(timer.observe(1, False, True))
        for _ in range(9):
            self.assertFalse(timer.observe(2, False, True))
        self.assertTrue(timer.observe(2, False, True))
        self.assertEqual((timer.attempts, timer.successful, len(timer.times)), (14, 13, 10))
        with self.assertRaises(FloatingPointError):
            timer.observe(1, True, True)
        timer = UpdateTimer()
        for _ in range(9):
            timer.observe(1, True, False)
        with self.assertRaises(FloatingPointError):
            timer.observe(1, True, False)
        with self.assertRaises(FloatingPointError):
            UpdateTimer().observe(1, False, False)

    def test_t4_profiles_never_enable_bf16_tf32_or_large_initial_batches(self):
        original = json.loads(Path("configs/nllb_zh_vi.json").read_text())
        config = profile_config(original, select_profile("Tesla T4", 15 * 1024**3), True)
        self.assertEqual((config["training"]["eval_steps"], config["training"]["save_steps"]), (2000, 2000))
        trials = list(trial_configs(config, True))
        self.assertEqual([row["training"]["per_device_train_batch_size"] for row in trials], [1, 1, 2, 4])
        for trial in trials:
            training = trial["training"]
            self.assertFalse(training["bf16"])
            self.assertFalse(training["tf32"])
            self.assertTrue(training["fp16"])
            self.assertTrue(training["gradient_checkpointing"])
            self.assertEqual(training["per_device_train_batch_size"] * training["gradient_accumulation_steps"], 32)

    def test_worker_train_and_eval_paths_with_tiny_cpu_model_and_mocked_cuda_probes(self):
        import torch
        from datasets import Dataset, DatasetDict
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, M2M100Config, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments
        from nllb_lora.train.profile_speed import worker
        torch.set_num_threads(1)
        vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, "zho_Hans": 3, "vie_Latn": 4, "hello": 5}
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
            unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        config = json.loads(Path("configs/nllb_zh_vi.json").read_text())
        config["training"].update(per_device_train_batch_size=8, gradient_accumulation_steps=4,
            fp16=False, bf16=False, gradient_checkpointing=False)
        config["generation"].update(max_new_tokens=3, num_beams=1, use_cache=True)
        rows = [{"id": f"val:{i}", "source": "hello", "target": "hello",
                 "input_ids": [3, 5, 2], "attention_mask": [1, 1, 1], "labels": [4, 5, 2]} for i in range(416)]
        data = DatasetDict(train=Dataset.from_list(rows), val=Dataset.from_list(rows[:128]))
        def model(*args, **kwargs):
            return M2M100ForConditionalGeneration(M2M100Config(vocab_size=6, d_model=8,
                encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
                encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=512,
                decoder_start_token_id=2, pad_token_id=1, eos_token_id=2, dropout=0.0))
        def arguments(config, output, steps):
            return Seq2SeqTrainingArguments(output_dir=str(output), use_cpu=True, max_steps=steps,
                per_device_train_batch_size=8, gradient_accumulation_steps=4,
                learning_rate=2e-5, dataloader_pin_memory=False, report_to="none", predict_with_generate=True)
        with tempfile.TemporaryDirectory() as directory:
            for purpose, t4 in (("train", False), ("eval", False), ("eval", True)):
                trial = json.loads(json.dumps(config))
                if t4:
                    trial["training"].update(gpu_profile="t4", fp16=True, bf16=False, tf32=False)
                output = Path(directory) / (purpose + str(t4) + ".json")
                with patch("nllb_lora.train.train_nllb.load_training_model", side_effect=model), \
                     patch("nllb_lora.train.train_nllb.training_arguments", side_effect=arguments), \
                     patch("nllb_lora.train.prepare_data.load_tokenizer", return_value=tokenizer), \
                     patch("datasets.load_from_disk", return_value=data), \
                     patch("torch.cuda.is_available", return_value=True), \
                     patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                     patch("torch.cuda.empty_cache"), patch("torch.cuda.max_memory_reserved", return_value=10), \
                     patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=100)):
                    worker({"config": trial, "cache": directory, "purpose": purpose, "output": str(output)})
                result = json.loads(output.read_text())
                self.assertTrue(viable(result), result)
                if purpose == "eval":
                    self.assertEqual([row["count"] for row in result["batches"]], [32]*4 if t4 else [128]*3)
                    self.assertTrue(all("eval_loss" in row["metrics"] for row in result["batches"]))

    def measured(self, seconds=1.0, peak=10, **extra):
        return {"status": "OK", "updates": 13, "measured_updates": 10, "finite_gradients": True,
            "optimizer_present": True, "headroom": 0.2, "seconds_per_update": seconds,
            "peak_reserved_bytes": peak, **extra}

    def test_measured_selection_requires_stability_and_prefers_memory_within_five_percent(self):
        a, b = self.measured(1, 20), self.measured(1.04, 10)
        self.assertEqual(choose_fastest([a, b]), b)
        self.assertEqual(choose_fastest([a, self.measured(1.06, 5)]), a)
        for field, value in (("updates", 12), ("finite_gradients", False), ("optimizer_present", False),
                             ("headroom", 0.09), ("seconds_per_update", float("nan"))):
            self.assertFalse(viable({**a, field: value}))
        with self.assertRaises(ValueError):
            choose_fastest([{"status": "FAILED", "error": "OutOfMemoryError"}])

    def test_tuning_candidates_only_change_execution_settings(self):
        original = json.loads(Path("configs/nllb_zh_vi.json").read_text())
        for bf16 in (True, False):
            candidates = list(trial_configs(original, bf16))
            self.assertEqual(len(candidates), 4)
            for candidate in candidates:
                self.assertEqual(candidate["data"], original["data"])
                self.assertEqual(candidate["model"], original["model"])
                training = candidate["training"]
                self.assertEqual(training["per_device_train_batch_size"] * training["gradient_accumulation_steps"], 32)
                self.assertEqual(training["learning_rate"], original["training"]["learning_rate"])
                self.assertEqual(training["num_train_epochs"], 3)
                self.assertTrue(candidate["generation"]["use_cache"])
        self.assertNotIn("use_cache", original["generation"])

    def test_orchestration_isolates_oom_and_keeps_completed_eval_on_timeout(self):
        from nllb_lora.train.prepare_data import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path("configs/nllb_zh_vi.json").read_text())
            config = profile_config(config, select_profile("RTX PRO 6000 Blackwell", 95*1024**3), True)
            write_json(root / "config.json", config)
            def run(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                result = {"status": "OK", "bf16_supported": True}
                if job["purpose"] == "train":
                    batch = job["config"]["training"]["per_device_train_batch_size"]
                    result = self.measured(8 / batch, batch)
                    if batch == 32:
                        result = {"status": "FAILED", "error": "OutOfMemoryError"}
                if job["purpose"] == "eval":
                    result = {"status": "OK", "batches": [dict(status="OK", seconds=10,
                        headroom=0.2, peak_reserved_bytes=20, batch=8, count=128)]}
                write_json(job["output"], result)
                if job["purpose"] == "eval":
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            with patch("nllb_lora.train.profile_speed.subprocess.run", side_effect=run):
                report = tune(root / "config.json", root / "cache", root / "speed-tuning", {"run": "fixture"})
            self.assertEqual(report["status"], "MEASURED")
            self.assertEqual(report["selected_config"]["training"]["per_device_train_batch_size"], 16)
            self.assertEqual(report["selected_config"]["training"]["per_device_eval_batch_size"], 8)
            self.assertTrue(any(row.get("error") == "OutOfMemoryError" for row in report["trials"]))

    def test_time_budget_keeps_measured_baseline_and_does_not_open_more_trials(self):
        from nllb_lora.train.prepare_data import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path("configs/nllb_zh_vi.json").read_text())
            write_json(root / "config.json", config)
            clock = [0]
            purposes = []
            def run(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                purposes.append(job["purpose"])
                row = {"status": "OK", "bf16_supported": False}
                if job["purpose"] == "train":
                    row = self.measured()
                    clock[0] = 700
                if job["purpose"] == "eval":
                    row["batches"] = [dict(status="OK", seconds=5, headroom=0.5,
                                           peak_reserved_bytes=10, batch=8, count=128)]
                write_json(job["output"], row)
            with patch("nllb_lora.train.profile_speed.subprocess.run", side_effect=run), \
                 patch("nllb_lora.train.profile_speed.time.monotonic", side_effect=lambda: clock[0]):
                report = tune(root / "config.json", root / "cache", root / "speed-tuning", {})
            self.assertEqual(purposes, ["hardware", "train", "eval"])
            self.assertEqual(report["status"], "MEASURED")

    def test_gpu_presets_preserve_batch_budget_and_reference_config(self):
        original = json.loads(Path("configs/nllb_zh_vi.json").read_text())
        for gpu, gib, batch in (("NVIDIA A100-SXM4-40GB", 40, 4),
            ("NVIDIA A100-SXM4-80GB", 80, 8), ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 95, 8),
            ("Tesla T4", 15, 1)):
            profile = select_profile(gpu, gib * 1024**3)
            self.assertEqual(profile["per_device_train_batch_size"], batch)
            self.assertEqual(profile["gradient_accumulation_steps"] * batch, 32)
            for full, interval in ((False, 500), (True, 2000)):
                config = profile_config(original, profile, full)
                self.assertEqual(config["training"]["save_steps"], interval)
                self.assertEqual(full_training_budget(913094, batch, 32 // batch, 3), 85605)
                self.assertEqual(config["generation"], original["generation"])
                self.assertEqual(config["data"], original["data"])
        self.assertEqual(original["training"]["per_device_train_batch_size"], 1)

    def test_overrides_rejected_and_timing_counts_final_validation(self):
        for batch in (3, 0, 64, True):
            with self.assertRaises(ValueError):
                select_profile("NVIDIA A100", 40 * 1024**3, train_batch=batch)
        with self.assertRaises(ValueError):
            select_profile("Tesla T4", 12 * 1024**3)
        report = estimate_full({"train_step_seconds": 100, "optimizer_updates": 100,
            "evaluation_seconds": 100}, 85605, 2000)
        self.assertEqual(report["checkpoint_evaluations"], 43)
        self.assertEqual(report["all_validation_passes"], 46)

    def test_memory_shuffle_keeps_saved_arrow_cache_unchanged(self):
        from datasets import Dataset, load_from_disk

        data = Dataset.from_list([{"id": f"train:{i}"} for i in range(10)])
        expected = data.shuffle(seed=42)["id"]
        with tempfile.TemporaryDirectory() as directory:
            data.save_to_disk(directory)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()}
            selected = load_from_disk(directory).shuffle(seed=42, keep_in_memory=True)
            self.assertEqual(selected["id"], expected)
            self.assertEqual({p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()}, before)
