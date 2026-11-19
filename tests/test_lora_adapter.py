"""Tests for LoRA adapter updates, export, resume and early stopping."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

from nllb_lora.train.lora_adapter import apply_lora, verify_adapter
from nllb_lora.train.train_nllb import load_training_model
from nllb_lora.train.prepare_data import write_json
from nllb_lora.train.sync_checkpoints import verify_checkpoint, restore_verified, resume_signature


class LoraTests(unittest.TestCase):
    def config(self):
        return json.loads(Path("configs/nllb_lora_zh_vi.json").read_text(encoding="utf-8"))

    def tiny(self):
        return M2M100ForConditionalGeneration(M2M100Config(vocab_size=32, d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
            encoder_ffn_dim=16, decoder_ffn_dim=16, dropout=0.0))

    def test_official_cadence_arguments_and_t4_trials_keep_lora(self):
        from nllb_lora.train.train_nllb import training_arguments
        from nllb_lora.train.profile_speed import profile_config, select_profile, trial_configs

        config = profile_config(self.config(), select_profile("Tesla T4", 15 * 1024**3), full=True)
        with tempfile.TemporaryDirectory() as directory:
            args = training_arguments(config, directory, 85605)
            self.assertEqual((args.eval_steps, args.save_steps), (10000, 2000))
            self.assertFalse(args.load_best_model_at_end)
            self.assertEqual(args.gradient_checkpointing_kwargs, {"use_reentrant": False})
        trials = list(trial_configs(config, True))
        self.assertEqual({t["training"]["per_device_train_batch_size"] for t in trials}, {1, 2, 4, 8, 16, 32})
        for trial in trials:
            self.assertEqual(trial["model"], config["model"])
            self.assertEqual(trial["training"]["learning_rate"], 1e-4)
            self.assertEqual(trial["training"]["per_device_train_batch_size"] * trial["training"]["gradient_accumulation_steps"], 32)
            self.assertFalse(trial["training"]["bf16"] or trial["training"]["tf32"])

    def test_only_adapter_updates_export_reload_and_revision_guard(self):
        torch.set_num_threads(1)
        config = self.config()
        base = self.tiny()
        original = copy.deepcopy(base)
        model = apply_lora(base, config)
        self.assertEqual(model.lora_inventory["module_count"], 6)
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        batch = {"input_ids": torch.tensor([[4, 5, 2]]), "labels": torch.tensor([[6, 7, 2]])}
        model(**batch).loss.backward()
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(frozen[name], p), name)
        self.assertTrue(any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if "lora_B" in n))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            write_json(Path(directory) / "base_model.json", {"model": config["model"]})
            verify_adapter(directory, config["model"])
            with patch("nllb_lora.train.train_nllb.AutoModelForSeq2SeqLM.from_pretrained", return_value=original) as loader:
                restored = load_training_model(config, local_files_only=True, model_path=directory)
            self.assertEqual(loader.call_args.kwargs["revision"], config["model"]["revision"])
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(**batch).logits, restored(**batch).logits, rtol=0, atol=0)
            wrong = copy.deepcopy(config["model"])
            wrong["revision"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_adapter(directory, wrong)
        with self.assertRaisesRegex(ValueError, "adapter export"):
            load_training_model(config, model_path="full-checkpoint")

    def test_fresh_process_lora_resume_before_validation_and_grouped(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = dict(os.environ, USE_TORCH="1", USE_TF="0", HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")

                def run(name, drive, *extra):
                    command = [sys.executable, "-m", "tests.helpers.nllb_resume_worker", "--lora",
                        "--run", str(root / name), "--drive", str(root / drive), *extra]
                    if grouped:
                        command.append("--group-by-length")
                    result = subprocess.run(command, capture_output=True, text=True, env=env, encoding="utf-8", timeout=180)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    return json.loads((root / name / "comparison.json").read_text())

                continuous = run("continuous", "continuous-drive")
                initial = run("initial", "resume-drive", "--stop", "2")
                checkpoint = root / "resume-drive/checkpoint-2"
                verify_checkpoint(checkpoint)
                self.assertFalse((checkpoint / "validation").exists())
                self.assertIsNone(json.loads((checkpoint / "trainer_state.json").read_text())["best_model_checkpoint"])
                resumed = run("resumed", "resume-drive", "--resume")
                for key in ("step", "weights", "optimizer", "scheduler", "rng", "early_stopping", "baseline_prediction_sha256"):
                    self.assertEqual(continuous[key], resumed[key], key)
                self.assertEqual(continuous["training_order"][len(initial["training_order"]):], resumed["training_order"])
                self.assertTrue(initial["base_frozen"] and resumed["base_frozen"])
                self.assertNotEqual(initial["weights"], resumed["weights"])
                probe = json.loads((root / "resumed/resume_verification.json").read_text())
                self.assertTrue(probe["adapter_changed"])
                self.assertTrue(probe["loaded_state_checks"])
                for step in (2, 4, 6, 7):
                    verify_checkpoint(root / f"resume-drive/checkpoint-{step}")
                self.assertEqual({p.name for p in (root / "resumed/validation").glob("step-*")}, {"step-0", "step-6", "step-7"})
                from nllb_lora.eval.finalize_model import export_verified_best, verify_inference_export
                export = root / "export"
                export_verified_best(root / "resume-drive", export)
                verify_inference_export(export)
                self.assertTrue((export / "adapter_model.safetensors").is_file())
                self.assertFalse((export / "model.safetensors").exists())
                provenance = json.loads((export / "provenance.json").read_text())
                best = root / "resume-drive" / provenance["best_checkpoint"]
                self.assertEqual((export / "adapter_model.safetensors").read_bytes(), (best / "adapter_model.safetensors").read_bytes())
                manifest = json.loads((checkpoint / "run_manifest.json").read_text())
                incompatible = resume_signature(manifest)
                incompatible["model"]["lora"]["r"] = 8
                with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                    restore_verified(root / "resume-drive", root / "wrong-rank", incompatible)

    def test_early_stop_saves_final_state_and_reloads_evaluated_best(self):
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Seq2SeqTrainingArguments
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        from nllb_lora.train.train_nllb import build_trainer, CheckpointSyncCallback, MetricArtifactsCallback
        from nllb_lora.eval.finalize_model import export_verified_best

        torch.set_num_threads(1)
        config = self.config()
        config["training"].update(eval_steps=3, save_steps=1, total_optimizer_steps=7,
            early_stopping_patience=1, fp16=False)
        config["generation"].update(max_new_tokens=3, num_beams=1)
        config["run"]["phase"] = "tiny_fixture"
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {"<unk>": 0, "<pad>": 1, "</s>": 2, "zho_Hans": 3, "vie_Latn": 4, "hello": 5}, unk_token="<unk>")),
            unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        data = Dataset.from_list([{"id": "val:1", "target": "hello", "input_ids": [3, 5, 2],
            "attention_mask": [1, 1, 1], "labels": [4, 5, 2]}])
        with tempfile.TemporaryDirectory() as directory:
            run, drive = Path(directory) / "run", Path(directory) / "drive"
            write_json(run / "resolved_config.json", config)
            write_json(run / "run_manifest.json", {"config": config})
            model = apply_lora(self.tiny(), config)
            args = Seq2SeqTrainingArguments(output_dir=str(run), use_cpu=True, max_steps=7,
                eval_strategy="steps", save_strategy="steps", eval_steps=3, save_steps=1,
                load_best_model_at_end=False, predict_with_generate=True, metric_for_best_model="chrf",
                per_device_train_batch_size=1, per_device_eval_batch_size=1, report_to="none", disable_tqdm=True)
            trainer = build_trainer(model, tokenizer, data, data, args, run, config, official=False)
            artifacts = next(c for c in trainer.callback_handler.callbacks if isinstance(c, MetricArtifactsCallback))
            original_metrics = trainer.compute_metrics

            def controlled_scores(prediction):
                result = original_metrics(prediction)
                result["chrf"] = artifacts.report["chrf"] = 10.0 if trainer.state.global_step == 3 else 9.0
                return result

            trainer.compute_metrics = controlled_scores
            trainer.add_callback(CheckpointSyncCallback(run, drive))
            trainer.train()
            self.assertEqual(trainer.state.global_step, 6)
            self.assertEqual(Path(trainer.state.best_model_checkpoint).name, "checkpoint-3")
            self.assertEqual(trainer.state.best_metric, 10.0)
            verify_checkpoint(drive / "checkpoint-6")
            expected = load_file(str(drive / "checkpoint-3/adapter_model.safetensors"))
            for name, tensor in get_peft_model_state_dict(model).items():
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
            export = Path(directory) / "export"
            export_verified_best(drive, export)
            provenance = json.loads((export / "provenance.json").read_text())
            self.assertEqual(provenance["completion_reason"], "early_stopping")
            self.assertEqual(provenance["best_weight_step"], 3)


if __name__ == "__main__":
    unittest.main()
