"""Tests for the trainer: length sampler, speed callbacks, ETA and final metrics."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

import torch
from datasets import Dataset
from transformers import AutoConfig, M2M100Config, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments

from nllb_lora.train.train_nllb import build_trainer, CheckpointSyncCallback, MetricArtifactsCallback, sanity_summary
from nllb_lora.train.prepare_data import load_tokenizer, write_json
from nllb_lora.train.sync_checkpoints import verify_checkpoint, choose_verified


class TrainerTests(unittest.TestCase):
    def test_length_sampler_keeps_all_pairs_and_is_deterministic_by_epoch(self):
        from types import SimpleNamespace
        from nllb_lora.train.train_nllb import SourceOnlySeq2SeqTrainer, pair_lengths
        rows = [{"input_ids": [3]*(i % 5 + 1), "labels": [4]*(i % 7 + 1)} for i in range(17)]
        data = Dataset.from_list(rows)
        self.assertEqual(list(pair_lengths(data)), [max(len(row["input_ids"]), len(row["labels"])) for row in rows])
        args = SimpleNamespace(group_by_length=True, train_batch_size=2, gradient_accumulation_steps=2,
                               data_seed=42, seed=42)
        fixture = SimpleNamespace(args=args, train_dataset=data)
        sampler = SourceOnlySeq2SeqTrainer._get_train_sampler(fixture, data)
        self.assertEqual(list(SourceOnlySeq2SeqTrainer._get_eval_sampler(fixture, data)), list(range(17)))
        first = list(sampler)
        self.assertEqual(sorted(first), list(range(17)))
        torch.rand(100)
        self.assertEqual(list(sampler), first)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertEqual(sorted(second), list(range(17)))
        self.assertNotEqual(first, second)
        restored = SourceOnlySeq2SeqTrainer._get_train_sampler(SimpleNamespace(args=args, train_dataset=data), data)
        restored.set_epoch(1)
        self.assertEqual(list(restored)[6:], second[6:])

    def test_full_speed_does_not_synchronize_each_update(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from nllb_lora.train.train_nllb import FullSpeedCallback
        with tempfile.TemporaryDirectory() as directory:
            callback = FullSpeedCallback(directory)
            args = SimpleNamespace(max_steps=85605, save_steps=2000, eval_steps=2000)
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize") as sync:
                for step in range(1, 101):
                    state = SimpleNamespace(global_step=step)
                    callback.on_step_begin(args, state, None)
                    callback.on_step_end(args, state, None)
                self.assertEqual(sync.call_count, 2)
            self.assertEqual(len(callback.steps), 1)

    def test_full_eta_includes_validation_save_and_sync(self):
        from types import SimpleNamespace
        from nllb_lora.train.train_nllb import FullSpeedCallback
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            callback = FullSpeedCallback(path / "local", path / "drive/performance_estimate.json")
            callback.steps = [1, 1, 1]
            callback.evaluations = [100]
            callback.saves = [10]
            args = SimpleNamespace(max_steps=85605, save_steps=2000, eval_steps=2000)
            callback.write(args, SimpleNamespace(global_step=2000))
            report = json.loads((path / "drive/performance_estimate.json").read_text())
            self.assertAlmostEqual(report["full_budget_hours_estimate"], (85605 + 46*100 + 43*10)/3600)
            self.assertEqual(report["seconds_per_update"], 1)

    def test_seven_steps_final_metrics_updates_and_no_target_leak(self):
        torch.set_num_threads(1)
        config = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
        config["generation"].update(max_new_tokens=4, num_beams=1)
        config["training"]["early_stopping_patience"] = 100
        tokenizer = load_tokenizer(config, local_files_only=True)
        original = AutoConfig.from_pretrained(config["model"]["name_or_path"], revision=config["model"]["revision"], local_files_only=True)
        torch.manual_seed(42)
        model = M2M100ForConditionalGeneration(M2M100Config(vocab_size=len(tokenizer), d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
            encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=32,
            decoder_start_token_id=original.decoder_start_token_id, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0))
        model.config.use_cache = False
        rows = []
        for index, (source, target) in enumerate((("你好", "Xin chào Việt Nam và các bạn!"), ("谢谢", "Cảm ơn tất cả những người bạn."))):
            rows.append({"id": f"val:{index+1}", "source": source, "target": target, **dict(tokenizer(source, text_target=target))})
        dataset = Dataset.from_list(rows)
        before = model.model.shared.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True, max_steps=7,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                learning_rate=0.001, warmup_steps=1, eval_strategy="steps", save_strategy="steps",
                eval_steps=3, save_steps=3, save_total_limit=3, predict_with_generate=True,
                load_best_model_at_end=True, metric_for_best_model="chrf", greater_is_better=True,
                report_to="none", disable_tqdm=True, logging_steps=1,
                restore_callback_states_from_checkpoint=True, save_safetensors=True)
            trainer = build_trainer(model, tokenizer, dataset, dataset, args, directory, config, official=False)
            from nllb_lora.train.train_nllb import SmokeHealthCallback
            trainer.add_callback(SmokeHealthCallback(trainer, directory))
            export_context = copy.deepcopy(config)
            export_context["run"]["phase"] = "tiny_fixture"
            export_context["training"]["total_optimizer_steps"] = 7
            write_json(Path(directory) / "resolved_config.json", export_context)
            write_json(Path(directory) / "run_manifest.json", {"phase": "tiny_cpu_test"})
            drive_temp = tempfile.TemporaryDirectory()
            self.addCleanup(drive_temp.cleanup)
            drive = Path(drive_temp.name) / "drive"
            trainer.add_callback(CheckpointSyncCallback(directory, drive))
            before_metrics = trainer.evaluate()
            result = trainer.train()
            self.assertTrue(json.loads((Path(directory) / "smoke_health.json").read_text())["passed"])
            self.assertEqual(trainer.state.global_step, 7)
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            self.assertFalse(torch.equal(before, model.model.shared.weight.detach()))
            self.assertEqual(choose_verified(drive).name, "checkpoint-7")
            self.assertEqual(verify_checkpoint(drive / "checkpoint-7")["step"], 7)
            from nllb_lora.eval.finalize_model import export_verified_best
            from transformers import AutoModelForSeq2SeqLM
            export = Path(directory) / "verified-export"
            export_verified_best(drive, export)
            restored_model = AutoModelForSeq2SeqLM.from_pretrained(str(export), local_files_only=True)
            self.assertTrue(torch.equal(restored_model.model.shared.weight, model.model.shared.weight))
            for step in (3, 6, 7):
                metrics = json.loads((Path(directory) / "validation" / f"step-{step}" / "metrics.json").read_text())
                self.assertEqual(metrics["step"], step)
                self.assertEqual(metrics["count"], 2)
                self.assertFalse(metrics["official_full_validation"])
                self.assertIn("bleu", metrics)
                state = json.loads((Path(directory) / f"checkpoint-{step}" / "trainer_state.json").read_text())
                self.assertEqual(state["global_step"], step)
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, MetricArtifactsCallback):
                    callback.scope = "sanity-best"
            report = sanity_summary(before_metrics, trainer.evaluate(), dataset["id"])
            selected_metric = json.loads((Path(directory) / "validation/sanity-best/metrics.json").read_text())
            self.assertEqual(selected_metric["step"], int(Path(trainer.state.best_model_checkpoint).name.rsplit("-", 1)[1]))
            self.assertEqual(selected_metric["training_global_step"], 7)
            from nllb_lora.eval.evaluate import generate_translations
            _, _, _, independent = generate_translations(restored_model, tokenizer, rows, config)
            for metric in ("bleu", "chrf"):
                self.assertAlmostEqual(independent[metric], selected_metric[metric], delta=1e-6)
            self.assertTrue(report["learning_diagnostic_improved"])
            self.assertEqual(report["status"], "REQUIRES_MANUAL_EXAMPLE_REVIEW")
            trainer.model.eval()
            batch = trainer.data_collator([{k: v for k, v in row.items() if k in ("input_ids", "attention_mask", "labels")} for row in rows])
            _, predictions, _ = trainer.prediction_step(trainer.model, batch, False)
            changed = {key: value.clone() for key, value in batch.items()}
            changed["labels"][:] = tokenizer.eos_token_id
            changed["decoder_input_ids"] = torch.full_like(changed["labels"], tokenizer.convert_tokens_to_ids("zho_Hans"))
            _, changed_predictions, _ = trainer.prediction_step(trainer.model, changed, False)
            self.assertTrue(torch.equal(predictions, changed_predictions))
            self.assertEqual(predictions.ndim, 2)
            self.assertTrue((predictions[:, 1] == tokenizer.convert_tokens_to_ids("vie_Latn")).all())


if __name__ == "__main__":
    unittest.main()
