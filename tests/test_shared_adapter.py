"""Tests for the shared zh<->vi adapter: config, cache, prompts, trainer and notebook."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoConfig, M2M100Config, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments

import test_prepare_data
from nllb_lora.eval.translation_metrics import make_direction_compute_metrics, scoring_options, text_fingerprint
from nllb_lora.train.sync_checkpoints import expected_validation_ids
from nllb_lora.train.train_nllb import build_trainer, decoder_prompt, target_by_source
from nllb_lora.train.check_readiness import full_training_budget
from nllb_lora.train.prepare_data import (balanced_indices, direction_specs, is_shared, load_tokenizer,
    prepare_cache, tokenizer_contract, train_example_count, validation_layout)

PROJECT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((PROJECT / name).read_text(encoding="utf-8"))


class SharedConfigTests(unittest.TestCase):
    def test_shared_differs_from_zh_vi_only_in_shared_fields(self):
        single, shared = read("configs/nllb_lora_zh_vi.json"), read("configs/nllb_lora_shared.json")
        expected = copy.deepcopy(single)
        expected["experiment"] = "nllb_600m_zhvi_shared_lora_r16_qv"
        expected["model"]["directions"] = [
            {"name": "zh-vi", "source_language": "zho_Hans", "target_language": "vie_Latn"},
            {"name": "vi-zh", "source_language": "vie_Latn", "target_language": "zho_Hans"}]
        expected["generation"].update(forced_bos_language=None, target_selection="decoder_prompt_from_source_language")
        expected["training"]["eval_steps"] = 20000
        expected["evaluation"].update(every_optimizer_updates=20000, aggregate="mean", scoring={
            "zh-vi": {"bleu_tokenize": "13a", "text_normalization": None},
            "vi-zh": {"bleu_tokenize": "zh", "text_normalization": "nfkc"}})
        expected["run"]["cache_fingerprint"] = None
        expected["run"]["required_before_execution"] = shared["run"]["required_before_execution"]
        self.assertEqual(shared, expected)
        self.assertEqual(shared["run"]["required_before_execution"][:-1], single["run"]["required_before_execution"])

    def test_directions_budget_and_layout(self):
        shared = read("configs/nllb_lora_shared.json")
        directions = direction_specs(shared)
        self.assertEqual([(d["name"], d["swap"]) for d in directions], [("zh-vi", False), ("vi-zh", True)])
        self.assertTrue(is_shared(shared))
        self.assertEqual(train_example_count(shared), 2 * 913094)
        training = shared["training"]
        self.assertEqual(full_training_budget(train_example_count(shared), training["per_device_train_batch_size"],
            training["gradient_accumulation_steps"], training["num_train_epochs"]), 171207)
        ids = validation_layout(shared)
        self.assertEqual((len(ids), ids[0], ids[4999], ids[5000], ids[-1]), (10000, "val:1", "val:5000", "val:1@vi-zh", "val:5000@vi-zh"))
        self.assertEqual(expected_validation_ids(shared), ids)
        self.assertEqual(scoring_options(shared, "vi-zh"), {"bleu_tokenize": "zh", "normalization": "nfkc"})
        self.assertEqual(scoring_options(shared, "zh-vi"), {"bleu_tokenize": "13a", "normalization": None})

    def test_single_direction_configs_keep_legacy_behavior(self):
        for name in ("configs/nllb_lora_zh_vi.json", "configs/nllb_lora_vi_zh.json"):
            config = read(name)
            self.assertFalse(is_shared(config))
            self.assertEqual(direction_specs(config)[0]["name"], None)
            self.assertEqual(train_example_count(config), 913094)
            self.assertEqual(validation_layout(config), [f"val:{i}" for i in range(1, 5001)])
            self.assertEqual(expected_validation_ids({}), [f"val:{i}" for i in range(1, 5001)])

    def test_invalid_directions_rejected(self):
        config = read("configs/nllb_lora_shared.json")
        for mutate in (lambda m: m["directions"].reverse(),
                       lambda m: m["directions"].append(dict(m["directions"][0], name="again-zh")),
                       lambda m: m["directions"][1].update(name="zh-vi"),
                       lambda m: m["directions"][1].update(target_language="eng_Latn")):
            candidate = copy.deepcopy(config)
            mutate(candidate["model"])
            with self.subTest(), self.assertRaises(ValueError):
                direction_specs(candidate)

    def test_balanced_indices_cover_every_direction(self):
        shared = read("configs/nllb_lora_shared.json")
        indices = balanced_indices(shared, "val", 16)
        self.assertEqual(indices, list(range(8)) + list(range(5000, 5008)))
        self.assertEqual(balanced_indices(read("configs/nllb_lora_zh_vi.json"), "train", 64), list(range(64)))


class SharedCacheTests(unittest.TestCase):
    pin_manifest = test_prepare_data.DataTests.pin_manifest

    def setUp(self):
        test_prepare_data.DataTests.setUp(self)
        self.config["model"] = read("configs/nllb_lora_shared.json")["model"]
        self.cache = Path(self.temp.name) / "cache"

    def test_both_directions_are_cached_with_own_language_prefixes(self):
        from datasets import load_from_disk

        manifest = prepare_cache(self.config, self.root, self.cache, 512, 512, True, 1)
        data = load_from_disk(str(self.cache / "dataset"))
        self.assertEqual((len(data["train"]), len(data["val"])), (4, 4))
        self.assertEqual(data["val"]["id"], ["val:1", "val:2", "val:1@vi-zh", "val:2@vi-zh"])
        self.assertEqual(data["val"]["direction"], ["zh-vi", "zh-vi", "vi-zh", "vi-zh"])
        zh, vi = 256200, 256193  # zho_Hans, vie_Latn
        rows = data["val"]
        self.assertEqual((rows[0]["input_ids"][0], rows[0]["labels"][0]), (zh, vi))
        self.assertEqual((rows[2]["input_ids"][0], rows[2]["labels"][0]), (vi, zh))
        self.assertEqual((rows[0]["source"], rows[0]["target"]), ("你好", "Xin chào"))
        self.assertEqual((rows[2]["source"], rows[2]["target"]), ("Xin chào", "你好"))
        self.assertEqual(manifest["status"], "COMPLETE_CACHE")
        from nllb_lora.train.prepare_data import verify_cache
        self.assertEqual(prepare_cache(self.config, self.root, self.cache, 512, 512, True)["cache_key"], manifest["cache_key"])
        verify_cache(self.cache, manifest["specification"])

    def test_cache_key_differs_from_single_direction(self):
        single = copy.deepcopy(self.config)
        single["model"] = read("configs/nllb_lora_zh_vi.json")["model"]
        other = Path(self.temp.name) / "other"
        first = prepare_cache(self.config, self.root, self.cache, 512, 512, True)["cache_key"]
        self.assertNotEqual(first, prepare_cache(single, self.root, other, 512, 512, True)["cache_key"])


class DirectionMetricsTests(unittest.TestCase):
    class Tokenizer:
        pad_token_id, eos_token_id = 1, 2

        def batch_decode(self, rows, skip_special_tokens=True):
            return [" ".join(str(int(x)) for x in row if x > 2) for row in rows]

    ids = ["val:1", "val:2", "val:1@vi-zh", "val:2@vi-zh"]
    refs = ["10 11 12 13 14 15", "20 21 22 23 24 25", "30 31 32 33 34 35", "40 41 42 43 44 45"]
    directions = ["zh-vi", "zh-vi", "vi-zh", "vi-zh"]
    options = {"zh-vi": {"bleu_tokenize": "13a", "normalization": None}, "vi-zh": {"bleu_tokenize": "13a", "normalization": None}}

    def predictions(self, rows):
        width = max(len(r.split()) for r in rows) + 2
        generated = np.full((len(rows), width), 1)
        for index, text in enumerate(rows):
            tokens = [2] + [int(x) for x in text.split()]
            generated[index, :len(tokens)] = tokens
        return SimpleNamespace(predictions=generated)

    def test_mean_over_directions_and_fingerprints(self):
        captured = []
        compute = make_direction_compute_metrics(self.Tokenizer(), self.ids, self.refs, self.directions, self.options,
            lambda ids, predictions, report: captured.append((ids, predictions, report)), generation_limit=50)
        wrong = list(self.refs)
        wrong[2] = "1 2 3 4 5 6"
        wrong[3] = "7 8 9 10 11 12"
        metrics = compute(self.predictions(wrong))
        report = captured[0][2]
        self.assertAlmostEqual(metrics["bleu_zh-vi"], 100)
        self.assertLess(metrics["bleu_vi-zh"], 100)
        self.assertAlmostEqual(metrics["bleu"], (metrics["bleu_zh-vi"] + metrics["bleu_vi-zh"]) / 2)
        self.assertAlmostEqual(metrics["chrf"], (metrics["chrf_zh-vi"] + metrics["chrf_vi-zh"]) / 2)
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["reference_sha256"], text_fingerprint(self.ids, self.refs))
        self.assertEqual(set(report["directions"]), {"zh-vi", "vi-zh"})
        self.assertEqual(report["directions"]["vi-zh"]["count"], 2)
        self.assertEqual(report["at_generation_limit"], 0)

    def test_rejects_missing_policy_or_wrong_counts(self):
        with self.assertRaises(ValueError):
            make_direction_compute_metrics(self.Tokenizer(), self.ids, self.refs, self.directions, {"zh-vi": self.options["zh-vi"]})
        compute = make_direction_compute_metrics(self.Tokenizer(), self.ids, self.refs, self.directions, self.options)
        with self.assertRaises(ValueError):
            compute(self.predictions(self.refs[:2]))


class DecoderPromptTests(unittest.TestCase):
    def test_prompt_depends_only_on_source_language_token(self):
        table = {7: 8, 8: 7}
        prompt = decoder_prompt(torch.tensor([[7, 1, 2], [8, 5, 2]]), table, 2)
        self.assertEqual(prompt.tolist(), [[2, 8], [2, 7]])
        with self.assertRaisesRegex(ValueError, "not a configured source language"):
            decoder_prompt(torch.tensor([[9, 1]]), table, 2)

    def test_tokenizer_table_and_contract_cover_both_directions(self):
        config = read("configs/nllb_lora_shared.json")
        tokenizer = load_tokenizer(config, local_files_only=True)
        self.assertEqual(target_by_source(tokenizer, config), {256200: 256193, 256193: 256200})
        self.assertIsNone(target_by_source(tokenizer, read("configs/nllb_lora_zh_vi.json")))
        contract = tokenizer_contract(config, local_files_only=True)
        self.assertEqual([(d["source_language_id"], d["target_language_id"]) for d in contract["directions"]],
                         [(256200, 256193), (256193, 256200)])
        self.assertEqual((contract["source_language_id"], contract["target_language_id"]), (256200, 256193))


class SharedTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = read("configs/nllb_lora_shared.json")
        self.config["generation"].update(max_new_tokens=4, num_beams=1)
        self.config["training"]["early_stopping_patience"] = 100
        self.tokenizer = load_tokenizer(self.config, local_files_only=True)
        original = AutoConfig.from_pretrained(self.config["model"]["name_or_path"], revision=self.config["model"]["revision"], local_files_only=True)
        torch.manual_seed(42)
        self.model = M2M100ForConditionalGeneration(M2M100Config(vocab_size=len(self.tokenizer), d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
            encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=32,
            decoder_start_token_id=original.decoder_start_token_id, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0))
        self.model.config.use_cache = False
        rows = []
        for position, direction in enumerate(direction_specs(self.config)):
            tokenizer = load_tokenizer(self.config, True, direction)
            for index, (zh, vi) in enumerate((("你好", "Xin chào Việt Nam và các bạn!"), ("谢谢", "Cảm ơn tất cả những người bạn."))):
                source, target = (vi, zh) if direction["swap"] else (zh, vi)
                rows.append({"id": f"val:{index + 1}" + ("@vi-zh" if position else ""), "direction": direction["name"],
                             "source": source, "target": target, **dict(tokenizer(source, text_target=target))})
        self.rows = rows
        self.dataset = Dataset.from_list(rows)

    def arguments(self, directory):
        return Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True, max_steps=3,
            per_device_train_batch_size=2, per_device_eval_batch_size=3, learning_rate=0.001, warmup_steps=1,
            eval_strategy="steps", save_strategy="steps", eval_steps=3, save_steps=3, save_total_limit=1,
            predict_with_generate=True, load_best_model_at_end=True, metric_for_best_model="chrf",
            greater_is_better=True, report_to="none", disable_tqdm=True, logging_steps=1, save_safetensors=True)

    def test_validation_scores_each_direction_and_forces_correct_target_language(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_trainer(self.model, self.tokenizer, self.dataset, self.dataset, self.arguments(directory),
                                    directory, self.config, official=False)
            self.assertIsNone(self.model.generation_config.forced_bos_token_id)
            metrics = trainer.evaluate()
            for key in ("eval_bleu", "eval_chrf", "eval_bleu_zh-vi", "eval_chrf_zh-vi", "eval_bleu_vi-zh", "eval_chrf_vi-zh"):
                self.assertIn(key, metrics)
            self.assertAlmostEqual(metrics["eval_chrf"], (metrics["eval_chrf_zh-vi"] + metrics["eval_chrf_vi-zh"]) / 2)
            saved = json.loads((Path(directory) / "validation/step-0/metrics.json").read_text())
            self.assertEqual(saved["count"], 4)
            self.assertEqual(set(saved["directions"]), {"zh-vi", "vi-zh"})
            self.assertEqual(saved["reference_sha256"], text_fingerprint(self.dataset["id"], self.dataset["target"]))
            self.model.eval()
            batch = trainer.data_collator([{k: v for k, v in row.items() if k in ("input_ids", "attention_mask", "labels")} for row in self.rows])
            _, predictions, _ = trainer.prediction_step(self.model, batch, False)
            targets = [self.tokenizer.convert_tokens_to_ids(name) for name in ("vie_Latn", "vie_Latn", "zho_Hans", "zho_Hans")]
            self.assertEqual(predictions[:, 1].tolist(), targets)
            changed = {key: value.clone() for key, value in batch.items()}
            changed["labels"][:] = self.tokenizer.eos_token_id
            _, changed_predictions, _ = trainer.prediction_step(self.model, changed, False)
            self.assertTrue(torch.equal(predictions, changed_predictions))

    def test_independent_evaluator_matches_trainer_per_direction(self):
        from nllb_lora.eval.evaluate import direction_rows, generate_translations

        with tempfile.TemporaryDirectory() as directory:
            trainer = build_trainer(self.model, self.tokenizer, self.dataset, self.dataset, self.arguments(directory),
                                    directory, self.config, official=False)
            trainer.evaluate()
            saved = json.loads((Path(directory) / "validation/step-0/metrics.json").read_text())
            root = Path(directory) / "pairs"
            root.mkdir()
            for direction in direction_specs(self.config):
                rows = [r for r in self.rows if r["direction"] == direction["name"]]
                source, target = root / f"{direction['name']}.src", root / f"{direction['name']}.tgt"
                source.write_text("".join(r["source"] + "\n" for r in rows), encoding="utf-8", newline="\n")
                target.write_text("".join(r["target"] + "\n" for r in rows), encoding="utf-8", newline="\n")
                tokenizer = load_tokenizer(self.config, True, direction)
                _, _, _, report = generate_translations(self.model, tokenizer,
                    direction_rows(self.config, direction, source, target, "val"), self.config, 2, direction)
                expected = saved["directions"][direction["name"]]
                self.assertEqual(report["reference_sha256"], expected["reference_sha256"])
                self.assertEqual(report["prediction_sha256"], expected["prediction_sha256"])
                for metric in ("bleu", "chrf"):
                    self.assertAlmostEqual(report[metric], expected[metric], delta=1e-6)

    def test_official_checkpoint_verifies_direction_blocks(self):
        from nllb_lora.train.sync_checkpoints import validate_payload, verify_checkpoint
        from nllb_lora.train.train_nllb import CheckpointSyncCallback
        from nllb_lora.train.prepare_data import write_json

        context = copy.deepcopy(self.config)
        context["model"]["method"] = "full_fine_tuning"
        context["run"]["phase"] = "tiny_fixture"
        context["training"]["total_optimizer_steps"] = 3
        context["data"]["splits"]["val"]["expected_pairs"] = 2
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as drive_directory:
            trainer = build_trainer(self.model, self.tokenizer, self.dataset, self.dataset, self.arguments(directory),
                                    directory, self.config, official=True)
            write_json(Path(directory) / "resolved_config.json", context)
            write_json(Path(directory) / "run_manifest.json", {"phase": "tiny_cpu_test"})
            trainer.add_callback(CheckpointSyncCallback(directory, Path(drive_directory) / "drive"))
            trainer.train()
            manifest = verify_checkpoint(Path(drive_directory) / "drive" / "checkpoint-3")
            self.assertEqual(manifest["step"], 3)
            rows = (Path(drive_directory) / "drive/checkpoint-3/validation/predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(row)["id"] for row in rows], ["val:1", "val:2", "val:1@vi-zh", "val:2@vi-zh"])
            # A single-direction layout (first block only) is no longer an acceptable official validation.
            broken = copy.deepcopy(context)
            broken["model"].pop("directions")
            write_json(Path(drive_directory) / "drive/checkpoint-3/resolved_config.json", broken)
            with self.assertRaisesRegex(ValueError, "Official checkpoint requires full 2 validation predictions"):
                validate_payload(Path(drive_directory) / "drive" / "checkpoint-3")

    def test_training_step_updates_weights_on_mixed_directions(self):
        before = self.model.model.shared.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_trainer(self.model, self.tokenizer, self.dataset, self.dataset, self.arguments(directory),
                                    directory, self.config, official=False)
            result = trainer.train()
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertFalse(torch.equal(before, self.model.model.shared.weight.detach()))


class SharedPackagingTests(unittest.TestCase):
    def test_code_files_include_shared_config(self):
        from nllb_lora.train import colab_workflow as wf
        self.assertIn("configs/nllb_lora_shared.json", wf.CODE_FILES)
        self.assertTrue((PROJECT / "configs/nllb_lora_shared.json").is_file())

    def test_notebook_selects_shared_config_and_isolates_runs(self):
        notebook = json.loads((PROJECT / "notebooks/nllb_lora_shared_train.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        for text in ('CONFIG_NAME = "nllb_lora_shared.json"', 'ID_PREFIX = "shared-lora-"',
                     'ACTIVE_PREFIX = "active-lora-shared-"', "nllb_lora_colab_source.zip",
                     "DACNTT-lora-shared-source-20261106", "saved.get(\"source_archive\")", "evaluate_test"):
            self.assertIn(text, source)
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        for text in ("DIRECTION", "zhvi-lora-", "active-full", "__BUNDLE_SHA256__"):
            self.assertNotIn(text, code)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "cell", "exec")
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
