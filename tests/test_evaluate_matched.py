"""Tests for matched evaluation against the trainer's padding and raw references."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from nllb_lora.eval.evaluate_matched import compare_scores, generate


class MatchedEvaluationTests(unittest.TestCase):
    def test_eos_equivalence_preserves_decoding_and_score_guards(self):
        baseline = {"count": 5000, "reference_sha256": "same", "bleu": 29.5, "chrf": 48.8,
            "generation": {"eos_token_id": 2, "num_beams": 4, "max_new_tokens": 256}}
        actual = copy.deepcopy(baseline)
        actual["generation"]["eos_token_id"] = [2]
        compare_scores(actual, baseline)
        for key, value in (("num_beams", 1), ("eos_token_id", [2, 3])):
            changed = copy.deepcopy(actual)
            changed["generation"][key] = value
            with self.assertRaises(ValueError):
                compare_scores(changed, baseline)
        for key, value in (("bleu", 29.51), ("count", 4999), ("reference_sha256", "other")):
            with self.assertRaises(ValueError):
                compare_scores(dict(actual, **{key: value}), baseline)

    def test_tiny_generation_matches_trainer_padding_and_raw_references(self):
        import torch
        from datasets import Dataset
        from transformers import M2M100Config, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments
        from nllb_lora.train.train_nllb import build_trainer
        from nllb_lora.train.prepare_data import load_tokenizer

        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
        config["generation"].update(max_new_tokens=4, num_beams=1)
        tokenizer = load_tokenizer(config, local_files_only=True)
        model = M2M100ForConditionalGeneration(M2M100Config(vocab_size=len(tokenizer), d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
            encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=64,
            decoder_start_token_id=2, pad_token_id=1, eos_token_id=2,
            dropout=0.0, attention_dropout=0.0, activation_dropout=0.0))
        rows = [{"id": f"val:{i+1}", "source": source, "target": target,
            **dict(tokenizer(source, text_target=target))}
            for i, (source, target) in enumerate((("你好", "Xin chào."), ("谢谢你的帮助", "Cảm ơn bạn đã giúp.")))]
        dataset = Dataset.from_list(rows)
        with tempfile.TemporaryDirectory() as directory:
            args = Seq2SeqTrainingArguments(output_dir=directory, use_cpu=True,
                per_device_eval_batch_size=1, predict_with_generate=True,
                eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
                metric_for_best_model="chrf", greater_is_better=True,
                report_to="none", disable_tqdm=True)
            trainer = build_trainer(model, tokenizer, dataset, dataset, args, directory, config, official=False)
            trainer.evaluate()
            expected = json.loads((Path(directory) / "validation/step-0/metrics.json").read_text(encoding="utf-8"))
            *_, actual = generate(model, tokenizer, rows, config, torch.device("cpu"))
            compare_scores(actual, expected)
            changed = [{**r, "target": "Một tham chiếu khác."} for r in rows]
            first = generate(model, tokenizer, rows, config, torch.device("cpu"))
            second = generate(model, tokenizer, changed, config, torch.device("cpu"))
            self.assertEqual(first[1], second[1])
            self.assertNotEqual(first[-1]["reference_sha256"], second[-1]["reference_sha256"])
