"""Tests for translation export and overflow handling in evaluate.py."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from nllb_lora.eval.evaluate import generate_translations
from nllb_lora.train.prepare_data import load_tokenizer


class EvaluationTests(unittest.TestCase):
    def test_source_only_export_and_overflow(self):
        import torch
        from transformers import M2M100Config, M2M100ForConditionalGeneration, AutoModelForSeq2SeqLM

        torch.set_num_threads(1)
        config = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
        config["generation"].update(max_new_tokens=3, num_beams=1)
        tokenizer = load_tokenizer(config, True)
        model = M2M100ForConditionalGeneration(M2M100Config(vocab_size=len(tokenizer), d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
            encoder_ffn_dim=16, decoder_ffn_dim=16, decoder_start_token_id=2, pad_token_id=1,
            eos_token_id=2, max_position_embeddings=32)).eval()
        rows = [{"id": "val:1", "source": "你好", "target": "Xin chào Việt Nam và tất cả bạn bè."}]
        first = generate_translations(model, tokenizer, rows, config)
        cached_config = copy.deepcopy(config)
        cached_config["generation"]["use_cache"] = True
        model.config.use_cache = False
        cached = generate_translations(model, tokenizer, rows, cached_config)
        uncached_config = copy.deepcopy(config)
        uncached_config["generation"]["use_cache"] = False
        uncached = generate_translations(model, tokenizer, rows, uncached_config)
        self.assertEqual(cached[1], uncached[1])
        self.assertFalse(model.config.use_cache)
        changed = [{**rows[0], "target": "Một bản tham chiếu hoàn toàn khác nội dung."}]
        self.assertEqual(first[1], generate_translations(model, tokenizer, changed, config)[1])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            restored = AutoModelForSeq2SeqLM.from_pretrained(directory, local_files_only=True)
            second = generate_translations(restored, tokenizer, rows, config)
            self.assertEqual(first[1], second[1])
            self.assertEqual(first[3]["chrf"], second[3]["chrf"])
        short = copy.deepcopy(config)
        short["data"]["max_source_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            generate_translations(model, tokenizer, rows, short)
        with self.assertRaises(ValueError):
            generate_translations(model, tokenizer, rows, config, 0)


if __name__ == "__main__":
    unittest.main()
