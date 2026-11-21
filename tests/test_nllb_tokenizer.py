"""Tests for the NLLB tokenizer contract, revision pin and collator shifting."""
import copy
import json
import os
from pathlib import Path
import unittest

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

from nllb_lora.train.prepare_data import load_tokenizer, tokenizer_contract


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
        cls.tokenizer = load_tokenizer(cls.config, local_files_only=True)

    def test_actual_contract(self):
        contract = tokenizer_contract(self.config, local_files_only=True)
        self.assertEqual(contract["source_language_id"], 256200)
        self.assertEqual(contract["target_language_id"], 256193)
        self.assertNotEqual(contract["decoder_start_token_id"], contract["forced_bos_token_id"])
        for row in contract["examples"]:
            self.assertEqual(row["decoded_target"], row["target"])
            self.assertNotIn("vie_Latn", row["decoded_target"])

    def test_bad_revision_and_language(self):
        for field, value in (("revision", "main"), ("tokenizer_revision", "0" * 40), ("target_language", "bad"), ("source_language", "vie_Latn")):
            config = copy.deepcopy(self.config)
            config["model"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_tokenizer(config, local_files_only=True)

    def test_collator_shift_and_forced_generation(self):
        import torch
        torch.set_num_threads(1)
        from transformers import AutoConfig, DataCollatorForSeq2Seq, M2M100Config, M2M100ForConditionalGeneration
        from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

        original = AutoConfig.from_pretrained(self.config["model"]["name_or_path"], revision=self.config["model"]["revision"], local_files_only=True)
        config = M2M100Config(vocab_size=len(self.tokenizer), d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=1,
            decoder_attention_heads=1, encoder_ffn_dim=16, decoder_ffn_dim=16,
            decoder_start_token_id=original.decoder_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            max_position_embeddings=32)
        torch.manual_seed(42)
        model = M2M100ForConditionalGeneration(config).eval()
        examples = [self.tokenizer("你好", text_target="Xin chào!"), self.tokenizer("中国", text_target="Việt Nam có 2026 cuốn sách.")]
        batch = DataCollatorForSeq2Seq(self.tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)(examples)
        self.assertTrue((batch["labels"] == -100).any())
        # M2M100 shifts inside forward; collator may not materialize decoder inputs.
        decoder_inputs = shift_tokens_right(batch["labels"], self.tokenizer.pad_token_id, original.decoder_start_token_id)
        self.assertTrue((decoder_inputs[:, 0] == original.decoder_start_token_id).all())
        expected = batch["labels"][:, :-1].clone()
        expected[expected == -100] = self.tokenizer.pad_token_id
        self.assertTrue(torch.equal(decoder_inputs[:, 1:], expected))
        with torch.no_grad():
            output = model(**batch)
            self.assertTrue(torch.isfinite(output.loss))
            predictions = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("vie_Latn"), max_new_tokens=3, num_beams=1)
        self.assertTrue((predictions[:, 1] == self.tokenizer.convert_tokens_to_ids("vie_Latn")).all())


if __name__ == "__main__":
    unittest.main()
