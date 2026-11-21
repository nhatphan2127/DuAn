"""Tests for the vi-zh/zh-vi configs, metric tokenizer options and the language-direction contract."""
import copy
import json
import os
from pathlib import Path
import unicodedata
import unittest

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

from nllb_lora.eval.translation_metrics import score_translations, scoring_options
from nllb_lora.train import colab_workflow as wf
from nllb_lora.train.prepare_data import load_tokenizer, tokenizer_contract

PROJECT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((PROJECT / name).read_text(encoding="utf-8"))


class ConfigTests(unittest.TestCase):
    def test_vi_zh_differs_only_in_direction_fields(self):
        forward, reverse = read("configs/nllb_lora_zh_vi.json"), read("configs/nllb_lora_vi_zh.json")
        expected = copy.deepcopy(forward)
        expected["experiment"] = "nllb_600m_vi_zh_lora_r16_qv"
        expected["model"].update(source_language="vie_Latn", target_language="zho_Hans")
        for split in expected["data"]["splits"].values():
            split["source"], split["target"] = split["target"], split["source"]
        expected["generation"]["forced_bos_language"] = "zho_Hans"
        expected["evaluation"]["bleu"]["tokenize"] = "zh"
        expected["evaluation"]["text_normalization"] = "nfkc"
        expected["run"]["cache_fingerprint"] = None
        self.assertEqual(reverse, expected)

    def test_zh_vi_config_keeps_original_bleu_and_direction(self):
        forward = read("configs/nllb_lora_zh_vi.json")
        self.assertEqual(forward["evaluation"]["bleu"]["tokenize"], "13a")
        self.assertEqual((forward["model"]["source_language"], forward["model"]["target_language"]), ("zho_Hans", "vie_Latn"))


class MetricsTokenizerTests(unittest.TestCase):
    ids = ["val:1", "val:2"]
    refs = ["今天我们研究越南语到中文的机器翻译模型。", "越南的首都是河内，这里有很多历史悠久的建筑。"]

    def test_default_is_13a_and_zh_is_selectable(self):
        default = score_translations(self.ids, self.refs, self.ids, self.refs)
        chinese = score_translations(self.ids, self.refs, self.ids, self.refs, "zh")
        self.assertIn("tok:13a", default["signatures"]["bleu"])
        self.assertIn("tok:zh", chinese["signatures"]["bleu"])
        self.assertAlmostEqual(chinese["bleu"], 100)
        self.assertAlmostEqual(chinese["chrf"], 100)

    def test_zh_tokenizer_scores_partial_overlap_above_13a(self):
        predictions = ["今天我们研究机器翻译模型。", "越南的首都是河内，这里有很多建筑。"]
        zh = score_translations(self.ids, predictions, self.ids, self.refs, "zh")["bleu"]
        a13 = score_translations(self.ids, predictions, self.ids, self.refs, "13a")["bleu"]
        self.assertGreater(zh, a13)

    def test_unknown_tokenizer_rejected(self):
        with self.assertRaises(ValueError):
            score_translations(self.ids, self.refs, self.ids, self.refs, "char")
        with self.assertRaises(ValueError):
            score_translations(self.ids, self.refs, self.ids, self.refs, "zh", "nfc")

    def test_nfkc_makes_ascii_and_fullwidth_punctuation_equivalent(self):
        references = ["越南的首都是河内，这里很美！", "今天天气很好。我们去公园。"]
        predictions = [unicodedata.normalize("NFKC", text) for text in references]
        raw = score_translations(self.ids, predictions, self.ids, references, "zh")
        normalized = score_translations(self.ids, predictions, self.ids, references, "zh", "nfkc")
        self.assertLess(raw["chrf"], 100)
        self.assertAlmostEqual(normalized["bleu"], 100)
        self.assertAlmostEqual(normalized["chrf"], 100)
        self.assertEqual(normalized["text_normalization"], "nfkc")
        self.assertIn("NFKC", normalized["text_policy"])
        # Fingerprints still describe the unmodified texts.
        self.assertEqual(normalized["reference_sha256"], raw["reference_sha256"])
        self.assertEqual(normalized["prediction_sha256"], raw["prediction_sha256"])

    def test_default_scoring_unchanged_for_zh_vi(self):
        report = score_translations(["a"], ["Xin chào các bạn, Việt Nam!"], ["a"], ["Xin chào các bạn, Việt Nam!"])
        self.assertIsNone(report["text_normalization"])
        self.assertEqual(scoring_options(read("configs/nllb_lora_zh_vi.json")), {"bleu_tokenize": "13a", "normalization": None})
        self.assertEqual(scoring_options(read("configs/nllb_lora_vi_zh.json")), {"bleu_tokenize": "zh", "normalization": "nfkc"})


class TokenizerDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read("configs/nllb_lora_vi_zh.json")

    def test_vi_zh_contract(self):
        contract = tokenizer_contract(self.config, local_files_only=True)
        self.assertEqual(contract["source_language_id"], 256193)
        self.assertEqual(contract["target_language_id"], 256200)
        self.assertEqual(contract["forced_bos_token_id"], 256200)
        self.assertNotEqual(contract["decoder_start_token_id"], contract["forced_bos_token_id"])
        for row in contract["examples"]:
            self.assertNotIn("zho_Hans", row["decoded_target"])
        first, second = contract["examples"]
        # NLLB's tokenizer NFKC-normalizes Chinese; scoring applies the same on both sides.
        self.assertEqual(first["decoded_target"], unicodedata.normalize("NFKC", first["target"]))
        # Documented limitation: the surname character 阮 is outside NLLB's vocabulary (<unk>, dropped on decode).
        self.assertEqual(second["decoded_target"], "文安在河内有2026本书。")
        self.assertEqual(contract["examples"][0]["source"], "Xin chào, Việt Nam!")

    def test_unsupported_pairs_rejected(self):
        for source, target in (("vie_Latn", "vie_Latn"), ("zho_Hans", "zho_Hans"), ("eng_Latn", "zho_Hans")):
            config = copy.deepcopy(self.config)
            config["model"].update(source_language=source, target_language=target)
            with self.subTest(pair=(source, target)), self.assertRaises(ValueError):
                load_tokenizer(config, local_files_only=True)


class WorkflowPackagingTests(unittest.TestCase):
    def test_code_files_include_both_directions(self):
        for name in ("configs/nllb_lora_zh_vi.json", "configs/nllb_lora_vi_zh.json"):
            self.assertIn(name, wf.CODE_FILES)
            self.assertTrue((PROJECT / name).is_file())

if __name__ == "__main__":
    unittest.main()
