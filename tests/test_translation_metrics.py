"""Tests for BLEU/chrF scoring checks and adapter reference handling."""
from types import SimpleNamespace
import unittest

import numpy as np

from nllb_lora.eval.translation_metrics import make_compute_metrics, score_translations


class MetricsTests(unittest.TestCase):
    ids = ["val:1", "val:2"]
    refs = ["Xin chào Việt Nam và tất cả bạn bè trên thế giới.", "Hôm nay chúng tôi nghiên cứu mô hình dịch máy tiếng Hoa sang tiếng Việt."]

    def test_perfect_and_empty_wrong_predictions(self):
        perfect = score_translations(self.ids, self.refs, self.ids, self.refs)
        self.assertAlmostEqual(perfect["bleu"], 100)
        self.assertAlmostEqual(perfect["chrf"], 100)
        for predictions in (["", ""], ["a b c d e", "x y z"]):
            worse = score_translations(self.ids, predictions, self.ids, self.refs)
            self.assertLess(worse["bleu"], perfect["bleu"])
            self.assertLess(worse["chrf"], perfect["chrf"])
        self.assertIn("tok:13a", perfect["signatures"]["bleu"])
        self.assertIn("nw:0", perfect["signatures"]["chrf"])

    def test_mismatch_order_duplicate_empty_reference_rejected(self):
        for ids, texts, refids, refs in (([], [], [], []), (self.ids[:1], self.refs, self.ids, self.refs), (self.ids[::-1], self.refs, self.ids, self.refs), (["val:1"] * 2, self.refs, self.ids, self.refs), (self.ids, self.refs, self.ids, ["", "x"])):
            with self.subTest(ids=ids, refs=refs), self.assertRaises(ValueError):
                score_translations(ids, texts, refids, refs)

    def test_corpus_score_is_not_mean_sentence_score(self):
        predictions = [self.refs[0], "không đúng"]
        corpus = score_translations(self.ids, predictions, self.ids, self.refs)["bleu"]
        mean_sentence = sum(score_translations([key], [hyp], [key], [ref])["bleu"] for key, hyp, ref in zip(self.ids, predictions, self.refs)) / 2
        self.assertGreater(abs(corpus - mean_sentence), 1)

    def test_adapter_original_references_ignores_labels_and_matches_offline(self):
        class Tokenizer:
            pad_token_id = 1

            def batch_decode(inner, ids, skip_special_tokens):
                self.assertTrue(skip_special_tokens)
                self.assertEqual(ids.tolist(), [[3, 1], [4, 1]])
                return self.refs

        observed = []
        adapter = make_compute_metrics(Tokenizer(), self.ids, self.refs, lambda ids, text, report: observed.append(report))
        # Truncated/wrong labels cannot alter raw reference closure.
        metric = adapter(SimpleNamespace(predictions=np.array([[3, -100], [4, -100]]), label_ids=np.array([[99], [99]])))
        offline = score_translations(self.ids, self.refs, self.ids, self.refs)
        self.assertAlmostEqual(metric["bleu"], offline["bleu"], delta=1e-6)
        self.assertAlmostEqual(metric["chrf"], offline["chrf"], delta=1e-6)
        self.assertEqual(observed[0]["reference_sha256"], offline["reference_sha256"])
        with self.assertRaisesRegex(ValueError, "integer generated IDs"):
            adapter(SimpleNamespace(predictions=np.zeros((2, 3, 5))))


if __name__ == "__main__":
    unittest.main()
