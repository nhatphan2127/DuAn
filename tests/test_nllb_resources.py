"""Tests for resource summaries: CPU rejection, missing evaluation and generation caps."""
import time
from types import SimpleNamespace
import unittest

import numpy as np

from nllb_lora.eval.translation_metrics import make_compute_metrics
from nllb_lora.train.train_nllb import resource_summary


class ResourceTests(unittest.TestCase):
    def test_cpu_never_passes_and_missing_eval_fails(self):
        report = resource_summary(time.monotonic() - 2, 100, 1.0, 0.5, False, 0, 0, 0, 100)
        self.assertFalse(report["resource_probe_passed"])
        self.assertGreaterEqual(report["elapsed_seconds"], 2)
        self.assertIsNone(report["minimum_reserved_headroom_fraction"])
        self.assertFalse(resource_summary(time.monotonic(), 100, 1, 0, True, 80, 90, 100, 100)["resource_probe_passed"])
        passing = resource_summary(time.monotonic(), 100, 1, 2, True, 70, 80, 100, 100)
        self.assertTrue(passing["resource_probe_passed"])
        self.assertAlmostEqual(passing["minimum_reserved_headroom_fraction"], .2)

    def test_generated_cap_counts_eos_and_padding(self):
        class Tokenizer:
            pad_token_id, eos_token_id = 1, 2
            def batch_decode(self, ids, skip_special_tokens):
                return ["Một câu dịch đủ dài để chấm điểm."] * len(ids)
        observed = []
        scorer = make_compute_metrics(Tokenizer(), ["a", "b", "c"], ["Một câu dịch đủ dài để chấm điểm."] * 3,
            lambda ids, text, report: observed.append(report), generation_limit=3)
        scorer(SimpleNamespace(predictions=np.array([[2, 5, 6, 2], [2, 5, 2, -100], [2, 5, 6, 7]])))
        self.assertEqual(observed[0]["at_generation_limit"], 2)
        self.assertAlmostEqual(observed[0]["at_generation_limit_ratio"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
