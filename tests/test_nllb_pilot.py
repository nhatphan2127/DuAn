"""Tests for pilot budget validation and baseline/candidate comparison."""
import unittest

from nllb_lora.train.train_nllb import pilot_summary, validate_pilot


class PilotTests(unittest.TestCase):
    def test_official_budget_drive_and_full_validation(self):
        validate_pilot(1000, 500, "drive/run", 5000)
        for args in ((999, 500, "drive/run", 5000), (1000, 500, None, 5000), (1000, 500, "drive/run", 16)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_pilot(*args)

    def test_comparison_requires_same_refs_and_generation(self):
        baseline = {"reference_sha256": "original", "generation": {"num_beams": 4}, "bleu": 10, "chrf": 20}
        best = dict(baseline, bleu=9, chrf=19)
        report = pilot_summary(baseline, best, "checkpoint-1000")
        self.assertEqual(report["bleu_delta"], -1)
        self.assertEqual(report["status"], "REQUIRES_MANUAL_QUALITY_REVIEW")
        for key, value in (("reference_sha256", "wrong"), ("generation", {"num_beams": 1})):
            changed = dict(best, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot_summary(baseline, changed, "checkpoint-1000")


if __name__ == "__main__":
    unittest.main()
