"""Tests for the training sanity summary (loss reduction and example review)."""
import unittest

from nllb_lora.train.train_nllb import sanity_summary


class SanityTests(unittest.TestCase):
    def test_reduced_loss_requires_example_review(self):
        report = sanity_summary({"eval_loss": 5.0}, {"eval_loss": 4.0}, ["train:1"])
        self.assertEqual(report["loss_reduction"], 1)
        self.assertTrue(report["learning_diagnostic_improved"])
        self.assertEqual(report["status"], "REQUIRES_MANUAL_EXAMPLE_REVIEW")
        self.assertIn("unofficial", report["scope"])

    def test_bad_loss_cannot_pass(self):
        for loss in (5.0, 6.0, float("inf"), float("nan")):
            with self.subTest(loss=loss):
                report = sanity_summary({"eval_loss": 5.0}, {"eval_loss": loss}, ["train:1"])
                self.assertFalse(report["learning_diagnostic_improved"])
                self.assertEqual(report["status"], "LEARNING_DIAGNOSTIC_FAILED")


if __name__ == "__main__":
    unittest.main()
