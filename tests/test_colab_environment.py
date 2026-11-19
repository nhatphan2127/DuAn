"""Tests for the Colab environment pin reader, quota evidence and inventory."""
import tempfile
import unittest
from pathlib import Path

from nllb_lora.train.check_colab_environment import collect_environment, drive_quota_evidence, read_pins


class EnvironmentTests(unittest.TestCase):
    def test_exact_pins_ignore_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pins"
            path.write_text("# candidate\ntorch==2.8.0\ntransformers==4.57.1 # test\n")
            self.assertEqual(read_pins(path), {"torch": "2.8.0", "transformers": "4.57.1"})

    def test_unpinned_requirement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pins"
            path.write_text("torch>=2\n")
            with self.assertRaises(ValueError):
                read_pins(path)

    def test_quota_requires_account_evidence(self):
        for free, source in ((None, None), (-1, "Google One"), (32, " "), (float('nan'), "quota")):
            with self.subTest(free=free, source=source), self.assertRaises(ValueError):
                drive_quota_evidence(free, source)
        self.assertEqual(drive_quota_evidence(40, "Google One storage page")['free_gib'], 40)

    def test_inventory_covers_run_manifest_versions(self):
        report = collect_environment("nllb_lora/requirements-colab.lock", ".", inspect_only=True)
        for name in ("torch", "transformers", "datasets", "accelerate", "sacrebleu", "sentencepiece", "numpy", "tokenizers", "safetensors", "huggingface-hub", "psutil"):
            self.assertIn(name, report["packages"])
            self.assertIsNotNone(report["packages"][name]["actual"])
        self.assertEqual(report["status"], "INSPECTED_NOT_ACCEPTED")


if __name__ == "__main__":
    unittest.main()
