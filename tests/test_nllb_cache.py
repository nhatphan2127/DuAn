"""Tests for tokenized cache preparation, restart, corruption and overflow audit."""
import json
from pathlib import Path
import unittest

import test_prepare_data
from nllb_lora.train.prepare_data import prepare_cache, verify_cache


class CacheTests(unittest.TestCase):
    pin_manifest = test_prepare_data.DataTests.pin_manifest

    def setUp(self):
        test_prepare_data.DataTests.setUp(self)
        actual = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
        self.config["model"] = actual["model"]
        self.cache = Path(self.temp.name) / "cache"

    def test_prepare_restart_corrupt_and_mismatch(self):
        from datasets import load_from_disk

        manifest = prepare_cache(self.config, self.root, self.cache, 512, 512, True, 1)
        self.assertEqual(manifest["length_report"]["overflows"], {"train": 0, "val": 0})
        dataset = load_from_disk(str(self.cache / "dataset"))
        self.assertEqual(set(dataset), {"train", "val"})
        self.assertEqual(dataset["val"][0]["id"], "val:1")
        self.assertEqual(dataset["val"][0]["target"], "Xin chào")
        self.assertEqual(prepare_cache(self.config, self.root, self.cache, 512, 512, True)["cache_key"], manifest["cache_key"])
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            prepare_cache(self.config, self.root, self.cache, 256, 512, True)
        artifact = next(p for p in (self.cache / "dataset").rglob("*.arrow"))
        with artifact.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt cache"):
            verify_cache(self.cache, manifest["specification"])

    def test_overflow_preserves_audit_without_completion_marker(self):
        with self.assertRaisesRegex(ValueError, "Token overflow"):
            prepare_cache(self.config, self.root, self.cache, 1, 1, True)
        self.assertTrue((self.cache / "length_report.json").is_file())
        self.assertIn('"id": "train:1"', (self.cache / "overflow_ids.jsonl").read_text())
        self.assertFalse((self.cache / "cache_manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "Cache incomplete"):
            prepare_cache(self.config, self.root, self.cache, 1, 1, True)


if __name__ == "__main__":
    unittest.main()
