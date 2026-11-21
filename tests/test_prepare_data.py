"""Tests for input verification, bundle copying and pair parsing in prepare_data.py."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from nllb_lora.train.prepare_data import copy_bundle, file_sha256, iter_pairs, verify_inputs


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "frozen"
        self.root.mkdir()
        self.config = {"data": {"manifest": "checksums.json", "manifest_sha256_file": "checksums.json.sha256", "semantic_review": "not_performed", "splits": {}}}
        self.entries = []
        for split in ("train", "val", "test"):
            self.config["data"]["splits"][split] = {"source": f"final/{split}.zh", "target": f"final/{split}.vi", "expected_pairs": 2}
            for language, text in (("zh", "你好\n谢谢\n"), ("vi", "Xin chào\nCảm ơn\n")):
                relative = f"final/{split}.{language}"
                path = self.root / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")
                self.entries.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size, "lines": 2})
        self.pin_manifest()

    def pin_manifest(self):
        path = self.root / "checksums.json"
        path.write_text(json.dumps({"files": self.entries}), encoding="utf-8")
        digest = file_sha256(path)
        self.config["data"]["expected_manifest_sha256"] = digest
        (self.root / "checksums.json.sha256").write_text(f"{digest}  checksums.json\n", encoding="utf-8")

    def test_verify_and_bundle_roundtrip(self):
        report = verify_inputs(self.config, self.root)
        self.assertEqual(report["splits"]["train"]["pairs"], 2)
        bundle = Path(self.temp.name) / "bundle"
        self.assertEqual(copy_bundle(self.config, self.root, bundle)["status"], "VERIFIED_INPUTS")
        self.assertEqual(len(list(bundle.rglob("*.*"))), 8)
        with self.assertRaises(FileExistsError):
            copy_bundle(self.config, self.root, bundle)

    def test_corrupt_byte_and_wrong_manifest_pin(self):
        path = self.root / "final/train.zh"
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            verify_inputs(self.config, self.root)
        self.config["data"]["expected_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_inputs(self.config, self.root)

    def test_pair_errors_and_stable_id(self):
        source, target = self.root / "final/train.zh", self.root / "final/train.vi"
        self.assertEqual(next(iter_pairs(source, target, "train"))["id"], "train:1")
        for content, error in (("only one\n", "unequal"), ("\nsecond\n", "empty"), ("\ufefffirst\nsecond\n", "BOM")):
            with self.subTest(content=content):
                target.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    list(iter_pairs(source, target, "train"))

    def test_invalid_utf8(self):
        path = self.root / "final/train.zh"
        path.write_bytes(b"\xff\nabc\n")
        self.entries[0].update(sha256=file_sha256(path), bytes=path.stat().st_size)
        self.pin_manifest()
        with self.assertRaisesRegex(ValueError, "invalid UTF-8"):
            verify_inputs(self.config, self.root)

    def test_duplicate_manifest_and_path_traversal(self):
        self.entries.append(copy.deepcopy(self.entries[0]))
        self.pin_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_inputs(self.config, self.root)
        self.entries.pop()
        for escape in ("../outside", "/etc/passwd", "C:/outside", "final\\train.zh"):
            with self.subTest(escape=escape):
                self.entries[0]["path"] = escape
                self.pin_manifest()
                with self.assertRaisesRegex(ValueError, "Unsafe manifest"):
                    verify_inputs(self.config, self.root)

    def test_bundle_cannot_write_inside_source(self):
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            copy_bundle(self.config, self.root, self.root / "bundle")


if __name__ == "__main__":
    unittest.main()
