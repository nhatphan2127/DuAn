"""Tests for build_source_bundle.py: contents match CODE_FILES and the ZIP is reproducible."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from nllb_lora.train.build_source_bundle import build_bundle
from nllb_lora.train.colab_workflow import CODE_FILES

PROJECT = Path(__file__).resolve().parents[1]


class BundleTests(unittest.TestCase):
    def test_contents_match_code_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bundle.zip"
            digest = build_bundle(PROJECT, output)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            with zipfile.ZipFile(output) as bundle:
                self.assertEqual(sorted(bundle.namelist()), sorted([*CODE_FILES, "SOURCE_MANIFEST.json"]))
                for info in bundle.infolist():
                    self.assertEqual((info.create_system, info.date_time), (3, (1980, 1, 1, 0, 0, 0)))
                manifest = json.loads(bundle.read("SOURCE_MANIFEST.json"))
                self.assertEqual(sorted(manifest), sorted(CODE_FILES))
                for name, entry in manifest.items():
                    data = (PROJECT / name).read_bytes()
                    self.assertEqual(bundle.read(name), data)
                    self.assertEqual(entry, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def test_same_sources_give_same_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(build_bundle(PROJECT, Path(folder) / "a.zip"), build_bundle(PROJECT, Path(folder) / "b.zip"))

    def test_missing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                build_bundle(Path(folder) / "empty", Path(folder) / "bundle.zip")


if __name__ == "__main__":
    unittest.main()
