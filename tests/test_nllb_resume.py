"""Tests that a fresh-process resume matches continuous training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nllb_lora.train.sync_checkpoints import restore_verified, resume_signature


class ResumeTests(unittest.TestCase):
    def test_fresh_process_resume_matches_continuous(self):
        self.compare_resume(False)

    def test_grouped_fresh_process_resume_matches_continuous(self):
        self.compare_resume(True)

    def compare_resume(self, grouped):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, USE_TORCH="1", USE_TF="0", HF_HUB_OFFLINE="1")
            def run(name, drive, *extra):
                if grouped:
                    extra = (*extra, "--group-by-length")
                result = subprocess.run([sys.executable, "-m", "tests.helpers.nllb_resume_worker", "--run", str(root / name),
                    "--drive", str(root / drive), *extra], capture_output=True, text=True, env=env, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads((root / name / "comparison.json").read_text())
            continuous = run("continuous", "continuous-drive")
            interrupted = run("interrupted", "resumed-drive", "--stop", "3")
            self.assertEqual(interrupted["step"], 3)
            resumed = run("resumed", "resumed-drive", "--resume")
            for key in ("step", "weights", "optimizer", "scheduler", "rng", "early_stopping", "baseline_prediction_sha256"):
                self.assertEqual(continuous[key], resumed[key], key)
            self.assertEqual(continuous["training_order"][3:], resumed["training_order"])
            self.assertEqual(resumed["early_stopping"]["attributes"]["early_stopping_patience_counter"], 2)
            self.assertTrue(resumed["best"].startswith(str(root / "resumed")))
            mapping = json.loads((root / "resumed/resume_mapping.json").read_text())
            self.assertTrue(mapping["verified_before_remap"])
            probe = json.loads((root / "resumed/resume_verification.json").read_text())
            self.assertEqual(probe["status"], "RESUME_PROBE_PASSED_REVIEW_REQUIRED")
            self.assertTrue(probe["loaded_state_checks"])
            latest = root / "resumed-drive/checkpoint-7"
            manifest = json.loads((latest / "run_manifest.json").read_text())
            incompatible = resume_signature(manifest)
            incompatible["cache_key"] = "wrong-cache"
            with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                restore_verified(root / "resumed-drive", root / "rejected", incompatible)
            # Corrupt newest; restore chooses checkpoint6 with verified best3.
            (latest / "optimizer.pt").write_text("corrupt")
            fallback = restore_verified(root / "resumed-drive", root / "fallback", resume_signature(manifest))
            self.assertEqual(fallback.name, "checkpoint-6")
            self.assertTrue((root / "fallback/checkpoint-3").exists())


if __name__ == "__main__":
    unittest.main()
