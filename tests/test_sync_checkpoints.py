"""Tests for checkpoint publication, retention, verification and path safety."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nllb_lora.train.sync_checkpoints import choose_verified, synchronize_checkpoint, verify_checkpoint
from nllb_lora.train.prepare_data import write_json
from nllb_lora.eval.translation_metrics import text_fingerprint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.drive = self.root / "run", self.root / "drive"
        self.run.mkdir()
        write_json(self.run / "resolved_config.json", {})
        write_json(self.run / "run_manifest.json", {})

    def checkpoint(self, step, best=1):
        checkpoint = self.run / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
            (checkpoint / name).write_text("fixture")
        write_json(checkpoint / "trainer_state.json", {"global_step": step, "best_model_checkpoint": str(self.run / f"checkpoint-{best}")})
        folder = self.run / "validation" / f"step-{step}"
        write_json(folder / "metrics.json", {"step": step, "count": 1, "official_full_validation": False,
            "prediction_sha256": text_fingerprint(["val:1"], ["Xin chào"])})
        (folder / "predictions.jsonl").write_text('{"id":"val:1","text":"Xin chào"}\n', encoding="utf-8")
        return checkpoint

    def test_publish_idempotent_retention_and_corrupt_selection(self):
        for step in (1, 2, 3, 4):
            synchronize_checkpoint(self.checkpoint(step), self.drive, self.run)
        self.assertEqual({p.name for p in self.drive.glob("checkpoint-*")}, {"checkpoint-1", "checkpoint-3", "checkpoint-4"})
        synchronize_checkpoint(self.run / "checkpoint-4", self.drive, self.run)
        self.assertEqual(choose_verified(self.drive).name, "checkpoint-4")
        (self.drive / "checkpoint-4/optimizer.pt").write_text("corrupt")
        self.assertEqual(choose_verified(self.drive).name, "checkpoint-3")
        with self.assertRaises(ValueError):
            verify_checkpoint(self.drive / "checkpoint-4")

    def test_missing_wrongstep_and_official_count(self):
        checkpoint = self.checkpoint(1)
        (checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            synchronize_checkpoint(checkpoint, self.drive, self.run)
        (checkpoint / "optimizer.pt").write_text("fixture")
        write_json(checkpoint / "trainer_state.json", {"global_step": 2})
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            synchronize_checkpoint(checkpoint, self.drive, self.run)
        write_json(checkpoint / "trainer_state.json", {"global_step": 1})
        write_json(self.run / "validation/step-1/metrics.json", {"step": 1, "count": 1, "official_full_validation": True,
            "prediction_sha256": text_fingerprint(["val:1"], ["Xin chào"])})
        with self.assertRaisesRegex(ValueError, "5000"):
            synchronize_checkpoint(checkpoint, self.drive, self.run)

    def test_partial_copy_failure_preserves_previous(self):
        synchronize_checkpoint(self.checkpoint(1), self.drive, self.run)
        checkpoint = self.checkpoint(2)
        import shutil
        real_copy = shutil.copyfile
        def failing_copy(source, target):
            if Path(target).parent == self.drive / "checkpoint-2":
                raise OSError("simulated upload failure")
            return real_copy(source, target)
        with patch("nllb_lora.train.sync_checkpoints.shutil.copyfile", side_effect=failing_copy):
            with self.assertRaises(OSError):
                synchronize_checkpoint(checkpoint, self.drive, self.run)
        self.assertFalse((self.drive / "checkpoint-2/COMPLETE.json").exists())
        self.assertEqual(choose_verified(self.drive).name, "checkpoint-1")

    def test_manifest_path_escape_rejected(self):
        synchronize_checkpoint(self.checkpoint(1), self.drive, self.run)
        marker = self.drive / "checkpoint-1/COMPLETE.json"
        manifest = json.loads(marker.read_text())
        manifest["files"]["../outside"] = {"bytes": 1, "sha256": "x"}
        write_json(marker, manifest)
        with self.assertRaises(ValueError):
            verify_checkpoint(marker.parent)


if __name__ == "__main__":
    unittest.main()
