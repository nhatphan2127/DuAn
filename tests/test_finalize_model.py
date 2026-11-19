"""Tests for exporting the best verified checkpoint and comparing decoding results."""
import json
import unittest

import test_sync_checkpoints
from nllb_lora.eval.finalize_model import export_verified_best
from nllb_lora.eval.compare_translations import compare_translations
from nllb_lora.eval.translation_metrics import decoding_settings
from nllb_lora.train.sync_checkpoints import synchronize_checkpoint
from nllb_lora.train.prepare_data import write_json


class FinalizationTests(unittest.TestCase):
    def test_decoding_comparison_ignores_only_library_metadata(self):
        baseline = {"num_beams": 4, "forced_bos_token_id": 256193, "_from_model_config": True}
        exported = dict(baseline, _from_model_config=False, transformers_version="4.57.1")
        self.assertEqual(decoding_settings(baseline), decoding_settings(exported))
        self.assertNotEqual(decoding_settings(baseline), decoding_settings(dict(exported, num_beams=1)))

    def test_export_best_only_and_reject_incomplete(self):
        fixture = test_sync_checkpoints.CheckpointTests("test_publish_idempotent_retention_and_corrupt_selection")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        context = {"run": {"phase": "tiny_fixture"}, "model": {"name": "fixture"},
            "data": {"frozen_root": str(fixture.root / "frozen")},
            "training": {"total_optimizer_steps": 2, "early_stopping_patience": 3}}
        write_json(fixture.run / "resolved_config.json", context)
        synchronize_checkpoint(fixture.checkpoint(1), fixture.drive, fixture.run)
        with self.assertRaisesRegex(ValueError, "before budget"):
            export_verified_best(fixture.drive, fixture.root / "unfinished")
        checkpoint = fixture.checkpoint(2)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        state["best_metric"] = 1.7
        write_json(checkpoint / "trainer_state.json", state)
        synchronize_checkpoint(checkpoint, fixture.drive, fixture.run)
        exported = fixture.root / "export"
        report = export_verified_best(fixture.drive, exported)
        self.assertEqual(report["best_weight_step"], 1)
        self.assertTrue((exported / "model.safetensors").is_file())
        self.assertFalse((exported / "optimizer.pt").exists())
        self.assertEqual(json.loads((exported / "provenance.json").read_text())["completion_reason"], "budget_complete")
        with self.assertRaises(ValueError):
            export_verified_best(fixture.drive, fixture.drive / "checkpoint-1/illegal")
        with self.assertRaises(FileExistsError):
            export_verified_best(fixture.drive, exported)

    def test_paired_bootstrap_reproducible_and_negative_delta(self):
        ids = [f"val:{i}" for i in range(1, 5)]
        references = ["Xin chào Việt Nam và tất cả những người bạn trên thế giới."] * 4
        candidate = ["Sai nghĩa và thiếu nội dung."] * 4
        first = compare_translations(ids, references, candidate, references, resamples=40)
        second = compare_translations(ids, references, candidate, references, resamples=40)
        self.assertEqual(first, second)
        self.assertLess(first["delta"]["bleu"], 0)
        self.assertLess(first["delta"]["chrf"], 0)
        self.assertEqual(first["status"], "COMPARISON_REQUIRES_QUALITY_REVIEW")
        for rows in first["paired_bootstrap"]["results"].values():
            self.assertTrue(0 <= rows[1]["p_value"] <= 1)
        with self.assertRaises(ValueError):
            compare_translations(ids, references, candidate, references, resamples=3)


if __name__ == "__main__":
    unittest.main()
