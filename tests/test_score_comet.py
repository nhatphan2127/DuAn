"""Tests for COMET scoring provenance, normalization policy and input checks."""
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nllb_lora.eval import score_comet
from nllb_lora.eval.translation_metrics import text_fingerprint, write_predictions


class FakeComet:
    def __init__(self):
        self.samples = None

    def predict(self, samples, batch_size, gpus):
        self.samples = samples
        scores = [0.1 * (index + 1) for index in range(len(samples))]
        return SimpleNamespace(scores=scores, system_score=sum(scores) / len(scores))


class CometTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.eval = self.root / "eval"
        self.eval.mkdir()
        self.ids = ["test:1@vi-zh", "test:2@vi-zh"]
        self.predictions = ["ＡＢＣ你好", "谢谢"]
        self.references = ["ＡＢＣ您好", "多谢"]
        self.write(["Ｘin chào", "Cảm ơn"], self.predictions, self.references)

    def write(self, sources, predictions, references, metrics=True, normalization="nfkc"):
        write_predictions(self.eval / "sources.jsonl", self.ids, sources)
        write_predictions(self.eval / "predictions.jsonl", self.ids, predictions)
        write_predictions(self.eval / "references.jsonl", self.ids, references)
        if metrics:
            (self.eval / "metrics.json").write_text(json.dumps({"count": len(self.ids), "split": "test",
                "direction": "vi-zh", "text_normalization": normalization,
                "prediction_sha256": text_fingerprint(self.ids, predictions),
                "reference_sha256": text_fingerprint(self.ids, references)}), encoding="utf-8")

    def run_main(self, *extra, model=None):
        model = model or FakeComet()
        output = self.root / "out"
        with patch.object(importlib.metadata, "version", return_value="2.2.7"):
            score_comet.main(["--eval-dir", str(self.eval), "--output-dir", str(output), "--gpus", "0", *extra],
                load_model=lambda: model)
        return model, output

    def test_fingerprint_matches_translation_metrics(self):
        self.assertEqual(score_comet.text_fingerprint(self.ids, self.references), text_fingerprint(self.ids, self.references))

    def test_scores_keep_order_and_record_provenance(self):
        _, output = self.run_main()
        report = json.loads((output / "comet_metrics.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(report["comet"], 0.15)
        self.assertEqual((report["count"], report["split"], report["direction"]), (2, "test", "vi-zh"))
        self.assertEqual((report["model"], report["revision"]), (score_comet.MODEL, score_comet.REVISION))
        self.assertEqual(report["prediction_sha256"], text_fingerprint(self.ids, self.predictions))
        segments = [json.loads(line) for line in (output / "comet_segments.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in segments], self.ids)
        self.assertAlmostEqual(segments[1]["comet"], 0.2)

    def test_nfkc_follows_metrics_and_leaves_source_alone(self):
        model, _ = self.run_main()
        self.assertEqual(model.samples[0], {"src": "Ｘin chào", "mt": "ABC你好", "ref": "ABC您好"})

    def test_no_normalization_keeps_raw_text(self):
        self.write(["Ｘin chào", "Cảm ơn"], self.predictions, self.references, normalization=None)
        model, _ = self.run_main()
        self.assertEqual(model.samples[0]["mt"], "ＡＢＣ你好")

    def test_baseline_folder_without_metrics_needs_explicit_policy(self):
        (self.eval / "metrics.json").unlink()
        with self.assertRaisesRegex(ValueError, "explicitly"):
            self.run_main()
        model, _ = self.run_main("--normalization", "nfkc")
        self.assertEqual(model.samples[1]["ref"], "多谢")

    def test_conflicting_normalization_refused(self):
        with self.assertRaisesRegex(ValueError, "differs from metrics"):
            self.run_main("--normalization", "none")

    def test_edited_prediction_refused(self):
        write_predictions(self.eval / "predictions.jsonl", self.ids, ["khác", "谢谢"])
        with self.assertRaisesRegex(ValueError, "metrics.json"):
            self.run_main()

    def test_misaligned_ids_refused(self):
        write_predictions(self.eval / "references.jsonl", list(reversed(self.ids)), self.references)
        with self.assertRaisesRegex(ValueError, "same order"):
            self.run_main()

    def test_nonfinite_score_writes_no_metrics(self):
        class Broken(FakeComet):
            def predict(self, samples, batch_size, gpus):
                return SimpleNamespace(scores=[float("nan"), 0.5], system_score=0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_main(model=Broken())
        self.assertFalse((self.root / "out").exists())
        self.run_main()  # A failed attempt leaves nothing that blocks the retry.
        self.assertTrue((self.root / "out/comet_metrics.json").is_file())

    def test_existing_or_inner_output_refused_before_loading_model(self):
        def unexpected():
            raise AssertionError("model must not load")
        (self.root / "taken").mkdir()
        with self.assertRaises(FileExistsError):
            score_comet.main(["--eval-dir", str(self.eval), "--output-dir", str(self.root / "taken")], load_model=unexpected)
        with self.assertRaises(SystemExit):
            score_comet.main(["--eval-dir", str(self.eval), "--output-dir", str(self.eval / "comet")], load_model=unexpected)


if __name__ == "__main__":
    unittest.main()
