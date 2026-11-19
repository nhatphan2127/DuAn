"""All accepting evidence below is synthetic and confined to temporary unit fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import importlib.metadata

from nllb_lora.train.check_readiness import full_training_budget, validate_readiness
from nllb_lora.train.prepare_data import file_sha256, write_json


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
        config["run"]["phase"] = "full"
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "accelerate", "sacrebleu", "sentencepiece", "numpy", "tokenizers", "safetensors", "huggingface-hub", "psutil")}
        self.expected = {"config": config, "cache_key": "SYNTHETIC_CACHE", "data": {"manifest_sha256": "SYNTHETIC_DATA"},
            "code": {"source_sha256": {"example.py": "SYNTHETIC_CODE"}},
            "environment": {"packages": packages, "gpu_total_bytes": 200}}
        metric = {"count": 5000, "reference_sha256": "SYNTHETIC_REF", "generation": config["generation"]}
        self.payloads = {
            "environment": {"status": "ENVIRONMENT_PROBE_PASSED_REVIEW_REQUIRED", "cuda_available": True,
                "api_checks": {"test": True}, "packages": {name: {"actual": version} for name, version in packages.items()}},
            "benchmark": {"resource_probe_passed": True, "cuda": True, "optimizer_present_during_eval": True,
                "finite_metrics": True, "validation_count": 5000, "optimizer_updates": 100, "peak_cuda_reserved_bytes": 100},
            "sanity": {"learning_diagnostic_improved": True, "finite_loss": True},
            "pilot_summary": {"status": "REQUIRES_MANUAL_QUALITY_REVIEW", "pretrained_step_zero": metric, "selected_best": metric},
            "resume": {"status": "RESUME_PROBE_PASSED_REVIEW_REQUIRED", "optimizer_loaded": True, "loaded_state_checks": True,
                "scheduler_loaded_step": 500, "initial_step": 500, "final_step": 1000, "best_checkpoint_present": True}}
        for name, phase in (("benchmark_manifest", "benchmark"), ("sanity_manifest", "sanity"), ("pilot_manifest", "pilot"), ("resume_manifest", "pilot")):
            self.payloads[name] = copy.deepcopy(self.expected)
            self.payloads[name]["config"]["run"]["phase"] = phase
        self.review = {"reviewer": {"name": "SYNTHETIC_UNIT_TEST_ONLY", "decision": "ACCEPTED"}, "artifacts": {}}
        self.persist()

    def persist(self):
        for name, payload in self.payloads.items():
            artifact = self.root / f"{name}.json"
            write_json(artifact, payload)
            self.review["artifacts"][name] = {"path": artifact.name, "sha256": file_sha256(artifact)}
        write_json(self.root / "review.json", self.review)

    def test_fixed_budget_and_synthetic_review_success(self):
        self.assertEqual(full_training_budget(913094, 1, 32, 3), 85605)
        with self.assertRaises(ValueError):
            full_training_budget(913094, 1, 32, 4)
        result = validate_readiness(self.root / "review.json", self.expected)
        self.assertEqual(result["status"], "REVIEWED_EVIDENCE_VALIDATED")

    def test_missing_bad_hash_and_cpu_fail_closed(self):
        (self.root / "environment.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_readiness(self.root / "review.json", self.expected)
        self.payloads["environment"]["cuda_available"] = False
        self.persist()
        with self.assertRaisesRegex(ValueError, "CUDA/API"):
            validate_readiness(self.root / "review.json", self.expected)
        self.review["artifacts"].pop("resume")
        write_json(self.root / "review.json", self.review)
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_readiness(self.root / "review.json", self.expected)

    def test_context_mismatch_and_insufficient_gpu_fail(self):
        self.payloads["pilot_manifest"]["cache_key"] = "wrong"
        self.persist()
        with self.assertRaisesRegex(ValueError, "Model/cache/data"):
            validate_readiness(self.root / "review.json", self.expected)
        self.payloads["pilot_manifest"]["cache_key"] = self.expected["cache_key"]
        self.persist()
        self.expected["environment"]["gpu_total_bytes"] = 104
        with self.assertRaisesRegex(ValueError, "headroom"):
            validate_readiness(self.root / "review.json", self.expected)

    def test_unfilled_public_template_rejected(self):
        with self.assertRaisesRegex(ValueError, "named reviewer"):
            validate_readiness("configs/colab_readiness.template.json", self.expected)

    def test_full_cadence_change_requires_specific_review(self):
        self.expected["config"]["training"].update(eval_steps=2000, save_steps=2000)
        with self.assertRaisesRegex(ValueError, "reviewed 500-to-2000"):
            validate_readiness(self.root / "review.json", self.expected)
        self.review["checkpoint_cadence_review"] = {"qualification_interval": 500,
            "full_interval": 2000, "reason": "SYNTHETIC_TEST: full-val cost; checkpoint loss window reviewed"}
        self.persist()
        self.assertEqual(validate_readiness(self.root / "review.json", self.expected)["status"], "REVIEWED_EVIDENCE_VALIDATED")
        self.expected["config"]["training"]["save_steps"] = 4000
        with self.assertRaisesRegex(ValueError, "cadence mismatch"):
            validate_readiness(self.root / "review.json", self.expected)

    def test_same_effective_batch_does_not_reuse_old_microbatch_evidence(self):
        self.expected["config"]["training"].update(per_device_train_batch_size=8, gradient_accumulation_steps=4)
        with self.assertRaisesRegex(ValueError, "Training/generation mismatch"):
            validate_readiness(self.root / "review.json", self.expected)


if __name__ == "__main__":
    unittest.main()
