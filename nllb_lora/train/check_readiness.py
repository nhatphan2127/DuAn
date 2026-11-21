"""Validate reviewed, hash-bound Colab evidence before a full experiment."""

import json
import math
from pathlib import Path

from nllb_lora.train.prepare_data import file_sha256


def full_training_budget(count, batch_size, accumulation, epochs):
    if any(not isinstance(value, int) or value <= 0 for value in (count, batch_size, accumulation, epochs)):
        raise ValueError("Full budget requires positive integer count/batch/accumulation/epochs")
    if epochs > 3:
        raise ValueError("Full experiment budget is at most three epochs")
    return math.ceil(math.ceil(count / batch_size) / accumulation) * epochs


def validate_checkpoint_cadence(prior_training, current_training, review):
    prior = (prior_training["eval_steps"], prior_training["save_steps"])
    current = (current_training["eval_steps"], current_training["save_steps"])
    if prior == current:
        return
    cadence = review.get("checkpoint_cadence_review", {})
    if (prior != (500, 500) or current != (2000, 2000)
        or cadence.get("qualification_interval") != 500
        or cadence.get("full_interval") != 2000
        or not isinstance(cadence.get("reason"), str) or not cadence["reason"].strip()):
        raise ValueError("Pilot/full checkpoint cadence mismatch requires a reviewed 500-to-2000 transition")


def validate_readiness(path, expected_manifest):
    path = Path(path)
    review = json.loads(path.read_text(encoding="utf-8"))
    if not review.get("reviewer", {}).get("name") or review["reviewer"].get("decision") != "ACCEPTED":
        raise ValueError("Full run requires a named reviewer and ACCEPTED evidence review")
    loaded = {}
    keys = ("environment", "benchmark", "benchmark_manifest", "sanity", "sanity_manifest",
        "pilot_summary", "pilot_manifest", "resume", "resume_manifest")
    for name in keys:
        item = review.get("artifacts", {}).get(name, {})
        if not item.get("path") or not item.get("sha256"):
            raise ValueError(f"Missing readiness artifact: {name}")
        artifact = Path(item["path"])
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        if not artifact.is_file() or file_sha256(artifact) != item["sha256"]:
            raise ValueError(f"Readiness artifact hash mismatch: {name}")
        loaded[name] = json.loads(artifact.read_text(encoding="utf-8"))
    environment = loaded["environment"]
    if environment["status"] != "ENVIRONMENT_PROBE_PASSED_REVIEW_REQUIRED" or not environment["cuda_available"] or not all(environment["api_checks"].values()):
        raise ValueError("Environment evidence did not pass CUDA/API probe")
    benchmark = loaded["benchmark"]
    if not all(benchmark.get(name) for name in ("resource_probe_passed", "cuda", "optimizer_present_during_eval", "finite_metrics")) or benchmark["validation_count"] != 5000 or benchmark["optimizer_updates"] < 100:
        raise ValueError("Benchmark evidence lacks full-val/CUDA/optimizer/finite proof")
    if benchmark["peak_cuda_reserved_bytes"] <= 0 or expected_manifest["environment"]["gpu_total_bytes"] < benchmark["peak_cuda_reserved_bytes"] * 1.05:
        raise ValueError("Current GPU memory lacks 5% headroom over measured benchmark reserved peak")
    if not loaded["sanity"]["learning_diagnostic_improved"]:
        raise ValueError("Sanity evidence did not demonstrate learning")
    pilot = loaded["pilot_summary"]
    baseline, selected = pilot["pretrained_step_zero"], pilot["selected_best"]
    if pilot["status"] != "REQUIRES_MANUAL_QUALITY_REVIEW" or baseline["count"] != 5000 or selected["count"] != 5000 or baseline["reference_sha256"] != selected["reference_sha256"] or baseline["generation"] != selected["generation"]:
        raise ValueError("Pilot evidence has inconsistent full validation/reference/generation")
    resume = loaded["resume"]
    if resume["status"] != "RESUME_PROBE_PASSED_REVIEW_REQUIRED" or not resume["optimizer_loaded"] or resume["scheduler_loaded_step"] < resume["initial_step"] or resume["final_step"] <= resume["initial_step"] or not resume["best_checkpoint_present"]:
        raise ValueError("Resume evidence did not restore/progress full state")
    current = expected_manifest["config"]
    if not loaded["sanity"].get("finite_loss") or not resume.get("loaded_state_checks"):
        raise ValueError("Sanity finite-loss or resume scheduler/LR checks missing")
    for name, version in expected_manifest["environment"]["packages"].items():
        if environment["packages"].get(name, {}).get("actual") != version:
            raise ValueError(f"Environment package version mismatch: {name}")
    for name in ("max_new_tokens", "num_beams", "do_sample"):
        if baseline["generation"].get(name) != current["generation"][name]:
            raise ValueError("Pilot generation evidence differs from full configuration")
    fit_keys = ("optim", "learning_rate", "weight_decay", "per_device_train_batch_size",
        "gradient_accumulation_steps", "per_device_eval_batch_size", "fp16", "bf16",
        "gradient_checkpointing", "max_grad_norm", "label_smoothing_factor")
    for name in ("benchmark_manifest", "sanity_manifest", "pilot_manifest", "resume_manifest"):
        prior = loaded[name]
        expected_phase = {"benchmark_manifest": "benchmark", "sanity_manifest": "sanity", "pilot_manifest": "pilot", "resume_manifest": "pilot"}[name]
        if prior["config"]["run"]["phase"] != expected_phase:
            raise ValueError(f"Wrong evidence phase in {name}")
        if prior["config"]["model"] != current["model"] or prior["cache_key"] != expected_manifest["cache_key"] or prior["data"]["manifest_sha256"] != expected_manifest["data"]["manifest_sha256"]:
            raise ValueError(f"Model/cache/data mismatch in {name}")
        if prior["code"]["source_sha256"] != expected_manifest["code"]["source_sha256"] or prior["environment"]["packages"] != expected_manifest["environment"]["packages"]:
            raise ValueError(f"Code/environment mismatch in {name}")
        if prior["environment"].get("python") != expected_manifest["environment"].get("python"):
            raise ValueError(f"Python environment mismatch in {name}")
        if prior["config"]["generation"] != current["generation"] or any(prior["config"]["training"][key] != current["training"][key] for key in fit_keys):
            raise ValueError(f"Training/generation mismatch in {name}")
        if any(prior["config"]["data"][key] != current["data"][key] for key in ("max_source_tokens", "max_target_tokens")):
            raise ValueError(f"Token length mismatch in {name}")
        if name == "pilot_manifest":
            validate_checkpoint_cadence(prior["config"]["training"], current["training"], review)
    return {"status": "REVIEWED_EVIDENCE_VALIDATED", "reviewer": review["reviewer"],
        "readiness_sha256": file_sha256(path), "artifacts": review["artifacts"]}
