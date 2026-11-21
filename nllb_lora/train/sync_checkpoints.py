"""Publish verified Trainer state and full/adapter weights; mark completion last."""

import json
from pathlib import Path
import re
import shutil
import time

from nllb_lora.train.prepare_data import contained_file, file_sha256, validation_layout, write_json
from nllb_lora.eval.translation_metrics import text_fingerprint


def checkpoint_step(directory):
    match = re.fullmatch(r"checkpoint-(\d+)", Path(directory).name)
    if not match:
        raise ValueError("Expected checkpoint-N directory")
    return int(match.group(1))


def expected_validation_ids(context):
    """Official validation IDs from the resolved config; bare fixtures without data config use val:1..5000."""
    try:
        return validation_layout(context)
    except KeyError:
        return [f"val:{index}" for index in range(1, 5001)]


def validate_payload(directory):
    directory = Path(directory)
    step = checkpoint_step(directory)
    context = json.loads(contained_file(directory, "resolved_config.json").read_text(encoding="utf-8"))
    lora = context.get("model", {}).get("method") == "lora"
    evaluated = True
    if lora:
        from nllb_lora.train.lora_adapter import verify_adapter
        verify_adapter(directory, context["model"])
        evaluated = json.loads(contained_file(directory, "checkpoint_metadata.json").read_text(encoding="utf-8"))["evaluated"]
    required = ["generation_config.json", "tokenizer_config.json", "tokenizer.json",
        "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth",
        "resolved_config.json", "run_manifest.json"]
    required += (["validation/metrics.json", "validation/predictions.jsonl"] if evaluated else [])
    required += (["adapter_config.json", "adapter_model.safetensors", "base_model.json"] if lora else ["config.json"])
    if lora and context["training"]["fp16"]:
        required.append("scaler.pt")
    for name in required:
        contained_file(directory, name)
    if lora:
        pass
    elif (directory / "model.safetensors").is_file():
        contained_file(directory, "model.safetensors")
    elif (directory / "model.safetensors.index.json").is_file():
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        for name in set(index["weight_map"].values()):
            contained_file(directory, name)
    else:
        raise ValueError("Checkpoint lacks safetensors weights/index")
    state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    if state["global_step"] != step:
        raise ValueError("Checkpoint/state step mismatch")
    if not evaluated:
        if (directory / "validation").exists():
            raise ValueError("Recovery checkpoint must not claim validation metrics")
        if step % context["training"]["eval_steps"] == 0:
            raise ValueError("Scheduled validation checkpoint lacks metrics")
        if context.get("run", {}).get("phase") in ("pilot", "full"):
            original = json.loads(contained_file(directory, "pretrained_baseline/metrics.json").read_text(encoding="utf-8"))
            predictions = [json.loads(line) for line in contained_file(directory, "pretrained_baseline/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            expected_rows = len(expected_validation_ids(context))
            if original["step"] != 0 or original["count"] != expected_rows or len(predictions) != expected_rows:
                raise ValueError("Recovery checkpoint requires full pretrained validation baseline")
            if text_fingerprint([row["id"] for row in predictions], [row["text"] for row in predictions]) != original["prediction_sha256"]:
                raise ValueError("Baseline predictions fingerprint mismatch")
        return state
    metric = json.loads((directory / "validation/metrics.json").read_text(encoding="utf-8"))
    if state["global_step"] != step or metric["step"] != step:
        raise ValueError("Checkpoint/state/validation step mismatch")
    rows = [json.loads(line) for line in (directory / "validation/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != metric["count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Checkpoint predictions/count mismatch")
    if text_fingerprint([row["id"] for row in rows], [row["text"] for row in rows]) != metric["prediction_sha256"]:
        raise ValueError("Checkpoint prediction fingerprint mismatch")
    expected_ids = expected_validation_ids(context) if metric["official_full_validation"] else None
    if expected_ids is not None and len(rows) != len(expected_ids):
        raise ValueError(f"Official checkpoint requires full {len(expected_ids)} validation predictions")
    if lora and context.get("run", {}).get("phase") in ("pilot", "full") and not metric["official_full_validation"]:
        raise ValueError("Official LoRA checkpoint must use full validation")
    if expected_ids is not None and [row["id"] for row in rows] != expected_ids:
        raise ValueError("Official validation IDs must preserve the configured direction blocks in val:1..N order")
    baseline = directory / "pretrained_baseline"
    if context.get("run", {}).get("phase") in ("pilot", "full") and not baseline.is_dir():
        raise ValueError("Official pilot/full checkpoint requires original pretrained baseline")
    if baseline.is_dir():
        original = json.loads(contained_file(directory, "pretrained_baseline/metrics.json").read_text(encoding="utf-8"))
        predictions = [json.loads(line) for line in contained_file(directory, "pretrained_baseline/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if original["step"] != 0 or original["count"] != len(predictions) or original["reference_sha256"] != metric["reference_sha256"] or original["generation"] != metric["generation"]:
            raise ValueError("Baseline step/count/reference/generation mismatch")
        if text_fingerprint([row["id"] for row in predictions], [row["text"] for row in predictions]) != original["prediction_sha256"]:
            raise ValueError("Baseline predictions fingerprint mismatch")
    return state


def verify_checkpoint(directory):
    directory = Path(directory)
    marker = directory / "COMPLETE.json"
    if not marker.is_file():
        raise ValueError("Incomplete checkpoint: COMPLETE.json missing")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p != marker}
    if set(manifest["files"]) != files:
        raise ValueError("Checkpoint artifact set mismatch")
    for relative, entry in manifest["files"].items():
        path = contained_file(directory, relative)
        if file_sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Corrupt checkpoint artifact: {relative}")
    state = validate_payload(directory)
    if manifest["step"] != state["global_step"]:
        raise ValueError("Checkpoint manifest step mismatch")
    return manifest


def choose_verified(run_directory):
    valid = []
    for directory in Path(run_directory).glob("checkpoint-*"):
        if not directory.is_dir():
            continue
        try:
            manifest = verify_checkpoint(directory)
            valid.append((manifest["step"], directory))
        except (ValueError, OSError, KeyError, TypeError):
            continue
    if not valid:
        raise ValueError("No verified full checkpoint available")
    return max(valid, key=lambda item: item[0])[1]


def synchronize_checkpoint(source, drive_run_directory, run_directory):
    source, drive = Path(source).resolve(), Path(drive_run_directory).resolve()
    run_directory = Path(run_directory).resolve()
    step = checkpoint_step(source)
    if source.parent != run_directory:
        raise ValueError("Checkpoint source must be directly inside its local run directory")
    if drive == run_directory or run_directory in drive.parents or drive in run_directory.parents:
        raise ValueError("Checkpoint source and Drive destination must be separate")
    drive.mkdir(parents=True, exist_ok=True)
    destination = drive / source.name
    # Context/metrics copied into the local checkpoint before hashing.
    for name in ("resolved_config.json", "run_manifest.json"):
        shutil.copyfile(run_directory / name, source / name)
    if (run_directory / "source").is_dir():
        shutil.copytree(run_directory / "source", source / "source", dirs_exist_ok=True)
    context = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    evaluated = (run_directory / "validation" / f"step-{step}" / "metrics.json").is_file()
    if evaluated or context.get("model", {}).get("method") != "lora":
        validation = source / "validation"
        validation.mkdir(exist_ok=True)
        for name in ("metrics.json", "predictions.jsonl"):
            shutil.copyfile(run_directory / "validation" / f"step-{step}" / name, validation / name)
    baseline = run_directory / "validation" / "step-0"
    if baseline.is_dir():
        shutil.copytree(baseline, source / "pretrained_baseline", dirs_exist_ok=True)
    state = validate_payload(source)
    best_name = Path(state["best_model_checkpoint"]).name if state.get("best_model_checkpoint") else None
    if best_name and best_name != source.name:
        verify_checkpoint(drive / best_name)
    inventory = {p.relative_to(source).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in source.rglob("*") if p.is_file() and p.name != "COMPLETE.json"}
    if destination.exists():
        existing = verify_checkpoint(destination)
        if existing["files"] != inventory:
            raise ValueError("Refuse to overwrite a different published checkpoint")
        return existing
    size = sum(entry["bytes"] for entry in inventory.values())
    free = shutil.disk_usage(drive).free
    if size > free:
        raise ValueError("Insufficient destination filesystem space; Drive account quota must also be checked separately")
    started = time.monotonic()
    destination.mkdir(exist_ok=False)
    for relative in inventory:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained_file(source, relative), target)
    # Compare destination before publishing a marker. A failed copy stays incomplete.
    for relative, entry in inventory.items():
        target = contained_file(destination, relative)
        if target.stat().st_size != entry["bytes"] or file_sha256(target) != entry["sha256"]:
            raise ValueError(f"Upload verification failed: {relative}")
    manifest = {"step": step, "files": inventory, "bytes": size,
        "sync_seconds": time.monotonic() - started, "status": (
            "COMPLETE_LORA_CHECKPOINT" if context.get("model", {}).get("method") == "lora" else "COMPLETE_FULL_CHECKPOINT")}
    write_json(destination / "COMPLETE.json", manifest)
    verify_checkpoint(destination)
    best = drive / best_name if best_name else None
    if best:
        verify_checkpoint(best)
        if not (best / "validation/metrics.json").is_file():
            raise ValueError("Best checkpoint must have its own evaluated metrics")
    valid = []
    for candidate in drive.glob("checkpoint-*"):
        try:
            valid.append((verify_checkpoint(candidate)["step"], candidate))
        except (ValueError, OSError, KeyError, TypeError):
            continue
    valid.sort(reverse=True, key=lambda item: item[0])
    keep = {p for _, p in valid[:2]} | ({best} if best else set())
    # Update pointer only after new and best are verified. Never trust it without hashes.
    write_json(drive / "latest.json", {"checkpoint": valid[0][1].name, "step": valid[0][0],
        "best": best.name if best else None, "previous": valid[1][1].name if len(valid) > 1 else None})
    if context.get("checkpoint", {}).get("preserve_all"):
        return manifest
    for _, candidate in valid:
        resolved = candidate.resolve()
        if candidate not in keep:
            if resolved.parent != drive or not re.fullmatch(r"checkpoint-\d+", resolved.name):
                raise ValueError("Unsafe retention path")
            shutil.rmtree(resolved)
    return manifest


def resume_signature(manifest):
    config = manifest["config"]
    return {
        "model": config["model"], "phase": config["run"]["phase"],
        "training": config["training"], "generation": config["generation"],
        "lengths": [config["data"]["max_source_tokens"], config["data"]["max_target_tokens"]],
        "data_manifest": manifest["data"]["manifest_sha256"], "cache_key": manifest["cache_key"],
        "train_ids": manifest["train_ids_sha256"], "train_count": manifest["train_count"],
        "val_count": manifest["val_count"], "packages": manifest["environment"]["packages"],
        "python": manifest["environment"].get("python"),
        "code": {Path(name).name: digest for name, digest in manifest["code"]["source_sha256"].items()},
    }


def restore_verified(drive_run_directory, new_local_run, expected_signature):
    drive, destination = Path(drive_run_directory).resolve(), Path(new_local_run).resolve()
    if drive == destination or drive in destination.parents or destination in drive.parents:
        raise ValueError("Restore destination must be separate from Drive run")
    if destination.exists():
        raise FileExistsError("Resume requires a new local run directory")
    candidates = []
    for candidate in drive.glob("checkpoint-*"):
        try:
            manifest = verify_checkpoint(candidate)
            state = json.loads((candidate / "trainer_state.json").read_text(encoding="utf-8"))
            best_name = Path(state["best_model_checkpoint"]).name if state.get("best_model_checkpoint") else None
            if best_name:
                verify_checkpoint(drive / best_name)
            candidates.append((manifest["step"], candidate, best_name))
        except (ValueError, OSError, KeyError, TypeError):
            continue
    if not candidates:
        raise ValueError("No verified checkpoint with verified best dependency available")
    _, latest, best_name = max(candidates, key=lambda item: item[0])
    previous = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    if resume_signature(previous) != expected_signature:
        raise ValueError("Incompatible resume signature: data/cache/code/environment/phase/budget/hyperparameters changed")
    destination.mkdir(parents=True, exist_ok=False)
    dependencies = {latest.name} | ({best_name} if best_name else set())
    for name in dependencies:
        shutil.copytree(drive / name, destination / name)
        verify_checkpoint(destination / name)
    for name in ("resolved_config.json", "run_manifest.json"):
        shutil.copyfile(latest / name, destination / name)
    if (latest / "source").is_dir():
        shutil.copytree(latest / "source", destination / "source")
    local = destination / latest.name
    state_path = local / "trainer_state.json"
    original = json.loads(state_path.read_text(encoding="utf-8"))
    write_json(destination / "resume_original_trainer_state.json", original)
    remapped = dict(original, best_model_checkpoint=str(destination / best_name) if best_name else None)
    # This local state is intentionally edited after byte verification: remove the
    # COMPLETE marker, retain the original state and an explicit provenance map.
    (local / "COMPLETE.json").unlink()
    write_json(state_path, remapped)
    write_json(destination / "resume_mapping.json", {"source": str(latest), "checkpoint": str(local),
        "verified_before_remap": True, "old_best": original["best_model_checkpoint"],
        "new_best": remapped["best_model_checkpoint"], "signature": expected_signature})
    for name in dependencies:
        step = checkpoint_step(destination / name)
        target = destination / "validation" / f"step-{step}"
        if (destination / name / "validation").is_dir():
            shutil.copytree(destination / name / "validation", target)
    if (local / "pretrained_baseline").is_dir():
        shutil.copytree(local / "pretrained_baseline", destination / "validation" / "step-0")
    return local
