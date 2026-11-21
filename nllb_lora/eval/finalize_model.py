"""Export the best verified checkpoint after a completed training budget or early stop."""

import argparse
import json
from pathlib import Path
import shutil

from nllb_lora.train.sync_checkpoints import choose_verified, verify_checkpoint
from nllb_lora.train.prepare_data import contained_file, file_sha256, write_json


def verify_inference_export(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "export_manifest.json").read_text(encoding="utf-8"))
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p.name != "export_manifest.json"}
    if files != set(manifest["files"]):
        raise ValueError("Export artifact set differs from manifest")
    for name, entry in manifest["files"].items():
        path = contained_file(directory, name)
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Corrupt inference export: {name}")
    return manifest


def export_verified_best(drive_run, output):
    drive_run, output = Path(drive_run).resolve(), Path(output).resolve()
    latest = choose_verified(drive_run)
    latest_manifest = verify_checkpoint(latest)
    state = json.loads((latest / "trainer_state.json").read_text(encoding="utf-8"))
    context = json.loads((latest / "resolved_config.json").read_text(encoding="utf-8"))
    budget = context["training"]["total_optimizer_steps"]
    counter = state.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {}).get("attributes", {}).get("early_stopping_patience_counter", 0)
    complete = state["global_step"] >= budget
    early_stop = counter >= context["training"]["early_stopping_patience"]
    if not complete and not early_stop:
        raise ValueError("Run stopped before budget without early-stopping evidence; final export refused")
    best_name = Path(state["best_model_checkpoint"]).name
    best = drive_run / best_name
    best_manifest = verify_checkpoint(best)
    if context.get("model", {}).get("method") == "lora":
        metric = json.loads((best / "validation/metrics.json").read_text(encoding="utf-8"))
        if metric["step"] != best_manifest["step"] or metric[context["training"]["metric_for_best_model"]] != state["best_metric"]:
            raise ValueError("Best adapter checkpoint does not match the selected metric")
    if any(output == p or p in output.parents for p in drive_run.glob("checkpoint-*")):
        raise ValueError("Export must be outside checkpoint directories")
    frozen = Path(context["data"]["frozen_root"]).resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError("Export must be outside frozen inputs")
    lora = context.get("model", {}).get("method") == "lora"
    names = {"generation_config.json", "tokenizer_config.json", "tokenizer.json"}
    names.update(name for name in ("sentencepiece.bpe.model", "special_tokens_map.json", "added_tokens.json") if (best / name).is_file())
    if lora:
        names.update(("adapter_config.json", "adapter_model.safetensors", "base_model.json"))
    elif (best / "model.safetensors").is_file():
        names.add("config.json")
        names.add("model.safetensors")
    else:
        names.add("config.json")
        names.add("model.safetensors.index.json")
        names.update(json.loads((best / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"].values())
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(names):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained_file(best, name), target)
        if file_sha256(target) != best_manifest["files"][name]["sha256"]:
            raise ValueError(f"Export hash mismatch: {name}")
    write_json(output / "provenance.json", {"phase": context["run"]["phase"],
        "model": context["model"], "latest_step": state["global_step"], "total_budget": budget,
        "completion_reason": "budget_complete" if complete else "early_stopping",
        "best_checkpoint": best.name, "best_weight_step": best_manifest["step"],
        "best_metric": state["best_metric"], "source_checkpoint_manifest_sha256": file_sha256(best / "COMPLETE.json"),
        "latest_checkpoint_manifest_sha256": file_sha256(latest / "COMPLETE.json"),
        "latest_verified_bytes": latest_manifest["bytes"], "scope": "inference export; not a resume checkpoint"})
    files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in output.rglob("*") if p.is_file()}
    result = {"status": "VERIFIED_INFERENCE_EXPORT", "files": files, "best_weight_step": best_manifest["step"]}
    write_json(output / "export_manifest.json", result)
    verify_inference_export(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-run", required=True)
    parser.add_argument("--output", required=True, help="New inference export directory, preferably on Drive")
    args = parser.parse_args()
    result = export_verified_best(args.drive_run, args.output)
    print(json.dumps({"status": result["status"], "best_weight_step": result["best_weight_step"]}))


if __name__ == "__main__":
    main()
