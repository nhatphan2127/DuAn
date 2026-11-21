"""Record the runtime and fail closed when the Colab environment gate is unmet."""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def read_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            pins[name] = version
    return pins


def drive_quota_evidence(free_gib, source):
    if free_gib is None or not math.isfinite(free_gib) or free_gib < 0 or not source or not source.strip():
        raise ValueError("Provide nonnegative Drive account free GiB and its quota source; mount df is not account quota")
    return {"free_gib": free_gib, "source": source.strip(), "kind": "manual_account_quota"}


def collect_environment(pins_path, work_dir, inspect_only=False, drive_free_gib=None,
                        drive_quota_source=None, required_drive_gib=32):
    pins = read_pins(pins_path)
    disk = shutil.disk_usage(work_dir)
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "mode": "inspect_only" if inspect_only else "verify",
        "pin_status": "candidate_not_colab_validated",
        "packages": {},
        "local_disk": {"path": str(Path(work_dir).resolve()), "free_bytes": disk.free},
        "errors": [],
    }
    if Path("/proc/meminfo").exists():
        report["ram_meminfo"] = Path("/proc/meminfo").read_text().splitlines()[:3]
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        report["nvidia_smi"] = {"returncode": result.returncode, "output": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = {"error": str(exc)}
    inventory = dict(pins)
    inventory.update({name: None for name in ("numpy", "tokenizers", "safetensors", "huggingface-hub", "psutil")})
    for name, expected in inventory.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        report["packages"][name] = {"expected": expected, "actual": actual}
        # A local CUDA build suffix is compatible with the same release pin.
        if not inspect_only and expected is not None and (actual or "").split("+")[0] != expected:
            report["errors"].append(f"{name}: expected {expected}, found {actual}")
    if inspect_only:
        report["status"] = "INSPECTED_NOT_ACCEPTED"
        return report
    try:
        quota = drive_quota_evidence(drive_free_gib, drive_quota_source)
        quota["required_gib_proposal"] = required_drive_gib
        report["drive_quota"] = quota
        if quota["free_gib"] < required_drive_gib:
            report["errors"].append("Insufficient Drive quota for proposed retention/staging; measure real checkpoints at T08")
    except ValueError as exc:
        report["errors"].append(str(exc))
    # Text-only run: avoid activating TensorFlow; never install into data venv.
    os.environ["USE_TORCH"] = "1"
    os.environ["USE_TF"] = "0"
    try:
        torch = importlib.import_module("torch")
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            report["errors"].append("CUDA unavailable; CPU cannot accept the Colab gate")
        else:
            device = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": device.name, "total_memory_bytes": device.total_memory}
            report["cuda_tensor_result"] = (torch.ones(1, device="cuda") + 1).item()
            torch.cuda.synchronize()
        transformers = importlib.import_module("transformers")
        for module in ("datasets", "accelerate", "sentencepiece", "sacrebleu", "peft"):
            importlib.import_module(module)
        checks = {
            "peft_seq2seq_lora": all(hasattr(importlib.import_module("peft"), name)
                for name in ("LoraConfig", "PeftModel", "TaskType", "get_peft_model")),
            "eval_strategy": "eval_strategy" in inspect.signature(transformers.Seq2SeqTrainingArguments).parameters,
            "save_only_model": "save_only_model" in inspect.signature(transformers.Seq2SeqTrainingArguments).parameters,
            "restore_callback_states_from_checkpoint": "restore_callback_states_from_checkpoint" in inspect.signature(transformers.Seq2SeqTrainingArguments).parameters,
            "processing_class": "processing_class" in inspect.signature(transformers.Seq2SeqTrainer).parameters,
            "resume_from_checkpoint": "resume_from_checkpoint" in inspect.signature(transformers.Seq2SeqTrainer.train).parameters,
        }
        for name in ("AutoTokenizer", "AutoModelForSeq2SeqLM", "DataCollatorForSeq2Seq"):
            checks[name] = getattr(transformers, name) is not None
        report["api_checks"] = checks
        if not all(checks.values()):
            report["errors"].append("Required Trainer/tokenizer API missing")
    except Exception as exc:
        # Preserve useful diagnostics even when the candidate environment cannot import.
        report["errors"].append(f"Import/CUDA probe failed: {type(exc).__name__}: {exc}")
    report["status"] = "WAITING_FOR_COLAB" if report["errors"] else "ENVIRONMENT_PROBE_PASSED_REVIEW_REQUIRED"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", default="nllb_lora/requirements-colab.lock")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--drive-free-gib", type=float)
    parser.add_argument("--drive-quota-source")
    parser.add_argument("--required-drive-gib", type=float, default=32)
    args = parser.parse_args()
    if not math.isfinite(args.required_drive_gib) or args.required_drive_gib <= 0:
        parser.error("--required-drive-gib must be positive")
    report = collect_environment(args.pins, args.work_dir, args.inspect_only,
                                 args.drive_free_gib, args.drive_quota_source,
                                 args.required_drive_gib)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "errors": report["errors"]}))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
