"""Restart-free Colab pilot orchestration. The notebook kernel uses stdlib only."""

from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
import venv


CODE_FILES = (
    "nllb_lora/requirements-colab.lock", "configs/nllb_zh_vi.json", "configs/nllb_lora_zh_vi.json",
    "configs/nllb_lora_vi_zh.json", "configs/nllb_lora_shared.json",
    "configs/colab_readiness.template.json",
    *[f"nllb_lora/train/{name}.py" for name in (
        "colab_workflow", "check_colab_environment", "sync_checkpoints",
        "train_nllb", "check_readiness", "prepare_data", "profile_speed", "lora_adapter")],
    *[f"nllb_lora/eval/{name}.py" for name in (
        "translation_metrics", "evaluate", "evaluate_matched", "finalize_model", "compare_translations")],
)


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inventory(directory, names=None):
    directory = Path(directory)
    names = names if names is not None else sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())
    return {name: {"bytes": (directory / name).stat().st_size,
                   "sha256": digest(directory / name)} for name in names}


def verify_tree(directory, files):
    directory = Path(directory).resolve()
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError(f"Artifact set mismatch: {directory}")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if directory not in path.parents or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Hash/size mismatch: {name} in {directory}")


def quarantine(path):
    """Preserve an incomplete generated artifact next to its original location."""
    path = Path(path)
    target = path.with_name(".incomplete-" + path.name + "-" + uuid.uuid4().hex[:10])
    path.rename(target)
    print(f"Preserved incomplete artifact: {target}", flush=True)
    return target


def publish_tree(source, destination):
    """Copy, verify destination bytes, then rename. Never publish a partial copy."""
    source, destination = Path(source), Path(destination)
    files = inventory(source)
    if destination.exists():
        try:
            verify_tree(destination, files)
            return files
        except (ValueError, OSError):
            quarantine(destination)
    temporary = destination.with_name(destination.name + ".copying-" + uuid.uuid4().hex[:10])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    verify_tree(temporary, files)
    temporary.rename(destination)
    return files


def prepare_source(source, local_base):
    source = Path(source)
    missing = [name for name in CODE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing code files in " + str(source) + ":\n" + "\n".join(missing))
    files = inventory(source, CODE_FILES)
    identity = fingerprint(files)
    target = Path(local_base) / identity
    if target.exists():
        try:
            # Only selected source files count; Python may create __pycache__ later.
            if inventory(target, CODE_FILES) == files:
                return target, identity
        except OSError:
            pass
        quarantine(target)
    temporary = target.with_name(target.name + ".copying-" + uuid.uuid4().hex[:10])
    for name in CODE_FILES:
        output = temporary / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, output)
    if inventory(temporary, CODE_FILES) != files:
        raise ValueError("Source changed during snapshot copy; rerun after Drive finishes syncing")
    temporary.rename(target)
    return target, identity


def run_command(command, cwd, log):
    """Stream stdout+stderr to Colab and a durable UTF-8 log; preserve real errors."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", USE_TORCH="1", USE_TF="0", USE_FLAX="0")
    # Never inherit host notebook's module search path into the isolated Python.
    env.pop("PYTHONPATH", None)
    tail = deque(maxlen=35)
    print(f"Running: {command}\nLog: {log}", flush=True)
    with log.open("a", encoding="utf-8") as output:
        output.write("\nCOMMAND " + repr(command) + "\n")
        with subprocess.Popen([str(item) for item in command], cwd=cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                    tail.append(line)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if code:
        raise RuntimeError(f"Command failed (exit {code}). Log: {log}\n" + "".join(tail))


def data_api(root):
    spec = importlib.util.spec_from_file_location("workflow_data", Path(root) / "nllb_lora/train/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_data(root, bundle, destination):
    api = data_api(root)
    config = read_json(Path(root) / "configs/nllb_zh_vi.json")
    bundle, destination = Path(bundle), Path(destination)
    if destination.exists():
        try:
            return api.verify_inputs(config, destination)
        except (OSError, ValueError, KeyError):
            quarantine(destination)
    temporary = destination.with_name(destination.name + ".copying-" + uuid.uuid4().hex[:10])
    temporary.mkdir(parents=True)
    for name in (config["data"]["manifest"], config["data"]["manifest_sha256_file"]):
        shutil.copyfile(bundle / name, temporary / name)
    manifest_path = temporary / config["data"]["manifest"]
    if digest(manifest_path) != config["data"]["expected_manifest_sha256"]:
        raise ValueError("Bundle manifest differs from the pinned dataset")
    entries = {item["path"]: item for item in read_json(manifest_path)["files"]}
    for split in config["data"]["splits"].values():
        for side in ("source", "target"):
            name = split[side]
            source, target = bundle / name, temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                shutil.copyfile(source, target)
            elif name == "final/train.vi":
                parts = sorted(source.parent.glob("train.vi.part*"))
                if [p.name for p in parts] != ["train.vi.part000", "train.vi.part001"]:
                    raise ValueError("Expected exactly train.vi.part000 and train.vi.part001, or the original train.vi")
                with target.open("wb") as output:
                    for part in parts:
                        with part.open("rb") as stream:
                            shutil.copyfileobj(stream, output)
            else:
                raise FileNotFoundError(f"Missing bundle input: {source}")
            expected = entries[name]
            if target.stat().st_size != expected["bytes"] or digest(target) != expected["sha256"]:
                raise ValueError(f"Bundle input or joined parts have wrong hash/size: {name}")
    api.verify_inputs(config, temporary)
    temporary.rename(destination)
    return api.verify_inputs(config, destination)


def require_complete(directory, budget):
    completion = read_json(Path(directory) / "completion.json")
    if (not completion.get("experiment_complete") or
        completion.get("reason") not in ("budget_complete", "early_stopping") or
        completion.get("total_budget") != budget or
        (completion["reason"] == "budget_complete" and completion["global_step"] < budget)):
        raise ValueError("Training is incomplete; evaluation/export must wait for a complete run")
    return completion


class Workflow:
    def __init__(self, settings, root, source_identity):
        self.settings, self.root = settings, Path(root).resolve()
        self.config = self.root / settings.get("config", "configs/nllb_zh_vi.json")
        if settings.get("require_lora") and read_json(self.config)["model"]["method"] != "lora":
            raise ValueError("This workflow requires LoRA; full fine-tuning fallback refused")
        if settings["mode"] != "full" and len(data_api(self.root).direction_specs(read_json(self.config))) > 1:
            raise ValueError("A shared multi-direction adapter supports mode 'full' only")
        base = Path(settings["drive_output"]) / "run-all"
        base.mkdir(parents=True, exist_ok=True)
        probe = base / (".write-probe-" + uuid.uuid4().hex)
        probe.write_text("write probe", encoding="utf-8")
        probe.unlink()
        self.identity = {"source": source_identity, "mode": settings["mode"],
                         "config": digest(self.config)}
        if settings["mode"] not in ("pilot_complete", "full_after_review", "full"):
            raise ValueError("Mode must be pilot_complete, full_after_review or full")
        if settings["mode"] == "full_after_review" and not Path(settings["readiness"]).is_file():
            raise FileNotFoundError("Full mode requires an existing reviewed readiness.json")
        active = base / ("active-" + settings["mode"] + ".json")
        run_id = settings.get("experiment_id")
        if run_id is None:
            if active.is_file():
                run_id = read_json(active)["experiment_id"]
            else:
                run_id = "zhvi-" + settings["mode"] + "-" + uuid.uuid4().hex[:10]
                write_json(active, {"experiment_id": run_id})
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("experiment_id must contain only letters, digits, '_' or '-'")
        self.directory = base / run_id
        self.directory.mkdir(exist_ok=True)
        self.state_file = self.directory / "workflow.json"
        if self.state_file.is_file():
            self.state = read_json(self.state_file)
            if self.state["identity"] != self.identity:
                raise ValueError("Existing experiment has different code/config/mode. Restore its snapshot or set a new experiment_id; existing evidence is preserved")
        else:
            self.state = {"experiment_id": run_id, "identity": self.identity, "phases": {}}
            self.save()
        self.local = Path(settings["local_work"]) / run_id
        self.local.mkdir(parents=True, exist_ok=True)
        archive, _ = prepare_source(self.root, self.directory / "code")
        self.state["source_archive"] = str(archive)
        self.save()
        self.python = Path(settings["venv"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.frozen = self.local / "frozen"
        self.cache = self.local / "cache"
        self.inputs = None
        if self.state.get("speed_tuning"):
            self.use_tuned_config()

    def use_tuned_config(self):
        entry = self.state["speed_tuning"]
        record = self.directory / "speed_tuning.json"
        config = self.directory / "full-config.json"
        if digest(record) != entry["record_sha256"] or digest(config) != entry["config_sha256"]:
            raise ValueError("Locked GPU tuning/config changed; restore recorded files before resume")
        report = read_json(record)
        if report["status"] != "MEASURED" or read_json(config) != report["selected_config"]:
            raise ValueError("Locked full config differs from measured selection")
        self.config = config

    def tune_speed(self):
        if self.state.get("speed_tuning"):
            self.use_tuned_config()
            print("Reusing locked GPU speed configuration; no retuning on resume", flush=True)
            return
        if self.state["phases"] or any((self.directory / "checkpoints/full").glob("checkpoint-*")):
            raise ValueError("Speed tuning requires a new experiment; existing training stays frozen")
        context = {"identity": self.identity, "cache_key": self.state["cache_key"],
                   "environment": self.state["environment_signature"]}
        context_path = self.local / "speed-context.json"
        write_json(context_path, context)
        self.command("speed-tuning", ["-m", "nllb_lora.train.profile_speed", "--config", str(self.config),
            "--cache", str(self.cache), "--output", str(self.directory / "speed-tuning"),
            "--context", str(context_path)])
        record = self.directory / "speed_tuning.json"
        report = read_json(record)
        if report.get("status") != "MEASURED" or report.get("context") != context:
            raise ValueError("GPU speed measurement is incomplete or belongs to another run")
        selected = report["selected_config"]
        original = read_json(self.config)
        # Tuning may change execution settings only, never the experiment itself.
        allowed = {"per_device_train_batch_size", "gradient_accumulation_steps", "per_device_eval_batch_size",
            "fp16", "bf16", "gradient_checkpointing", "optim", "attn_implementation",
            "dataloader_num_workers", "dataloader_pin_memory", "dataloader_persistent_workers", "group_by_length"}
        expected = json.loads(json.dumps(original))
        for key in allowed:
            if key in selected["training"]:
                expected["training"][key] = selected["training"][key]
        expected["generation"]["use_cache"] = True
        training = selected["training"]
        batch = training["per_device_train_batch_size"]
        if (selected != expected or type(batch) is not int or batch not in (1, 2, 4, 8, 16, 32)
            or batch * training["gradient_accumulation_steps"] != 32
            or training["fp16"] == training["bf16"]
            or (training.get("gpu_profile") == "t4" and (training["bf16"] or training.get("tf32") is not False))):
            raise ValueError("Speed tuning changed protected method/data/budget settings")
        config = self.directory / "full-config.json"
        write_json(config, selected)
        self.state["speed_tuning"] = {"record_sha256": digest(record), "config_sha256": digest(config)}
        self.save()
        self.use_tuned_config()

    def save(self):
        write_json(self.state_file, self.state)

    def command(self, name, args, python=None):
        run_command([python or self.python, *args], self.root, self.directory / "logs" / f"{name}.log")

    def environment(self):
        stamp = self.python.parent.parent / "workflow_environment.json"
        expected = {"python": sys.version, "lock": digest(self.root / "nllb_lora/requirements-colab.lock")}
        frozen_lock = self.directory / "requirements-resolved.txt"
        if frozen_lock.is_file():
            expected["resolved_lock"] = digest(frozen_lock)
        reuse = self.python.is_file() and stamp.is_file() and read_json(stamp)["identity"] == expected
        if not reuse:
            location = self.python.parent.parent
            if location.exists():
                quarantine(location)
            # Colab's Debian Python may omit ensurepip. Host pip --python can
            # manage a stdlib venv without installing anything into the kernel.
            venv.EnvBuilder(with_pip=False, system_site_packages=False).create(location)
            requirements = frozen_lock if frozen_lock.is_file() else self.root / "nllb_lora/requirements-colab.lock"
            pip = [sys.executable, "-m", "pip", "--python", str(self.python)]
            run_command(pip + ["install", "--dry-run", "--only-binary=:all:", "-r", str(requirements)],
                        self.root, self.directory / "logs/environment-install.log")
            run_command(pip + ["install", "--only-binary=:all:", "-r", str(requirements)],
                        self.root, self.directory / "logs/environment-install.log")
            run_command(pip + ["check"], self.root, self.directory / "logs/environment-install.log")
            result = subprocess.run(pip + ["freeze"], cwd=self.root, check=True, capture_output=True, text=True)
            if not frozen_lock.is_file():
                frozen_lock.write_text(result.stdout, encoding="utf-8")
            expected["resolved_lock"] = digest(frozen_lock)
        pip = [sys.executable, "-m", "pip", "--python", str(self.python)]
        current = subprocess.run(pip + ["freeze"], cwd=self.root, check=True, capture_output=True, text=True)
        if sorted(current.stdout.splitlines()) != sorted(frozen_lock.read_text(encoding="utf-8").splitlines()):
            raise ValueError("Installed dependency inventory differs from requirements-resolved.txt")
        output = self.directory / "environment.json"
        self.command("environment-probe", ["-m", "nllb_lora.train.check_colab_environment",
            "--pins", str(self.root / "nllb_lora/requirements-colab.lock"), "--output", str(output),
            "--drive-free-gib", str(self.settings["drive_free_gib"]),
            "--drive-quota-source", self.settings["drive_quota_source"]])
        self.command("cuda-kernel-probe", ["-c", "import torch; "
            "x=torch.randn(16,16,device='cuda',requires_grad=True); "
            "y=(x.half() @ x.half()).float().square().mean(); y.backward(); "
            "torch.cuda.synchronize(); assert torch.isfinite(y).item() and torch.isfinite(x.grad).all().item(); "
            "print('CUDA FP16 forward/backward passed:',torch.cuda.get_device_name(0))"])
        report = read_json(output)
        signature = {"python": report["python"], "packages": report["packages"], "resolved_lock": digest(frozen_lock)}
        if "environment_signature" in self.state and self.state["environment_signature"] != signature:
            raise ValueError("Python/package versions changed since this experiment; restore the recorded environment")
        if reuse and read_json(stamp)["signature"] != signature:
            raise ValueError("The isolated environment was changed; restore it before continuing")
        self.state["environment_signature"] = signature
        self.save()
        write_json(stamp, {"identity": expected, "signature": signature})
        print("CUDA/API probe passed. Model memory and learning still require benchmark/sanity review.", flush=True)

    def prepare(self):
        if shutil.disk_usage(self.local).free < 32 * 1024**3:
            raise ValueError("Less than 32 GiB local disk free for cache/model/optimizer checkpoints")
        self.inputs = stage_data(self.root, self.settings["drive_bundle"], self.frozen)
        write_json(self.directory / "input_manifest.json", self.inputs)
        self.command("token-contract", ["-m", "nllb_lora.train.prepare_data", "--config", str(self.config),
            "--frozen-root", str(self.frozen), "--token-contract", "--output", str(self.directory / "token_contract.json")])
        config = read_json(self.config)
        versions = self.state["environment_signature"]["packages"]
        # Same schema as cache_specification; use versions from the isolated probe.
        specification = {"schema_version": 1, "input_manifest_sha256": self.inputs["manifest_sha256"],
            "splits": {name: self.inputs["splits"][name] for name in ("train", "val")},
            "model": config["model"], "legacy_behaviour": False,
            "versions": {name: versions[name]["actual"] for name in ("transformers", "tokenizers", "datasets", "sentencepiece")},
            "policy": {"max_source_tokens": config["data"]["max_source_tokens"], "max_target_tokens": config["data"]["max_target_tokens"],
                       "padding": False, "truncation": False, "overflow": "fail"}}
        api = data_api(self.root)
        cache_drive = Path(self.settings["drive_output"]) / "caches" / api.cache_key(specification)
        for candidate in (self.cache, cache_drive):
            if candidate.exists():
                try:
                    api.verify_cache(candidate, specification)
                except (ValueError, OSError, KeyError):
                    quarantine(candidate)
        if not self.cache.exists() and cache_drive.exists():
            publish_tree(cache_drive, self.cache)
        self.command("cache", ["-m", "nllb_lora.train.prepare_data", "--config", str(self.config),
            "--frozen-root", str(self.frozen), "--prepare-cache", "--cache-dir", str(self.cache),
            "--max-source-tokens", str(config["data"]["max_source_tokens"]),
            "--max-target-tokens", str(config["data"]["max_target_tokens"]),
            "--output", str(self.directory / "cache_manifest.json")])
        api.verify_cache(self.cache, specification)
        publish_tree(self.cache, cache_drive)
        api.verify_cache(cache_drive, specification)
        self.state["cache_key"] = api.cache_key(specification)
        self.save()

    def prior_phase(self, name):
        entry = self.state["phases"].get(name)
        if entry:
            if entry["identity"] != self.identity or entry["cache_key"] != self.state["cache_key"]:
                raise ValueError(f"Phase identity changed: {name}")
            try:
                verify_tree(self.directory / "artifacts" / name, entry["files"])
            except (OSError, ValueError):
                target = self.directory / "artifacts" / name
                if target.exists():
                    quarantine(target)
                del self.state["phases"][name]
                self.save()
                return None
        return entry

    def publish_phase(self, name, source):
        files = publish_tree(source, self.directory / "artifacts" / name)
        self.state["phases"][name] = {"identity": self.identity, "cache_key": self.state["cache_key"], "files": files}
        self.save()

    def train_phase(self, phase, steps):
        if self.prior_phase(phase):
            directory = self.directory / "artifacts" / phase
            require_complete(directory, steps)
            self.validate_phase(phase, directory)
            print(f"Verified completed phase; reusing {phase}", flush=True)
            return directory
        # Same run basename across fresh attempts is required by checkpoint identity.
        attempt = self.local / "attempts" / uuid.uuid4().hex[:10]
        run = attempt / (self.state["experiment_id"] + "-" + phase)
        drive = self.directory / "checkpoints" / phase
        command = ["-m", "nllb_lora.train.train_nllb", "--config", str(self.config),
            "--frozen-root", str(self.frozen), "--cache-dir", str(self.cache),
            "--phase", phase, "--max-steps", str(steps), "--run-dir", str(run)]
        if phase in ("pilot", "full"):
            command += ["--drive-run-dir", str(drive)]
            self.repair_partial_uploads(drive)
            checkpoints = list(drive.glob("checkpoint-*"))
            if checkpoints:
                latest = self.select_checkpoint(drive)
                previous = read_json(latest / "run_manifest.json")
                self.check_resume_context(previous, phase, steps)
                saved = read_json(latest / "trainer_state.json")
                patience = previous["config"]["training"]["early_stopping_patience"]
                counter = saved.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {}).get("attributes", {}).get("early_stopping_patience_counter", 0)
                if saved["global_step"] >= steps or counter >= patience:
                    return self.recover_completed(phase, steps, drive)
                command += ["--resume-from-drive"]
            if phase == "full" and self.settings.get("readiness"):
                command += ["--readiness", self.settings["readiness"]]
        self.command(phase, command)
        require_complete(run, steps)
        self.validate_phase(phase, run)
        # Keep reports and best export; full optimizer checkpoints stay in Drive.
        publish = attempt / "publish"
        publish.mkdir()
        for path in run.iterdir():
            if path.name.startswith("checkpoint-") or (path.name == "best_model" and phase not in ("pilot", "full")):
                continue
            if path.is_dir():
                shutil.copytree(path, publish / path.name)
            else:
                shutil.copyfile(path, publish / path.name)
        self.publish_phase(phase, publish)
        # Only remove this successful, generated local attempt after readback.
        if self.local.resolve() not in run.resolve().parents:
            raise ValueError("Unsafe local run cleanup path")
        shutil.rmtree(run)
        return self.directory / "artifacts" / phase

    def select_checkpoint(self, drive):
        from nllb_lora.train.sync_checkpoints import verify_checkpoint
        candidates = []
        for candidate in drive.glob("checkpoint-*"):
            try:
                info = verify_checkpoint(candidate)
                state = read_json(candidate / "trainer_state.json")
                if state.get("best_model_checkpoint"):
                    best_name = Path(state["best_model_checkpoint"]).name
                    verify_checkpoint(drive / best_name)
                candidates.append((info["step"], candidate))
            except (ValueError, OSError, KeyError, TypeError):
                continue
        if not candidates:
            raise ValueError("No checkpoint with a verified best dependency; inspect preserved Drive artifacts")
        return max(candidates, key=lambda item: item[0])[1]

    def repair_partial_uploads(self, drive):
        from nllb_lora.train.sync_checkpoints import verify_checkpoint
        for checkpoint in drive.glob("checkpoint-*"):
            try:
                verify_checkpoint(checkpoint)
            except (ValueError, OSError, KeyError, TypeError):
                # Keep the bytes and remove the checkpoint-N name so callbacks
                # can publish that step again. Never select an incomplete upload.
                quarantine(checkpoint)
        # A complete payload can still reference a failed/missing best upload.
        # Iterate because removing an orphan may expose another orphan.
        changed = True
        while changed:
            changed = False
            for checkpoint in drive.glob("checkpoint-*"):
                try:
                    state = read_json(checkpoint / "trainer_state.json")
                    if state.get("best_model_checkpoint"):
                        verify_checkpoint(drive / Path(state["best_model_checkpoint"]).name)
                except (ValueError, OSError, KeyError, TypeError):
                    quarantine(checkpoint)
                    changed = True

    def check_resume_context(self, previous, phase, steps):
        from nllb_lora.train.prepare_data import file_sha256
        config = read_json(self.config)
        resolved = previous["config"]
        expected_training = dict(config["training"], total_optimizer_steps=steps,
            warmup_steps=math.ceil(steps * config["training"]["warmup_ratio_proposal"]))
        if (resolved["model"] != config["model"] or resolved["generation"] != config["generation"] or
            resolved["run"]["phase"] != phase or resolved["training"] != expected_training or
            previous["cache_key"] != self.state["cache_key"] or
            previous["data"]["manifest_sha256"] != self.inputs["manifest_sha256"]):
            raise ValueError("Checkpoint model/config/data/cache/budget differs from this experiment")
        # Trainer performs the complete resume_signature check before loading.
        for name, value in previous["code"]["source_sha256"].items():
            source = next((self.root / p for p in CODE_FILES if Path(p).name == name), None)
            if source is None or file_sha256(source) != value:
                raise ValueError(f"Checkpoint code changed: {name}")
        versions = {name: item["actual"] for name, item in self.state["environment_signature"]["packages"].items()}
        if previous["environment"]["packages"] != versions:
            raise ValueError("Checkpoint package versions changed")
        if previous["environment"]["python"] != self.state["environment_signature"]["python"].split()[0]:
            raise ValueError("Checkpoint Python version changed")

    def recover_completed(self, phase, steps, drive):
        from nllb_lora.eval.finalize_model import export_verified_best
        latest = self.select_checkpoint(drive)
        state = read_json(latest / "trainer_state.json")
        best_name = Path(state["best_model_checkpoint"]).name
        publish = self.local / ("recovered-" + uuid.uuid4().hex[:10])
        export_verified_best(drive, publish / "best_model")
        metric_dir = publish / "validation" / f"{phase}-best"
        metric_dir.mkdir(parents=True)
        best = read_json(drive / best_name / "validation/metrics.json")
        baseline = read_json(latest / "pretrained_baseline/metrics.json")
        write_json(metric_dir / "metrics.json", best)
        shutil.copyfile(drive / best_name / "validation/predictions.jsonl", metric_dir / "predictions.jsonl")
        shutil.copytree(latest / "pretrained_baseline", publish / "validation/step-0")
        write_json(publish / "learning_history.json", state.get("log_history", []))
        write_json(publish / f"{phase}_summary.json", {"pretrained_step_zero": baseline, "selected_best": best,
            "best_checkpoint": best_name, "bleu_delta": best["bleu"] - baseline["bleu"],
            "chrf_delta": best["chrf"] - baseline["chrf"], "status": "REQUIRES_MANUAL_QUALITY_REVIEW"})
        reason = read_json(publish / "best_model/provenance.json")["completion_reason"]
        write_json(publish / "completion.json", {"reason": reason, "global_step": state["global_step"],
            "total_budget": steps, "experiment_complete": True, "recovered_from_verified_checkpoint": str(latest)})
        shutil.copyfile(latest / "run_manifest.json", publish / "run_manifest.json")
        shutil.copyfile(latest / "resolved_config.json", publish / "resolved_config.json")
        self.validate_phase(phase, publish)
        self.publish_phase(phase, publish)
        return self.directory / "artifacts" / phase

    def validate_phase(self, phase, run):
        expected_rows = len(data_api(self.root).validation_layout(read_json(self.config)))
        for path in (Path(run) / "validation").glob("*/metrics.json"):
            metric = read_json(path)
            if phase in ("pilot", "full", "benchmark") and metric["count"] != expected_rows:
                raise ValueError(f"Official validation must have {expected_rows} rows: {path}")
            for key in ("bleu", "chrf"):
                if not math.isfinite(metric[key]):
                    raise ValueError(f"Non-finite {key} in {path}")
            for key, value in metric.get("trainer_metrics", {}).items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"Non-finite {key} in {path}")
        if phase == "benchmark":
            report = read_json(Path(run) / "resource_benchmark.json")
            if not all(report.get(k) for k in ("resource_probe_passed", "finite_metrics", "optimizer_present_during_eval")) or report["validation_count"] != 5000:
                raise ValueError("Benchmark resource/full-validation check failed")
        if phase == "sanity":
            report = read_json(Path(run) / "sanity_learning.json")
            if not report["finite_loss"] or not report["learning_diagnostic_improved"]:
                raise ValueError("Sanity did not show finite, decreasing loss; inspect examples/log before continuing")
        if phase in ("pilot", "full"):
            weights = "best_model/adapter_config.json" if read_json(self.config)["model"]["method"] == "lora" else "best_model/config.json"
            for file in (weights, f"{phase}_summary.json", f"validation/{phase}-best/metrics.json"):
                if not (Path(run) / file).is_file():
                    raise FileNotFoundError(f"Missing completed {phase} artifact: {file}")

    def evaluate(self, phase, trained):
        from nllb_lora.eval.translation_metrics import verify_independent_scores
        name = phase + "-independent-val"
        expected = read_json(trained / "validation" / f"{phase}-best" / "metrics.json")
        if not self.prior_phase(name):
            output = self.local / (name + "-" + uuid.uuid4().hex[:10])
            require_complete(trained, 1000 if phase == "pilot" else self.full_budget())
            local_model = self.local / (phase + "-best-model")
            publish_tree(trained / "best_model", local_model)
            self.command(name, ["-m", "nllb_lora.eval.evaluate", "--config", str(self.config),
                "--model", str(local_model), "--frozen-root", str(self.frozen),
                "--split", "val", "--output-dir", str(output)])
            verify_independent_scores(read_json(output / "metrics.json"), expected)
            self.publish_phase(name, output)
        metrics = read_json(self.directory / "artifacts" / name / "metrics.json")
        if metrics["count"] != 5000 or metrics.get("split") != "val":
            raise ValueError("Independent evaluation must use the full validation split")
        verify_independent_scores(metrics, expected)
        return metrics

    def full_budget(self):
        from nllb_lora.train.check_readiness import full_training_budget
        config = read_json(self.config)
        return full_training_budget(data_api(self.root).train_example_count(config),
            config["training"]["per_device_train_batch_size"], config["training"]["gradient_accumulation_steps"],
            config["training"]["num_train_epochs"])

    def verify_t4_resume(self):
        name = "t4-smoke-resume"
        context = {"config_sha256": digest(self.config), "identity": self.identity,
                   "cache_key": self.state["cache_key"]}
        if self.prior_phase(name):
            report = read_json(self.directory / "artifacts" / name / "report.json")
            if report.get("context") != context or not report.get("passed"):
                raise ValueError("T4 smoke/resume context mismatch")
            return
        attempt = self.local / "attempts" / ("t4-probe-" + uuid.uuid4().hex[:10])
        config = read_json(self.config)
        config["training"].update(eval_steps=7, save_steps=7, early_stopping_patience=100)
        lora = config["model"]["method"] == "lora"
        stop = 2 if lora else 3
        if lora:
            config["training"].update(eval_steps=6, save_steps=2)
        config_path = attempt / "probe-config.json"
        write_json(config_path, config)
        drive = self.directory / "checkpoints" / ("t4-probe-" + attempt.name)
        basename = self.state["experiment_id"] + "-smoke-probe"
        first, resumed = attempt / "initial" / basename, attempt / "resumed" / basename
        common = ["-m", "nllb_lora.train.train_nllb", "--config", str(config_path),
            "--frozen-root", str(self.frozen), "--cache-dir", str(self.cache),
            "--phase", "smoke", "--max-steps", "7", "--drive-run-dir", str(drive)]
        self.command(name + "-initial", common + ["--run-dir", str(first), "--stop-after-step", str(stop)])
        from nllb_lora.train.sync_checkpoints import verify_checkpoint
        initial = verify_checkpoint(drive / f"checkpoint-{stop}")
        health = read_json(first / "smoke_health.json")
        if not health.get("passed") or not health.get("optimizer_present"):
            raise ValueError("T4 initial smoke failed")
        if self.local.resolve() not in first.resolve().parents:
            raise ValueError("Unsafe generated smoke cleanup path")
        shutil.rmtree(first)
        self.command(name + "-restore", common + ["--run-dir", str(resumed), "--resume-from-drive"])
        require_complete(resumed, 7)
        final = verify_checkpoint(drive / "checkpoint-7")
        probe = read_json(resumed / "resume_verification.json")
        restored_health = read_json(resumed / "smoke_health.json")
        self.validate_phase("smoke", resumed)
        if (probe.get("status") != "RESUME_PROBE_PASSED_REVIEW_REQUIRED" or
            not probe.get("loaded_state_checks") or not restored_health.get("passed")):
            raise ValueError("T4 fresh-process checkpoint resume failed")
        publish = attempt / "publish"
        write_json(publish / "report.json", {"passed": True, "context": context,
            "scope": "Fresh process on current GPU; fresh Colab runtime not established",
            "initial_health": health, "resumed_health": restored_health, "resume": probe,
            "validation": read_json(resumed / "validation/step-7/metrics.json"),
            "rng_payload_sha256": initial["files"]["rng_state.pth"]["sha256"],
            "checkpoint_steps": [initial["step"], final["step"]], "checkpoint_directory": str(drive)})
        self.publish_phase(name, publish)
        if self.local.resolve() not in attempt.resolve().parents:
            raise ValueError("Unsafe generated probe cleanup path")
        shutil.rmtree(attempt)

    def run(self):
        try:
            direct_full = self.settings["mode"] == "full"
            if direct_full and self.settings.get("auto_tune", False):
                self.tune_speed()
            if direct_full and not any((self.directory / "checkpoints/full").glob("checkpoint-*")):
                if read_json(self.config)["model"]["method"] == "lora" or read_json(self.config)["training"].get("gpu_profile") == "t4":
                    self.verify_t4_resume()
                else:
                    self.train_phase("smoke", 7)
            if self.settings["mode"] == "pilot_complete":
                for phase, steps in (("smoke", 7), ("benchmark", 100), ("sanity", 500), ("pilot", 1000)):
                    self.train_phase(phase, steps)
                phase = "pilot"
            else:
                phase = "full"
                self.train_phase(phase, self.full_budget())
            trained = self.directory / "artifacts" / phase
            metrics = read_json(trained / "validation/full-best/metrics.json") if direct_full else self.evaluate(phase, trained)
            summary = {"status": "COMPLETE" if direct_full else "COMPLETE_REQUIRES_MANUAL_QUALITY_REVIEW", "phase": phase,
                "experiment_id": self.state["experiment_id"], "bleu": metrics["bleu"], "chrf": metrics["chrf"],
                "count": metrics["count"], "best_model": str(trained / "best_model"),
                "results": str(self.directory), "fresh_runtime_resume_acceptance": "NOT_ESTABLISHED_BY_RUN_ALL"}
            config = read_json(self.config)
            directions = data_api(self.root).direction_specs(config)
            shared = len(directions) > 1
            if shared:
                summary["validation_directions"] = {name: {key: part[key] for key in ("bleu", "chrf", "count")}
                    for name, part in metrics["directions"].items()}
            if direct_full and self.settings.get("evaluate_test", True):
                tests = {}
                for direction in directions:
                    name = f"full-test-{direction['name']}" if shared else "full-test"
                    if not self.prior_phase(name):
                        output = self.local / (name + "-" + uuid.uuid4().hex[:10])
                        command = ["-m", "nllb_lora.eval.evaluate", "--config", str(self.config),
                            "--model", str(trained / "best_model"), "--frozen-root", str(self.frozen),
                            "--split", "test", "--output-dir", str(output)]
                        self.command(name, command + (["--direction", direction["name"]] if shared else []))
                        self.publish_phase(name, output)
                    test = read_json(self.directory / "artifacts" / name / "metrics.json")
                    expected_pairs = config["data"]["splits"]["test"]["expected_pairs"]
                    if test.get("split") != "test" or test["count"] != expected_pairs or not all(math.isfinite(test[k]) for k in ("bleu", "chrf")):
                        raise ValueError(f"Final test requires {expected_pairs} finite scored predictions")
                    tests[name] = test
                scores = list(tests.values())
                summary.update(test_bleu=sum(s["bleu"] for s in scores) / len(scores),
                    test_chrf=sum(s["chrf"] for s in scores) / len(scores),
                    test_count=sum(s["count"] for s in scores),
                    test_results=str(self.directory / "artifacts" / (next(iter(tests)) if not shared else "")))
                if shared:
                    summary["test_directions"] = {name: {key: test[key] for key in ("bleu", "chrf", "count")}
                        for name, test in tests.items()}
            write_json(self.directory / "summary.json", summary)
            return summary
        except Exception as exc:
            write_json(self.directory / "failure.json", {"error": str(exc), "type": type(exc).__name__,
                "recorded_at": datetime.now(timezone.utc).isoformat(), "logs": str(self.directory / "logs")})
            raise
