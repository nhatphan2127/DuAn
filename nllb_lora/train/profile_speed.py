"""GPU presets and bounded, isolated speed trials for full-data Colab training."""

import copy
import math
import json
from pathlib import Path
import subprocess
import sys
import time


def is_t4(config):
    return config["training"].get("gpu_profile") == "t4"


def select_profile(gpu_name, total_bytes, train_batch=None, eval_batch=None):
    gib = total_bytes / 1024**3
    name = gpu_name.upper()
    if name in ("TESLA T4", "NVIDIA T4", "T4") and gib >= 14:
        tag, batch = "t4", 1
    elif "A100" in name and gib >= 35:
        tag, batch = ("a100-80", 8) if gib >= 70 else ("a100-40", 4)
    elif ("RTX PRO 6000" in name or "BLACKWELL" in name) and gib >= 70:
        tag, batch = "g4-blackwell", 8
    else:
        raise ValueError(f"No initial profile for {gpu_name} ({gib:.1f} GiB); use T4, G4 Blackwell or A100")
    training = batch if train_batch is None else train_batch
    evaluation = batch if eval_batch is None else eval_batch
    if type(training) is not int or training not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Train microbatch must be a positive divisor of effective batch 32")
    if type(evaluation) is not int or not 1 <= evaluation <= 32:
        raise ValueError("Evaluation batch must be in [1,32]")
    return {"name": f"{tag}-t{training}-e{evaluation}", "gpu": gpu_name,
        "gpu_total_bytes": total_bytes, "per_device_train_batch_size": training,
        "gradient_accumulation_steps": 32 // training, "per_device_eval_batch_size": evaluation,
        "effective_batch": 32, "fp16": True, "bf16": False,
        "qualification_interval": 500, "full_interval": 2000,
        "status": "INITIAL_PROFILE_REQUIRES_GPU_QUALIFICATION"}


def profile_config(original, profile, full=False):
    config = copy.deepcopy(original)
    config["status"] = "fast_profile_requires_review"
    interval = profile["full_interval"] if full else profile["qualification_interval"]
    config["training"].update({key: profile[key] for key in (
        "per_device_train_batch_size", "gradient_accumulation_steps", "per_device_eval_batch_size", "fp16", "bf16")})
    config["training"].update(eval_steps=interval, save_steps=interval,
        total_optimizer_steps=None, warmup_steps=None)
    if config["model"]["method"] == "lora":
        config["training"].update(eval_steps=original["training"]["eval_steps"], save_steps=original["training"]["save_steps"])
    if profile["name"].startswith("t4-"):
        config["training"].update(gpu_profile="t4", tf32=False, fp16=True, bf16=False,
            gradient_checkpointing=True, group_by_length=False)
        if config["model"]["method"] == "lora":
            config["training"].update(per_device_train_batch_size=4, gradient_accumulation_steps=8,
                per_device_eval_batch_size=8, gradient_checkpointing=False)
    config["run"].update(run_id=None, phase=None, hyperparameters_locked=False, readiness_record=None)
    return config


def estimate_full(benchmark, steps, interval):
    training_seconds = benchmark["train_step_seconds"] / benchmark["optimizer_updates"] * steps
    evaluations = math.ceil(steps / interval) + 3  # pretrained, post-best, independent
    return {"optimizer_updates": steps, "checkpoint_evaluations": math.ceil(steps / interval),
        "all_validation_passes": evaluations,
        "train_hours_extrapolated": training_seconds / 3600,
        "validation_hours_extrapolated": evaluations * benchmark["evaluation_seconds"] / 3600,
        "scope": "measured GPU extrapolation; checkpoint sync, setup and interruptions excluded"}


def viable(result):
    return (result.get("status") == "OK" and result.get("updates") == 13
            and result.get("measured_updates") == 10
            and result.get("eligible", True)
            and result.get("finite_gradients") is True and result.get("optimizer_present") is True
            and result.get("headroom", 0) >= 0.10
            and math.isfinite(result.get("seconds_per_update", float("inf")))
            and result.get("seconds_per_update", 0) > 0)


def choose_fastest(results, key="seconds_per_update"):
    valid = [row for row in results if viable(row)]
    if not valid:
        raise ValueError("No measured stable GPU profile; inspect speed_tuning.json")
    fastest = min(row[key] for row in valid)
    return min((row for row in valid if row[key] <= fastest * 1.05),
               key=lambda row: (row["peak_reserved_bytes"], row[key]))


def trial_configs(config, bf16):
    """Keep the original as baseline; test complete configurations, not isolated effects."""
    base = copy.deepcopy(config)
    base["generation"]["use_cache"] = True
    base["training"].update(dataloader_pin_memory=True, dataloader_persistent_workers=False)
    yield base
    lora = config["model"]["method"] == "lora"
    for batch in ((2, 1, 8, 16, 32) if lora else ((1, 2, 4) if is_t4(config) else (8, 16, 32))):
        candidate = copy.deepcopy(base)
        candidate["training"].update(per_device_train_batch_size=batch,
            gradient_accumulation_steps=32 // batch, gradient_checkpointing=(batch == 1) if lora else is_t4(config),
            bf16=bf16 and not is_t4(config), fp16=not bf16 or is_t4(config),
            optim="adamw_torch_fused", attn_implementation="sdpa")
        yield candidate


class UpdateTimer:
    """AMP-skipped attempts never become measured optimizer updates."""

    def __init__(self):
        self.attempts = self.successful = 0
        self.times = []

    def observe(self, seconds, skipped, finite):
        self.attempts += 1
        if not skipped and not finite:
            raise FloatingPointError("Nonfinite gradients reached optimizer")
        if self.successful < 3:
            if not skipped and finite:
                self.successful += 1
            if self.attempts >= 10 and self.successful < 3:
                raise FloatingPointError("AMP warmup failed to complete 3 updates within 10 attempts")
        else:
            if skipped or not finite:
                raise FloatingPointError("Measured optimizer update skipped or nonfinite")
            self.successful += 1
            self.times.append(seconds)
        return len(self.times) == 10


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    from nllb_lora.train.prepare_data import write_json
    write_json(path, value)


def tune(config_path, cache, output, context, budget=900):
    """Called in the isolated training venv, never the Colab kernel."""
    config, output = _read(config_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample_indices = None

    def launch(name, candidate, purpose, timeout):
        job = output / (name + ".job.json")
        result = output / (name + ".json")
        result.unlink(missing_ok=True)
        _write(job, {"config": candidate, "cache": str(cache), "purpose": purpose, "output": str(result),
            "sample_indices": sample_indices})
        try:
            with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-m", "nllb_lora.train.profile_speed", "--worker", str(job)],
                    check=True, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, timeout))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            # Evaluation writes each completed batch result before starting the next.
            if purpose != "eval" or not result.is_file():
                _write(result, {"status": "FAILED", "error": type(exc).__name__})
        row = _read(result)
        row["config"] = candidate
        return row

    # Download/load weights before starting the 15-minute measurement allowance.
    probe = launch("hardware", config, "hardware", 1800)
    if probe.get("status") != "OK":
        raise ValueError("GPU/model preparation failed; see speed-tuning/hardware.log")
    if config["model"]["method"] == "lora":
        import heapq
        import random
        from datasets import load_from_disk
        from nllb_lora.train.train_nllb import pair_lengths
        data = load_from_disk(str(Path(cache) / "dataset"))["train"]
        lengths = pair_lengths(data)
        longest = heapq.nlargest(32, range(len(lengths)), key=lengths.__getitem__)
        rng = random.Random(config["training"]["seed"])
        diverse = [i for i in rng.sample(range(len(data)), 448) if i not in set(longest)][:384]
        sample_indices = longest + diverse
        rng.shuffle(sample_indices)
    started = time.monotonic()
    report = {"status": "MEASURING", "context": context, "hardware": probe,
              "budget_seconds": budget, "trials": [], "evaluation_trials": [],
              "sample_indices": sample_indices,
              "scope": "Short speed trials only; no quality improvement or global optimum claimed"}
    record = output.parent / "speed_tuning.json"

    def measure(candidate):
        remaining = min(600, budget - 300) - (time.monotonic() - started) if is_t4(config) else budget - 180 - (time.monotonic() - started)
        if remaining < 45:
            return None
        row = launch(f"train-{len(report['trials'])}", candidate, "train", min(180 if is_t4(config) else 150, remaining))
        report["trials"].append(row)
        _write(record, report)
        print("GPU speed trial:", len(report["trials"]), row.get("status"),
              row.get("seconds_per_update"), "seconds/update", flush=True)
        return row

    for candidate in trial_configs(config, probe["bf16_supported"]):
        result = measure(candidate)
        if result is None:
            break
        # Some CUDA/optimizer combinations do not support the fused/SDPA path.
        if not viable(result) and "OutOfMemory" not in result.get("error", ""):
            fallback = copy.deepcopy(candidate)
            fallback["training"].update(optim="adamw_torch")
            known = next((row for row in report["trials"] if viable(row)), None)
            fallback["training"]["attn_implementation"] = known.get("attention", "eager") if known else "eager"
            if fallback != candidate:
                measure(fallback)
    valid = [row for row in report["trials"] if viable(row)]
    if not valid:
        report["status"] = "NO_STABLE_PROFILE"
        _write(record, report)
        raise ValueError("No stable speed profile; see speed_tuning.json")
    best = choose_fastest(valid)
    alternate = copy.deepcopy(best["config"])
    if is_t4(config) or config["model"]["method"] == "lora":
        alternate["training"]["gradient_checkpointing"] = not alternate["training"]["gradient_checkpointing"]
        measure(alternate)
    elif probe["bf16_supported"]:
        alternate["training"].update(fp16=not alternate["training"]["fp16"], bf16=not alternate["training"]["bf16"])
        measure(alternate)
    best = choose_fastest(report["trials"])
    workers = copy.deepcopy(best["config"])
    workers["training"].update(dataloader_num_workers=2, dataloader_persistent_workers=True)
    measure(workers)
    best = choose_fastest(report["trials"])
    if is_t4(config) and best["config"]["training"]["per_device_train_batch_size"] > 1:
        grouped = copy.deepcopy(best["config"])
        grouped["training"]["group_by_length"] = True
        row = measure(grouped)
        if row and viable(row) and row["seconds_per_update"] >= best["seconds_per_update"] * 0.95:
            row.update(eligible=False, selection_note="Length grouping did not improve time by at least 5%")
        best = choose_fastest(report["trials"])
    remaining = budget - (time.monotonic() - started)
    if remaining >= 20:
        evaluation = launch("evaluation", best["config"], "eval", remaining)
        report["evaluation_trials"] = evaluation.get("batches", [])
    eval_valid = [row for row in report["evaluation_trials"]
                  if row.get("status") == "OK" and row.get("headroom", 0) >= 0.10]
    if not eval_valid:
        report["status"] = "NO_STABLE_EVALUATION"
        _write(record, report)
        raise ValueError("No measured evaluation batch completed; see speed-tuning/evaluation.log")
    eval_min = min(row["seconds"] for row in eval_valid)
    chosen_eval = min((row for row in eval_valid if row["seconds"] <= eval_min * 1.05),
                      key=lambda row: row["peak_reserved_bytes"])
    selected = copy.deepcopy(best["config"])
    selected["training"]["per_device_eval_batch_size"] = chosen_eval["batch"]
    from nllb_lora.train.check_readiness import full_training_budget
    from nllb_lora.train.prepare_data import train_example_count, validation_layout
    full_steps = full_training_budget(train_example_count(config), config["training"]["per_device_train_batch_size"],
        config["training"]["gradient_accumulation_steps"], config["training"]["num_train_epochs"])
    baseline = report["trials"][0]
    report.update(status="MEASURED", selected_config=selected,
        seconds_per_update=best["seconds_per_update"],
        train_speedup_vs_baseline=(baseline["seconds_per_update"] / best["seconds_per_update"] if viable(baseline) else None),
        elapsed_seconds=time.monotonic() - started,
        projected_full_hours=(full_steps * best["seconds_per_update"] +
            (math.ceil(full_steps / config["training"]["eval_steps"]) + 3) * len(validation_layout(config)) / chosen_eval["count"] * chosen_eval["seconds"]) / 3600,
        projection_excludes="Setup, checkpoint save/sync and interruptions; validation sample extrapolation")
    _write(record, report)
    print("Selected GPU profile:", selected["training"], flush=True)
    return report


def worker(job):
    """Each invocation starts with fresh pretrained weights and CUDA state."""
    import random
    import statistics
    from contextlib import nullcontext
    import torch
    from datasets import load_from_disk
    from transformers import TrainerCallback, DataCollatorForSeq2Seq, set_seed
    from torch.utils.data import SequentialSampler
    from nllb_lora.train.train_nllb import (load_training_model, training_arguments, configure_generation,
        SourceOnlySeq2SeqTrainer, target_by_source, validation_metrics)
    from nllb_lora.train.prepare_data import load_tokenizer

    config, output = job["config"], Path(job["output"])
    result = {"status": "FAILED"}
    try:
        if not torch.cuda.is_available():
            raise ValueError("GPU speed measurements require CUDA")
        if is_t4(config) and (config["training"]["bf16"] or not config["training"]["fp16"] or config["training"].get("tf32") is not False):
            raise ValueError("T4 requires FP16, BF16 off and TF32 off")
        set_seed(config["training"]["seed"])
        model = load_training_model(config, local_files_only=job["purpose"] != "hardware")
        if job["purpose"] == "hardware":
            result = {"status": "OK", "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "default_attention": model.config._attn_implementation}
            _write(output, result)
            return
        model.config.use_cache = False
        tokenizer = load_tokenizer(config, local_files_only=True)
        configure_generation(model, tokenizer, config)
        data = load_from_disk(str(Path(job["cache"]) / "dataset"))
        indices = job.get("sample_indices") or random.Random(config["training"]["seed"]).sample(range(len(data["train"])), 416)
        rows = data["train"].select(indices, keep_in_memory=True)
        columns = {"input_ids", "attention_mask", "labels"}
        rows = rows.remove_columns([key for key in rows.column_names if key not in columns])
        train = config["training"]
        args = training_arguments(config, output.parent / (output.stem + "-trainer"), 20)
        args.eval_strategy = args.save_strategy = "no"
        args.load_best_model_at_end = False
        args.logging_strategy = "no"
        args.logging_nan_inf_filter = False
        args.disable_tqdm = True
        args.warmup_steps = math.ceil(85605 * train["warmup_ratio_proposal"])

        class OrderedTrainer(SourceOnlySeq2SeqTrainer):
            def _get_train_sampler(self, train_dataset=None):
                if self.args.group_by_length:
                    return super()._get_train_sampler(train_dataset)
                return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        class Timer(TrainerCallback):
            def __init__(self):
                self.counter = UpdateTimer()
                self.finite = False

            def on_step_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                if self.counter.attempts == 0:
                    self.started = time.monotonic()

            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                gradients = [torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]
                self.finite = bool(gradients) and bool(torch.isfinite(torch.stack(gradients)).all())

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                ended = time.monotonic()
                if self.counter.observe(ended - self.started, trainer.accelerator.optimizer_step_was_skipped, self.finite):
                    control.should_training_stop = True
                self.started = ended

        timer = Timer()
        trainer = OrderedTrainer(model=model, args=args, train_dataset=rows, processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8), callbacks=[timer])
        trainer.target_by_source = target_by_source(tokenizer, config)
        torch.cuda.reset_peak_memory_stats()
        trained = trainer.train()
        if not math.isfinite(trained.training_loss):
            raise FloatingPointError("Nonfinite loss")
        # Check full 256-token activation shapes with optimizer state already allocated.
        long_rows = []
        for row in rows.select(range(train["per_device_train_batch_size"])):
            long_rows.append({"input_ids": row["input_ids"] + [tokenizer.pad_token_id] * (256 - len(row["input_ids"])),
                "attention_mask": row["attention_mask"] + [0] * (256 - len(row["attention_mask"])),
                "labels": row["labels"] + [-100] * (256 - len(row["labels"]))})
        batch = trainer._prepare_inputs(trainer.data_collator(long_rows))
        model.train()
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                    if config["model"]["method"] == "lora" else nullcontext())
        with profiler as trace:
            with trainer.compute_loss_context_manager():
                loss = trainer.model(**batch).loss
            trainer.accelerator.backward(loss)
        attention_ops = sorted(event.key for event in trace.key_averages()
            if "scaled_dot_product" in event.key or "flash_attention" in event.key or "efficient_attention" in event.key) if trace else []
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite long-sequence loss")
        trainer.model.zero_grad(set_to_none=True)
        del loss, batch, long_rows
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        peak = torch.cuda.max_memory_reserved()
        result = {"status": "OK", "updates": timer.counter.successful, "attempts": timer.counter.attempts,
            "measured_updates": len(timer.counter.times),
            "finite_gradients": timer.finite, "optimizer_present": bool(trainer.optimizer.state),
            "loss": trained.training_loss, "seconds_per_update": statistics.median(timer.counter.times),
            "examples_per_second": 32 / statistics.median(timer.counter.times),
            "amp_skipped_attempts": timer.counter.attempts - timer.counter.successful,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "method": config["model"]["method"],
            "sample_indices": indices,
            "long_shape_probe": "Padded 256-token source/target shapes after optimizer allocation; LoRA tuning sample also includes 32 longest real pairs",
            "peak_reserved_bytes": peak, "headroom": 1 - peak / total,
            "attention": model.config._attn_implementation}
        result["attention_kernel_ops"] = attention_ops
        if job["purpose"] == "eval":
            order = sorted(range(len(data["val"])), key=lambda i: max(
                len(data["val"][i]["input_ids"]), len(data["val"][i]["labels"])))
            count = 32 if is_t4(config) else 128
            selected = [order[round(i * (len(order) - 1) / (count - 1))] for i in range(count)]
            sample = data["val"].select(selected, keep_in_memory=True)
            evaluation = sample.remove_columns([key for key in sample.column_names if key not in columns])
            compute_metrics = validation_metrics(tokenizer, sample, config,
                                                 generation_limit=config["generation"]["max_new_tokens"])
            result["batches"] = []
            for size in ((8, 16, 32, 4, 2, 1) if config["model"]["method"] == "lora" else ((1, 2, 4, 8) if is_t4(config) else (8, 16, 32))):
                row = {"batch": size, "status": "FAILED"}
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    trainer.args.per_device_eval_batch_size = size
                    trainer._eval_dataloaders = {}
                    trainer.compute_metrics = lambda prediction: {}
                    trainer.evaluate(eval_dataset=evaluation.select(range(min(4, count))))
                    trainer._eval_dataloaders = {}
                    trainer.compute_metrics = compute_metrics
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    metrics = trainer.evaluate(eval_dataset=evaluation)
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved()
                    if not all(math.isfinite(metrics[key]) for key in ("eval_loss", "eval_bleu", "eval_chrf")):
                        raise ValueError("Incomplete/nonfinite evaluation")
                    row.update(status="OK", seconds=time.monotonic() - start,
                        peak_reserved_bytes=peak, headroom=1 - peak / total, count=count,
                        metrics=metrics, scope="Same trainer loss + generation + metrics as official validation")
                except torch.cuda.OutOfMemoryError:
                    row["error"] = "OutOfMemoryError"
                result["batches"].append(row)
                _write(output, result)
                if row.get("error") == "OutOfMemoryError" and config["model"]["method"] != "lora":
                    break
        _write(output, result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        _write(output, result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--config")
    parser.add_argument("--cache")
    parser.add_argument("--output")
    parser.add_argument("--context")
    args = parser.parse_args()
    if args.worker:
        worker(_read(args.worker))
    else:
        tune(args.config, args.cache, args.output, _read(args.context))


if __name__ == "__main__":
    main()
