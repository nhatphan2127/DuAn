"""Single-GPU training of explicit full or LoRA NLLB ZH→VI experiments."""

import argparse
import copy
import csv
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)

from nllb_lora.eval.translation_metrics import (make_compute_metrics, make_direction_compute_metrics,
    scoring_options, write_predictions)
from nllb_lora.train.prepare_data import (balanced_indices, cache_key, cache_specification, direction_specs,
    file_sha256, is_shared, load_tokenizer, train_example_count, verify_cache, verify_inputs, write_json)


def target_by_source(tokenizer, config):
    """Source language token ID -> target language token ID for a shared adapter, else None."""
    directions = direction_specs(config)
    if len(directions) < 2:
        return None
    table = {tokenizer.convert_tokens_to_ids(d["source_language"]): tokenizer.convert_tokens_to_ids(d["target_language"])
             for d in directions}
    if any(value is None or value == tokenizer.unk_token_id for pair in table.items() for value in pair):
        raise ValueError("Direction language token is unknown to tokenizer")
    return table


def decoder_prompt(input_ids, table, decoder_start_token_id):
    """Per-example decoder prompt [start, target language] derived only from the source language token."""
    try:
        targets = [table[int(value)] for value in input_ids[:, 0].tolist()]
    except KeyError as exc:
        raise ValueError(f"Source starts with token {exc.args[0]}, which is not a configured source language") from None
    return torch.tensor([[decoder_start_token_id, target] for target in targets], dtype=input_ids.dtype, device=input_ids.device)


def configure_generation(model, tokenizer, config):
    spec = config["generation"]
    limit = spec["max_new_tokens"]
    if not isinstance(limit, int) or not 1 <= limit <= 512:
        raise ValueError("Generation max_new_tokens must be explicitly resolved in [1,512]")
    if is_shared(config):
        # One forced token cannot serve both directions; a per-example decoder prompt replaces it.
        target_id = None
    else:
        target_id = tokenizer.convert_tokens_to_ids(config["model"]["target_language"])
        if target_id is None or target_id == tokenizer.unk_token_id:
            raise ValueError("Target language is unknown to tokenizer")
    model.generation_config.forced_bos_token_id = target_id
    model.generation_config.max_new_tokens = limit
    model.generation_config.num_beams = spec["num_beams"]
    model.generation_config.do_sample = False
    if "use_cache" in spec:
        model.generation_config.use_cache = spec["use_cache"]
    return model.generation_config


def load_training_model(config, local_files_only=False, model_path=None):
    method = config["model"]["method"]
    if method not in ("lora", "full_fine_tuning"):
        raise ValueError("Unknown training method; no full fine-tuning fallback")
    adapter_path = model_path if model_path and (Path(model_path) / "adapter_config.json").is_file() else None
    if adapter_path and method != "lora":
        raise ValueError("Adapter export requires an explicit LoRA experiment configuration")
    if method == "lora" and model_path and not adapter_path:
        raise ValueError("LoRA evaluation requires an adapter export, not full fine-tuned weights")
    if adapter_path:
        from nllb_lora.train.lora_adapter import verify_adapter
        verify_adapter(adapter_path, config["model"])
    options = {"local_files_only": local_files_only}
    if model_path is None or adapter_path:
        options["revision"] = config["model"]["revision"]
    if config["training"].get("attn_implementation"):
        options["attn_implementation"] = config["training"]["attn_implementation"]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        (None if adapter_path else model_path) or config["model"]["name_or_path"], **options)
    if adapter_path:
        from peft import PeftModel
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=local_files_only)
    if method == "lora":
        from nllb_lora.train.lora_adapter import apply_lora
        model = apply_lora(model, config)
    return model


class SourceOnlySeq2SeqTrainer(Seq2SeqTrainer):
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        # Transformers 4.57.1 rejects save_steps < eval_steps with automatic
        # best reload. Best selection still runs at each evaluated save; reload
        # explicitly only after the last optimizer/RNG checkpoint is persisted.
        if getattr(self, "reload_best_after_training", False) and self.state.best_model_checkpoint:
            self._load_best_model()
        return result

    def _get_eval_sampler(self, eval_dataset):
        # Training length grouping must never reorder predictions against raw references.
        from torch.utils.data import SequentialSampler
        return SequentialSampler(eval_dataset)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.args.group_by_length:
            from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
            if not hasattr(self, "_pair_lengths"):
                self._pair_lengths = pair_lengths(dataset)
            return DistributedLengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=self._pair_lengths, num_replicas=1, rank=0, drop_last=False,
                seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed)
        return super()._get_train_sampler(train_dataset)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs)
        prepared = self._prepare_inputs(inputs)
        labels = prepared.get("labels")
        loss = None
        if labels is not None:
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, prepared).detach().mean()
        source = {key: value for key, value in inputs.items() if key in ("input_ids", "attention_mask")}
        table = getattr(self, "target_by_source", None)
        if table:
            gen_kwargs["decoder_input_ids"] = decoder_prompt(source["input_ids"], table, model.config.decoder_start_token_id)
        _, generated, _ = super().prediction_step(model, source, False, ignore_keys=ignore_keys, **gen_kwargs)
        return loss, generated, labels


def pair_lengths(dataset):
    from array import array
    lengths = array("I")
    for batch in dataset.iter(batch_size=1024):
        lengths.extend(max(len(source), len(target)) for source, target in zip(batch["input_ids"], batch["labels"]))
    return lengths


class FinalEvaluationCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps or control.should_training_stop:
            # A deliberate recovery-only pause must not add official validation.
            if state.global_step >= state.max_steps or args.eval_steps == args.save_steps:
                control.should_evaluate = True
            control.should_save = True
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        # Early stopping is decided during evaluation; persist that final state.
        if control.should_training_stop:
            control.should_save = True
        return control


class StopAtCheckpointCallback(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step is not None and state.global_step >= self.step:
            control.should_training_stop = True
        return control


class MetricArtifactsCallback(TrainerCallback):
    def __init__(self, directory, generation, official):
        self.directory = Path(directory)
        self.generation = generation.to_dict()
        self.official = official
        self.trainer = None
        self.report = None
        self.scope = None


    def capture(self, ids, predictions, report):
        self.report = report
        target = self.directory / "validation" / (self.scope or f"step-{self.trainer.state.global_step}")
        write_predictions(target / "predictions.jsonl", ids, predictions)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.report is None:
            raise ValueError("Evaluation did not produce generated-text scores")
        selected_step = state.global_step
        weight_checkpoint = None
        if self.scope and self.scope.endswith("-best") and state.best_model_checkpoint:
            weight_checkpoint = state.best_model_checkpoint
            selected_step = int(Path(weight_checkpoint).name.rsplit("-", 1)[1])
        write_json(self.directory / "validation" / (self.scope or f"step-{state.global_step}") / "metrics.json",
            {**self.report, "step": selected_step, "training_global_step": state.global_step,
             "weight_checkpoint": weight_checkpoint, "epoch": state.epoch,
             "trainer_metrics": metrics, "generation": self.generation,
             "official_full_validation": self.official})
        self.report = None


class CheckpointSyncCallback(TrainerCallback):
    def __init__(self, run_dir, drive_run_dir):
        self.run_dir, self.drive_run_dir = Path(run_dir), drive_run_dir

    def on_save(self, args, state, control, **kwargs):
        from nllb_lora.train.sync_checkpoints import synchronize_checkpoint
        model = kwargs["model"]
        checkpoint = self.run_dir / f"checkpoint-{state.global_step}"
        model.generation_config.save_pretrained(checkpoint)
        config = json.loads((self.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        if config.get("model", {}).get("method") == "lora":
            write_json(checkpoint / "base_model.json", {"model": config["model"]})
            write_json(checkpoint / "checkpoint_metadata.json", {
                "evaluated": (self.run_dir / "validation" / f"step-{state.global_step}" / "metrics.json").is_file()})
        synchronize_checkpoint(self.run_dir / f"checkpoint-{state.global_step}", self.drive_run_dir, self.run_dir)


class SmokeHealthCallback(TrainerCallback):
    def __init__(self, trainer, run_dir):
        self.trainer, self.run_dir = trainer, Path(run_dir)
        self.updates = 0
        self.changed = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.parameter = next(p for name, p in model.named_parameters()
            if p.requires_grad and ("lora_B" in name or "decoder.layer_norm.bias" in name))
        self.before = self.parameter.detach().flatten()[:4096].float().cpu().clone()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        norms = [torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]
        self.finite = bool(norms) and bool(torch.isfinite(torch.stack(norms)).all())

    def on_step_end(self, args, state, control, **kwargs):
        if not self.trainer.accelerator.optimizer_step_was_skipped:
            if not self.finite:
                raise FloatingPointError("Nonfinite smoke gradients reached optimizer")
            self.updates += 1
            self.changed |= not torch.equal(self.before, self.parameter.detach().flatten()[:4096].float().cpu())

    def on_train_end(self, args, state, control, **kwargs):
        changed = self.changed
        passed = changed and self.updates > 0
        write_json(self.run_dir / "smoke_health.json", {"passed": passed,
            "successful_updates": self.updates, "weights_changed": changed,
            "optimizer_present": bool(self.trainer.optimizer.state)})
        if not passed:
            raise ValueError("Smoke did not demonstrate finite updates and changed weights")


class FullSpeedCallback(TrainerCallback):
    """Update a measured ETA; eval/save/sync are separate from training steps."""

    def __init__(self, run_dir, drive_report=None, trainer=None):
        self.run_dir = Path(run_dir)
        self.drive_report = Path(drive_report) if drive_report else None
        self.steps, self.evaluations, self.saves = [], [], []
        self.save_start = None
        self.step_start = None
        self.window_steps = 0
        self.trainer = trainer
        self.amp_skipped = self.attempts = 0

    def reset_window(self):
        self.step_start = None
        self.window_steps = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self.step_start is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.step_start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        self.attempts += 1
        if self.trainer is not None:
            self.amp_skipped += int(self.trainer.accelerator.optimizer_step_was_skipped)
        self.window_steps += 1
        if self.window_steps == 100 or state.global_step == args.max_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ended = time.monotonic()
            self.steps.append((ended - self.step_start) / self.window_steps)
            self.reset_window()
            self.write(args, state)
        if control is not None and control.should_save:
            self.save_start = time.monotonic()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_runtime") is not None:
            self.evaluations.append(metrics["eval_runtime"])
        self.save_start = time.monotonic()
        self.write(args, state)
        self.reset_window()

    def on_save(self, args, state, control, **kwargs):
        if self.save_start is not None:
            self.saves.append(time.monotonic() - self.save_start)
            self.save_start = None
        self.write(args, state)
        self.reset_window()

    def write(self, args, state):
        import statistics
        step = statistics.median(self.steps[-10:]) if self.steps else None
        evaluation = statistics.median(self.evaluations) if self.evaluations else None
        save = statistics.median(self.saves) if self.saves else None
        checkpoints = math.ceil(args.max_steps / args.save_steps)
        hours = ((args.max_steps * step + (math.ceil(args.max_steps / args.eval_steps) + 3) * evaluation + checkpoints * save) / 3600
                 if all(value is not None for value in (step, evaluation, save)) else None)
        report = {
            "global_step": state.global_step, "total_optimizer_steps": args.max_steps,
            "amp_skipped_attempts_this_process": self.amp_skipped,
            "successful_optimizer_updates_this_process": self.attempts - self.amp_skipped,
            "seconds_per_update": step, "full_validation_seconds": evaluation,
            "checkpoint_save_sync_seconds": save, "full_budget_hours_estimate": hours,
            "scope": "Measured extrapolation; test approximated by validation; excludes setup and interruptions; early stopping may finish sooner"}
        write_json(self.run_dir / "performance_estimate.json", report)
        if self.drive_report and (state.global_step % 500 == 0 or state.global_step == args.max_steps):
            write_json(self.drive_report, report)
        if hours is not None and (state.global_step % args.save_steps == 0 or state.global_step == args.max_steps):
            print(f"Measured full-budget estimate: {hours:.2f} GPU hours; step {state.global_step}/{args.max_steps}", flush=True)


class ResumeVerificationCallback(TrainerCallback):
    def __init__(self, trainer, run_dir, initial_step, checkpoint):
        self.trainer, self.run_dir, self.initial_step = trainer, Path(run_dir), initial_step
        self.checkpoint = Path(checkpoint)
        self.optimizer_loaded = False
        self.scheduler_step = -1
        self.loaded_state_checks = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.adapter_before = {name: p.detach().cpu().clone() for name, p in self.trainer.model.named_parameters()
                               if p.requires_grad and "lora_" in name}
        self.optimizer_loaded = bool(self.trainer.optimizer.state)
        self.scheduler_step = self.trainer.lr_scheduler.last_epoch
        saved_scheduler = torch.load(self.checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
        self.loaded_state_checks = (state.global_step == self.initial_step and
            self.scheduler_step == saved_scheduler["last_epoch"] and
            [group["lr"] for group in self.trainer.optimizer.param_groups] == saved_scheduler["_last_lr"])
        self.scaler_status = "not_saved_by_backend"
        if (self.checkpoint / "scaler.pt").is_file():
            saved_scaler = torch.load(self.checkpoint / "scaler.pt", map_location="cpu", weights_only=False)
            scaler = self.trainer.accelerator.scaler
            self.loaded_state_checks = self.loaded_state_checks and scaler is not None and scaler.state_dict() == saved_scaler
            self.scaler_status = "compared_with_checkpoint"

    def on_train_end(self, args, state, control, **kwargs):
        best_exists = bool(state.best_model_checkpoint and Path(state.best_model_checkpoint).is_dir())
        adapter_changed = (any(not torch.equal(self.adapter_before[name], p.detach().cpu())
            for name, p in self.trainer.model.named_parameters() if name in self.adapter_before)
            if self.adapter_before else None)
        passed = self.optimizer_loaded and self.loaded_state_checks and self.scheduler_step >= self.initial_step and state.global_step > self.initial_step and (best_exists or state.best_model_checkpoint is None) and adapter_changed is not False
        write_json(self.run_dir / "resume_verification.json", {"status": "RESUME_PROBE_PASSED_REVIEW_REQUIRED" if passed else "RESUME_PROBE_FAILED",
            "initial_step": self.initial_step, "final_step": state.global_step, "optimizer_loaded": self.optimizer_loaded,
            "scheduler_loaded_step": self.scheduler_step, "best_checkpoint_present": best_exists,
            "loaded_state_checks": self.loaded_state_checks, "scaler": self.scaler_status,
            "adapter_changed": adapter_changed,
            "rng_note": "Trainer restores RNG at data resumption; exact RNG/order proven separately by fresh-process source test"})


def sanity_summary(before, after, training_ids):
    finite = all(math.isfinite(value["eval_loss"]) for value in (before, after))
    improved = finite and after["eval_loss"] < before["eval_loss"]
    return {"before": before, "after_best": after, "loss_reduction": before["eval_loss"] - after["eval_loss"],
        "finite_loss": finite, "learning_diagnostic_improved": improved,
        "training_ids": list(training_ids), "count": len(training_ids),
        "status": "REQUIRES_MANUAL_EXAMPLE_REVIEW" if improved else "LEARNING_DIAGNOSTIC_FAILED",
        "scope": "overfit same training examples; unofficial; not research quality or validation evidence"}


def validate_pilot(total_steps, interval, drive_run_dir, validation_count):
    if total_steps < 2 * interval:
        raise ValueError("Pilot requires at least two complete evaluation/save cycles")
    if not drive_run_dir:
        raise ValueError("Official pilot requires persistent Drive checkpoint sync")
    if validation_count != 5000:
        raise ValueError("Official pilot requires full 5000 validation examples")


def pilot_summary(baseline, best, checkpoint):
    if baseline["reference_sha256"] != best["reference_sha256"] or baseline["generation"] != best["generation"]:
        raise ValueError("Pilot baseline/best must use identical original references and generation settings")
    return {"pretrained_step_zero": baseline, "selected_best": best, "best_checkpoint": checkpoint,
        "bleu_delta": best["bleu"] - baseline["bleu"], "chrf_delta": best["chrf"] - baseline["chrf"],
        "status": "REQUIRES_MANUAL_QUALITY_REVIEW", "scope": "pilot configuration selection by validation, not final test"}


def validation_metrics(tokenizer, dataset, config, on_result=None, generation_limit=None):
    """Official compute_metrics: one policy for single-direction runs, per-direction for shared ones."""
    if is_shared(config):
        names = list(dataset["direction"])
        options = {name: scoring_options(config, name) for name in dict.fromkeys(names)}
        return make_direction_compute_metrics(tokenizer, dataset["id"], dataset["target"], names, options,
                                              on_result, generation_limit)
    return make_compute_metrics(tokenizer, dataset["id"], dataset["target"], on_result,
                                generation_limit=generation_limit, **scoring_options(config))


def write_pilot_report(run_dir, trainer, training_history, phase="pilot"):
    run_dir = Path(run_dir)
    baseline = json.loads((run_dir / "validation/step-0/metrics.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / f"validation/{phase}-best/metrics.json").read_text(encoding="utf-8"))
    report = pilot_summary(baseline, selected, trainer.state.best_model_checkpoint)
    if phase == "full":
        report["scope"] = "full training validation comparison; final test follows model selection lock"
    write_json(run_dir / f"{phase}_summary.json", report)
    write_json(run_dir / "learning_history.json", training_history)
    directional = sorted(key for key in baseline["trainer_metrics"] if key.startswith(("eval_bleu_", "eval_chrf_")))
    with (run_dir / f"{phase}_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "epoch", "eval_loss", "eval_bleu", "eval_chrf", *directional])
        writer.writeheader()
        writer.writerow({"step": 0, **{name: baseline["trainer_metrics"].get(name) for name in writer.fieldnames if name != "step"}})
        for row in training_history:
            if "eval_bleu" in row and row.get("step", 0) > 0:
                writer.writerow({name: row.get(name) for name in writer.fieldnames})


def resource_summary(started, updates, step_seconds, eval_seconds, cuda, allocated, reserved, total, rss):
    return {"elapsed_seconds": time.monotonic() - started, "optimizer_updates": updates,
        "train_step_seconds": step_seconds, "evaluation_seconds": eval_seconds,
        "peak_cuda_allocated_bytes": allocated, "peak_cuda_reserved_bytes": reserved,
        "gpu_total_bytes": total, "minimum_reserved_headroom_fraction": (1 - reserved / total) if total else None,
        "peak_sampled_process_rss_bytes": rss, "cuda": cuda,
        "resource_probe_passed": bool(cuda and updates >= 100 and eval_seconds > 0 and allocated > 0 and total > reserved)}


class ResourceBenchmarkCallback(TrainerCallback):
    def __init__(self, run_dir, config):
        self.run_dir, self.config = Path(run_dir), config
        self.started = time.monotonic()
        self.step_start = self.started
        self.step_seconds = self.eval_seconds = 0.0
        self.rss = 0
        self.trainer = None
        self.finite_metrics = False

    def sample(self):
        import psutil
        self.rss = max(self.rss, psutil.Process().memory_info().rss)

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_seconds += time.monotonic() - self.step_start
        self.sample()
        self.eval_start = time.monotonic()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.eval_seconds += time.monotonic() - self.eval_start
        self.finite_metrics = all(key in metrics and math.isfinite(metrics[key]) for key in ("eval_loss", "eval_bleu", "eval_chrf"))
        self.sample()
        self.write(state.global_step)

    def write(self, updates, error=None):
        cuda = torch.cuda.is_available()
        report = resource_summary(self.started, updates, self.step_seconds, self.eval_seconds, cuda,
            torch.cuda.max_memory_allocated() if cuda else 0, torch.cuda.max_memory_reserved() if cuda else 0,
            torch.cuda.get_device_properties(0).total_memory if cuda else 0, self.rss)
        report.update(source_tokens=self.config["data"]["max_source_tokens"], target_tokens=self.config["data"]["max_target_tokens"],
            generation=self.config["generation"], error=error,
            validation_count=len(self.trainer.eval_dataset) if self.trainer else 0,
            finite_metrics=self.finite_metrics,
            optimizer_present_during_eval=bool(self.trainer and self.trainer.optimizer and self.trainer.optimizer.state))
        if error or not report["optimizer_present_during_eval"] or not self.finite_metrics or report["validation_count"] != self.config["evaluation"]["expected_pairs"]:
            report["resource_probe_passed"] = False
        report["status"] = "PROBE_PASSED_REVIEW_REQUIRED" if report["resource_probe_passed"] else "WAITING_FOR_COLAB"
        write_json(self.run_dir / "resource_benchmark.json", report)


def build_trainer(model, tokenizer, train_dataset, val_dataset, args, run_dir, config,
                  official=True, stop_after_step=None):
    generation = configure_generation(model, tokenizer, config)
    artifacts = MetricArtifactsCallback(run_dir, generation, official)
    metrics = validation_metrics(tokenizer, val_dataset, config, artifacts.capture, generation.max_new_tokens)
    columns = {"input_ids", "attention_mask", "labels"}
    trainer = SourceOnlySeq2SeqTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=train_dataset.remove_columns([name for name in train_dataset.column_names if name not in columns]),
        eval_dataset=val_dataset.remove_columns([name for name in val_dataset.column_names if name not in columns]),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8),
        compute_metrics=metrics,
        callbacks=[StopAtCheckpointCallback(stop_after_step), artifacts,
            EarlyStoppingCallback(early_stopping_patience=config["training"]["early_stopping_patience"],
                early_stopping_threshold=config["training"]["early_stopping_threshold"]), FinalEvaluationCallback()],
    )
    artifacts.trainer = trainer
    trainer.target_by_source = target_by_source(tokenizer, config)
    trainer.reload_best_after_training = config["training"].get("best_model_reload_policy") == "explicit_after_training"
    return trainer


def training_arguments(config, run_dir, total_steps):
    training = config["training"]
    keys = ("optim", "learning_rate", "weight_decay", "lr_scheduler_type", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "per_device_eval_batch_size",
        "fp16", "bf16", "gradient_checkpointing", "max_grad_norm", "label_smoothing_factor",
        "seed", "data_seed", "dataloader_num_workers", "logging_steps", "eval_strategy",
        "save_strategy", "eval_steps", "save_steps", "eval_accumulation_steps", "predict_with_generate",
        "metric_for_best_model", "greater_is_better", "load_best_model_at_end", "save_total_limit",
        "save_only_model", "report_to")
    return Seq2SeqTrainingArguments(output_dir=str(run_dir), max_steps=total_steps,
        warmup_steps=math.ceil(total_steps * training["warmup_ratio_proposal"]),
        restore_callback_states_from_checkpoint=True, dataloader_drop_last=False,
        **({"gradient_checkpointing_kwargs": {"use_reentrant": False}} if config["model"]["method"] == "lora" else {}),
        **{key: training[key] for key in ("dataloader_pin_memory", "dataloader_persistent_workers") if key in training},
        **{key: training[key] for key in ("tf32", "group_by_length") if key in training},
        **{key: training[key] for key in keys})


def code_snapshot(run_dir=None):
    files = [Path(__file__), Path("nllb_lora/train/prepare_data.py"),
             Path("nllb_lora/eval/translation_metrics.py"), Path("nllb_lora/train/sync_checkpoints.py"),
             Path("nllb_lora/train/check_readiness.py"), Path("nllb_lora/train/lora_adapter.py"), Path("nllb_lora/requirements-colab.lock")]
    hashes = {}
    for source in files:
        if run_dir is not None:
            target = Path(run_dir) / "source" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        hashes[source.name] = file_sha256(source)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"git_revision": revision or None, "source_sha256": hashes}


def main():
    from datasets import load_from_disk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nllb_zh_vi.json")
    parser.add_argument("--frozen-root")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("smoke", "sanity", "pilot", "benchmark", "full"), required=True)
    parser.add_argument("--max-steps", type=int, help="Fixed total phase budget; never session duration")
    parser.add_argument("--stop-after-step", type=int, help="Stop at a checkpoint without changing total scheduler budget")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--drive-run-dir", help="Persistent Drive folder for verified full checkpoints")
    parser.add_argument("--resume-from-drive", action="store_true", help="Restore full state into a fresh local run directory")
    parser.add_argument("--readiness", help="Optional named review with hashed Colab evidence")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.resume_from_drive and not args.drive_run_dir:
        parser.error("--resume-from-drive requires --drive-run-dir")
    if args.phase == "sanity" and args.resume_from_drive:
        parser.error("Sanity is a separate short diagnostic run; start fresh to preserve its step-zero comparison")
    if not torch.cuda.is_available():
        parser.error("Production entrypoint requires CUDA; use tiny CPU tests only for source validation")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This experiment supports one GPU only")
    if config["model"]["name_or_path"] != "facebook/nllb-200-distilled-600M" or config["model"]["method"] not in ("lora", "full_fine_tuning"):
        parser.error("Experiment requires explicit full fine-tuning or LoRA of pinned NLLB-600M")
    if config["model"]["method"] == "lora" and args.phase in ("pilot", "full") and (config["training"]["eval_steps"] not in (10000, 20000) or config["training"]["save_steps"] != 2000):
        parser.error("Official LoRA requires validation every 10000 or 20000 and recovery save every 2000 updates")
    if args.phase == "full":
        from nllb_lora.train.check_readiness import full_training_budget
        required_steps = full_training_budget(train_example_count(config),
            config["training"]["per_device_train_batch_size"], config["training"]["gradient_accumulation_steps"], config["training"]["num_train_epochs"])
        if args.max_steps not in (None, required_steps) or not args.drive_run_dir:
            parser.error("Full run requires Drive sync and the fixed complete epoch budget")
        args.max_steps = required_steps
    if args.phase == "benchmark":
        if args.max_steps not in (None, 100):
            parser.error("Benchmark budget is exactly 100 optimizer updates")
        args.max_steps = 100
        config = copy.deepcopy(config)
        config["training"].update(eval_steps=100, save_steps=100, early_stopping_patience=100)
    if args.max_steps is None or args.max_steps <= 0:
        parser.error("Smoke/pilot require a positive fixed --max-steps phase budget")
    training = config["training"]
    if training["eval_steps"] % training["save_steps"]:
        parser.error("Evaluation interval must be divisible by recovery checkpoint interval")
    if args.stop_after_step is not None and (args.stop_after_step <= 0 or args.stop_after_step > args.max_steps or
        (args.phase != "smoke" and args.stop_after_step % training["save_steps"] and args.stop_after_step != args.max_steps)):
        parser.error("Stop step must be an evaluation/checkpoint boundary within total budget")
    root = args.frozen_root or config["data"]["frozen_root"]
    inputs = verify_inputs(config, root)
    specification = cache_specification(config, inputs, config["data"]["max_source_tokens"], config["data"]["max_target_tokens"])
    cache_manifest = verify_cache(args.cache_dir, specification)
    cached = load_from_disk(str(Path(args.cache_dir) / "dataset"))
    if args.phase == "smoke" and is_shared(config):
        # Smoke must exercise every direction, not only the first cache block.
        train = cached["train"].select(balanced_indices(config, "train", 64))
        val = cached["val"].select(balanced_indices(config, "val", 16))
    elif args.phase == "smoke":
        train = cached["train"].select(range(min(64, len(cached["train"]))))
        val = cached["val"].select(range(min(16, len(cached["val"]))))
    elif args.phase == "sanity":
        train = cached["train"].shuffle(seed=training["seed"], keep_in_memory=True).select(range(min(64, len(cached["train"]))))
        val = train
    elif args.phase == "benchmark":
        lengths = []
        for batch in cached["train"].iter(batch_size=256):
            lengths.extend(max(len(ids), len(labels)) for ids, labels in zip(batch["input_ids"], batch["labels"]))
        longest = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)[:3200]
        train = cached["train"].select(longest)
        val = cached["val"]
    elif args.phase == "full":
        train, val = cached["train"], cached["val"]
    else:
        train = cached["train"].shuffle(seed=training["seed"], keep_in_memory=True).select(range(min(20000, len(cached["train"]))))
        val = cached["val"]
    if args.phase == "pilot":
        validate_pilot(args.max_steps, training["eval_steps"], args.drive_run_dir, len(val))
    run_dir = Path(args.run_dir).resolve()
    if any(run_dir == protected or protected in run_dir.parents for protected in (Path(root).resolve(), Path(args.cache_dir).resolve())):
        parser.error("Run output must be outside frozen inputs and cache")
    if args.drive_run_dir:
        drive_dir = Path(args.drive_run_dir).resolve()
        if any(drive_dir == protected or protected in drive_dir.parents for protected in (Path(root).resolve(), Path(args.cache_dir).resolve(), run_dir)):
            parser.error("Drive checkpoint destination must be outside frozen/cache/local run")
        if not args.resume_from_drive and any(drive_dir.glob("checkpoint-*")):
            parser.error("New run requires a Drive run folder without existing checkpoints; use resume explicitly")
    resolved = copy.deepcopy(config)
    resolved["run"].update(run_id=run_dir.name, phase=args.phase, cache_fingerprint=cache_manifest["cache_key"])
    resolved["training"]["total_optimizer_steps"] = args.max_steps
    resolved["training"]["warmup_steps"] = math.ceil(args.max_steps * training["warmup_ratio_proposal"])
    environment = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "accelerate", "sacrebleu", "sentencepiece", "numpy", "tokenizers", "safetensors", "huggingface-hub", "psutil", "peft")}, "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory}
    manifest = {"config": resolved, "environment": environment,
        "code": code_snapshot(), "data": inputs, "cache_key": cache_key(specification),
        "train_ids_sha256": sha256("\n".join(train["id"]).encode()).hexdigest(),
        "train_count": len(train), "val_count": len(val), "official_full_validation": args.phase not in ("smoke", "sanity")}
    readiness = None
    if args.phase == "full":
        if args.readiness:
            from nllb_lora.train.check_readiness import validate_readiness
            readiness = validate_readiness(args.readiness, manifest)
        resolved["status"] = "reviewed_full_run" if readiness else "full_run"
        resolved["run"]["hyperparameters_locked"] = True
        resolved["run"]["readiness_record"] = readiness
    resume = None
    if args.resume_from_drive:
        from nllb_lora.train.sync_checkpoints import restore_verified, resume_signature
        resume = restore_verified(args.drive_run_dir, run_dir, resume_signature(manifest))
        restored_step = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
        if restored_step >= args.max_steps or (args.stop_after_step is not None and args.stop_after_step <= restored_step):
            parser.error("Resume checkpoint already reached total budget or requested stop step")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    code_snapshot(run_dir)
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "run_manifest.json", manifest)
    if readiness:
        write_json(run_dir / "readiness_validated.json", readiness)
    set_seed(training["seed"])
    tokenizer = load_tokenizer(config, args.local_files_only)
    model = load_training_model(config, args.local_files_only)
    if config["model"]["method"] == "lora":
        manifest["lora_inventory"] = model.lora_inventory
        write_json(run_dir / "run_manifest.json", manifest)
    model.config.use_cache = False
    trainer = build_trainer(model, tokenizer, train, val, training_arguments(config, run_dir, args.max_steps), run_dir, config,
                            official=args.phase not in ("smoke", "sanity"), stop_after_step=args.stop_after_step)
    if args.phase == "smoke":
        trainer.add_callback(SmokeHealthCallback(trainer, run_dir))
    sanity_before = trainer.evaluate() if args.phase == "sanity" else None
    if args.phase in ("pilot", "full"):
        if resume is None:
            trainer.evaluate()
        elif not (run_dir / "validation/step-0/metrics.json").is_file():
            raise ValueError("Resumed pilot is missing its original pretrained step-zero baseline")
    if args.drive_run_dir:
        trainer.add_callback(CheckpointSyncCallback(run_dir, args.drive_run_dir))
    if args.phase == "full":
        trainer.add_callback(FullSpeedCallback(run_dir, Path(args.drive_run_dir).parent.parent / "performance_estimate.json", trainer))
    if resume:
        trainer.add_callback(ResumeVerificationCallback(trainer, run_dir, restored_step, resume))
    benchmark = None
    if args.phase == "benchmark":
        benchmark = ResourceBenchmarkCallback(run_dir, config)
        benchmark.trainer = trainer
        trainer.add_callback(benchmark)
    try:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except (torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
        if benchmark:
            benchmark.write(trainer.state.global_step, f"{type(exc).__name__}: {exc}")
        raise
    if benchmark:
        benchmark.write(trainer.state.global_step)
    if args.stop_after_step is not None and trainer.state.global_step < args.max_steps and trainer.state.global_step >= args.stop_after_step:
        write_json(run_dir / "completion.json", {"reason": "interrupted_at_checkpoint",
            "global_step": trainer.state.global_step, "total_budget": args.max_steps, "experiment_complete": False})
        trainer.save_metrics("train", result.metrics)
        return
    if args.phase == "sanity":
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, MetricArtifactsCallback):
                callback.scope = "sanity-best"
        sanity_after = trainer.evaluate()
        write_json(run_dir / "sanity_learning.json", sanity_summary(sanity_before, sanity_after, train["id"]))
    if args.phase in ("pilot", "full"):
        training_history = copy.deepcopy(trainer.state.log_history)
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, MetricArtifactsCallback):
                callback.scope = "pilot-best" if args.phase == "pilot" else "full-best"
        trainer.evaluate()
        write_pilot_report(run_dir, trainer, training_history, phase=args.phase)
    trainer.save_metrics("train", result.metrics)
    counter = next((callback.early_stopping_patience_counter for callback in trainer.callback_handler.callbacks if isinstance(callback, EarlyStoppingCallback)), 0)
    reason = "budget_complete" if trainer.state.global_step >= args.max_steps else (
        "interrupted_at_checkpoint" if args.stop_after_step is not None and trainer.state.global_step >= args.stop_after_step else (
        "early_stopping" if counter >= training["early_stopping_patience"] else "stopped_before_budget"))
    write_json(run_dir / "completion.json", {"reason": reason, "global_step": trainer.state.global_step,
        "total_budget": args.max_steps, "experiment_complete": reason in ("budget_complete", "early_stopping")})
    # Best weights are exported separately; never combine best weights with last optimizer.
    export = run_dir / "best_model"
    trainer.save_model(str(export))
    tokenizer.save_pretrained(str(export))
    model.generation_config.save_pretrained(str(export))
    if config["model"]["method"] == "lora":
        write_json(export / "base_model.json", {"model": config["model"]})
    write_json(export / "provenance.json", {"best_checkpoint": trainer.state.best_model_checkpoint,
        "model": config["model"], "best_metric": trainer.state.best_metric, "last_step": trainer.state.global_step})


if __name__ == "__main__":
    main()
