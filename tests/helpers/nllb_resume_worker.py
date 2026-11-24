"""Internal tiny CPU fixture executed in fresh processes by test_nllb_resume."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoConfig, M2M100Config, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments

from nllb_lora.train.sync_checkpoints import restore_verified, resume_signature
from nllb_lora.train.train_nllb import build_trainer, CheckpointSyncCallback, ResumeVerificationCallback
from nllb_lora.train.prepare_data import load_tokenizer, write_json


def tensor_hash(value):
    digest = sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(item.cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                add(entry)
        else:
            digest.update(repr(item).encode())
    add(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--drive", required=True)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = json.loads(Path("configs/nllb_zh_vi.json").read_text(encoding="utf-8"))
    config["generation"].update(max_new_tokens=4, num_beams=1)
    config["training"].update(early_stopping_patience=100, total_optimizer_steps=7, group_by_length=args.group_by_length)
    config["run"]["phase"] = "tiny_fixture"
    if args.lora:
        config["model"] = json.loads(Path("configs/nllb_lora_zh_vi.json").read_text(encoding="utf-8"))["model"]
        config["training"].update(eval_steps=6, save_steps=2, fp16=False, best_model_reload_policy="explicit_after_training")
        config["checkpoint"]["preserve_all"] = True
    manifest = {"config": config, "data": {"manifest_sha256": "fixture"}, "cache_key": "fixture",
        "train_ids_sha256": "fixture", "train_count": 2, "val_count": 2,
        "environment": {"packages": {"torch": torch.__version__}},
        "code": {"source_sha256": {"worker": "fixture"}}}
    run = Path(args.run)
    resume = restore_verified(args.drive, run, resume_signature(manifest)) if args.resume else None
    if not args.resume:
        run.mkdir()
    write_json(run / "resolved_config.json", config)
    write_json(run / "run_manifest.json", manifest)
    tokenizer = load_tokenizer(config, local_files_only=True)
    original = AutoConfig.from_pretrained(config["model"]["name_or_path"], revision=config["model"]["revision"], local_files_only=True)
    model = M2M100ForConditionalGeneration(M2M100Config(vocab_size=len(tokenizer), d_model=8,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=1, decoder_attention_heads=1,
        encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=32, dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0, decoder_start_token_id=original.decoder_start_token_id,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id))
    model.config.use_cache = False
    if args.lora:
        from nllb_lora.train.lora_adapter import apply_lora
        model = apply_lora(model, config)
    base_before = tensor_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
    training_order = []
    def record_inputs(module, positional, keywords):
        if module.training and "input_ids" in keywords:
            training_order.append(keywords["input_ids"].tolist())
    model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    rows = [{"id": f"val:{i+1}", "source": source, "target": target, **dict(tokenizer(source, text_target=target))}
        for i, (source, target) in enumerate((("你好", "Xin chào Việt Nam và các bạn!"), ("谢谢", "Cảm ơn tất cả những người bạn.")))]
    if args.lora:
        rows = [dict(rows[i % 2], id=f"val:{i + 1}") for i in range(7)]
    dataset = Dataset.from_list(rows)
    training = Seq2SeqTrainingArguments(output_dir=str(run), use_cpu=True, max_steps=7,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, learning_rate=.001,
        gradient_accumulation_steps=2 if args.lora else 1,
        warmup_steps=1, eval_strategy="steps", save_strategy="steps", eval_steps=6 if args.lora else 3, save_steps=2 if args.lora else 3,
        save_total_limit=3, predict_with_generate=True, load_best_model_at_end=not args.lora,
        metric_for_best_model="chrf", report_to="none", disable_tqdm=True,
        restore_callback_states_from_checkpoint=True, seed=42, data_seed=42, group_by_length=args.group_by_length)
    trainer = build_trainer(model, tokenizer, dataset, dataset, training, run, config, official=False, stop_after_step=args.stop)
    if resume is None:
        trainer.evaluate()
    trainer.add_callback(CheckpointSyncCallback(run, args.drive))
    if resume:
        restored_step = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
        trainer.add_callback(ResumeVerificationCallback(trainer, run, restored_step, resume))
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    step = trainer.state.global_step
    checkpoint = run / f"checkpoint-{step}"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    write_json(run / "comparison.json", {"step": step,
        "baseline_prediction_sha256": json.loads((run / "validation/step-0/metrics.json").read_text(encoding="utf-8"))["prediction_sha256"],
        "weights": tensor_hash(load_file(str(checkpoint / ("adapter_model.safetensors" if args.lora else "model.safetensors")))),
        "base_frozen": base_before == tensor_hash({n: p for n, p in model.named_parameters() if not p.requires_grad}),
        "optimizer": tensor_hash(torch.load(checkpoint / "optimizer.pt", weights_only=False)),
        "scheduler": tensor_hash(torch.load(checkpoint / "scheduler.pt", weights_only=False)),
        "rng": tensor_hash(torch.load(checkpoint / "rng_state.pth", weights_only=False)),
        "early_stopping": state["stateful_callbacks"]["EarlyStoppingCallback"],
        "training_order": training_order,
        "best": state["best_model_checkpoint"]})


if __name__ == "__main__":
    main()
