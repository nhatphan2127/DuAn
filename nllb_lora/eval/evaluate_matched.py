"""Independent text evaluation with the pilot Trainer's padding and AMP policy."""

import argparse
import copy
import json
from pathlib import Path
import time


def comparable_generation(settings):
    from nllb_lora.eval.translation_metrics import generation_settings

    result = generation_settings(settings)
    # Generation accepts either spelling for a single EOS token.
    if isinstance(result.get("eos_token_id"), list) and len(result["eos_token_id"]) == 1:
        result["eos_token_id"] = result["eos_token_id"][0]
    return result


def compare_scores(actual, expected):
    from nllb_lora.eval.translation_metrics import verify_independent_scores

    if actual["count"] != expected["count"]:
        raise ValueError("Independent validation count differs")
    left, right = copy.deepcopy(actual), copy.deepcopy(expected)
    left["generation"] = comparable_generation(left["generation"])
    right["generation"] = comparable_generation(right["generation"])
    verify_independent_scores(left, right, tolerance=1e-6)


def generate(model, tokenizer, rows, config, device):
    import torch
    from nllb_lora.eval.translation_metrics import score_translations, scoring_options
    from nllb_lora.train.train_nllb import configure_generation

    configure_generation(model, tokenizer, config)
    model.eval()
    ids, predictions, references, sources = [], [], [], []
    capped = 0
    batch_size = config["training"]["per_device_eval_batch_size"]
    multiple = config["training"]["pad_to_multiple_of"]
    pending = []

    def consume(batch):
        nonlocal capped
        encoded = tokenizer([r["source"] for r in batch], padding=True,
            pad_to_multiple_of=multiple, truncation=False, return_tensors="pt")
        if int(encoded["attention_mask"].sum(dim=1).max()) > config["data"]["max_source_tokens"]:
            raise ValueError("Source exceeds the audited length limit")
        with torch.inference_mode():
            generated = model.generate(**{key: value.to(device) for key, value in encoded.items()})
        for sequence in generated:
            tail = sequence[1:].tolist()
            eos = tokenizer.eos_token_id
            length = tail.index(eos) + 1 if eos in tail else sum(t != tokenizer.pad_token_id for t in tail)
            capped += length >= config["generation"]["max_new_tokens"]
        ids.extend(r["id"] for r in batch)
        sources.extend(r["source"] for r in batch)
        references.extend(r["target"] for r in batch)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    for row in rows:
        pending.append(row)
        if len(pending) == batch_size:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    report = score_translations(ids, predictions, ids, references, **scoring_options(config))
    report.update(generation=model.generation_config.to_dict(), generation_cap_count=capped,
        source_padding_multiple=multiple, eval_batch_size=batch_size)
    return ids, predictions, references, sources, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    import torch
    from accelerate import Accelerator
    from transformers import AutoTokenizer
    from nllb_lora.eval.translation_metrics import write_predictions
    from nllb_lora.train.prepare_data import file_sha256, iter_pairs, verify_inputs, write_json

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from nllb_lora.train.prepare_data import is_shared
    if is_shared(config):
        raise ValueError("Matched evaluation compares one direction; use nllb_lora.eval.evaluate --direction for a shared adapter")
    expected = json.loads(Path(args.expected_metrics).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise ValueError("Matched Colab evaluation requires CUDA")
    if not config["training"]["fp16"] or config["training"]["bf16"]:
        raise ValueError("This evaluated profile requires the recorded FP16 Trainer policy")
    root, model_path = Path(args.frozen_root).resolve(), Path(args.model).resolve()
    destination = Path(args.output_dir).resolve()
    if any(destination == p or p in destination.parents for p in (root, model_path)):
        raise ValueError("Evaluation output must be outside inputs/model")
    inputs = verify_inputs(config, root)
    spec = config["data"]["splits"]["val"]
    destination.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), src_lang=config["model"]["source_language"],
        tgt_lang=config["model"]["target_language"], local_files_only=True)
    from nllb_lora.train.train_nllb import load_training_model
    model = load_training_model(config, model_path=str(model_path))
    # Seq2SeqTrainer delegates its model-forward AMP wrapping to Accelerate.
    accelerator = Accelerator(mixed_precision="fp16")
    model = accelerator.prepare_model(model, evaluation_mode=True)
    started = time.monotonic()
    ids, predictions, references, sources, report = generate(model, tokenizer,
        iter_pairs(root / spec["source"], root / spec["target"], "val"), config, accelerator.device)
    for name, texts in (("predictions", predictions), ("references", references), ("sources", sources)):
        write_predictions(destination / f"{name}.jsonl", ids, texts)
    report.update(split="val", elapsed_seconds=time.monotonic() - started,
        inference_precision="accelerate_fp16_forward_autocast_fp32_weights", gpu=torch.cuda.get_device_name(0),
        input_manifest_sha256=inputs["manifest_sha256"], model_weights_sha256=file_sha256(model_path / (
            "adapter_model.safetensors" if config["model"]["method"] == "lora" else "model.safetensors")),
        evaluator_sha256=file_sha256(__file__), expected_metrics_sha256=file_sha256(args.expected_metrics))
    write_json(destination / "metrics.json", report)
    comparison = {"status": "FAILED", "bleu_delta": report["bleu"] - expected["bleu"],
        "chrf_delta": report["chrf"] - expected["chrf"], "tolerance": 1e-6,
        "eos_normalization": "singleton list and integer are equivalent; all other decoding settings compared"}
    try:
        if len(ids) != 5000:
            raise ValueError("Full validation must contain 5000 rows")
        compare_scores(report, expected)
    except ValueError as error:
        comparison["error"] = str(error)
        write_json(destination / "comparison.json", comparison)
        raise
    comparison["status"] = "MATCHED"
    write_json(destination / "comparison.json", comparison)
    print(json.dumps({key: report[key] for key in ("bleu", "chrf", "count", "generation_cap_count")}))


if __name__ == "__main__":
    main()
