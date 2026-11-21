"""Generate translations (ZH→VI or VI→ZH per config) from source text only and score original references."""

import argparse
import json
from pathlib import Path
import time

from nllb_lora.eval.translation_metrics import score_translations, scoring_options, write_predictions
from nllb_lora.train.prepare_data import (direction_specs, file_sha256, is_shared, iter_pairs, load_tokenizer,
    verify_inputs, write_json)


def direction_rows(config, direction, source, reference, split):
    """Rows of one direction; later directions carry the ``@name`` ID suffix used in shared validation."""
    position = direction_specs(config).index(direction)
    for row in iter_pairs(source, reference, split):
        yield {**row, "id": row["id"] if position == 0 else f"{row['id']}@{direction['name']}"}


def select_direction(config, name):
    directions = direction_specs(config)
    if len(directions) == 1:
        if name not in (None, directions[0]["name"]):
            raise ValueError("--direction is only valid for a shared multi-direction config")
        return directions[0]
    chosen = next((d for d in directions if d["name"] == name), None)
    if chosen is None:
        raise ValueError(f"A shared config requires --direction in {[d['name'] for d in directions]}")
    return chosen


def generate_translations(model, tokenizer, rows, config, batch_size=1, direction=None):
    import torch
    from nllb_lora.train.train_nllb import configure_generation, decoder_prompt, target_by_source

    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    configure_generation(model, tokenizer, config)
    table = target_by_source(tokenizer, config)
    options = scoring_options(config, direction["name"] if direction and direction["name"] else None)
    model.eval()
    ids, predictions, references = [], [], []
    capped = 0
    pending = []

    def consume(batch):
        nonlocal capped
        encoded = tokenizer([r["source"] for r in batch], padding=True, truncation=False,
            pad_to_multiple_of=config["training"]["pad_to_multiple_of"], return_tensors="pt")
        if int(encoded["attention_mask"].sum(dim=1).max()) > config["data"]["max_source_tokens"]:
            raise ValueError("Source exceeds audited token limit; refusing silent truncation")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        extra = {"decoder_input_ids": decoder_prompt(encoded["input_ids"], table, model.config.decoder_start_token_id)} if table else {}
        with torch.inference_mode():
            generated = model.generate(**encoded, **extra)
        limit = config["generation"]["max_new_tokens"]
        for sequence in generated:
            # An EOS at the last allowed position still reached the generation cap.
            content = sequence[1:].tolist()
            eos = tokenizer.eos_token_id
            length = content.index(eos) + 1 if eos in content else len(content)
            capped += length >= limit
        ids.extend(r["id"] for r in batch)
        references.extend(r["target"] for r in batch)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    for row in rows:
        pending.append(row)
        if len(pending) == batch_size:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    report = score_translations(ids, predictions, ids, references, **options)
    report["generation_cap_count"] = capped
    report["generation"] = model.generation_config.to_dict()
    return ids, predictions, references, report


def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nllb_zh_vi.json")
    parser.add_argument("--model", help="Local best_model export; omitted uses pinned pretrained baseline")
    parser.add_argument("--frozen-root")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--source", help="External benchmark source; requires --reference and --expected-pairs")
    parser.add_argument("--reference")
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, help="Defaults to the training configuration's evaluation batch")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--direction", help="Direction name; required for a shared multi-direction adapter")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        direction = select_direction(config, args.direction)
    except ValueError as error:
        parser.error(str(error))
    if bool(args.source) != bool(args.reference) or (args.source and not args.expected_pairs):
        parser.error("Benchmark requires --source, --reference and positive --expected-pairs")
    if args.source:
        source, reference = Path(args.source), Path(args.reference)
        expected = args.expected_pairs
        split = "benchmark"
    else:
        root = Path(args.frozen_root or config["data"]["frozen_root"])
        verify_inputs(config, root)
        spec = config["data"]["splits"][args.split]
        source, reference = root / spec["source"], root / spec["target"]
        if direction["swap"]:
            source, reference = reference, source
        expected, split = spec["expected_pairs"], args.split
    count = sum(1 for _ in direction_rows(config, direction, source, reference, split))
    if count != expected:
        parser.error(f"Expected {expected} pairs, found {count}")
    destination = Path(args.output_dir).resolve()
    protected = []
    if not args.source:
        protected.append(root.resolve())
    if args.model:
        protected.append(Path(args.model).resolve())
    if any(destination == parent or parent in destination.parents for parent in protected):
        parser.error("Evaluation output must be outside frozen inputs and model export")
    destination.mkdir(parents=True, exist_ok=False)
    if args.model:
        model_path = Path(args.model)
        if (model_path / "export_manifest.json").is_file():
            from nllb_lora.eval.finalize_model import verify_inference_export
            verify_inference_export(model_path)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), src_lang=direction["source_language"],
            tgt_lang=direction["target_language"], local_files_only=True)
        from nllb_lora.train.train_nllb import load_training_model
        model = load_training_model(config, local_files_only=args.local_files_only, model_path=str(model_path))
        metadata = "base_model.json" if config["model"]["method"] == "lora" else "config.json"
        identity = {"export": str(model_path.resolve()), "config_sha256": file_sha256(model_path / metadata),
            "base_model": config["model"]}
    else:
        tokenizer = load_tokenizer(config, args.local_files_only, direction)
        from nllb_lora.train.train_nllb import load_training_model
        model = load_training_model(config, args.local_files_only)
        identity = config["model"]
    precision = "no"
    if torch.cuda.is_available():
        from accelerate import Accelerator
        precision = "fp16" if config["training"]["fp16"] else ("bf16" if config["training"]["bf16"] else "no")
        model = Accelerator(mixed_precision=precision).prepare_model(model, evaluation_mode=True)
    else:
        model.to("cpu")
    started = time.monotonic()
    ids, predictions, references, report = generate_translations(model, tokenizer,
        direction_rows(config, direction, source, reference, split), config,
        args.batch_size if args.batch_size is not None else config["training"]["per_device_eval_batch_size"], direction)
    write_predictions(destination / "predictions.jsonl", ids, predictions)
    write_predictions(destination / "references.jsonl", ids, references)
    write_predictions(destination / "sources.jsonl", ids, [row["source"] for row in direction_rows(config, direction, source, reference, split)])
    write_json(destination / "metrics.json", {**report, "model": identity, "split": split, "direction": direction["name"],
        "source_sha256": file_sha256(source), "reference_file_sha256": file_sha256(reference),
        "elapsed_seconds": time.monotonic() - started, "inference_precision": precision})
    print(json.dumps({key: report[key] for key in ("bleu", "chrf", "count", "generation_cap_count")}))


if __name__ == "__main__":
    main()
