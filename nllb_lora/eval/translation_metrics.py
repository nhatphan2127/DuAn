"""Corpus BLEU/chrF for generated text (Vietnamese or Chinese) and original references."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import math
import unicodedata
from pathlib import Path


def generation_settings(values):
    """Ignore serialization metadata while retaining every decoding parameter."""
    return {key: value for key, value in values.items() if not key.startswith("_") and key != "transformers_version"}


def verify_independent_scores(actual, expected, tolerance=1e-6):
    if actual["reference_sha256"] != expected["reference_sha256"] or generation_settings(actual["generation"]) != generation_settings(expected["generation"]):
        raise ValueError("Independent evaluation uses different references or generation settings")
    for metric in ("bleu", "chrf"):
        if not math.isfinite(actual[metric]) or not math.isfinite(expected[metric]) or abs(actual[metric] - expected[metric]) > tolerance:
            raise ValueError(f"Independent {metric} differs from selected checkpoint beyond {tolerance}")


def validate_rows(ids, texts, role):
    if not ids or len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError(f"{role}: empty corpus, count mismatch or duplicate IDs")
    if any(not isinstance(key, str) or not key.strip() for key in ids):
        raise ValueError(f"{role}: invalid ID")
    if any(not isinstance(text, str) or "\n" in text or "\r" in text for text in texts):
        raise ValueError(f"{role}: each text must be one string without newlines")
    if role == "reference" and any(not text.strip() for text in texts):
        raise ValueError("reference: empty text")


def text_fingerprint(ids, texts):
    payload = json.dumps(list(zip(ids, texts)), ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def decoding_settings(settings):
    """Exclude library provenance flags that do not affect decoding."""
    metadata = {"_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version"}
    return {key: value for key, value in settings.items() if key not in metadata}


BLEU_TOKENIZERS = ("13a", "zh")
TEXT_NORMALIZATIONS = (None, "nfkc")


def scoring_options(config, direction=None):
    """Scoring settings from a training config; defaults reproduce the original zh->vi policy.

    A shared adapter lists one policy per direction name in ``evaluation.scoring``.
    """
    evaluation = config["evaluation"]
    if direction is not None:
        entry = evaluation["scoring"][direction]
        return {"bleu_tokenize": entry["bleu_tokenize"], "normalization": entry.get("text_normalization")}
    return {"bleu_tokenize": evaluation["bleu"]["tokenize"], "normalization": evaluation.get("text_normalization")}


def score_translations(prediction_ids, predictions, reference_ids, references, bleu_tokenize="13a", normalization=None):
    from sacrebleu.metrics import BLEU, CHRF

    if bleu_tokenize not in BLEU_TOKENIZERS:
        raise ValueError(f"bleu_tokenize must be one of {BLEU_TOKENIZERS}")
    if normalization not in TEXT_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {TEXT_NORMALIZATIONS}")

    validate_rows(prediction_ids, predictions, "prediction")
    validate_rows(reference_ids, references, "reference")
    if list(prediction_ids) != list(reference_ids):
        raise ValueError("Prediction/reference ID count or order mismatch")
    bleu = BLEU(tokenize=bleu_tokenize, lowercase=False, effective_order=False)
    chrf = CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False)
    # Fingerprints stay on the raw texts; only the scored strings are normalized, identically on both sides.
    scored_predictions, scored_references = predictions, references
    if normalization == "nfkc":
        scored_predictions = [unicodedata.normalize("NFKC", text) for text in predictions]
        scored_references = [unicodedata.normalize("NFKC", text) for text in references]
    bleu_score = bleu.corpus_score(scored_predictions, [scored_references]).score
    chrf_score = chrf.corpus_score(scored_predictions, [scored_references]).score
    return {
        "bleu": bleu_score, "chrf": chrf_score, "count": len(references),
        "empty_predictions": sum(not text.strip() for text in predictions),
        "reference_sha256": text_fingerprint(reference_ids, references),
        "prediction_sha256": text_fingerprint(prediction_ids, predictions),
        "signatures": {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature())},
        "sacrebleu_version": importlib.metadata.version("sacrebleu"),
        "text_policy": "original full references; case-sensitive; SacreBLEU corpus normalization; chrF whitespace=False"
            + ("; NFKC applied to predictions and references before scoring" if normalization == "nfkc" else ""),
        "text_normalization": normalization,
    }


def generated_token_ids(evaluation_prediction, tokenizer):
    """Validated generated IDs [examples, tokens] with Trainer padding mapped to the pad token."""
    import numpy as np

    generated = evaluation_prediction.predictions
    if isinstance(generated, tuple):
        raise ValueError("Expected generated token IDs, not model-output tuples/logits")
    generated = np.asarray(generated)
    if generated.ndim != 2 or not np.issubdtype(generated.dtype, np.integer):
        raise ValueError("Expected integer generated IDs [examples, tokens]; do not score logits/argmax")
    if ((generated < 0) & (generated != -100)).any():
        raise ValueError("Unexpected negative token ID")
    return np.where(generated == -100, tokenizer.pad_token_id, generated)


def count_at_generation_limit(generated, tokenizer, generation_limit):
    """Count generated positions after decoder start, including EOS if present."""
    import numpy as np

    at_limit = 0
    for row in generated:
        tail = row[1:]
        eos = np.flatnonzero(tail == tokenizer.eos_token_id)
        length = int(eos[0]) + 1 if len(eos) else int(np.count_nonzero(tail != tokenizer.pad_token_id))
        at_limit += length >= generation_limit
    return at_limit


def make_compute_metrics(tokenizer, reference_ids, references, on_result=None, generation_limit=None, bleu_tokenize="13a", normalization=None):
    """Close over raw references; label_ids are deliberately never decoded."""
    reference_ids, references = list(reference_ids), list(references)
    validate_rows(reference_ids, references, "reference")

    def compute_metrics(evaluation_prediction):
        generated = generated_token_ids(evaluation_prediction, tokenizer)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        report = score_translations(reference_ids, predictions, reference_ids, references, bleu_tokenize, normalization)
        if generation_limit is not None:
            at_limit = count_at_generation_limit(generated, tokenizer, generation_limit)
            report["generation_limit"] = generation_limit
            report["at_generation_limit"] = at_limit
            report["at_generation_limit_ratio"] = at_limit / len(predictions)
        if on_result is not None:
            on_result(reference_ids, predictions, report)
        return {"bleu": report["bleu"], "chrf": report["chrf"]}

    return compute_metrics


def make_direction_compute_metrics(tokenizer, reference_ids, references, directions, options, on_result=None, generation_limit=None):
    """Score one shared validation set per direction; ``bleu``/``chrf`` are the means over directions.

    ``directions`` names the direction of every row (any order); ``options`` maps each
    name to ``scoring_options(config, name)``. Fingerprints cover all rows in dataset order.
    """
    reference_ids, references, directions = list(reference_ids), list(references), list(directions)
    validate_rows(reference_ids, references, "reference")
    names = list(dict.fromkeys(directions))
    if len(directions) != len(references) or set(names) != set(options):
        raise ValueError("Every direction needs one scoring policy and every row one direction")

    def compute_metrics(evaluation_prediction):
        generated = generated_token_ids(evaluation_prediction, tokenizer)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(predictions) != len(references):
            raise ValueError("Prediction count differs from validation rows")
        parts, metrics = {}, {}
        for name in names:
            rows = [index for index, value in enumerate(directions) if value == name]
            ids = [reference_ids[index] for index in rows]
            parts[name] = score_translations(ids, [predictions[index] for index in rows], ids,
                [references[index] for index in rows], **options[name])
            metrics[f"bleu_{name}"], metrics[f"chrf_{name}"] = parts[name]["bleu"], parts[name]["chrf"]
        bleu = sum(part["bleu"] for part in parts.values()) / len(parts)
        chrf = sum(part["chrf"] for part in parts.values()) / len(parts)
        report = {
            "bleu": bleu, "chrf": chrf, "count": len(references),
            "empty_predictions": sum(part["empty_predictions"] for part in parts.values()),
            "reference_sha256": text_fingerprint(reference_ids, references),
            "prediction_sha256": text_fingerprint(reference_ids, predictions),
            "signatures": {name: part["signatures"] for name, part in parts.items()},
            "sacrebleu_version": importlib.metadata.version("sacrebleu"),
            "text_policy": "per direction: " + "; ".join(f"{name}: {part['text_policy']}" for name, part in parts.items()),
            "text_normalization": {name: part["text_normalization"] for name, part in parts.items()},
            "aggregate": "unweighted mean of per-direction corpus scores",
            "directions": parts,
        }
        if generation_limit is not None:
            at_limit = count_at_generation_limit(generated, tokenizer, generation_limit)
            report["generation_limit"] = generation_limit
            report["at_generation_limit"] = at_limit
            report["at_generation_limit_ratio"] = at_limit / len(predictions)
        if on_result is not None:
            on_result(reference_ids, predictions, report)
        return {"bleu": bleu, "chrf": chrf, **metrics}

    return compute_metrics


def write_predictions(path, ids, texts):
    validate_rows(ids, texts, "prediction")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for key, text in zip(ids, texts):
            stream.write(json.dumps({"id": key, "text": text}, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [row["id"] for row in rows], [row["text"] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL rows with id,text")
    parser.add_argument("--references", required=True, help="JSONL rows with id,text; original full references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bleu-tokenize", choices=BLEU_TOKENIZERS, default="13a")
    parser.add_argument("--normalization", choices=("nfkc",), default=None, help="Apply to predictions and references before scoring")
    args = parser.parse_args()
    if Path(args.output).resolve() in (Path(args.predictions).resolve(), Path(args.references).resolve()):
        parser.error("Output must not overwrite predictions or references")
    report = score_translations(*read_jsonl(args.predictions), *read_jsonl(args.references), args.bleu_tokenize, args.normalization)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("bleu", "chrf", "count", "empty_predictions")}))


if __name__ == "__main__":
    main()
