"""COMET for one evaluator output folder (sources/predictions/references.jsonl).

Standalone on purpose (standard library + COMET): unbabel-comet needs numpy<2 and torchmetrics<0.11,
so it runs in its own environment, not in the training lock and not in the frozen run snapshot.
"""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import time
import unicodedata


MODEL = "Unbabel/wmt22-comet-da"
REVISION = "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
SYSTEM_SCORE_TOLERANCE = 1e-6


def text_fingerprint(ids, texts):
    """Same hash as nllb_lora.eval.translation_metrics.text_fingerprint (a test keeps them equal)."""
    payload = json.dumps(list(zip(ids, texts)), ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    return [row["id"] for row in rows], [row["text"] for row in rows]


def load_inputs(eval_dir, normalization=None):
    """Aligned COMET samples and their provenance; refuses anything that is not the scored output."""
    eval_dir = Path(eval_dir)
    ids, sources = read_rows(eval_dir / "sources.jsonl")
    prediction_ids, predictions = read_rows(eval_dir / "predictions.jsonl")
    reference_ids, references = read_rows(eval_dir / "references.jsonl")
    if not ids or len(set(ids)) != len(ids) or ids != prediction_ids or ids != reference_ids:
        raise ValueError("Source/prediction/reference IDs must be non-empty, unique and in the same order")
    if any(not text.strip() for text in sources + references):
        raise ValueError("Source and reference text must not be empty")
    prediction_hash, reference_hash = text_fingerprint(ids, predictions), text_fingerprint(ids, references)
    info = {}
    metrics_file = eval_dir / "metrics.json"
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if (metrics["count"] != len(ids) or metrics["prediction_sha256"] != prediction_hash
                or metrics["reference_sha256"] != reference_hash):
            raise ValueError("JSONL files differ from the scored outputs recorded in metrics.json")
        recorded = metrics.get("text_normalization") or "none"
        if normalization not in (None, recorded):
            raise ValueError(f"--normalization {normalization} differs from metrics.json ({recorded})")
        normalization = recorded
        info = {key: metrics[key] for key in ("split", "direction") if key in metrics}
    if normalization is None:
        raise ValueError("No metrics.json: pass --normalization none or nfkc explicitly")
    if normalization == "nfkc":
        # Same scored strings as BLEU/chrF of this direction; the source stays as the system received it.
        predictions = [unicodedata.normalize("NFKC", text) for text in predictions]
        references = [unicodedata.normalize("NFKC", text) for text in references]
    samples = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(sources, predictions, references)]
    return ids, samples, {**info, "text_normalization": normalization, "prediction_sha256": prediction_hash,
        "reference_sha256": reference_hash, "source_sha256": text_fingerprint(ids, sources),
        "empty_predictions": sum(not text.strip() for text in predictions)}


def load_pinned_model():
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=MODEL, revision=REVISION)
    return load_from_checkpoint(str(Path(snapshot) / "checkpoints" / "model.ckpt"))


def score_samples(model, samples, batch_size, gpus):
    output = model.predict(samples, batch_size=batch_size, gpus=gpus)
    scores = [float(value) for value in output.scores]
    if len(scores) != len(samples) or not all(math.isfinite(value) for value in scores):
        raise ValueError("COMET must return one finite score per sample")
    mean = sum(scores) / len(scores)
    if abs(mean - output.system_score) > SYSTEM_SCORE_TOLERANCE:
        raise ValueError("COMET system score differs from the mean of its segment scores")
    return mean, scores


def main(argv=None, load_model=load_pinned_model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, help="Folder holding sources/predictions/references.jsonl")
    parser.add_argument("--output-dir", required=True, help="New folder for comet_metrics.json and comet_segments.jsonl")
    parser.add_argument("--normalization", choices=("none", "nfkc"), help="Required only without metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1, help="0 runs on CPU")
    args = parser.parse_args(argv)
    ids, samples, info = load_inputs(args.eval_dir, args.normalization)
    destination = Path(args.output_dir).resolve()
    source = Path(args.eval_dir).resolve()
    if destination == source or source in destination.parents:
        parser.error("Output must be outside the evaluated folder")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    model = load_model()
    started = time.monotonic()
    mean, scores = score_samples(model, samples, args.batch_size, args.gpus)
    # Created only after scoring, so an out-of-memory failure can be retried with a smaller batch.
    destination.mkdir(parents=True)
    with (destination / "comet_segments.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for key, value in zip(ids, scores):
            stream.write(json.dumps({"id": key, "comet": value}, ensure_ascii=False) + "\n")
    report = {"comet": mean, "count": len(scores), "model": MODEL, "revision": REVISION,
        "comet_version": importlib.metadata.version("unbabel-comet"), "batch_size": args.batch_size,
        "gpus": args.gpus, "elapsed_seconds": time.monotonic() - started, **info}
    # Written last: its presence marks a complete result.
    (destination / "comet_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("comet", "count")}))


if __name__ == "__main__":
    main()
