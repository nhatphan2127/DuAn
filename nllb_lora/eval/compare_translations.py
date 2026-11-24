"""Compare aligned pretrained/fine-tuned predictions with paired bootstrap."""

import argparse
import csv
import json
import os
from pathlib import Path

from nllb_lora.eval.translation_metrics import decoding_settings, read_jsonl, score_translations
from nllb_lora.train.prepare_data import file_sha256, write_json


def compare_translations(ids, baseline, candidate, references, resamples=1000, seed=42, bleu_tokenize="13a", normalization=None):
    from sacrebleu.metrics import BLEU, CHRF
    from sacrebleu.significance import PairedTest

    if not isinstance(resamples, int) or resamples < 40:
        raise ValueError("At least 40 bootstrap resamples required; use 1000 for final reporting")
    before = score_translations(ids, baseline, ids, references, bleu_tokenize, normalization)
    after = score_translations(ids, candidate, ids, references, bleu_tokenize, normalization)
    if normalization == "nfkc":
        import unicodedata
        baseline = [unicodedata.normalize("NFKC", text) for text in baseline]
        candidate = [unicodedata.normalize("NFKC", text) for text in candidate]
        references = [unicodedata.normalize("NFKC", text) for text in references]
    old_seed = os.environ.get("SACREBLEU_SEED")
    os.environ["SACREBLEU_SEED"] = str(seed)
    try:
        signatures, results = PairedTest(named_systems=[("pretrained", baseline), ("fine_tuned", candidate)],
            metrics={"BLEU": BLEU(tokenize=bleu_tokenize, lowercase=False),
                     "chrF": CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False)},
            references=[references], test_type="bs", n_samples=resamples, n_jobs=1)()
    finally:
        if old_seed is None:
            os.environ.pop("SACREBLEU_SEED", None)
        else:
            os.environ["SACREBLEU_SEED"] = old_seed
    statistics = {}
    for metric, rows in results.items():
        if metric == "System":
            continue
        statistics[metric] = [{key: float(value) if value is not None else None for key, value in row.__dict__.items()} for row in rows]
    return {"status": "COMPARISON_REQUIRES_QUALITY_REVIEW", "baseline": before, "fine_tuned": after,
        "delta": {name: after[name] - before[name] for name in ("bleu", "chrf")},
        "paired_bootstrap": {"resamples": resamples, "seed": seed, "systems": ["pretrained", "fine_tuned"],
            "signatures": {name: str(value) for name, value in signatures.items()}, "results": statistics,
            "ci_note": "ci is the half-width of each system's 95% resampled score interval, not a delta interval"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Prediction JSONL id,text")
    parser.add_argument("--candidate", required=True, help="Prediction JSONL id,text")
    parser.add_argument("--references", required=True, help="Original reference JSONL id,text")
    parser.add_argument("--sources", help="Optional source JSONL id,text for manual error analysis")
    parser.add_argument("--baseline-metrics", required=True, help="Evaluator metrics.json; guards decoding/reference comparability")
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    ids, references = read_jsonl(args.references)
    bid, baseline = read_jsonl(args.baseline)
    cid, candidate = read_jsonl(args.candidate)
    if ids != bid or ids != cid:
        parser.error("Baseline/candidate/reference IDs must have exactly the same order")
    before = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    after = json.loads(Path(args.candidate_metrics).read_text(encoding="utf-8"))
    if decoding_settings(before["generation"]) != decoding_settings(after["generation"]) or before["reference_sha256"] != after["reference_sha256"]:
        parser.error("Different generation settings or reference corpus; rescore under identical conditions")
    tokenizers = {metric["signatures"]["bleu"].split("tok:")[1].split("|")[0] for metric in (before, after)}
    if len(tokenizers) != 1:
        parser.error("Baseline and candidate metrics use different BLEU tokenizers")
    normalizations = {metric.get("text_normalization") for metric in (before, after)}
    if len(normalizations) != 1:
        parser.error("Baseline and candidate metrics use different text normalization")
    report = compare_translations(ids, baseline, candidate, references, args.resamples,
        bleu_tokenize=tokenizers.pop(), normalization=normalizations.pop())
    for supplied, actual in ((before, report["baseline"]), (after, report["fine_tuned"])):
        if supplied["prediction_sha256"] != actual["prediction_sha256"] or supplied["reference_sha256"] != actual["reference_sha256"]:
            parser.error("Metric metadata does not belong to the supplied predictions/references")
    sources = [""] * len(ids)
    if args.sources:
        sid, sources = read_jsonl(args.sources)
        if sid != ids:
            parser.error("Source IDs do not match reference order")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    report["inputs"] = {name: {"path": str(Path(value).resolve()), "sha256": file_sha256(value)}
        for name, value in vars(args).items() if name not in ("output_dir", "resamples") and value}
    write_json(destination / "comparison.json", report)
    with (destination / "examples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "source", "reference", "pretrained", "fine_tuned", "manual_error_type", "manual_note"])
        for row in zip(ids, sources, references, baseline, candidate):
            writer.writerow([*row, "", ""])
    print(json.dumps(report["delta"]))


if __name__ == "__main__":
    main()
