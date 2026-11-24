"""Đánh giá sâu trên results/matched/test: paired bootstrap và kết quả theo độ dài câu.

Dùng đúng cấu hình chấm của NLLB + LoRA (BLEU tok zh + NFKC cho vi-zh, 13a cho zh-vi; chrF 6/0/2).
Paired bootstrap (Koehn, 2004): lấy mẫu lại 5.000 câu có hoàn lại, cùng chỉ số cho mọi hệ thống,
tính BLEU/chrF từ thống kê đủ (sufficient statistics) và COMET là trung bình điểm từng câu.

    .venv-lora/bin/python scripts/deep_eval_matched.py --out results/matched/test/deep_eval.json
"""
import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path("results/matched/test")
SYSTEMS = ("baseline", "direct", "pivot")
PAIRS = (("baseline", "direct"), ("direct", "pivot"), ("baseline", "pivot"))
BUCKETS = {
    "vi-zh": [(1, 10, "1–10 từ"), (11, 20, "11–20 từ"), (21, 30, "21–30 từ"), (31, 40, "31–40 từ"), (41, 10**9, "trên 40 từ")],
    "zh-vi": [(1, 15, "1–15 ký tự"), (16, 30, "16–30 ký tự"), (31, 45, "31–45 ký tự"), (46, 60, "46–60 ký tự"), (61, 10**9, "trên 60 ký tự")],
}


def read_jsonl(path, key):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["id"] for r in rows], [r[key] for r in rows]


def source_length(text, direction):
    return len(text.split()) if direction == "vi-zh" else len(text)


def load(direction):
    data = {}
    for name in SYSTEMS:
        folder = ROOT / f"{name}_{direction}"
        ids, hyps = read_jsonl(folder / "predictions.jsonl", "text")
        ref_ids, refs = read_jsonl(folder / "references.jsonl", "text")
        _, srcs = read_jsonl(folder / "sources.jsonl", "text")
        cid, comet = read_jsonl(ROOT / "comet" / f"{name}_{direction}" / "comet_segments.jsonl", "comet")
        assert ids == ref_ids == cid, name
        data[name] = {"ids": ids, "hyp": hyps, "ref": refs, "src": srcs, "comet": np.array(comet)}
    first = data[SYSTEMS[0]]
    for name in SYSTEMS[1:]:
        assert data[name]["ids"] == first["ids"] and data[name]["ref"] == first["ref"], name
    return data


def metrics(direction):
    return {
        "bleu": BLEU(tokenize="zh" if direction == "vi-zh" else "13a", lowercase=False, effective_order=False),
        "chrf": CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False),
    }


def prep(texts, direction):
    return [unicodedata.normalize("NFKC", t) for t in texts] if direction == "vi-zh" else list(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default=str(ROOT / "deep_eval.json"))
    args = parser.parse_args()

    report = {"n_bootstrap": args.n, "seed": args.seed, "directions": {}}
    for direction in ("vi-zh", "zh-vi"):
        data = load(direction)
        refs = prep(data["baseline"]["ref"], direction)
        n = len(refs)
        mets = metrics(direction)
        stats, full = {}, {}
        for name in SYSTEMS:
            hyps = prep(data[name]["hyp"], direction)
            stats[name] = {m: np.array(met._extract_corpus_statistics(hyps, [refs])) for m, met in mets.items()}
            full[name] = {m: met.corpus_score(hyps, [refs]).score for m, met in mets.items()}
            full[name]["comet"] = float(data[name]["comet"].mean())

        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, n, size=(args.n, n))
        samples = {name: {m: np.empty(args.n) for m in ("bleu", "chrf", "comet")} for name in SYSTEMS}
        for b in range(args.n):
            for name in SYSTEMS:
                for m, met in mets.items():
                    samples[name][m][b] = met._compute_score_from_stats(list(stats[name][m][idx[b]].sum(axis=0))).score
                samples[name]["comet"][b] = data[name]["comet"][idx[b]].mean()

        pairs = []
        for a, c in PAIRS:
            row = {"a": a, "b": c}
            for m in ("bleu", "chrf", "comet"):
                diff = samples[a][m] - samples[c][m]
                observed = full[a][m] - full[c][m]
                # p: tỉ lệ mẫu mà chiều chênh lệch ngược với chiều quan sát được
                p = float(np.mean(diff <= 0)) if observed > 0 else float(np.mean(diff >= 0))
                row[m] = {"diff": observed, "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "p": p}
            pairs.append(row)

        buckets = []
        lengths = [source_length(s, direction) for s in data["baseline"]["src"]]
        for lo, hi, label in BUCKETS[direction]:
            sel = [i for i, L in enumerate(lengths) if lo <= L <= hi]
            row = {"label": label, "count": len(sel)}
            for name in SYSTEMS:
                hyps = prep([data[name]["hyp"][i] for i in sel], direction)
                sub_refs = [refs[i] for i in sel]
                row[name] = {m: met.corpus_score(hyps, [sub_refs]).score for m, met in mets.items()}
                row[name]["comet"] = float(data[name]["comet"][sel].mean())
            buckets.append(row)

        report["directions"][direction] = {"system": full, "pairs": pairs, "length_buckets": buckets}

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for direction, d in report["directions"].items():
        print(direction, {k: {m: round(v, 4) for m, v in s.items()} for k, s in d["system"].items()})
        for p in d["pairs"]:
            print("  ", p["a"], "vs", p["b"], {m: (round(p[m]["diff"], 4), [round(x, 4) for x in p[m]["ci95"]], p[m]["p"]) for m in ("bleu", "chrf", "comet")})
        for b in d["length_buckets"]:
            print("  ", b["label"], b["count"], {s: {m: round(b[s][m], 2 if m != "comet" else 4) for m in ("bleu", "chrf", "comet")} for s in SYSTEMS})


if __name__ == "__main__":
    main()
