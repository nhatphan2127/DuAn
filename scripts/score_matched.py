"""Chấm BLEU/chrF cho một thư mục đầu ra bằng đúng hàm chấm của NLLB + LoRA.

Dùng score_translations() và scoring_options() của nllb_lora/eval/translation_metrics.py với
configs/nllb_lora_shared.json: sacreBLEU 2.5.1 (môi trường .venv-lora), Vi->Zh tokenizer "zh" + NFKC,
Zh->Vi tokenizer "13a" không chuẩn hoá, chrF char 6 / word 0 / beta 2, phân biệt hoa thường.
Trước khi chấm, đối chiếu id, câu nguồn và tham chiếu với dữ liệu đóng băng (direction_rows),
nên mọi hệ thống chắc chắn được chấm trên cùng 5.000 cặp câu theo cùng thứ tự.
Ghi metrics.json cùng định dạng với evaluate.py, để nllb_lora/eval/score_comet.py chấm COMET được.

    .venv-lora/bin/python scripts/score_matched.py --eval_dir results/matched/test/baseline_vi-zh --direction vi-zh
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json


def main():
    from sacrebleu.metrics import CHRF
    from nllb_lora.eval.evaluate import direction_rows, select_direction
    from nllb_lora.eval.translation_metrics import read_jsonl, score_translations, scoring_options
    from nllb_lora.train.prepare_data import verify_inputs, write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nllb_lora_shared.json")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--direction", required=True, choices=("vi-zh", "zh-vi"))
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--allow_subset", action="store_true", help="cho phép chấm N câu đầu (chạy thử)")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["data"]["frozen_root"])
    verify_inputs(config, root)
    direction = select_direction(config, args.direction)
    spec = config["data"]["splits"][args.split]
    source, reference = root / spec["source"], root / spec["target"]
    if direction["swap"]:
        source, reference = reference, source
    official = list(direction_rows(config, direction, source, reference, args.split))

    eval_dir = Path(args.eval_dir)
    ids, predictions = read_jsonl(eval_dir / "predictions.jsonl")
    ref_ids, references = read_jsonl(eval_dir / "references.jsonl")
    src_ids, sources = read_jsonl(eval_dir / "sources.jsonl")
    if len(ids) != len(official) and not args.allow_subset:
        raise ValueError(f"Cần {len(official)} câu, thư mục có {len(ids)} câu")
    official = official[:len(ids)]
    if not (ids == ref_ids == src_ids == [r["id"] for r in official]):
        raise ValueError("Id hoặc thứ tự câu khác dữ liệu đóng băng")
    if references != [r["target"] for r in official] or sources != [r["source"] for r in official]:
        raise ValueError("Câu nguồn hoặc tham chiếu khác dữ liệu đóng băng")

    options = scoring_options(config, direction["name"])
    report = score_translations(ids, predictions, ids, references, **options)
    # chrF++ chỉ để tham khảo (LoRA không báo cáo chrF++); cùng chuẩn hoá với chrF chính
    scored_hyp, scored_ref = predictions, references
    if options["normalization"] == "nfkc":
        import unicodedata
        scored_hyp = [unicodedata.normalize("NFKC", t) for t in predictions]
        scored_ref = [unicodedata.normalize("NFKC", t) for t in references]
    chrf_pp = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False)
    report["chrf++_reference_only"] = chrf_pp.corpus_score(scored_hyp, [scored_ref]).score

    decode_info = eval_dir / "decode_info.json"
    extra = json.loads(decode_info.read_text(encoding="utf-8")) if decode_info.is_file() else {}
    result = {**report, "split": args.split, "direction": direction["name"],
              "scorer": "scripts/score_matched.py -> nllb_lora.eval.translation_metrics.score_translations",
              "decode_info": extra}
    metrics_file = eval_dir / "metrics.json"
    if metrics_file.is_file() and "scorer" not in json.loads(metrics_file.read_text(encoding="utf-8")):
        # metrics.json do nllb_lora/eval/evaluate.py ghi: giữ nguyên, chỉ đối chiếu rồi ghi kết quả ra file riêng
        original = json.loads(metrics_file.read_text(encoding="utf-8"))
        for key in ("bleu", "chrf", "prediction_sha256", "reference_sha256"):
            if original[key] != report[key]:
                raise ValueError(f"{key} khác metrics.json gốc của evaluate.py")
        write_json(eval_dir / "metrics_matched.json", result)
    else:
        write_json(metrics_file, result)
    print(json.dumps({"dir": str(eval_dir), "bleu": round(report["bleu"], 2), "chrf": round(report["chrf"], 2),
                      "chrf++": round(report["chrf++_reference_only"], 2), "count": report["count"],
                      "empty": report["empty_predictions"], "sacrebleu": report["sacrebleu_version"]}))


if __name__ == "__main__":
    main()
