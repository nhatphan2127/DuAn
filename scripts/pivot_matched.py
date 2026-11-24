"""Dịch pivot qua tiếng Anh bằng NLLB gốc với đúng quy trình suy luận của NLLB + LoRA.

Mỗi bước (nguồn -> en, en -> đích) gọi nguyên hàm generate_translations() của
nllb_lora/eval/evaluate.py với cùng config (configs/nllb_lora_shared.json): cùng revision NLLB,
cùng cách nạp mô hình (load_training_model), fp16 qua Accelerate, beam 4, tối đa 256 token,
batch 8, pad bội số 8, không cắt câu nguồn, chọn ngôn ngữ đích bằng decoder prompt.
Chỉ khác cặp ngôn ngữ của từng bước. Bước 2 dịch từ chính bản dịch tiếng Anh của bước 1.

Chạy bằng môi trường .venv-lora:
    .venv-lora/bin/python scripts/pivot_matched.py --direction vi-zh --output_dir results/matched/test/pivot_vi-zh
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import time

LANG = {"vi": "vie_Latn", "zh": "zho_Hans", "en": "eng_Latn"}


def hop_config(config, src, tgt):
    """Config một bước dịch: chỉ thay danh sách chiều dịch, giữ nguyên mọi thiết lập khác."""
    hop = copy.deepcopy(config)
    name = f"{src}-{tgt}"
    # Giữ dạng "shared" (2 chiều) để ngôn ngữ đích được chọn bằng decoder prompt như lúc chạy LoRA.
    # Chiều thứ hai chỉ để đủ dạng shared, không dùng tới.
    hop["model"]["source_language"], hop["model"]["target_language"] = LANG[src], LANG[tgt]
    hop["model"]["directions"] = [
        {"name": name, "source_language": LANG[src], "target_language": LANG[tgt]},
        {"name": f"{tgt}-{src}", "source_language": LANG[tgt], "target_language": LANG[src]},
    ]
    policy = {"bleu_tokenize": "zh", "text_normalization": "nfkc"} if tgt == "zh" else \
        {"bleu_tokenize": "13a", "text_normalization": None}
    hop["evaluation"]["scoring"] = {name: policy, f"{tgt}-{src}": policy}
    return hop, hop["model"]["directions"][0]


def load_hop_tokenizer(config, direction, local_files_only):
    """Giống load_tokenizer() của nllb_lora, chỉ bỏ ràng buộc cặp Hoa–Việt để dùng được tiếng Anh."""
    from transformers import AutoTokenizer

    spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["name_or_path"], revision=spec["revision"],
        src_lang=direction["source_language"], tgt_lang=direction["target_language"],
        legacy_behaviour=False, local_files_only=local_files_only)
    for language in (direction["source_language"], direction["target_language"]):
        token_id = tokenizer.convert_tokens_to_ids(language)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Unknown language token: {language}")
    return tokenizer


def main():
    import torch
    from nllb_lora.train import prepare_data
    # nllb_lora chỉ chấp nhận cặp Hoa–Việt; mở rộng trong tiến trình này (không sửa file) cho 4 bước qua tiếng Anh
    prepare_data.LANGUAGE_PAIRS = prepare_data.LANGUAGE_PAIRS + tuple(
        (LANG[a], LANG[b]) for a, b in (("vi", "en"), ("en", "zh"), ("zh", "en"), ("en", "vi")))
    from nllb_lora.eval.evaluate import direction_rows, generate_translations, select_direction
    from nllb_lora.eval.translation_metrics import write_predictions
    from nllb_lora.train.prepare_data import file_sha256, verify_inputs, write_json
    from nllb_lora.train.train_nllb import load_training_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nllb_lora_shared.json")
    parser.add_argument("--direction", required=True, choices=("vi-zh", "zh-vi"))
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--pivot", default="en")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["data"]["frozen_root"])
    verify_inputs(config, root)
    final = select_direction(config, args.direction)
    spec = config["data"]["splits"][args.split]
    source, reference = root / spec["source"], root / spec["target"]
    if final["swap"]:
        source, reference = reference, source
    rows = list(direction_rows(config, final, source, reference, args.split))
    if args.num_samples:
        rows = rows[:args.num_samples]
    elif len(rows) != spec["expected_pairs"]:
        raise ValueError(f"Expected {spec['expected_pairs']} pairs, found {len(rows)}")

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    src, tgt = args.direction.split("-")
    batch_size = config["training"]["per_device_eval_batch_size"]

    model = load_training_model(config, args.local_files_only)   # cùng cách nạp NLLB gốc như evaluate.py
    precision = "no"
    if torch.cuda.is_available():
        from accelerate import Accelerator
        precision = "fp16" if config["training"]["fp16"] else ("bf16" if config["training"]["bf16"] else "no")
        model = Accelerator(mixed_precision=precision).prepare_model(model, evaluation_mode=True)

    started = time.monotonic()
    hops, current = [], rows
    for hop_src, hop_tgt in ((src, args.pivot), (args.pivot, tgt)):
        hop, direction = hop_config(config, hop_src, hop_tgt)
        tokenizer = load_hop_tokenizer(hop, direction, args.local_files_only)
        over_limit = None
        if hop_src == args.pivot:
            # Câu trung gian chạm trần max_new_tokens (lặp vô hạn ở bước 1) dài tới max_new_tokens + 2 token
            # (thêm token ngôn ngữ và [EOS]). Chỉ ở bước 2, nới giới hạn câu nguồn lên đúng mức đó để
            # dịch nguyên câu trung gian, không cắt; câu nguồn gốc vẫn theo giới hạn 256 của LoRA.
            limit = config["data"]["max_source_tokens"]
            over_limit = [r["id"] for r in current if len(tokenizer(r["source"])["input_ids"]) > limit]
            hop["data"]["max_source_tokens"] = config["generation"]["max_new_tokens"] + 2
        _, predictions, _, report = generate_translations(model, tokenizer, current, hop, batch_size, direction)
        hops.append({"hop": direction["name"], "generation_cap_count": report["generation_cap_count"],
                     "empty_predictions": report["empty_predictions"], "generation": report["generation"],
                     "max_source_tokens": hop["data"]["max_source_tokens"],
                     "sources_over_original_limit": over_limit})
        if hop_tgt == args.pivot:
            write_predictions(destination / f"intermediate.{args.pivot}.jsonl", [r["id"] for r in rows], predictions)
        # Bước sau dịch từ bản dịch của bước trước; "target" vẫn là tham chiếu cuối (chỉ để đủ định dạng)
        current = [{**r, "source": p} for r, p in zip(rows, predictions)]

    ids = [r["id"] for r in rows]
    write_predictions(destination / "predictions.jsonl", ids, predictions)
    write_predictions(destination / "references.jsonl", ids, [r["target"] for r in rows])
    write_predictions(destination / "sources.jsonl", ids, [r["source"] for r in rows])
    write_json(destination / "decode_info.json", {
        "system": "nllb_pivot", "direction": args.direction, "pivot": args.pivot, "split": args.split,
        "count": len(rows), "model": config["model"]["name_or_path"], "revision": config["model"]["revision"],
        "hops": hops, "inference_precision": precision, "batch_size": batch_size,
        "source_sha256": file_sha256(source), "reference_file_sha256": file_sha256(reference),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"direction": args.direction, "count": len(rows),
                      "generation_cap_count": [h["generation_cap_count"] for h in hops]}))


if __name__ == "__main__":
    main()
