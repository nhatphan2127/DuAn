"""Dịch tập test bằng Transformer baseline với đúng cấu hình suy luận của NLLB + LoRA.

Cấu hình lấy từ configs/nllb_lora_shared.json (mục "generation"): beam 4, length penalty 1,0 kiểu
Hugging Face (điểm / số token sinh ra), tối đa 256 token sinh ra, không cắt câu nguồn.
Kết quả ghi theo đúng định dạng thư mục đầu ra của nllb_lora/eval/evaluate.py
(sources/predictions/references.jsonl, id dạng "test:<số dòng>"), để chấm bằng cùng một quy trình:

    python scripts/translate_matched.py --base_dir vi_zh_bidir --out_root results/matched/test
    .venv-lora/bin/python scripts/score_matched.py --eval_dir results/matched/test/baseline_vi-zh ...

Chạy bằng môi trường .venv của baseline. Không ghi đè test.hyp.* hay eval_log.jsonl của baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được nmt/ khi chạy scripts/*.py

import argparse
import json
import time

import torch
from tqdm import tqdm

from nmt.config import Config
from nmt.data import build_dataset
from nmt.decoding import beam_search
from nmt.model import load_model
from nmt.tokenizer import load_tokenizers
from nmt.utils import DEVICE

NO_TRUNCATION = 10 ** 9


def read_raw_rows(source_file, target_file, split, suffix=''):
    """Giống direction_rows() của nllb_lora: chỉ bỏ ký tự xuống dòng, giữ nguyên tham chiếu gốc.
    suffix: chiều thứ hai trong config shared (vi-zh) có id dạng "test:N@vi-zh"."""
    sources = Path(source_file).read_text(encoding='utf-8').split('\n')
    targets = Path(target_file).read_text(encoding='utf-8').split('\n')
    if sources and sources[-1] == '':
        sources, targets = sources[:-1], targets[:-1]
    if len(sources) != len(targets):
        raise ValueError(f"{source_file} và {target_file} lệch số dòng")
    return [{"id": f"{split}:{i}{suffix}", "source": s.rstrip('\r'), "target": t.rstrip('\r')}
            for i, (s, t) in enumerate(zip(sources, targets), 1)]


def write_jsonl(path, ids, texts):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        for key, text in zip(ids, texts):
            stream.write(json.dumps({"id": key, "text": text}, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='vi_zh_bidir')
    parser.add_argument('--data_dir', default=Config.data_dir)
    parser.add_argument('--split', default='test')
    parser.add_argument('--directions', nargs='+', default=['vi-zh', 'zh-vi'])
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty_style', default='hf', choices=('hf', 'gnmt'))
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--max_source_tokens', type=int, default=256,
                        help='chỉ để đếm số câu nguồn vượt ngưỡng; câu nguồn không bị cắt')
    parser.add_argument('--num_samples', type=int, default=0, help='0 = toàn bộ split')
    args = parser.parse_args()

    paths = Config(base_dir=args.base_dir)
    model, cfg = load_model(paths.best_ckpt, DEVICE)
    cfg.base_dir, cfg.data_dir = paths.base_dir, args.data_dir
    tokenizers = load_tokenizers(cfg, train_if_missing=False)
    trained_max_len = cfg.max_len
    cfg.max_len = NO_TRUNCATION   # TranslationDataset cắt câu theo cfg.max_len; tắt để giữ nguyên câu nguồn

    for direction in args.directions:
        src_lang, tgt_lang = direction.split('-')
        if (src_lang, tgt_lang) not in cfg.lang_pairs:
            raise ValueError(f"Checkpoint không có chiều {direction}")
        destination = Path(args.out_root) / f'baseline_{direction}'
        destination.mkdir(parents=True, exist_ok=False)

        rows = read_raw_rows(cfg.split_file(args.split, src_lang), cfg.split_file(args.split, tgt_lang), args.split,
                             '' if direction == 'zh-vi' else f'@{direction}')
        dataset = build_dataset(cfg, args.split, tokenizers, src_lang, tgt_lang)
        if len(dataset) != len(rows):
            raise ValueError("Dataset đã bỏ câu rỗng nên lệch hàng với tham chiếu")
        n = min(args.num_samples, len(rows)) if args.num_samples else len(rows)
        rows = rows[:n]

        started = time.monotonic()
        hyp_ids, capped, long_sources = [], 0, 0
        for i in tqdm(range(n), desc=f"Baseline {direction} (k={args.beam_size})"):
            src = dataset[i][0].unsqueeze(0).to(DEVICE)
            long_sources += src.size(1) - 2 > args.max_source_tokens   # bỏ token đầu câu và [EOS]
            out = beam_search(model, src, args.beam_size, NO_TRUNCATION, args.length_penalty,
                              args.length_penalty_style, args.max_new_tokens)
            capped += len(out) + 1 >= args.max_new_tokens   # +1: [EOS]; chạm trần nếu không kịp sinh [EOS]
            hyp_ids.append(out)
        predictions = tokenizers[tgt_lang].decode_batch(hyp_ids)
        elapsed = time.monotonic() - started

        ids = [r['id'] for r in rows]
        write_jsonl(destination / 'predictions.jsonl', ids, predictions)
        write_jsonl(destination / 'references.jsonl', ids, [r['target'] for r in rows])
        write_jsonl(destination / 'sources.jsonl', ids, [r['source'] for r in rows])
        (destination / 'decode_info.json').write_text(json.dumps({
            "system": "transformer_baseline", "checkpoint": str(paths.best_ckpt), "direction": direction,
            "split": args.split, "count": n,
            "generation": {"num_beams": args.beam_size, "length_penalty": args.length_penalty,
                           "length_penalty_style": args.length_penalty_style,
                           "max_new_tokens": args.max_new_tokens, "do_sample": False,
                           "source_truncation": False},
            "trained_max_len": trained_max_len,
            "sources_over_max_source_tokens": long_sources, "max_source_tokens": args.max_source_tokens,
            "generation_cap_count": capped, "inference_precision": "fp32",
            "device": torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
            "elapsed_seconds": elapsed,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"[{direction}] {n} câu, {capped} câu chạm trần {args.max_new_tokens} token, "
              f"{long_sources} câu nguồn > {args.max_source_tokens} token, {elapsed:.0f}s -> {destination}")


if __name__ == '__main__':
    main()
