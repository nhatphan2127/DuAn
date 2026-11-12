"""Tính BLEU / chrF / chrF++ / COMET của best checkpoint trên một split (mặc định: test).

Mô hình hai chiều được đánh giá riêng từng chiều.

Trên Colab:
    !python scripts/evaluate.py                                # mô hình Việt -> Trung, beam search k=5
    !python scripts/evaluate.py --direction zh-vi              # mô hình Trung -> Việt
    !python scripts/evaluate.py --direction both               # mô hình hai chiều, chấm cả hai chiều
    !python scripts/evaluate.py --split val --beam_size 1
    !python scripts/evaluate.py --comet_model ''               # bỏ qua COMET
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được nmt/ khi chạy scripts/*.py

import argparse
import gc

import torch

from nmt.config import DIRECTIONS, Config
from nmt.data import build_dataset
from nmt.evaluation import compute_comet, evaluate_bleu
from nmt.logger import logger
from nmt.model import load_model
from nmt.tokenizer import load_tokenizers
from nmt.utils import DEVICE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test')
    parser.add_argument('--direction', default='vi-zh', choices=DIRECTIONS,
                        help='chỉ dùng để chọn base_dir mặc định; chiều dịch thật lấy từ checkpoint')
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--data_dir', default=Config.data_dir)
    parser.add_argument('--beam_size', type=int, default=Config.beam_size)
    parser.add_argument('--num_samples', type=int, default=0, help='0 = toàn bộ split')
    parser.add_argument('--comet_model', default='Unbabel/wmt22-comet-da', help="'' = bỏ qua COMET")
    parser.add_argument('--comet_batch_size', type=int, default=64)
    args = parser.parse_args()

    paths = Config(direction=args.direction, base_dir=args.base_dir)
    model, cfg = load_model(paths.best_ckpt, DEVICE)
    cfg.base_dir, cfg.data_dir = paths.base_dir, args.data_dir
    tokenizers = load_tokenizers(cfg, train_if_missing=False)

    results = []
    for src_lang, tgt_lang in cfg.lang_pairs:
        for lang in (src_lang, tgt_lang):
            if not cfg.split_file(args.split, lang).exists():
                raise FileNotFoundError(f"Không tìm thấy {cfg.split_file(args.split, lang)}")
        dataset = build_dataset(cfg, args.split, tokenizers, src_lang, tgt_lang)
        scores, hyps, refs = evaluate_bleu(model, dataset, tokenizers[tgt_lang], tgt_lang, cfg.max_len,
                                           num_samples=args.num_samples, beam_size=args.beam_size)

        # Ghi bản dịch trước khi tính COMET để lỗi ở COMET không làm mất kết quả beam search
        out_path = Path(cfg.base_dir) / f'{args.split}.hyp.{tgt_lang}'
        out_path.write_text('\n'.join(hyps) + '\n', encoding='utf-8')
        srcs = dataset.src_texts[:len(hyps)]
        results.append(dict(src_lang=src_lang, tgt_lang=tgt_lang, scores=scores,
                            srcs=srcs, hyps=hyps, refs=refs, out_path=out_path))

    if args.comet_model:
        # Giải phóng model dịch khỏi GPU trước khi nạp model COMET
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # Gộp mọi chiều vào một lần predict để chỉ nạp model COMET một lần, rồi tách điểm theo chiều
        _, seg_scores = compute_comet(sum((r['srcs'] for r in results), []),
                                      sum((r['hyps'] for r in results), []),
                                      sum((r['refs'] for r in results), []),
                                      args.comet_model, args.comet_batch_size)
        start = 0
        for r in results:
            n = len(r['hyps'])
            r['scores']['comet'] = round(sum(seg_scores[start:start + n]) / n, 4)
            start += n

    # Mỗi lần đánh giá thêm một dòng (mỗi chiều) vào eval_log.jsonl để so sánh giữa các lần
    logger.set_file(Path(cfg.base_dir) / 'eval_log.jsonl')
    for r in results:
        scores = r['scores']
        msg = (f"[{args.split} | {r['src_lang']}→{r['tgt_lang']}] {len(r['hyps'])} câu"
               f" | BLEU: {scores['bleu']:.2f} | chrF: {scores['chrf']:.2f} | chrF++: {scores['chrf++']:.2f}")
        if 'comet' in scores:
            msg += f" | COMET: {scores['comet']:.4f}"
        logger.log(
            "eval_result", f"{msg}\nĐã lưu bản dịch vào {r['out_path']}",
            split=args.split, direction=f"{r['src_lang']}-{r['tgt_lang']}", num_sentences=len(r['hyps']),
            beam_size=args.beam_size, checkpoint=paths.best_ckpt, hyp_file=r['out_path'],
            comet_model=args.comet_model or None, normalization='NFKC', **scores,
        )


if __name__ == '__main__':
    main()
