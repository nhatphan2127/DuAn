"""NLLB-200 dịch Việt <-> Trung qua tiếng Anh làm ngôn ngữ trung gian (pivot), không train gì thêm:
    pivot:   vi ──NLLB──▶ en ──NLLB──▶ zh
(chiều Trung -> Việt tương tự). Điểm được chấm bằng đúng normalize_text() + compute_scores() + COMET
của baseline, nên so được với bảng kết quả của baseline và của mô hình NLLB fine-tune / zero-shot.

NLLB dịch thẳng (zero-shot) do nhóm B chạy; truyền file bản dịch đó qua --compare để kiểm định
paired bootstrap pivot so với dịch thẳng. --modes direct pivot thì script tự dịch thẳng để đối chứng.

Ví dụ:
    python scripts/pivot_nllb.py                                  # cả hai chiều, pivot, beam k=5
    python scripts/pivot_nllb.py --compare vi-zh=zs/test.zh zh-vi=zs/test.vi   # + kiểm định với bản zero-shot
    python scripts/pivot_nllb.py --num_samples 50 --comet_model ''  # chạy thử nhanh
    python scripts/pivot_nllb.py --model ./nllb_finetuned         # pivot bằng NLLB đã fine-tune (đo mức "quên" tiếng Anh)

Bản dịch đã có trong --out_dir được dùng lại (chạy tiếp được khi bị ngắt); --overwrite để dịch lại.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được nmt/ khi chạy scripts/*.py

import argparse
import gc

import torch
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest
from tqdm.auto import tqdm

from nmt.data import read_lines
from nmt.evaluation import compute_comet, compute_scores, normalize_text
from nmt.logger import logger
from nmt.utils import DEVICE

NLLB_CODES = {'vi': 'vie_Latn', 'zh': 'zho_Hans', 'en': 'eng_Latn'}
PIVOT_LANG = 'en'


def translate(texts, model, tokenizer, src_lang, tgt_lang, beam_size, batch_size, max_len):
    """Dịch theo batch, sắp câu theo độ dài để giảm padding, trả về đúng thứ tự ban đầu."""
    tokenizer.src_lang = NLLB_CODES[src_lang]
    forced_bos = tokenizer.convert_tokens_to_ids(NLLB_CODES[tgt_lang])
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    hyps = [None] * len(texts)
    for start in tqdm(range(0, len(order), batch_size), desc=f"NLLB {src_lang}→{tgt_lang}", leave=False):
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], return_tensors='pt', padding=True,
                          truncation=True, max_length=max_len).to(DEVICE)
        with torch.inference_mode():
            out = model.generate(**batch, forced_bos_token_id=forced_bos, num_beams=beam_size,
                                 max_new_tokens=max_len)
        for i, h in zip(idx, tokenizer.batch_decode(out, skip_special_tokens=True)):
            hyps[i] = h.strip()
    return hyps


def cached_translate(path, overwrite, *args, **kwargs):
    """Dùng lại file bản dịch nếu đã có đủ số dòng, không thì dịch rồi ghi ra `path`."""
    texts = args[0]
    if path.exists() and not overwrite:
        lines = read_lines(path)
        if len(lines) == len(texts):
            logger.log("reuse_hyp", f"Dùng lại {path}", hyp_file=path)
            return lines
    hyps = translate(*args, **kwargs)
    path.write_text('\n'.join(hyps) + '\n', encoding='utf-8')
    return hyps


def paired_bootstrap(direct, pivot, refs, tgt_lang, n_samples=1000):
    """Paired bootstrap (Koehn, 2004) của sacreBLEU, hệ thống đối chứng là bản dịch thẳng. Trả về {metric: (Δ, p)}."""
    metrics = {'bleu': BLEU(tokenize='zh' if tgt_lang == 'zh' else '13a'),
               'chrf': CHRF(), 'chrf++': CHRF(word_order=2)}
    norm = lambda xs: [normalize_text(x) for x in xs]
    test = PairedTest([('direct', norm(direct)), ('pivot', norm(pivot))], metrics, [norm(refs)],
                      test_type='bs', n_samples=n_samples)
    # Kết quả đặt tên theo sacreBLEU ('BLEU', 'chrF2', ...) theo đúng thứ tự `metrics`, cột đầu là 'System'
    _, results = test()
    per_metric = [res for name, res in results.items() if name != 'System']
    return {name: (round(res[1].score - res[0].score, 2), round(res[1].p_value, 4))
            for name, res in zip(metrics, per_metric)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/nllb-200-distilled-600M')
    parser.add_argument('--split', default='test')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--out_dir', default='./nllb_pivot')
    parser.add_argument('--directions', nargs='+', default=['vi-zh', 'zh-vi'], choices=['vi-zh', 'zh-vi'])
    parser.add_argument('--modes', nargs='+', default=['pivot'], choices=['direct', 'pivot'])
    parser.add_argument('--compare', nargs='+', default=[], metavar='CHIỀU=FILE',
                        help='file bản dịch thẳng (vd. NLLB zero-shot của nhóm B) để kiểm định với pivot, '
                             'vd. vi-zh=zs/test.zh zh-vi=zs/test.vi')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=256, help='số token tối đa của câu vào và câu ra')
    parser.add_argument('--num_samples', type=int, default=0, help='0 = toàn bộ split')
    parser.add_argument('--overwrite', action='store_true', help='dịch lại kể cả khi đã có file bản dịch')
    parser.add_argument('--comet_model', default='Unbabel/wmt22-comet-da', help="'' = bỏ qua COMET")
    parser.add_argument('--comet_batch_size', type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    logger.set_file(out_dir / 'eval_log.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype).to(DEVICE).eval()
    gen = dict(model=model, tokenizer=tokenizer, beam_size=args.beam_size,
               batch_size=args.batch_size, max_len=args.max_len)

    results = []
    for direction in args.directions:
        src_lang, tgt_lang = direction.split('-')
        srcs = read_lines(Path(args.data_dir) / f'{args.split}.{src_lang}')
        refs = read_lines(Path(args.data_dir) / f'{args.split}.{tgt_lang}')
        if args.num_samples:
            srcs, refs = srcs[:args.num_samples], refs[:args.num_samples]

        hyps = {}
        if 'direct' in args.modes:
            path = out_dir / f'{args.split}.direct.{src_lang}-{tgt_lang}.{tgt_lang}'
            hyps['direct'] = cached_translate(path, args.overwrite, srcs, src_lang=src_lang,
                                              tgt_lang=tgt_lang, **gen)
        if 'pivot' in args.modes:
            # Giữ lại bản dịch tiếng Anh trung gian để phân tích lỗi sinh ra ở bước nào
            mid_path = out_dir / f'{args.split}.pivot.{src_lang}-{PIVOT_LANG}.{PIVOT_LANG}'
            mids = cached_translate(mid_path, args.overwrite, srcs, src_lang=src_lang,
                                    tgt_lang=PIVOT_LANG, **gen)
            path = out_dir / f'{args.split}.pivot.{src_lang}-{tgt_lang}.{tgt_lang}'
            hyps['pivot'] = cached_translate(path, args.overwrite, mids, src_lang=PIVOT_LANG,
                                             tgt_lang=tgt_lang, **gen)

        for mode, h in hyps.items():
            results.append(dict(direction=direction, src_lang=src_lang, tgt_lang=tgt_lang, mode=mode,
                                scores=compute_scores(h, refs, tgt_lang), srcs=srcs, hyps=h, refs=refs))

    if args.comet_model:
        # Giải phóng NLLB khỏi GPU trước khi nạp COMET; gộp mọi hệ thống vào một lần predict.
        # src của COMET luôn là câu nguồn gốc, kể cả với pivot.
        del model, gen
        gc.collect()
        torch.cuda.empty_cache()
        _, seg_scores = compute_comet(sum((r['srcs'] for r in results), []),
                                      sum(([normalize_text(h) for h in r['hyps']] for r in results), []),
                                      sum(([normalize_text(x) for x in r['refs']] for r in results), []),
                                      args.comet_model, args.comet_batch_size)
        start = 0
        for r in results:
            n = len(r['hyps'])
            r['scores']['comet'] = round(sum(seg_scores[start:start + n]) / n, 4)
            start += n

    for r in results:
        scores = r['scores']
        msg = (f"[{args.split} | {r['src_lang']}→{r['tgt_lang']} | {r['mode']}] {len(r['hyps'])} câu"
               f" | BLEU: {scores['bleu']:.2f} | chrF: {scores['chrf']:.2f} | chrF++: {scores['chrf++']:.2f}")
        if 'comet' in scores:
            msg += f" | COMET: {scores['comet']:.4f}"
        logger.log("eval_result", msg, split=args.split, direction=r['direction'], mode=r['mode'],
                   pivot_lang=PIVOT_LANG if r['mode'] == 'pivot' else None, num_sentences=len(r['hyps']),
                   model=args.model, beam_size=args.beam_size, comet_model=args.comet_model or None,
                   normalization='NFKC', **scores)

    # Đối chứng: file truyền qua --compare, nếu không có thì bản dịch thẳng script vừa chạy
    compare = dict(item.split('=', 1) for item in args.compare)
    by_key = {(r['direction'], r['mode']): r for r in results}
    for direction in args.directions:
        if (direction, 'pivot') not in by_key:
            continue
        p = by_key[direction, 'pivot']
        if direction in compare:
            direct, source = read_lines(compare[direction])[:len(p['hyps'])], compare[direction]
            if len(direct) != len(p['hyps']):
                raise ValueError(f"{source} có {len(direct)} dòng, cần {len(p['hyps'])}")
        elif (direction, 'direct') in by_key:
            direct, source = by_key[direction, 'direct']['hyps'], 'direct (pivot_nllb.py)'
        else:
            continue
        sig = paired_bootstrap(direct, p['hyps'], p['refs'], p['tgt_lang'])
        msg = f"[{direction}] pivot - dịch thẳng ({source}): " + ' | '.join(
            f"{name}: {delta:+.2f} (p = {pval:.4f})" for name, (delta, pval) in sig.items())
        logger.log("paired_bootstrap", msg, split=args.split, direction=direction, n_samples=1000,
                   compare_with=source,
                   **{f'delta_{k}': v[0] for k, v in sig.items()}, **{f'p_{k}': v[1] for k, v in sig.items()})


if __name__ == '__main__':
    main()
