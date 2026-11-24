import unicodedata

import sacrebleu
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .decoding import beam_search, greedy_decode
from .tokenizer import PAD_IDX
from .utils import DEVICE


def normalize_text(text):
    """Chuẩn hoá dùng chung cho mọi hệ thống trước khi chấm điểm: NFKC (dấu câu full-width -> ASCII,
    dấu tiếng Việt dạng dựng sẵn) và gộp khoảng trắng. Áp cho cả hyp lẫn ref, không phụ thuộc tokenizer
    của mô hình, nên điểm của baseline so được với hệ thống khác (vd. NLLB)."""
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def compute_scores(hyps, refs, tgt_lang):
    """BLEU / chrF / chrF++ (sacreBLEU). Đích tiếng Trung tách theo ký tự (tokenize='zh'),
    đích tiếng Việt dùng tokenizer mặc định '13a'."""
    hyps = [normalize_text(h) for h in hyps]
    refs = [normalize_text(r) for r in refs]
    return {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh' if tgt_lang == 'zh' else '13a').score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
    }


def translate_dataset(model, dataset, tok_tgt, max_len, num_samples=0, beam_size=1, batch_size=64):
    """Dịch `num_samples` câu đầu của dataset (0 = toàn bộ), trả về list câu dịch.

    beam_size=1 dùng greedy theo batch (nhanh, dùng trong lúc train);
    beam_size>1 dùng beam search từng câu (chậm hơn, dùng khi đánh giá cuối).
    """
    model.eval()
    n = min(num_samples, len(dataset)) if num_samples else len(dataset)

    hyp_ids = []
    if beam_size <= 1:
        for start in tqdm(range(0, n, batch_size), desc="Decoding", leave=False):
            srcs = [dataset[i][0] for i in range(start, min(start + batch_size, n))]
            src = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
            hyp_ids.extend(greedy_decode(model, src.to(DEVICE), max_len))
    else:
        for i in tqdm(range(n), desc=f"Beam search (k={beam_size})", leave=False):
            src = dataset[i][0].unsqueeze(0).to(DEVICE)
            hyp_ids.append(beam_search(model, src, beam_size, max_len))

    return tok_tgt.decode_batch(hyp_ids)


def evaluate_bleu(model, dataset, tok_tgt, tgt_lang, max_len, num_samples=0, beam_size=1, batch_size=64):
    """Dịch rồi chấm điểm. Trả về (scores, hyps, refs), refs đã chuẩn hoá bằng normalize_text."""
    hyps = translate_dataset(model, dataset, tok_tgt, max_len, num_samples, beam_size, batch_size)
    refs = [normalize_text(r) for r in dataset.tgt_texts[:len(hyps)]]
    return compute_scores(hyps, refs, tgt_lang), hyps, refs


def compute_comet(srcs, hyps, refs, model_name='Unbabel/wmt22-comet-da', batch_size=64):
    """Điểm COMET (0-1) của cả hệ thống = trung bình điểm từng câu.

    Model COMET (~2,3 GB) được tải về cache của HuggingFace ở lần chạy đầu.
    Import trong hàm để train.py không phải nạp thư viện comet.
    """
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    output = model.predict(data, batch_size=batch_size, gpus=1 if DEVICE.type == 'cuda' else 0)
    del model
    torch.cuda.empty_cache()
    return round(output.system_score, 4), output.scores
