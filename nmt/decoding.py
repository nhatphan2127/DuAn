"""Sinh câu dịch: greedy (theo batch, dùng khi tính BLEU) và beam search (từng câu)."""
import torch

from .tokenizer import BOS_IDX, EOS_IDX, PAD_IDX


def _strip_special(ids):
    """Bỏ [BOS] ở đầu, cắt tại [EOS]."""
    out = []
    for token in ids[1:]:
        if token == EOS_IDX:
            break
        out.append(token)
    return out


@torch.no_grad()
def greedy_decode(model, src, max_len):
    """src: (B, S) -> list B danh sách id token (đã bỏ token đặc biệt)."""
    memory, src_mask = model.encode(src)
    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        logits = model.decode(ys, memory, src_mask)[:, -1]
        next_token = logits.argmax(dim=-1).masked_fill(finished, PAD_IDX)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == EOS_IDX
        if finished.all():
            break

    return [_strip_special(seq) for seq in ys.tolist()]


@torch.no_grad()
def beam_search(model, src, beam_size, max_len, length_penalty=0.6, length_penalty_style='gnmt',
                max_new_tokens=None):
    """src: (1, S) -> danh sách id token của câu dịch tốt nhất.

    length_penalty_style='gnmt' (mặc định, dùng cho các kết quả đã báo cáo): điểm / ((5 + |y|) / 6) ** α.
    length_penalty_style='hf': điểm / (số token sinh ra) ** α, giống `model.generate()` của Hugging Face,
    dùng khi cần giải mã cùng cấu hình với NLLB (beam 4, α = 1,0).
    max_new_tokens: số token sinh tối đa (kể cả [EOS]); None = max_len - 1 như trước.
    """
    if length_penalty_style not in ('gnmt', 'hf'):
        raise ValueError("length_penalty_style phải là 'gnmt' hoặc 'hf'")
    memory, src_mask = model.encode(src)
    beams = [([BOS_IDX], 0.0)]   # (chuỗi token, tổng log-prob)
    finished = []

    for _ in range(max_len - 1 if max_new_tokens is None else max_new_tokens):
        k = len(beams)
        ys = torch.tensor([seq for seq, _ in beams], dtype=torch.long, device=src.device)
        logits = model.decode(ys, memory.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1))[:, -1]
        log_probs = logits.float().log_softmax(dim=-1)

        prev_scores = torch.tensor([score for _, score in beams], device=src.device)
        scores = (prev_scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_idx = scores.topk(beam_size)

        vocab_size = log_probs.size(-1)
        prev_beams, beams = beams, []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            beam_id, token = divmod(idx, vocab_size)
            candidate = (prev_beams[beam_id][0] + [token], score)
            (finished if token == EOS_IDX else beams).append(candidate)

        if not beams or len(finished) >= beam_size:
            break

    finished.extend(beams)

    # Length penalty để không thiên vị câu ngắn
    def normalized(candidate):
        seq, score = candidate
        if length_penalty_style == 'hf':
            return score / ((len(seq) - 1) ** length_penalty)   # len(seq) - 1: bỏ [BOS], tính cả [EOS]
        return score / (((5 + len(seq)) / 6) ** length_penalty)

    return _strip_special(max(finished, key=normalized)[0])


def translate_sentence(sentence, model, tok_src, tok_tgt, max_len=128, beam_size=5, src_tag=None):
    """src_tag: id token <2xx> chọn ngôn ngữ đích (mô hình hai chiều), None = mô hình một chiều."""
    model.eval()
    device = next(model.parameters()).device
    src_start = BOS_IDX if src_tag is None else src_tag
    src_ids = [src_start] + tok_src.encode(sentence).ids[:max_len - 2] + [EOS_IDX]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    if beam_size > 1:
        out_ids = beam_search(model, src, beam_size, max_len)
    else:
        out_ids = greedy_decode(model, src, max_len)[0]
    return tok_tgt.decode(out_ids)
