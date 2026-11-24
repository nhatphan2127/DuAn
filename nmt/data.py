import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .logger import logger
from .tokenizer import BOS_IDX, EOS_IDX, PAD_IDX, lang_tag_id
from .utils import DEVICE


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


# ---------------------------------------------------------------- Thống kê
def analyze_corpus(vi_path, zh_path):
    vi_lines = read_lines(vi_path)
    zh_lines = read_lines(zh_path)
    assert len(vi_lines) == len(zh_lines), "Source and Target line counts mismatch!"

    num_samples = len(vi_lines)
    vi_word_lens = [len(s.split()) for s in vi_lines]
    zh_char_lens = [len(s) for s in zh_lines]

    return {
        "Total Pairs": num_samples,
        "Empty Pairs": sum(1 for v, z in zip(vi_lines, zh_lines) if not v or not z),
        "VI - Min Words": min(vi_word_lens),
        "VI - Max Words": max(vi_word_lens),
        "VI - Avg Words": round(sum(vi_word_lens) / num_samples, 2),
        "ZH - Min Chars": min(zh_char_lens),
        "ZH - Max Chars": max(zh_char_lens),
        "ZH - Avg Chars": round(sum(zh_char_lens) / num_samples, 2),
    }


def log_corpus_stats(cfg):
    for split in ('train', 'val'):
        stats = analyze_corpus(*cfg.split_paths(split))
        logger.log("corpus_stats",
                   f"--- {split.capitalize()} Dataset Stats ---\n{json.dumps(stats, indent=4)}",
                   split=split, stats=stats)


# ---------------------------------------------------------------- Dataset
class TranslationDataset(Dataset):
    """Tokenize toàn bộ corpus một lần lúc khởi tạo và bỏ các cặp câu rỗng.

    src_tag: id token <2xx> đặt thay [BOS] ở đầu câu nguồn (mô hình hai chiều), None = [BOS].
    """

    def __init__(self, src_file, tgt_file, tok_src, tok_tgt, max_len=128, src_tag=None):
        src_lines, tgt_lines = read_lines(src_file), read_lines(tgt_file)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"{src_file} có {len(src_lines)} dòng nhưng {tgt_file} có {len(tgt_lines)} dòng")

        pairs = [(s, t) for s, t in zip(src_lines, tgt_lines) if s and t]
        n_dropped = len(src_lines) - len(pairs)
        if n_dropped:
            logger.log("empty_pairs_dropped", f"{src_file}: bỏ {n_dropped} cặp câu rỗng",
                       file=src_file, dropped=n_dropped)

        self.src_texts = [s for s, _ in pairs]
        self.tgt_texts = [t for _, t in pairs]
        self.max_len = max_len
        src_start = BOS_IDX if src_tag is None else src_tag
        self.src = [self._wrap(enc.ids, src_start) for enc in tok_src.encode_batch(self.src_texts)]
        self.tgt = [self._wrap(enc.ids) for enc in tok_tgt.encode_batch(self.tgt_texts)]

    def _wrap(self, ids, start=BOS_IDX):
        # int32 numpy tiết kiệm RAM hơn nhiều so với list Python
        return np.asarray([start] + ids[:self.max_len - 2] + [EOS_IDX], dtype=np.int32)

    def extend(self, other):
        """Nối thêm một dataset khác (dùng để gộp hai chiều dịch thành một tập train)."""
        self.src_texts += other.src_texts
        self.tgt_texts += other.tgt_texts
        self.src += other.src
        self.tgt += other.tgt
        return self

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.from_numpy(self.src[idx]).long(), torch.from_numpy(self.tgt[idx]).long()

    def lengths(self):
        return [max(len(s), len(t)) for s, t in zip(self.src, self.tgt)]


def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return src, tgt


class TokenBatchSampler(Sampler):
    """Gom các câu có độ dài gần nhau vào cùng batch, giới hạn theo tổng số token (kể cả padding).

    So với batch_size cố định + shuffle hoàn toàn ngẫu nhiên, cách này giảm padding đáng kể.
    Gọi set_epoch(epoch) trước mỗi epoch để có thứ tự batch mới.
    """

    def __init__(self, lengths, max_tokens, shuffle, seed=0, pool_size=100_000):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.pool_size = pool_size
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        # Chỉ sắp xếp trong từng pool để vẫn giữ tính ngẫu nhiên giữa các epoch
        pool_size = self.pool_size if self.shuffle else max(len(indices), 1)
        batches = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start:start + pool_size], key=self.lengths.__getitem__)
            batches.extend(self._chunk(pool))

        if self.shuffle:
            rng.shuffle(batches)
        self.batches = batches

    def _chunk(self, sorted_indices):
        batch, batch_max = [], 0
        for idx in sorted_indices:
            new_max = max(batch_max, self.lengths[idx])
            if batch and new_max * (len(batch) + 1) > self.max_tokens:
                yield batch
                batch, new_max = [], self.lengths[idx]
            batch.append(idx)
            batch_max = new_max
        if batch:
            yield batch

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def build_dataset(cfg, split, tokenizers, src_lang, tgt_lang):
    """Dataset của một chiều dịch."""
    return TranslationDataset(
        cfg.split_file(split, src_lang), cfg.split_file(split, tgt_lang),
        tokenizers[src_lang], tokenizers[tgt_lang], cfg.max_len,
        src_tag=lang_tag_id(cfg, tokenizers, tgt_lang),
    )


def build_dataloader(cfg, split, tokenizers, shuffle):
    """DataLoader gộp mọi chiều dịch của mô hình (mô hình hai chiều: mỗi cặp câu thành hai mẫu)."""
    dataset = None
    for src_lang, tgt_lang in cfg.lang_pairs:
        part = build_dataset(cfg, split, tokenizers, src_lang, tgt_lang)
        dataset = part if dataset is None else dataset.extend(part)
    sampler = TokenBatchSampler(dataset.lengths(), cfg.max_tokens, shuffle=shuffle, seed=cfg.seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=DEVICE.type == 'cuda',
    )
