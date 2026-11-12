"""Cấu hình toàn bộ dự án.

Mọi field đều override được qua CLI, ví dụ:
    !python train.py --epochs 30 --d_model 256 --max_tokens 8192
    !python train.py --direction zh-vi
"""
import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def _str2bool(s):
    return s.lower() in ('1', 'true', 'yes', 'y')


DIRECTIONS = ('vi-zh', 'zh-vi', 'both')
# Thư mục mặc định theo chiều dịch, để không vô tình train tiếp checkpoint của chiều khác
DEFAULT_BASE_DIRS = {
    'vi-zh': './vi_zh_translation',
    'zh-vi': './zh_vi_translation',
    'both': './vi_zh_bidir',
}


@dataclass
class Config:
    # Chiều dịch: 'vi-zh' / 'zh-vi' (một chiều) hoặc 'both' (một mô hình dịch cả hai chiều,
    # tokenizer chung, thêm token <2vi>/<2zh> vào đầu câu nguồn để chọn ngôn ngữ đích)
    direction: str = 'vi-zh'

    # Đường dẫn (tương đối so với thư mục chạy lệnh; trên Colab truyền --base_dir/--data_dir trỏ vào Drive)
    base_dir: str = ''            # '' = DEFAULT_BASE_DIRS[direction]
    data_dir: str = './data'

    # Tokenizer / dữ liệu
    vocab_size: int = 0           # 0 = 16000 cho mỗi ngôn ngữ khi dịch một chiều, 32000 cho tokenizer chung
    max_len: int = 128            # độ dài tối đa mỗi câu, tính cả [BOS] và [EOS]

    # Mô hình
    d_model: int = 512
    n_heads: int = 8
    num_enc: int = 6
    num_dec: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Huấn luyện
    epochs: int = 20
    max_tokens: int = 4096        # số token (kể cả padding) tối đa trong một batch
    accum_steps: int = 1          # gradient accumulation: batch hiệu dụng = max_tokens * accum_steps
    lr_factor: float = 1.0        # hệ số nhân cho lịch Noam (đỉnh LR ≈ 7e-4 với d_model=512, warmup=4000)
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    clip_grad_norm: float = 1.0
    patience: int = 5             # early stopping theo val loss, 0 = tắt
    use_amp: bool = True
    seed: int = 42
    num_workers: int = 2

    # Đánh giá
    bleu_samples: int = 500       # số câu val (mỗi chiều) dùng tính BLEU sau mỗi epoch, 0 = bỏ qua
    beam_size: int = 5

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction phải là một trong {DIRECTIONS}, nhận được {self.direction!r}")
        if not self.base_dir:
            self.base_dir = DEFAULT_BASE_DIRS[self.direction]
        if not self.vocab_size:
            self.vocab_size = 32000 if self.bidirectional else 16000

    @property
    def bidirectional(self) -> bool:
        return self.direction == 'both'

    @property
    def lang_pairs(self):
        """Các chiều dịch mô hình học: [(nguồn, đích), ...]."""
        if self.bidirectional:
            return [('vi', 'zh'), ('zh', 'vi')]
        return [tuple(self.direction.split('-'))]

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.base_dir) / 'checkpoints'

    @property
    def last_ckpt(self) -> Path:
        return self.checkpoint_dir / 'last_checkpoint.pt'

    @property
    def best_ckpt(self) -> Path:
        return self.checkpoint_dir / 'best_checkpoint.pt'

    @property
    def log_file(self) -> Path:
        return Path(self.base_dir) / 'training_metrics.json'

    @property
    def run_log_file(self) -> Path:
        return Path(self.base_dir) / 'train_log.jsonl'

    @property
    def config_file(self) -> Path:
        return Path(self.base_dir) / 'config.json'

    def tok_path(self, lang) -> Path:
        """Đường dẫn tokenizer của một ngôn ngữ ('vi' / 'zh') hoặc tokenizer chung ('joint')."""
        name = {
            'vi': 'tokenizer_vi.json',          # byte-level BPE
            'zh': 'tokenizer_zh.model',         # SentencePiece
            'joint': 'tokenizer_joint.model',   # SentencePiece chung cho cả hai ngôn ngữ
        }[lang]
        return Path(self.base_dir) / name

    def split_file(self, split, lang):
        return Path(self.data_dir) / f'{split}.{lang}'

    def split_paths(self, split):
        """Trả về (file .vi, file .zh) của một split: train / val / test."""
        return self.split_file(split, 'vi'), self.split_file(split, 'zh')

    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_json(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def diff(self, other):
        """Các field khác nhau giữa hai config: {field: [giá trị của other, giá trị của self]}."""
        mine, theirs = self.to_dict(), other.to_dict()
        return {k: [theirs[k], v] for k, v in mine.items() if theirs.get(k) != v}

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_args(cls, argv=None):
        parser = argparse.ArgumentParser()
        for f in fields(cls):
            arg_type = _str2bool if f.type is bool else f.type
            choices = DIRECTIONS if f.name == 'direction' else None
            parser.add_argument(f'--{f.name}', type=arg_type, default=f.default, choices=choices)
        return cls.from_dict(vars(parser.parse_args(argv)))
