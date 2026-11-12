"""Dịch câu Việt <-> Trung bằng best checkpoint.

Trên Colab:
    !python scripts/translate.py "Tôi thích học trí tuệ nhân tạo." "Hôm nay trời đẹp."
    !python scripts/translate.py --direction zh-vi "我喜欢学习人工智能。"
    !python scripts/translate.py --direction both "Hôm nay trời đẹp." "我喜欢学习人工智能。"   # tự nhận ngôn ngữ

Hoặc trong một cell notebook (không cần load lại model mỗi lần):
    from translate import Translator
    t = Translator(direction='both')
    t("Tôi thích học trí tuệ nhân tạo.")          # -> tiếng Trung
    t("我喜欢学习人工智能。")                       # -> tiếng Việt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được nmt/ khi chạy scripts/*.py

import argparse
import re

from nmt.config import DIRECTIONS, Config
from nmt.decoding import translate_sentence
from nmt.model import load_model
from nmt.tokenizer import lang_tag_id, load_tokenizers
from nmt.utils import DEVICE

HAN_CHARS = re.compile(r'[㐀-䶿一-鿿豈-﫿]')


def detect_lang(sentence):
    """Có chữ Hán -> 'zh', ngược lại -> 'vi'."""
    return 'zh' if HAN_CHARS.search(sentence) else 'vi'


class Translator:
    def __init__(self, direction='vi-zh', base_dir='', beam_size=Config.beam_size):
        """direction chỉ dùng để chọn base_dir mặc định; chiều dịch thật lấy từ checkpoint."""
        paths = Config(direction=direction, base_dir=base_dir)
        self.model, self.cfg = load_model(paths.best_ckpt, DEVICE)
        self.cfg.base_dir = paths.base_dir
        self.tokenizers = load_tokenizers(self.cfg, train_if_missing=False)
        self.beam_size = beam_size

    def __call__(self, sentence, src_lang=None):
        """src_lang = None: mô hình một chiều dùng ngôn ngữ nguồn của nó, mô hình hai chiều tự nhận diện."""
        pairs = dict(self.cfg.lang_pairs)   # {nguồn: đích}
        if src_lang is None:
            src_lang = detect_lang(sentence) if self.cfg.bidirectional else next(iter(pairs))
        if src_lang not in pairs:
            raise ValueError(f"Mô hình {self.cfg.direction} không dịch được từ '{src_lang}'")
        tgt_lang = pairs[src_lang]
        return translate_sentence(sentence, self.model, self.tokenizers[src_lang], self.tokenizers[tgt_lang],
                                  self.cfg.max_len, self.beam_size,
                                  src_tag=lang_tag_id(self.cfg, self.tokenizers, tgt_lang))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sentences', nargs='+')
    parser.add_argument('--direction', default='vi-zh', choices=DIRECTIONS)
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--beam_size', type=int, default=Config.beam_size)
    args = parser.parse_args()

    translator = Translator(args.direction, args.base_dir, args.beam_size)
    pairs = dict(translator.cfg.lang_pairs)
    for sentence in args.sentences:
        src_lang = detect_lang(sentence) if translator.cfg.bidirectional else next(iter(pairs))
        print(f"{src_lang.upper()}: {sentence}\n{pairs[src_lang].upper()}: {translator(sentence, src_lang)}\n")


if __name__ == '__main__':
    main()
