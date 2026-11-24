from collections import namedtuple
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from .logger import logger

# Thứ tự trong SPECIAL_TOKENS quyết định id: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = range(len(SPECIAL_TOKENS))

# Token chọn ngôn ngữ đích của mô hình hai chiều, đặt thay [BOS] ở đầu câu nguồn
LANG_TAGS = {'vi': '<2vi>', 'zh': '<2zh>'}


# ---------------------------------------------------------------- Byte-level BPE (tiếng Việt)
def train_bpe_tokenizer(corpus_path, vocab_size, save_path):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    tokenizer.save(str(save_path))
    return tokenizer


# ---------------------------------------------------------------- SentencePiece (tiếng Trung)
Encoding = namedtuple('Encoding', ['ids'])


class SentencePieceTokenizer:
    """Bọc SentencePiece theo đúng phần interface của `tokenizers.Tokenizer` mà project dùng
    (encode, encode_batch, decode, decode_batch, normalizer.normalize_str, ...)."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.normalizer = self

    def normalize_str(self, text):
        # normalize() trả về dạng nội bộ của SentencePiece, khoảng trắng là '▁'
        return self.sp.normalize(text).replace('\u2581', ' ').strip()

    def encode(self, text):
        return Encoding(self.sp.encode(text))

    def encode_batch(self, texts):
        return [Encoding(ids) for ids in self.sp.encode(texts)]

    def decode(self, ids):
        return self.sp.decode(ids)

    def decode_batch(self, batch_ids):
        return self.sp.decode(batch_ids)

    def token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def get_vocab_size(self):
        return self.sp.get_piece_size()


def train_sentencepiece_tokenizer(corpus_path, vocab_size, save_path):
    """Đơn vị gốc là ký tự Unicode: các chữ Hán phủ 99.95% corpus luôn có sẵn trong vocab,
    chỉ chữ hiếm hơn mới bị tách thành byte UTF-8 (byte_fallback) thay vì thành [UNK]."""
    return _train_sentencepiece(
        [corpus_path], vocab_size, save_path,
        add_dummy_prefix=False,     # tiếng Trung gần như không có dấu cách -> không thêm '▁' vào đầu câu
    )


def train_joint_tokenizer(corpus_paths, vocab_size, save_path):
    """SentencePiece chung cho tiếng Việt và tiếng Trung (mô hình hai chiều).

    Giữ add_dummy_prefix mặc định để từ đầu câu tiếng Việt được tách giống từ giữa câu.
    <2vi>/<2zh> là user_defined_symbols nên luôn là một token nguyên vẹn (id 4, 5).
    """
    return _train_sentencepiece(
        corpus_paths, vocab_size, save_path,
        user_defined_symbols=list(LANG_TAGS.values()),
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )


def _train_sentencepiece(corpus_paths, vocab_size, save_path, **kwargs):
    save_path = Path(save_path)
    spm.SentencePieceTrainer.train(
        input=','.join(str(p) for p in corpus_paths),
        model_prefix=str(save_path.with_suffix('')),
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name='nmt_nfkc',
        pad_id=PAD_IDX, unk_id=UNK_IDX, bos_id=BOS_IDX, eos_id=EOS_IDX,
        pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
        num_threads=8,
        minloglevel=2,
        **kwargs,
    )
    return SentencePieceTokenizer(save_path)


# ---------------------------------------------------------------- Load
def load_or_train_tokenizer(train_fn, corpus, vocab_size, save_path, train_if_missing=True):
    """Dùng lại tokenizer đã lưu để id token luôn khớp với checkpoint khi train tiếp.

    File .model -> SentencePiece, file .json -> byte-level BPE của thư viện `tokenizers`.
    """
    save_path = Path(save_path)
    is_spm = save_path.suffix == '.model'
    if save_path.exists():
        logger.log("tokenizer_loaded", f"Loading tokenizer from {save_path}", path=save_path)
        tokenizer = SentencePieceTokenizer(save_path) if is_spm else Tokenizer.from_file(str(save_path))
    elif train_if_missing:
        logger.log("tokenizer_training", f"Training tokenizer on {corpus}...", corpus=corpus)
        tokenizer = train_fn(corpus, vocab_size, save_path)
        logger.log("tokenizer_trained", path=save_path, vocab_size=tokenizer.get_vocab_size())
    else:
        raise FileNotFoundError(f"Không tìm thấy tokenizer: {save_path}")

    if not is_spm:
        # File tokenizer từ bản cũ không có decoder -> gắn lại (không ảnh hưởng tới id)
        tokenizer.decoder = decoders.ByteLevel()

    for idx, token in enumerate(SPECIAL_TOKENS):
        assert tokenizer.token_to_id(token) == idx, f"{save_path}: {token} phải có id {idx}"
    return tokenizer


def load_tokenizers(cfg, train_if_missing=True):
    """Trả về {'vi': tokenizer, 'zh': tokenizer}. Mô hình hai chiều dùng một tokenizer chung cho cả hai."""
    train_vi, train_zh = cfg.split_paths('train')
    if cfg.bidirectional:
        tok = load_or_train_tokenizer(train_joint_tokenizer, [train_vi, train_zh], cfg.vocab_size,
                                      cfg.tok_path('joint'), train_if_missing)
        for tag in LANG_TAGS.values():
            assert tok.token_to_id(tag) != UNK_IDX, f"Tokenizer chung thiếu token {tag}"
        return {'vi': tok, 'zh': tok}
    return {
        'vi': load_or_train_tokenizer(train_bpe_tokenizer, train_vi, cfg.vocab_size,
                                      cfg.tok_path('vi'), train_if_missing),
        'zh': load_or_train_tokenizer(train_sentencepiece_tokenizer, train_zh, cfg.vocab_size,
                                      cfg.tok_path('zh'), train_if_missing),
    }


def lang_tag_id(cfg, tokenizers, tgt_lang):
    """Id token <2xx> đặt ở đầu câu nguồn, None nếu là mô hình một chiều (đầu câu là [BOS])."""
    return tokenizers[tgt_lang].token_to_id(LANG_TAGS[tgt_lang]) if cfg.bidirectional else None
