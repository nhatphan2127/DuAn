"""Transformer encoder-decoder (Pre-LN) cài đặt từ đầu."""
import math

import torch
import torch.nn as nn

from .config import Config
from .tokenizer import PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        return x.view(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        # Chia scale trước khi nhân để tránh tràn số khi chạy fp16
        q = self._split_heads(self.w_q(q)) / math.sqrt(self.d_k)
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))

        scores = q @ k.transpose(-2, -1)
        if mask is not None:
            # Không dùng -1e9 vì sẽ tràn số ở fp16 (min ≈ -65504)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(q.size(0), -1, self.d_model)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_sz, tgt_vocab_sz, d_model=512, n_heads=8,
                 num_enc=6, num_dec=6, d_ff=2048, dropout=0.1, pad_idx=PAD_IDX, tie_embeddings=False):
        super().__init__()
        self.src_vocab_sz = src_vocab_sz
        self.tgt_vocab_sz = tgt_vocab_sz
        self.pad_idx = pad_idx
        self.scale = math.sqrt(d_model)

        self.src_embed = nn.Embedding(src_vocab_sz, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_sz, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_enc)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_dec)])

        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_sz)

        if tie_embeddings:
            # Từ vựng chung (mô hình hai chiều): embedding nguồn, embedding đích và lớp output
            # dùng chung một ma trận (Press & Wolf, 2017; Vaswani et al., 2017)
            assert src_vocab_sz == tgt_vocab_sz, "tie_embeddings cần từ vựng nguồn và đích giống nhau"
            self.tgt_embed.weight = self.src_embed.weight
            self.fc_out.weight = self.src_embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # (B, 1, 1, S): che các vị trí PAD của source
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # (B, 1, T, T): che PAD + che các token tương lai
        seq_len = tgt.size(1)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device, dtype=torch.bool))
        return pad_mask & causal_mask

    def _embed(self, embedding, tokens):
        return self.dropout(self.pos_enc(embedding(tokens) * self.scale))

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        x = self._embed(self.src_embed, src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x), src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        x = self._embed(self.tgt_embed, tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.fc_out(self.dec_norm(x))

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)


def build_model(cfg, src_vocab_sz, tgt_vocab_sz, device):
    return Seq2SeqTransformer(
        src_vocab_sz=src_vocab_sz,
        tgt_vocab_sz=tgt_vocab_sz,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_enc=cfg.num_enc,
        num_dec=cfg.num_dec,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        tie_embeddings=cfg.bidirectional,
    ).to(device)


def load_model(ckpt_path, device):
    """Dựng lại model từ best checkpoint (kiến trúc lấy từ config lưu trong checkpoint)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config.from_dict(ckpt['config'])
    model = build_model(cfg, ckpt['src_vocab_sz'], ckpt['tgt_vocab_sz'], device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg
