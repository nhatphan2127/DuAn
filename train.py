"""Huấn luyện mô hình dịch Việt <-> Trung.

Trên Colab:
    !python train.py                                   # Việt -> Trung, lưu vào ./vi_zh_translation
    !python train.py --direction zh-vi                 # Trung -> Việt, lưu vào ./zh_vi_translation
    !python train.py --direction both                  # một mô hình hai chiều, lưu vào ./vi_zh_bidir
    !python train.py --epochs 30 --max_tokens 8192     # override cấu hình

Kết quả ghi lại trong base_dir:
    config.json            config của lần chạy gần nhất
    train_log.jsonl        toàn bộ log (mỗi dòng một sự kiện JSON, gồm cả config của mọi lần chạy)
    training_metrics.json  metric theo từng epoch
"""
import sys

import torch

from nmt.config import Config
from nmt.data import build_dataloader, build_dataset, log_corpus_stats
from nmt.decoding import translate_sentence
from nmt.logger import logger
from nmt.model import build_model, load_model
from nmt.tokenizer import lang_tag_id, load_tokenizers
from nmt.trainer import Trainer
from nmt.utils import DEVICE, set_seed

SAMPLE_SENTENCES = {'vi': "Tôi thích học trí tuệ nhân tạo.", 'zh': "我喜欢学习人工智能。"}


def main():
    cfg = Config.from_args()
    set_seed(cfg.seed)
    # Kiểm tra trước khi ghi config.json, để chạy nhầm chiều không ghi đè config của lần train trước
    if cfg.config_file.exists():
        prev_direction = Config.load_json(cfg.config_file).direction
        if prev_direction != cfg.direction:
            raise RuntimeError(f"{cfg.base_dir} đang chứa mô hình chiều {prev_direction}, nhưng đang train "
                               f"chiều {cfg.direction}. Hãy dùng --base_dir khác.")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.set_file(cfg.run_log_file)
    cfg.save_json(cfg.config_file)
    logger.log(
        "run_start", f"Using device: {DEVICE}",
        command=' '.join(sys.argv),
        device=DEVICE,
        gpu=torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
        torch_version=torch.__version__,
        config=cfg.to_dict(),
    )

    log_corpus_stats(cfg)
    tokenizers = load_tokenizers(cfg)
    train_loader = build_dataloader(cfg, 'train', tokenizers, shuffle=True)
    val_loader = build_dataloader(cfg, 'val', tokenizers, shuffle=False)
    eval_sets = [(src, tgt, build_dataset(cfg, 'val', tokenizers, src, tgt), tokenizers[tgt])
                 for src, tgt in cfg.lang_pairs]

    # Mô hình hai chiều: nguồn và đích dùng chung tokenizer nên hai vocab size bằng nhau
    first_src, first_tgt = cfg.lang_pairs[0]
    src_vocab_sz = tokenizers[first_src].get_vocab_size()
    tgt_vocab_sz = tokenizers[first_tgt].get_vocab_size()
    model = build_model(cfg, src_vocab_sz, tgt_vocab_sz, DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_built", f"Total Trainable Parameters: {total_params:,}",
               total_params=total_params, direction=cfg.direction,
               src_vocab_sz=src_vocab_sz, tgt_vocab_sz=tgt_vocab_sz,
               train_batches=len(train_loader), val_batches=len(val_loader))

    Trainer(cfg, model, train_loader, val_loader, eval_sets).fit()

    # Dịch thử bằng best model (không phải model ở epoch cuối)
    best_model, _ = load_model(cfg.best_ckpt, DEVICE)
    for src_lang, tgt_lang in cfg.lang_pairs:
        sample = SAMPLE_SENTENCES[src_lang]
        hyp = translate_sentence(sample, best_model, tokenizers[src_lang], tokenizers[tgt_lang],
                                 cfg.max_len, cfg.beam_size, src_tag=lang_tag_id(cfg, tokenizers, tgt_lang))
        logger.log("sample_translation", f"[{src_lang}→{tgt_lang}] Input: {sample}\nTranslation: {hyp}",
                   direction=f"{src_lang}-{tgt_lang}", src=sample, hyp=hyp)


if __name__ == '__main__':
    main()
