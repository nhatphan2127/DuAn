import json
import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import Config
from .evaluation import evaluate_bleu
from .logger import logger
from .tokenizer import PAD_IDX
from .utils import DEVICE, atomic_save, get_amp_dtype

NUM_LOGGED_SAMPLES = 3   # số câu dịch mẫu ghi vào log mỗi epoch (mỗi chiều)


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, eval_sets):
        """eval_sets: [(src_lang, tgt_lang, dataset val của chiều đó, tokenizer đích), ...],
        dùng tính BLEU/chrF riêng từng chiều sau mỗi epoch."""
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_sets = eval_sets

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg.label_smoothing)
        # lr=1.0 vì LambdaLR lấy lr gốc nhân với giá trị lambda -> lr thực tế = đúng công thức Noam
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._noam)

        self.amp_dtype = get_amp_dtype(cfg.use_amp)
        # GradScaler chỉ cần cho fp16; bf16 có dải số đủ rộng
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_dtype == torch.float16)

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.bad_epochs = 0
        self.history = []

    def _noam(self, step):
        cfg = self.cfg
        step = max(step, 1)
        return cfg.lr_factor * cfg.d_model ** -0.5 * min(step ** -0.5, step * cfg.warmup_steps ** -1.5)

    # ------------------------------------------------------------ Một bước
    def _compute_loss(self, src, tgt):
        """Trả về (loss trung bình theo token, số token không phải PAD)."""
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        tgt_input, tgt_expected = tgt[:, :-1], tgt[:, 1:]

        with torch.autocast(device_type=DEVICE.type, dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None):
            logits = self.model(src, tgt_input)
        loss = self.criterion(logits.float().reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
        n_tokens = (tgt_expected != PAD_IDX).sum().item()
        return loss, n_tokens

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------ Một epoch
    def train_epoch(self, desc):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, total_tokens = 0.0, 0
        num_batches = len(self.train_loader)

        bar = tqdm(self.train_loader, desc=desc, leave=False)
        for i, (src, tgt) in enumerate(bar, start=1):
            loss, n_tokens = self._compute_loss(src, tgt)
            self.scaler.scale(loss / self.cfg.accum_steps).backward()
            if i % self.cfg.accum_steps == 0 or i == num_batches:
                self._optimizer_step()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

        return total_loss / total_tokens

    @torch.no_grad()
    def validate(self, desc):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in tqdm(self.val_loader, desc=desc, leave=False):
            loss, n_tokens = self._compute_loss(src, tgt)
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        return total_loss / total_tokens

    # ------------------------------------------------------------ Checkpoint
    def _save_last(self, epoch):
        atomic_save({
            'epoch': epoch,
            'config': self.cfg.to_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'bad_epochs': self.bad_epochs,
        }, self.cfg.last_ckpt)

    def _save_best(self, epoch, val_loss):
        atomic_save({
            'epoch': epoch,
            'val_loss': val_loss,
            'config': self.cfg.to_dict(),
            'src_vocab_sz': self.model.src_vocab_sz,
            'tgt_vocab_sz': self.model.tgt_vocab_sz,
            'model_state_dict': self.model.state_dict(),
        }, self.cfg.best_ckpt)

    def _write_log(self):
        with open(self.cfg.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)

    def resume(self):
        cfg = self.cfg
        if not cfg.last_ckpt.exists():
            logger.log("train_from_scratch", "No checkpoint found. Training from scratch.")
            return

        logger.log("checkpoint_found", f"Found checkpoint at {cfg.last_ckpt}. Loading...",
                   path=cfg.last_ckpt)
        ckpt = torch.load(cfg.last_ckpt, map_location=DEVICE)
        if 'config' not in ckpt:
            raise RuntimeError(
                f"{cfg.last_ckpt} được tạo bởi phiên bản code cũ (không tương thích). "
                f"Hãy xoá thư mục {cfg.checkpoint_dir} và {cfg.log_file} rồi train lại.")

        ckpt_cfg = Config.from_dict(ckpt['config'])
        if ckpt_cfg.direction != cfg.direction:
            raise RuntimeError(
                f"{cfg.last_ckpt} là checkpoint chiều {ckpt_cfg.direction}, nhưng đang train chiều "
                f"{cfg.direction}. Hãy dùng --base_dir khác.")
        changed = cfg.diff(ckpt_cfg)
        if changed:
            logger.log("config_changed",
                       f"⚠️ Config khác với lần train trước: {json.dumps(changed, ensure_ascii=False)}",
                       changes=changed)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.bad_epochs = ckpt['bad_epochs']

        if cfg.log_file.exists():
            with open(cfg.log_file, 'r', encoding='utf-8') as f:
                # Cắt bỏ các epoch đã log nhưng chưa kịp lưu checkpoint
                self.history = json.load(f)[:self.start_epoch]
        logger.log("resume",
                   f"Resuming training from Epoch {self.start_epoch + 1} "
                   f"(Best Val Loss: {self.best_val_loss:.4f})",
                   start_epoch=self.start_epoch + 1, best_val_loss=self.best_val_loss,
                   bad_epochs=self.bad_epochs)

    # ------------------------------------------------------------ Theo dõi BLEU
    def _suffix(self, src_lang, tgt_lang):
        """Mô hình một chiều giữ tên cột 'bleu'/'chrf' như cũ; mô hình hai chiều thêm hậu tố chiều dịch."""
        return f"_{src_lang}-{tgt_lang}" if len(self.eval_sets) > 1 else ""

    def _monitor_bleu(self, record):
        """Greedy trên `bleu_samples` câu val mỗi chiều, ghi điểm vào record.

        Mô hình hai chiều có thêm 'bleu'/'chrf' là trung bình hai chiều (chỉ để theo dõi).
        Trả về vài câu dịch mẫu để ghi log.
        """
        samples = []
        for src_lang, tgt_lang, dataset, tok_tgt in self.eval_sets:
            scores, hyps, refs = evaluate_bleu(self.model, dataset, tok_tgt, tgt_lang, self.cfg.max_len,
                                               num_samples=self.cfg.bleu_samples)
            suffix = self._suffix(src_lang, tgt_lang)
            record.update({k + suffix: v for k, v in scores.items()})
            samples += [{"direction": f"{src_lang}-{tgt_lang}", "src": dataset.src_texts[i],
                         "ref": refs[i], "hyp": hyps[i]}
                        for i in range(min(NUM_LOGGED_SAMPLES, len(hyps)))]
        if len(self.eval_sets) > 1:
            for metric in ('bleu', 'chrf'):
                values = [record[f"{metric}_{s}-{t}"] for s, t, _, _ in self.eval_sets]
                record[metric] = round(sum(values) / len(values), 2)
        return samples

    # ------------------------------------------------------------ Vòng lặp chính
    def fit(self):
        self.resume()
        cfg = self.cfg

        for epoch in range(self.start_epoch, cfg.epochs):
            start_time = time.time()
            tag = f"Epoch {epoch + 1}/{cfg.epochs}"

            self.train_loader.batch_sampler.set_epoch(epoch)
            train_loss = self.train_epoch(f"{tag} [Train]")
            val_loss = self.validate(f"{tag} [Val]")

            record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "lr": self.scheduler.get_last_lr()[0],
            }
            samples = []
            if cfg.bleu_samples > 0:
                samples = self._monitor_bleu(record)
            record["time_seconds"] = round(time.time() - start_time, 2)
            self.history.append(record)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.bad_epochs = 0
                self._save_best(epoch, val_loss)
            else:
                self.bad_epochs += 1
            # Lưu last SAU khi cập nhật best để best_val_loss trong checkpoint luôn đúng
            self._save_last(epoch)
            self._write_log()

            msg = f"Epoch {epoch + 1:02d}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            if cfg.bleu_samples > 0:
                for src_lang, tgt_lang, _, _ in self.eval_sets:
                    suffix = self._suffix(src_lang, tgt_lang)
                    label = f" {src_lang}→{tgt_lang}" if suffix else ""
                    msg += (f" | BLEU{label}: {record['bleu' + suffix]:.2f}"
                            f" | chrF{label}: {record['chrf' + suffix]:.2f}")
            msg += f" | Time: {record['time_seconds']:.1f}s"
            # Câu mẫu đầu tiên của mỗi chiều
            for sample in samples[::NUM_LOGGED_SAMPLES]:
                msg += (f"\n   SRC [{sample['direction']}]: {sample['src']}"
                        f"\n   REF: {sample['ref']}"
                        f"\n   HYP: {sample['hyp']}")

            logger.log("epoch_end", msg, **record, best=improved, bad_epochs=self.bad_epochs,
                       samples=samples)
            if improved:
                logger.log("best_saved", f"⭐ New Best Model saved at Epoch {epoch + 1}",
                           epoch=epoch + 1, val_loss=val_loss, path=cfg.best_ckpt)

            if cfg.patience and self.bad_epochs >= cfg.patience:
                logger.log("early_stopping",
                           f"Early stopping: val loss không cải thiện sau {cfg.patience} epoch.",
                           epoch=epoch + 1, best_val_loss=self.best_val_loss)
                break
        else:
            logger.log("train_finished", epochs=cfg.epochs, best_val_loss=self.best_val_loss)
