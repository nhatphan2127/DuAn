"""Vẽ biểu đồ quá trình huấn luyện từ training_metrics.json và train_log.jsonl.

    uv run --with matplotlib python scripts/plot_metrics.py
    !python scripts/plot_metrics.py                       # Colab có sẵn matplotlib

Ghi ra <base_dir>/figures/: loss.png, bleu_chrf.png, lr_schedule.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được nmt/ khi chạy scripts/*.py

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from nmt.config import Config

# Màu và mực theo bảng màu đã kiểm tra độ phân biệt (kể cả người mù màu)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834']
LANG_NAMES = {'vi': 'Việt', 'zh': 'Hoa'}


def vn(x, decimals=0):
    """Định dạng số kiểu Việt: 4.000 và 3,47."""
    return f'{x:,.{decimals}f}'.translate(str.maketrans(',.', '.,'))


def style_axes(ax, title, ylabel, xlabel='Epoch', y_decimals=0):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, y_decimals)))


def plot_line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, solid_joinstyle='round', solid_capstyle='round',
            label=label, zorder=3)
    # Nhãn trực tiếp ở điểm cuối, chữ dùng màu mực, chấm tròn mang màu series
    ax.scatter([x[-1]], [y[-1]], s=60, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax.annotate(f'{label}  {vn(y[-1], 2)}', (x[-1], y[-1]), xytext=(10, 0),
                textcoords='offset points', va='center', color=INK, fontsize=9)


def mark_best(ax, epoch, at_top=True):
    ax.axvline(epoch, color=AXIS, linewidth=1, zorder=1)
    ax.annotate(f'Best (epoch {epoch})', (epoch, 1 if at_top else 0), xycoords=('data', 'axes fraction'),
                xytext=(-6, -4 if at_top else 6), textcoords='offset points', ha='right',
                va='top' if at_top else 'bottom', color=INK_2, fontsize=9)


def finish(ax, out_path):
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=INK_2, fontsize=9,
              ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(ax.figure)
    print(f'Đã lưu {out_path}')


def steps_per_epoch(run_log_file):
    """Số batch train mỗi epoch, lấy từ sự kiện model_built gần nhất trong log."""
    batches = None
    if run_log_file.exists():
        with open(run_log_file, encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record['event'] == 'model_built':
                    batches = record['train_batches']
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=Config().base_dir)
    args = parser.parse_args()

    cfg = Config.load_json(Path(args.base_dir) / 'config.json')
    cfg.base_dir = args.base_dir
    with open(cfg.log_file, encoding='utf-8') as f:
        history = json.load(f)

    out_dir = Path(cfg.base_dir) / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [r['epoch'] for r in history]
    best_epoch = min(history, key=lambda r: r['val_loss'])['epoch']

    # 1. Train loss / val loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_line(ax, epochs, [r['train_loss'] for r in history], SERIES[0], 'Train loss')
    plot_line(ax, epochs, [r['val_loss'] for r in history], SERIES[1], 'Val loss')
    mark_best(ax, best_epoch)
    style_axes(ax, 'Loss theo epoch (cross-entropy, label smoothing 0,1)', 'Loss', y_decimals=1)
    ax.set_xticks(epochs)
    ax.set_xlim(0.5, epochs[-1] + 3)
    finish(ax, out_dir / 'loss.png')

    # 2. BLEU / chrF (greedy, 500 câu val). Mô hình hai chiều: mỗi độ đo một biểu đồ, mỗi chiều một đường
    pair_keys = [f'{s}-{t}' for s, t in cfg.lang_pairs]
    if len(pair_keys) > 1 and f'bleu_{pair_keys[0]}' in history[0]:
        for metric, name in (('bleu', 'BLEU'), ('chrf', 'chrF')):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for color, key in zip(SERIES, pair_keys):
                plot_line(ax, epochs, [r[f'{metric}_{key}'] for r in history], color,
                          ' – '.join(LANG_NAMES[lang] for lang in key.split('-')))
            mark_best(ax, best_epoch, at_top=False)
            style_axes(ax, f'{name} theo chiều dịch trên {cfg.bleu_samples} câu val mỗi chiều (greedy)',
                       'Điểm (0–100)')
            ax.set_xticks(epochs)
            ax.set_xlim(0.5, epochs[-1] + 3)
            finish(ax, out_dir / f'{metric}.png')
    elif 'bleu' in history[0]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_line(ax, epochs, [r['bleu'] for r in history], SERIES[0], 'BLEU')
        plot_line(ax, epochs, [r['chrf'] for r in history], SERIES[1], 'chrF')
        mark_best(ax, best_epoch, at_top=False)
        style_axes(ax, f'BLEU và chrF trên {cfg.bleu_samples} câu val (greedy)', 'Điểm (0–100)')
        ax.set_xticks(epochs)
        ax.set_xlim(0.5, epochs[-1] + 3)
        finish(ax, out_dir / 'bleu_chrf.png')

    # 3. Lịch tốc độ học Noam theo bước
    per_epoch = steps_per_epoch(cfg.run_log_file)
    if per_epoch:
        total = per_epoch * epochs[-1]
        steps = range(1, total + 1, 50)
        noam = lambda s: cfg.lr_factor * cfg.d_model ** -0.5 * min(s ** -0.5, s * cfg.warmup_steps ** -1.5)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(list(steps), [noam(s) * 1e4 for s in steps], color=SERIES[0], linewidth=2, zorder=3)
        peak = noam(cfg.warmup_steps) * 1e4
        ax.scatter([cfg.warmup_steps], [peak], s=60, color=SERIES[0], edgecolors=SURFACE, linewidths=2,
                   zorder=4)
        ax.annotate(f'Đỉnh {vn(peak, 2)}·10⁻⁴ tại bước {vn(cfg.warmup_steps)}',
                    (cfg.warmup_steps, peak), xytext=(10, 0), textcoords='offset points', va='center',
                    color=INK, fontsize=9)
        style_axes(ax, f'Lịch tốc độ học Noam (d_model = {cfg.d_model}, warmup = {vn(cfg.warmup_steps)})',
                   'Tốc độ học (×10⁻⁴)', xlabel=f'Bước cập nhật ({vn(per_epoch)} bước / epoch)')
        epoch_ticks = range(0, total + 1, per_epoch * 4)
        ax.set_xticks(list(epoch_ticks), [f'{s // 1000}K' for s in epoch_ticks])
        ax.set_xlim(0, total)
        ax.set_ylim(0, peak * 1.15)
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / 'lr_schedule.png', dpi=200, facecolor=SURFACE)
        plt.close(ax.figure)
        print(f'Đã lưu {out_dir / "lr_schedule.png"}')


if __name__ == '__main__':
    main()
