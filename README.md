# Dịch máy Hoa ↔ Việt: Transformer baseline và NLLB-200

Mã nguồn đồ án *Nghiên cứu và đánh giá hiệu quả của mô hình học chuyển đổi trong dịch máy cho cặp ngôn ngữ Hoa–Việt*. Repo gồm bốn hệ thống dịch hai chiều Việt → Trung và Trung → Việt, một web app để dùng thử và so sánh chúng, cùng mã tái lập toàn bộ kết quả đánh giá.

| Hệ thống | Mô tả | Mã |
|---|---|---|
| Transformer baseline | Encoder–decoder cài đặt bằng PyTorch, huấn luyện từ đầu trên dữ liệu của đồ án; một mô hình cho cả hai chiều | `nmt/`, `scripts/train.py` |
| NLLB-600M dịch thẳng | `facebook/nllb-200-distilled-600M` chưa tinh chỉnh (zero-shot) | `nllb_lora/eval/evaluate.py` |
| NLLB-600M + LoRA | NLLB tinh chỉnh bằng LoRA, một adapter chung cho hai chiều | `nllb_lora/` |
| NLLB-600M pivot | NLLB gốc dịch qua tiếng Anh: nguồn → en → đích | `scripts/pivot_matched.py` |

## Kết quả trên tập test

5.000 cặp câu, cùng thiết lập giải mã (beam 4, length penalty 1,0, tối đa 256 token) và cùng hàm chấm điểm cho mọi hệ thống. COMET là bản `Unbabel/wmt22-comet-da`.

| Hệ thống | Việt → Trung: BLEU / chrF / COMET | Trung → Việt: BLEU / chrF / COMET |
|---|---|---|
| Transformer baseline | 32,74 / 31,26 / 0,8193 | 30,50 / 49,76 / 0,8374 |
| NLLB-600M dịch thẳng | 28,75 / 27,26 / 0,8016 | 27,09 / 46,85 / 0,8252 |
| NLLB-600M + LoRA | 33,00 / 31,11 / 0,8201 | 29,91 / 49,26 / 0,8387 |
| NLLB-600M pivot | 26,84 / 25,53 / 0,7860 | 26,83 / 46,34 / 0,8201 |

Bản dịch và điểm chi tiết: `results/matched/test/` (baseline, dịch thẳng, pivot) và `results/lora/` (LoRA). Kiểm định paired bootstrap và điểm theo độ dài câu: `results/matched/test/deep_eval.json`.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `nmt/` | Transformer baseline: tokenizer, dữ liệu, mô hình, huấn luyện, beam search, chấm điểm |
| `scripts/` | lệnh chạy baseline (`train.py`, `evaluate.py`, `translate.py`, `plot_metrics.py`), các script đánh giá matched (`*_matched.py`) và `download_assets.py` |
| `nllb_lora/` | huấn luyện và đánh giá NLLB + LoRA trên Colab (xem [nllb_lora/README.md](nllb_lora/README.md)) |
| `data_pipeline/` | dựng lại và kiểm tra bộ dữ liệu song ngữ (xem [data_pipeline/README.md](data_pipeline/README.md)) |
| `app/` | web app dịch và so sánh mô hình (xem [app/README.md](app/README.md)) |
| `configs/` | config NLLB/LoRA và config đóng băng dữ liệu |
| `notebooks/` | notebook Colab huấn luyện LoRA và chấm COMET |
| `data/` | `val.*`, `test.*` (5.000 cặp mỗi tập); `train.*` tải riêng |
| `vi_zh_bidir/` | baseline hai chiều đã huấn luyện: tokenizer, config, log, hình; checkpoint tải riêng |
| `results/` | bản dịch, điểm BLEU/chrF/COMET, hình của các lần đánh giá |
| `nllb_pivot/` | kết quả pivot đợt đầu (quy trình chấm của baseline) |
| `figures/` | hình dùng trong báo cáo |
| `tests/` | test kiểm chứng mã (chạy offline, không cần GPU) |

## 1. Cài đặt

Cần: Git, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), và GPU NVIDIA (khuyến nghị; web app chạy được trên CPU nhưng chậm hơn). Mọi lệnh chạy từ thư mục gốc repo.

```bash
git clone https://github.com/nhatphan2127/DuAn.git
cd DuAn
```

Trên Windows, mở bằng Git Bash hoặc WSL. Repo có `.gitattributes` giữ nguyên ký tự xuống dòng LF, vì notebook Colab kiểm SHA-256 của mã nguồn.

### 1.1 Tải checkpoint và dữ liệu train

Checkpoint baseline (241 MB) và `data/train.*` (113 MB + 73 MB) vượt giới hạn 100 MB của GitHub, nên được đặt trên Hugging Face Hub:

- Model: [nhatpha/vi-zh-transformer-baseline](https://huggingface.co/nhatpha/vi-zh-transformer-baseline)
- Dữ liệu train: [nhatpha/vi-zh-parallel-20261016](https://huggingface.co/datasets/nhatpha/vi-zh-parallel-20261016)

```bash
python scripts/download_assets.py                    # cả checkpoint và dữ liệu
python scripts/download_assets.py --only checkpoint  # chỉ cần để chạy app / dịch
```

Script kiểm SHA-256 từng file và tạo:

- `vi_zh_bidir/checkpoints/best_checkpoint.pt`
- `data/train.vi`, `data/train.zh`
- `data/rebuild_20261016/frozen/` (bản dữ liệu đóng băng mà `configs/nllb_lora_*.json` và `scripts/score_matched.py` đọc)

Script chỉ dùng thư viện chuẩn của Python, không cần cài gì thêm. Repo dữ liệu có giới hạn quyền truy cập (dữ liệu tổng hợp từ OPUS, một phần có giấy phép phi thương mại): bấm xin quyền trên trang dataset, rồi đặt biến môi trường `HF_TOKEN=<token>` hoặc chạy `.venv-lora/bin/hf auth login` trước khi tải. Không có quyền thì vẫn dựng lại được dữ liệu bằng `data_pipeline/`.

### 1.2 Hai môi trường Python

Baseline và NLLB dùng hai phiên bản thư viện khác nhau, nên tách thành hai môi trường:

| Môi trường | Dùng cho | Cài đặt |
|---|---|---|
| `.venv` | huấn luyện/đánh giá baseline, chấm COMET | `uv sync` (đọc `pyproject.toml` + `uv.lock`; torch 2.11 bản CUDA 12.8) |
| `.venv-lora` | NLLB (dịch thẳng, pivot, LoRA), chấm điểm matched, web app, test | xem lệnh bên dưới |

```bash
uv sync

uv venv .venv-lora --python 3.12
uv pip install --python .venv-lora/bin/python -r requirements-lora.txt
```

Trên Windows: thay `.venv-lora/bin/python` bằng `.venv-lora\Scripts\python.exe`, và thêm `--extra-index-url https://download.pytorch.org/whl/cu128` khi cài để có torch bản GPU.

Lần đầu chạy NLLB, Hugging Face tự tải `facebook/nllb-200-distilled-600M` (khoảng 2,5 GB) về `~/.cache/huggingface`, đúng revision ghim trong `configs/nllb_lora_shared.json`.

## 2. Chạy web app

```bash
.venv-lora/bin/python -m app.server
```

Mở http://127.0.0.1:8000. Giao diện có ba chức năng: dịch (tự nhận diện Việt/Trung), so sánh một câu qua nhiều mô hình, và dịch nhiều dòng hoặc cả tệp `.txt`.

| Tham số | Ý nghĩa |
|---|---|
| `--preload baseline,nllb` | nạp sẵn mô hình khi khởi động (mặc định nạp ở lần dùng đầu) |
| `--lora-adapter <thư_mục>` | bật mô hình NLLB + LoRA; thư mục phải có `adapter_config.json` |
| `--device cpu` | chạy trên CPU |
| `--local-files-only` | không tải NLLB từ mạng, chỉ dùng cache |
| `--host 0.0.0.0 --port 8000` | cho máy khác trong mạng LAN truy cập |

Adapter LoRA không có trong repo (là đầu ra của notebook huấn luyện trên Google Drive). Thiếu adapter thì app vẫn chạy với ba mô hình còn lại. Chi tiết API và cấu trúc: [app/README.md](app/README.md).

## 3. Dịch bằng dòng lệnh (baseline)

```bash
uv run python scripts/translate.py --direction both "Tôi thích học trí tuệ nhân tạo." "我喜欢学习人工智能。"
```

## 4. Huấn luyện lại baseline

Cần `data/train.*` (mục 1.1).

```bash
uv run python scripts/train.py --direction both          # ghi vào vi_zh_bidir/, tự train tiếp nếu bị ngắt
uv run python scripts/evaluate.py --direction both       # BLEU, chrF, chrF++, COMET trên tập test
uv run --with matplotlib python scripts/plot_metrics.py --base_dir ./vi_zh_bidir
```

`--direction vi-zh` hoặc `zh-vi` huấn luyện mô hình một chiều (ghi vào `vi_zh_translation/`, `zh_vi_translation/`). Các siêu tham số xem `uv run python scripts/train.py --help` và `nmt/config.py`.

Huấn luyện vào `vi_zh_bidir/` sẽ ghi đè `best_checkpoint.pt` đã tải (tokenizer có sẵn thì được dùng lại). Muốn giữ bản đã phát hành, truyền `--base_dir <thư_mục_khác>`.

Trên Colab: `pip install -r requirements.txt` rồi chạy các lệnh trên với `!python` thay cho `uv run python`.

## 5. Tái lập kết quả đánh giá matched

Các lệnh tạo ra `results/matched/test/`. Mỗi hệ thống được chấm bằng cùng hàm chấm của `nllb_lora` (`scripts/score_matched.py`). Cần `data/rebuild_20261016/frozen/` (mục 1.1) và GPU.

```bash
# Baseline: dịch (.venv) rồi chấm (.venv-lora)
.venv/bin/python scripts/translate_matched.py --base_dir vi_zh_bidir --out_root results/matched/test
.venv-lora/bin/python scripts/score_matched.py --eval_dir results/matched/test/baseline_vi-zh --direction vi-zh
.venv-lora/bin/python scripts/score_matched.py --eval_dir results/matched/test/baseline_zh-vi --direction zh-vi

# NLLB dịch thẳng (không truyền --model = NLLB gốc)
.venv-lora/bin/python -m nllb_lora.eval.evaluate --config configs/nllb_lora_shared.json --split test \
    --direction vi-zh --output-dir results/matched/test/direct_vi-zh

# NLLB pivot qua tiếng Anh
.venv-lora/bin/python scripts/pivot_matched.py --direction vi-zh --output_dir results/matched/test/pivot_vi-zh

# COMET (môi trường .venv có unbabel-comet)
.venv/bin/python nllb_lora/eval/score_comet.py --eval-dir results/matched/test/pivot_vi-zh \
    --output-dir results/matched/test/comet/pivot_vi-zh

# Paired bootstrap và điểm theo độ dài câu
.venv-lora/bin/python scripts/deep_eval_matched.py --out results/matched/test/deep_eval.json
```

Chiều `zh-vi` chạy tương tự (đổi `vi-zh` thành `zh-vi`). Các thư mục đầu ra đã có sẵn trong repo; chạy lại thì trỏ `--output-dir` sang thư mục mới.

## 6. NLLB + LoRA và dữ liệu

- Huấn luyện LoRA trên Colab, dựng ZIP mã nguồn, chấm COMET: [nllb_lora/README.md](nllb_lora/README.md).
- Dựng lại bộ dữ liệu từ nguồn OPUS và FLORES-200 (T01–T10): [data_pipeline/README.md](data_pipeline/README.md). Không cần làm bước này để chạy app hay tái lập điểm: `scripts/download_assets.py` đã tải đúng bản dữ liệu đóng băng.

## 7. Kiểm thử

```bash
.venv-lora/bin/python -m pytest tests -q
```

Test chạy offline, không cần GPU. Các test của `nllb_lora` cần cache Hugging Face đã có tokenizer NLLB (chạy app hoặc `evaluate.py` một lần là có). Test `test_colab_workflow::...corrupt_local_copy_restores_from_drive` yêu cầu ổ đĩa còn trống ít nhất 32 GiB, giống điều kiện trên Colab.
