# nllb_lora

Huấn luyện và đánh giá **NLLB-200 distilled 600M** (`facebook/nllb-200-distilled-600M`, revision được ghim trong config) bằng LoRA cho cặp Hoa–Việt, theo cách chạy trên Google Colab. Cấu hình chính là **một adapter chung cho cả hai chiều** (`configs/nllb_lora_shared.json`). Thư mục này không chứa dữ liệu, checkpoint hay kết quả đánh giá.

Mọi lệnh dưới đây chạy từ **thư mục gốc của repo** (mã dùng import dạng `nllb_lora.train.*`, `nllb_lora.eval.*`).

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `train/train_nllb.py` | huấn luyện một GPU (full fine-tuning hoặc LoRA) |
| `train/prepare_data.py` | kiểm dữ liệu đóng băng, tạo/chép bundle dữ liệu tối thiểu, tạo cache token |
| `train/lora_adapter.py` | dựng LoRA và nạp lại adapter theo hợp đồng đã ghim |
| `train/sync_checkpoints.py` | ghi/kiểm checkpoint (Trainer state, trọng số) lên thư mục đồng bộ (Drive) |
| `train/check_readiness.py`, `train/check_colab_environment.py` | kiểm điều kiện trước khi chạy đầy đủ; ghi lại và kiểm môi trường Colab |
| `train/profile_speed.py` | cấu hình theo GPU và thử tốc độ có giới hạn |
| `train/colab_workflow.py` | điều phối toàn bộ quy trình trên Colab (chỉ dùng thư viện chuẩn trong kernel notebook); chứa danh sách `CODE_FILES` |
| `train/build_source_bundle.py` | dựng ZIP mã nguồn cho Colab từ `CODE_FILES` |
| `eval/evaluate.py` | sinh bản dịch chỉ từ câu nguồn rồi chấm BLEU/chrF với tham chiếu gốc (`--direction` cho adapter chung) |
| `eval/evaluate_matched.py` | đánh giá độc lập theo cùng chính sách padding/AMP của Trainer |
| `eval/finalize_model.py` | xuất checkpoint tốt nhất đã kiểm chứng sau khi hết ngân sách hoặc dừng sớm |
| `eval/compare_translations.py` | so sánh hai bản dịch thẳng hàng bằng paired bootstrap |
| `eval/translation_metrics.py` | BLEU/chrF cho văn bản tiếng Việt hoặc tiếng Trung |
| `eval/score_comet.py` | chấm COMET cho một thư mục đầu ra của evaluator (chạy trong môi trường riêng) |
| `requirements-colab.lock` | các phiên bản gói cho môi trường Colab |
| `../configs/` | config: `nllb_lora_shared.json` (adapter chung hai chiều), `nllb_lora_zh_vi.json` và `nllb_lora_vi_zh.json` (một chiều), `nllb_zh_vi.json` (full fine-tuning), `colab_readiness.template.json`, và `data_freeze.json` (cho `data_pipeline`) |
| `../notebooks/` | `nllb_lora_shared_train.ipynb` (huấn luyện) và `nllb_lora_shared_score_comet.ipynb` (chấm COMET) |
| `../tests/` | test kiểm chứng mã |

## 1. Chuẩn bị dữ liệu

Config đọc dữ liệu ở `data.frozen_root = data/rebuild_20261016/frozen`. Thư mục này cần có `final/{train,val,test}.{zh,vi}` **và** `checksums.json` cùng `checksums.json.sha256`: `prepare_data.py` đối chiếu SHA-256 của `checksums.json` với giá trị ghim trong config rồi kiểm từng file. Hai file `checksums.json*` do `data_pipeline/freeze_release.py` sinh ra và **không có trong repo**; chỉ chép ba cặp file văn bản vào `final/` là chưa đủ. Cách dựng `train.*`: xem `data_pipeline/README.md`.

Kiểm tra dữ liệu và tạo bundle để đưa lên Drive (notebook đọc bundle ở `DACNTT-data/nllb_zh_vi_20261016` trên My Drive):

```bash
python -m nllb_lora.train.prepare_data --config configs/nllb_lora_shared.json --output input_manifest.json --verify-only
python -m nllb_lora.train.prepare_data --config configs/nllb_lora_shared.json --output input_manifest.json --bundle-output <thư_mục_bundle_mới>
```

## 2. Dựng ZIP mã nguồn cho Colab

Notebook huấn luyện không lấy mã từ repo mà đọc một ZIP trên Drive và kiểm hash:

```bash
python -m nllb_lora.train.build_source_bundle --root . --output dist/nllb_lora_colab_source.zip
```

Lệnh in SHA-256 của ZIP và ghi `dist/nllb_lora_colab_source.zip.sha256`. Giá trị này phải bằng `BUNDLE_SHA256` trong `notebooks/nllb_lora_shared_train.ipynb`; sửa mã hoặc config trong `CODE_FILES` thì phải dựng lại ZIP và cập nhật hash trong notebook. ZIP chỉ chứa các file trong `CODE_FILES` và `SOURCE_MANIFEST.json`, được dựng tái lập (không nén, mốc thời gian cố định).

Hash chỉ khớp khi các file được checkout **đúng từng byte**. Trên Windows hãy đặt `git config core.autocrlf false` trước khi clone/checkout (hoặc dựng ZIP trên Linux/Colab); nếu không, Git đổi ký tự xuống dòng và hash sẽ khác.

Run đã bắt đầu bằng ZIP/bố cục cũ (mã nằm trong thư mục `scripts/`) trên Drive không *resume* được bằng mã này; hãy bắt đầu run mới.

## 3. Huấn luyện

1. Đặt `dist/nllb_lora_colab_source.zip` và `notebooks/nllb_lora_shared_train.ipynb` ở thư mục gốc My Drive.
2. Mở notebook trên Colab, chọn GPU (T4/G4/A100), Run all, cấp quyền Google/Drive khi được hỏi.
3. Notebook kiểm hash ZIP và từng file, giải nén, lấy quota Drive, chọn cấu hình theo GPU rồi chạy `colab_workflow`. Đặt `EVALUATE_TEST = True` chỉ sau khi đã khóa adapter và cấu hình đánh giá.

Các bước của `colab_workflow` (môi trường, bundle dữ liệu, cache token, huấn luyện, đánh giá) đều gọi lại các module ở trên bằng `python -m nllb_lora.train.<module>` / `python -m nllb_lora.eval.<module>`.

## 4. Chấm COMET

`notebooks/nllb_lora_shared_score_comet.ipynb` chạy trong môi trường riêng (cài `unbabel-comet`):

1. Chép `nllb_lora/eval/score_comet.py` lên thư mục gốc My Drive. Notebook kiểm `SCRIPT_SHA256`, nên file phải đúng từng byte như trong repo.
2. Sửa biến `RUN` trong notebook thành thư mục run của chính bạn trên Drive (giá trị hiện có là đường dẫn run của người dựng notebook).
3. Run all.

## 5. Adapter đã huấn luyện

Adapter chung hai chiều của đồ án (run 171.207 bước, LoRA r = 16, α = 32, `q_proj`/`v_proj`, 2.359.296 tham số) được đăng công khai tại [nhonguyen25/nllb-600m-vi-zh-lora](https://huggingface.co/nhonguyen25/nllb-600m-vi-zh-lora), kèm model card. Giấy phép CC-BY-NC 4.0 như NLLB-200: chỉ dùng cho nghiên cứu và học tập.

```bash
python scripts/download_assets.py --only lora      # tạo nllb_lora/adapter/ (revision ghim, kiểm SHA-256)
```

Điểm trên tập test (5.000 cặp mỗi chiều, beam 4, tối đa 256 token), báo riêng từng chiều:

| Chiều | BLEU | chrF | COMET |
|---|---|---|---|
| Trung → Việt (`zho_Hans` → `vie_Latn`) | 29,91 | 49,26 | 0,8387 |
| Việt → Trung (`vie_Latn` → `zho_Hans`) | 33,00 | 31,11 | 0,8201 |

BLEU Việt → Trung dùng tokenizer `zh` và NFKC; Trung → Việt dùng `13a`. Hai chiều không trung bình với nhau. Bản dịch và điểm chi tiết: `results/lora/`. Dùng trong web app: `python -m app.server --lora-adapter nllb_lora/adapter` (xem [app/README.md](../app/README.md)).

## Kiểm thử

```bash
python -m pytest tests -q
```

Cần môi trường có `torch`, `transformers`, `peft`, `sentencepiece` và `protobuf` (`protobuf` không có trong `requirements-colab.lock`). Test chạy offline (`local_files_only=True`), nên cache Hugging Face phải có sẵn các file tokenizer của `facebook/nllb-200-distilled-600M` ở revision ghi trong config.
