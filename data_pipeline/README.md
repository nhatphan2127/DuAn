# data_pipeline

Mã dựng lại và kiểm tra tập dữ liệu song song Hoa–Việt dùng để huấn luyện `nllb_lora`. Thư mục này **chỉ chứa mã**; không chứa dữ liệu thô, dữ liệu trung gian hay kết quả chạy.

## Dữ liệu trong repo và dữ liệu phải dựng lại

- `data/val.*` và `data/test.*` của repo trùng từng byte (đã so SHA-256) với bản đóng băng mà pipeline này tạo ra.
- `data/train.*` không có trong repo (`train.vi` khoảng 113 MB, vượt giới hạn GitHub và bị `.gitignore` bỏ qua). Muốn có lại phải chạy pipeline bên dưới.
- Config huấn luyện (`configs/nllb_lora_*.json`) đọc dữ liệu ở `data/rebuild_20261016/frozen/` (xem `nllb_lora/README.md`).

## Môi trường

- Python **3.12.14** (`rebuild_dataset.py` kiểm đúng phiên bản và các gói ở `configs/data_freeze.json`: `opencc==1.1.7`, `fasttext-wheel==0.9.2`, `numpy==1.26.4`).
- `requirements.pins` liệt kê phiên bản; `requirements.lock` thêm hash cho cài đặt có kiểm hash (`pip install --require-hashes -r data_pipeline/requirements.lock`). Nên dùng một venv riêng.
- `setup_environment.ps1` dựng venv offline từ `data/tools/python-base.zip` và `data/tools/wheelhouse` cùng các file bằng chứng trong `reports/data_freeze_20261016/`. Các thứ đó **không nằm trong repo**, nên script này chỉ chạy được trên máy đã có chúng.
- `resources/opencc/1.1.7/` là bảng chuyển phồn–giản (t2s) cố định, kèm `checksums.json`. Bảng lấy từ [OpenCC](https://github.com/BYVoid/OpenCC) 1.1.7 (BYVoid/OpenCC, giấy phép Apache 2.0); giữ nguyên nguồn và giấy phép khi phân phối lại.

Mọi lệnh dưới đây chạy từ **thư mục gốc của repo**.

## Luồng chạy

1. Tải nguồn thô: `python data_pipeline/download_sources.py` (OPUS vi-zh: ALT, TED2020, WikiMatrix, CCMatrix; và FLORES-200) vào `data/raw/`.
2. Chuẩn bị các đầu vào mà script tải không tạo ra: `data/benchmark/`, `data/processed/` (gồm mô hình nhận dạng ngôn ngữ `lid.176.bin` và các file trung gian của lần xử lý trước) và `data/final/`. Các thư mục này không có trong repo.
3. `python data_pipeline/snapshot_dataset.py` chép và kiểm các đầu vào vào `data/snapshots/20261016/` (tạo `input_manifest.json`).
4. `python data_pipeline/rebuild_dataset.py --run-id run-a` chạy cả chuỗi T01–T10 vào `data/rebuild_20261016/run-a/`. Dùng `--stage T04` để chạy riêng một stage và `--resume` để bỏ qua stage đã xong (có kiểm hash). `--run-id run-b` chạy lần thứ hai để đối chiếu.
5. Kiểm tra: `audit_frozen_data.py --run-id run-a|run-b`, `verify_data_contracts.py`, `verify_original_data.py`, rồi `compare_replays.py` (so run-a với run-b).
6. Đóng băng: `freeze_release.py` ghi `data/rebuild_20261016/frozen/` (`final/` và `metadata/`, `checksums.json`). `verify_manifest.py --manifest <checksums.json>` kiểm lại, `check_release_gates.py` kiểm các điều kiện kỹ thuật.

Các stage của `rebuild_dataset.py`:

| Stage | Nội dung |
|---|---|
| T01 | gộp cặp vàng (gold) |
| T02 | lấy mẫu CCMatrix |
| T03 | gộp corpus |
| T04 | làm sạch bằng regex (loại cặp còn ký tự điều khiển) |
| T05 | chuẩn hóa ngôn ngữ (NFC, khoảng trắng, Hoa giản thể) |
| T06 | lọc heuristic |
| T07 | lọc ngôn ngữ |
| T08 | khử trùng toàn cục |
| T09 | loại phần trùng với tập benchmark (decontamination) |
| T10 | chia train/val/test |

## Các file

| File | Chức năng |
|---|---|
| `download_sources.py` | tải corpus OPUS và FLORES-200 |
| `snapshot_dataset.py` | chép và kiểm đầu vào trước khi dựng lại |
| `check_environment.py` | khóa wheel đã cài và kiểm môi trường (`--prepare`, `--output`) |
| `rebuild_dataset.py` | dựng lại T01–T10 vào thư mục riêng, không ghi đè dữ liệu cũ |
| `invalidate_rebuild.py` | giữ bằng chứng audit lỗi, vô hiệu marker của các stage từ `--from-stage` |
| `audit_frozen_data.py` | đếm lại độc lập và replay kiểm tra |
| `audit_rebuilt_variants.py`, `audit_translation_variants.py` | mô tả cấu trúc các biến thể bản dịch; tìm câu đã chuẩn hóa trùng nhau nhưng có bản dịch khác nhau giữa các split |
| `compare_replays.py` | so sánh artifact của hai lần chạy độc lập |
| `verify_data_contracts.py`, `verify_original_data.py`, `verify_manifest.py` | kiểm hồi quy chuẩn hóa/rò rỉ giữa split; kiểm đầu vào gốc không đổi; kiểm checksum của bản đóng băng |
| `freeze_release.py`, `check_release_gates.py` | đóng băng bản đã kiểm; kiểm điều kiện phát hành kỹ thuật |
| `archive_python_runtime.py`, `setup_environment.ps1` | lưu trình thông dịch Python gốc; dựng lại venv offline |
| `freeze_utils.py`, `text_normalization.py` | hàm dùng chung (đường dẫn, hash, chuẩn hóa văn bản) |

## Giới hạn cần biết

- Không dựng lại được từ repo một mình: các bước 2–3 cần dữ liệu đầu vào ngoài repo (xem trên).
- `check_release_gates.py` đọc bộ bằng chứng `reports/data_freeze_20261016/` (sinh ra khi chạy các bước trên, không có trong repo). Không dùng `--include-report`: cờ này cần `DuAn.docx` và `document_qa.json`, vốn do các script dựng báo cáo (không đưa lên) tạo ra.
- `audit_translation_variants.py` ghi kết quả vào `reports/chapter4/evidence` và đọc `data/final/<split>.*`; đường dẫn này gõ cứng.
- Phần xem xét ngữ nghĩa của bản dịch **chưa được thực hiện** (`semantic_review` trong config); các kiểm tra ở đây chỉ là kiểm tra kỹ thuật.
