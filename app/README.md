# app — web app dịch Việt ↔ Trung

Giao diện web chạy local để dùng và so sánh các mô hình của đồ án:

| Mô hình | id | Cần gì |
|---|---|---|
| Transformer baseline (hai chiều, huấn luyện từ đầu) | `baseline` | `vi_zh_bidir/checkpoints/best_checkpoint.pt` + `tokenizer_joint.model` |
| NLLB-600M dịch thẳng (zero-shot) | `nllb` | NLLB đúng revision trong `configs/nllb_lora_shared.json` (cache HF hoặc tải về) |
| NLLB-600M pivot qua tiếng Anh | `pivot` | như trên, dùng chung một bản NLLB với `nllb` |
| NLLB-600M + LoRA | `lora` | thư mục adapter (có `adapter_config.json`), truyền bằng `--lora-adapter` |

## Chạy

```bash
python scripts/download_assets.py --only checkpoint                      # lần đầu: checkpoint baseline
uv venv .venv-lora --python 3.12                                         # lần đầu: môi trường
uv pip install --python .venv-lora/bin/python -r requirements-lora.txt
.venv-lora/bin/python -m app.server                                       # mở http://127.0.0.1:8000
```

`requirements-lora.txt` (gốc repo) đã gồm cả FastAPI/Uvicorn. Môi trường `.venv-lora` có sẵn torch, transformers, peft thì chỉ cần thêm `app/requirements.txt`.

Tùy chọn:

| Tham số | Ý nghĩa |
|---|---|
| `--lora-adapter <thư_mục>` | bật mô hình LoRA (hoặc đặt biến môi trường `NMT_LORA_ADAPTER`); adapter được kiểm khớp config bằng `load_training_model()` của `nllb_lora` |
| `--preload baseline,nllb` | nạp sẵn mô hình khi khởi động; mặc định mô hình được nạp ở lần dùng đầu tiên |
| `--local-files-only` | không tải NLLB từ Hugging Face, chỉ dùng cache |
| `--device cpu` | chạy trên CPU (khi GPU đang bận, chậm hơn) |
| `--host`, `--port` | mặc định `127.0.0.1:8000`; đặt `--host 0.0.0.0` để máy khác trong mạng LAN truy cập |

Bộ nhớ GPU: baseline khoảng 0,3 GB, NLLB fp16 khoảng 1,2 GB (dùng chung cho `nllb` và `pivot`), LoRA thêm khoảng 1,2 GB.

## Tính năng

- **Dịch**: tự nhận diện Việt/Trung, đảo chiều, tự dịch khi ngừng gõ (hoặc Ctrl+Enter), sao chép kết quả; mô hình pivot hiện thêm câu tiếng Anh trung gian.
- **So sánh mô hình**: cùng một câu qua nhiều mô hình, mỗi thẻ ghi thời gian dịch và điểm BLEU/chrF trên tập test (đọc từ `results/matched/test/<mô hình>_<chiều>/metrics.json`).
- **Dịch nhiều dòng / tệp**: dán hoặc kéo thả tệp `.txt`, mỗi dòng dịch riêng (tự nhận diện từng dòng), có thanh tiến độ, nút dừng, tải về `.txt` hoặc `.tsv` song ngữ.
- **Tùy chọn giải mã**: beam size, số token sinh tối đa.

Văn bản được tách thành câu trước khi dịch (các mô hình học trên từng câu) rồi ghép lại theo đúng dòng. Thiết lập giải mã mặc định giống lúc chấm điểm trong `results/matched/`: beam 4, length penalty 1,0 kiểu Hugging Face, tối đa 256 token, không cắt câu nguồn. Vì vậy câu dịch trong app khớp với bản dịch dùng để tính điểm.

## API

| Endpoint | Body | Trả về |
|---|---|---|
| `GET /api/models` | | danh sách mô hình, trạng thái nạp, điểm test |
| `POST /api/translate` | `{text, model, source: auto/vi/zh, beam_size, max_new_tokens}` | `{translation, pivot, source, target, sentences, elapsed_ms}` |
| `POST /api/compare` | `{text, models: [...], ...}` | `{results: [...]}`; lỗi của một mô hình nằm trong `error` của mục đó |
| `POST /api/batch` | `{lines: [...] (≤ 64), model, ...}` | `{translations, sources, elapsed_ms}` |

Tài liệu tương tác: `http://127.0.0.1:8000/api/docs`.

## Cấu trúc

| File | Nội dung |
|---|---|
| `text.py` | nhận diện ngôn ngữ, tách câu, ghép câu |
| `engines.py` | bốn engine với cùng giao diện `translate(sentences, src, tgt, options)`, nạp lười |
| `service.py` | tách câu, chọn chiều, gom batch, khóa GPU (mỗi lúc một yêu cầu) |
| `server.py` | FastAPI + phục vụ `static/` |
| `static/` | giao diện HTML/CSS/JS thuần, không cần build |

Test (engine giả, không cần GPU): `.venv-lora/bin/python -m pytest tests/test_app.py -q`
