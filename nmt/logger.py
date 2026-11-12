"""Ghi log ra console và file JSON Lines (mỗi dòng một object JSON) để xem lại sau.

Dùng JSON Lines thay vì một mảng JSON vì chỉ cần append, Colab có ngắt giữa chừng
thì các dòng đã ghi vẫn còn nguyên. Đọc lại:

    import json
    records = [json.loads(line) for line in open('vi_zh_translation/train_log.jsonl')]
    epochs = [r for r in records if r['event'] == 'epoch_end']
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm


class RunLogger:
    def __init__(self):
        self.path = None
        self.run_id = uuid.uuid4().hex[:8]   # phân biệt các lần chạy trong cùng một file log

    def set_file(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event, message=None, **data):
        """In `message` ra console (nếu có) và ghi {time, run_id, event, message, **data} vào file."""
        if message:
            # tqdm.write để không làm vỡ thanh tiến trình đang chạy
            tqdm.write(message)
        if self.path is None:
            return

        record = {
            "time": datetime.now().isoformat(timespec='seconds'),
            "run_id": self.run_id,
            "event": event,
        }
        if message:
            record["message"] = message
        record.update(data)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


logger = RunLogger()
