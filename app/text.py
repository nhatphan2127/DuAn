"""Xử lý văn bản đầu vào: nhận diện ngôn ngữ, tách câu, ghép lại bản dịch.

Các mô hình được huấn luyện trên từng câu (trung bình ~20 từ), nên đoạn văn dài được tách
thành câu trước khi dịch rồi ghép lại theo đúng thứ tự dòng.
"""
import re

LANGS = ('vi', 'zh')
OTHER = {'vi': 'zh', 'zh': 'vi'}

HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
LATIN = re.compile(r'[A-Za-zÀ-ỹ]')

# Dấu kết câu: tiếng Trung (。！？；…) tách ngay, kể cả khi không có khoảng trắng phía sau;
# dấu chấm Latinh chỉ tách khi theo sau là khoảng trắng (để giữ "3.5", "v.v.", URL...).
_BOUNDARY = re.compile(r'([。！？!?]+[”’"」』）)]*|…+|\.{3,}|\.(?=\s))\s*')
_ABBREVIATIONS = {'tp', 'q', 'p', 'ts', 'pgs', 'gs', 'ths', 'bs', 'mr', 'mrs', 'ms', 'dr', 'st', 'no', 'vs', 'v.v'}


def detect_lang(text):
    """'zh' nếu chữ Hán nhiều hơn chữ Latinh, ngược lại 'vi'.

    Đếm cả hai loại ký tự để một tên riêng tiếng Trung lẫn trong câu tiếng Việt không làm đổi chiều dịch.
    """
    han = len(HAN.findall(text))
    latin = len(LATIN.findall(text))
    return 'zh' if han > 0 and han >= latin / 2 else 'vi'


def _is_abbreviation(segment):
    words = segment.rstrip('.').split()
    if not words:
        return False
    last = words[-1].lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_sentences(line):
    """Tách một dòng thành các câu (giữ dấu câu ở cuối mỗi câu)."""
    parts, start, pending = [], 0, ''
    for match in _BOUNDARY.finditer(line):
        segment = pending + line[start:match.end(1)]
        start = match.end()
        if match.group(1) == '.' and _is_abbreviation(segment):
            pending = segment + line[match.end(1):match.end()]   # "TP. Hồ Chí Minh": chưa phải hết câu
            continue
        pending = ''
        if segment.strip():
            parts.append(segment.strip())
    tail = (pending + line[start:]).strip()
    if tail:
        parts.append(tail)
    return parts


def segment_text(text):
    """Văn bản -> danh sách dòng, mỗi dòng là danh sách câu (dòng trống -> danh sách rỗng)."""
    return [split_sentences(line) for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')]


def join_sentences(sentences, lang):
    """Tiếng Trung viết liền, tiếng Việt/Anh cách nhau một dấu cách."""
    return ('' if lang == 'zh' else ' ').join(s.strip() for s in sentences if s.strip())
