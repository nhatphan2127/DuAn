"""Tầng nghiệp vụ: tách câu, chọn chiều dịch, gọi engine, ghép kết quả. Không phụ thuộc web."""
import threading
import time

from .engines import Options
from .text import OTHER, detect_lang, join_sentences, segment_text, split_sentences


class UnknownModel(KeyError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class Translator:
    def __init__(self, engines, scores=None):
        self.engines = {engine.id: engine for engine in engines}
        self.scores = scores or {}
        # Các mô hình dùng chung một GPU: mỗi lúc chỉ một yêu cầu nạp/dịch
        self._lock = threading.Lock()

    def engine(self, model_id):
        engine = self.engines.get(model_id)
        if engine is None:
            raise UnknownModel(model_id)
        ok, reason = engine.availability()
        if not ok:
            raise ModelUnavailable(reason)
        return engine

    def models(self):
        out = []
        for engine in self.engines.values():
            ok, reason = engine.availability()
            out.append({'id': engine.id, 'name': engine.name, 'description': engine.description,
                        'available': ok, 'reason': reason, 'loaded': engine.loaded,
                        'scores': self.scores.get(engine.score_prefix, {})})
        return out

    def _run(self, engine, sentences, src, options):
        with self._lock:
            return engine.translate(sentences, src, OTHER[src], options)

    def translate(self, model_id, text, source='auto', options=Options()):
        """Dịch một văn bản (nhiều dòng, nhiều câu) theo một chiều duy nhất."""
        engine = self.engine(model_id)
        src = detect_lang(text) if source == 'auto' else source
        tgt = OTHER[src]
        lines = segment_text(text)
        flat = [sentence for line in lines for sentence in line]
        started = time.perf_counter()
        results = self._run(engine, flat, src, options) if flat else []
        elapsed = time.perf_counter() - started

        out_lines, pivot_lines, i = [], [], 0
        for line in lines:
            chunk = results[i:i + len(line)]
            i += len(line)
            out_lines.append(join_sentences([r.text for r in chunk], tgt))
            pivot_lines.append(join_sentences([r.pivot or '' for r in chunk], 'en'))
        has_pivot = any(r.pivot for r in results)
        return {'model': engine.id, 'source': src, 'target': tgt, 'detected': source == 'auto',
                'translation': '\n'.join(out_lines).strip('\n'),
                'pivot': '\n'.join(pivot_lines).strip('\n') if has_pivot else None,
                'sentences': len(flat), 'elapsed_ms': round(elapsed * 1000)}

    def compare(self, model_ids, text, source='auto', options=Options()):
        """Cùng một văn bản qua nhiều mô hình; lỗi của một mô hình không làm hỏng các mô hình khác."""
        results = []
        for model_id in model_ids:
            try:
                results.append(self.translate(model_id, text, source, options))
            except (UnknownModel, ModelUnavailable) as error:
                results.append({'model': model_id, 'error': str(error.args[0] if error.args else error)})
            except Exception as error:   # lỗi nạp/chạy mô hình: báo về giao diện thay vì làm hỏng cả lượt so sánh
                results.append({'model': model_id, 'error': f'{type(error).__name__}: {error}'})
        return results

    def batch(self, model_id, lines, source='auto', options=Options()):
        """Dịch từng dòng độc lập (dòng trống giữ nguyên); mỗi dòng tự nhận diện ngôn ngữ nếu source='auto'."""
        engine = self.engine(model_id)
        started = time.perf_counter()
        sources = [None if not line.strip() else (detect_lang(line) if source == 'auto' else source)
                   for line in lines]
        translations = [''] * len(lines)
        for src in ('vi', 'zh'):
            indices = [i for i, s in enumerate(sources) if s == src]
            if not indices:
                continue
            segmented = [split_sentences(' '.join(lines[i].splitlines())) for i in indices]
            flat = [sentence for sentences in segmented for sentence in sentences]
            results = self._run(engine, flat, src, options)
            k = 0
            for i, sentences in zip(indices, segmented):
                translations[i] = join_sentences([r.text for r in results[k:k + len(sentences)]], OTHER[src])
                k += len(sentences)
        return {'model': engine.id, 'translations': translations, 'sources': sources,
                'elapsed_ms': round((time.perf_counter() - started) * 1000)}
