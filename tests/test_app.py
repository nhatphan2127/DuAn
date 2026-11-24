"""Test web app dịch (app/): tách câu, nhận diện ngôn ngữ, tầng service và API với engine giả (không cần GPU)."""
import pytest

from app.engines import Engine, Options, Result, load_scores
from app.service import ModelUnavailable, Translator, UnknownModel
from app.text import detect_lang, join_sentences, segment_text, split_sentences


class FakeEngine(Engine):
    """Dịch giả: '<chiều>:<câu>' để kiểm được câu nào đi chiều nào; pivot=True thì trả thêm câu trung gian."""

    def __init__(self, id='fake', available=True, pivot=False, fail=None):
        super().__init__()
        self.id, self.name, self.description, self.score_prefix = id, id.title(), 'giả', id
        self._available, self._pivot, self._fail = available, pivot, fail
        self.calls = []

    def availability(self):
        return (True, None) if self._available else (False, 'chưa có adapter')

    def _load(self):
        if self._fail:
            raise self._fail

    def _translate(self, sentences, src, tgt, options):
        self.calls.append((list(sentences), src, tgt, options))
        return [Result(f'{src}>{tgt}:{s}', pivot=f'en:{s}' if self._pivot else None) for s in sentences]


# ---------- app/text.py ----------

@pytest.mark.parametrize('text, lang', [
    ('Tôi thích học trí tuệ nhân tạo.', 'vi'),
    ('我喜欢学习人工智能。', 'zh'),
    ('Tôi đã gặp ông 王 ở Hà Nội hôm qua.', 'vi'),   # một chữ Hán lẫn trong câu Việt không đổi chiều
    ('我在 Hà Nội 工作了三年，非常喜欢这里。', 'zh'),
    ('12345', 'vi'),
])
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang


@pytest.mark.parametrize('line, expected', [
    ('Tôi đi học. Hôm nay trời đẹp!', ['Tôi đi học.', 'Hôm nay trời đẹp!']),
    ('我喜欢学习。今天天气很好！你呢？', ['我喜欢学习。', '今天天气很好！', '你呢？']),
    ('TP. Hồ Chí Minh rất đông. Tôi thích.', ['TP. Hồ Chí Minh rất đông.', 'Tôi thích.']),
    ('Giá là 3.5 triệu, v.v. và nhiều thứ khác.', ['Giá là 3.5 triệu, v.v. và nhiều thứ khác.']),
    ('他说：“你好。”然后走了。', ['他说：“你好。”', '然后走了。']),
    ('Không có dấu câu', ['Không có dấu câu']),
    ('   ', []),
])
def test_split_sentences(line, expected):
    assert split_sentences(line) == expected


def test_segment_text_keeps_blank_lines_and_crlf():
    assert segment_text('Tôi đi. Bạn ở.\r\n\r\nHết') == [['Tôi đi.', 'Bạn ở.'], [], ['Hết']]


def test_join_sentences():
    assert join_sentences(['我喜欢。', '你呢？'], 'zh') == '我喜欢。你呢？'
    assert join_sentences(['Tôi thích.', 'Còn bạn?'], 'vi') == 'Tôi thích. Còn bạn?'


# ---------- app/service.py ----------

def test_translate_detects_direction_and_rebuilds_lines():
    engine = FakeEngine()
    out = Translator([engine]).translate('fake', 'Xin chào. Tạm biệt.\n\nCảm ơn')
    assert (out['source'], out['target'], out['detected']) == ('vi', 'zh', True)
    assert out['translation'] == 'vi>zh:Xin chào.vi>zh:Tạm biệt.\n\nvi>zh:Cảm ơn'
    assert out['sentences'] == 3 and out['pivot'] is None
    assert len(engine.calls) == 1   # mọi câu đi chung một lượt gọi engine


def test_translate_explicit_source_and_options():
    engine = FakeEngine()
    out = Translator([engine]).translate('fake', 'Xin chào', 'zh', Options(beam_size=1, max_new_tokens=32))
    assert (out['source'], out['target'], out['detected']) == ('zh', 'vi', False)
    assert engine.calls[0][3] == Options(beam_size=1, max_new_tokens=32)


def test_translate_empty_text_does_not_call_engine():
    engine = FakeEngine()
    out = Translator([engine]).translate('fake', '  \n ')
    assert out['translation'] == '' and engine.calls == [] and not engine.loaded


def test_translate_returns_pivot_text():
    out = Translator([FakeEngine(pivot=True)]).translate('fake', '我喜欢。你呢？')
    assert out['pivot'] == 'en:我喜欢。 en:你呢？'
    assert out['translation'] == 'zh>vi:我喜欢。 zh>vi:你呢？'


def test_unknown_and_unavailable_models():
    translator = Translator([FakeEngine('off', available=False)])
    with pytest.raises(UnknownModel):
        translator.translate('nope', 'x')
    with pytest.raises(ModelUnavailable, match='chưa có adapter'):
        translator.translate('off', 'x')


def test_compare_isolates_failures():
    translator = Translator([FakeEngine('a'), FakeEngine('off', available=False),
                             FakeEngine('broken', fail=RuntimeError('GPU hết bộ nhớ'))])
    results = translator.compare(['a', 'off', 'broken', 'nope'], 'Xin chào')
    assert results[0]['translation'] == 'vi>zh:Xin chào'
    assert results[1] == {'model': 'off', 'error': 'chưa có adapter'}
    assert 'GPU hết bộ nhớ' in results[2]['error']
    assert results[3] == {'model': 'nope', 'error': 'nope'}


def test_failed_load_is_retried():
    engine = FakeEngine(fail=RuntimeError('lần đầu lỗi'))
    translator = Translator([engine])
    with pytest.raises(RuntimeError):
        translator.translate('fake', 'Xin chào')
    engine._fail = None
    assert translator.translate('fake', 'Xin chào')['translation'] == 'vi>zh:Xin chào'


def test_batch_detects_each_line_and_keeps_order():
    engine = FakeEngine()
    out = Translator([engine]).batch('fake', ['Xin chào.', '', '你好。再见。', 'Tạm biệt'])
    assert out['translations'] == ['vi>zh:Xin chào.', '', 'zh>vi:你好。 zh>vi:再见。', 'vi>zh:Tạm biệt']
    assert out['sources'] == ['vi', None, 'zh', 'vi']
    assert [(c[1], c[2]) for c in engine.calls] == [('vi', 'zh'), ('zh', 'vi')]   # gom theo chiều


def test_models_listing_includes_scores_and_status():
    translator = Translator([FakeEngine('a'), FakeEngine('off', available=False)],
                            {'a': {'vi-zh': {'bleu': 30.0, 'chrf': 40.0}}})
    listing = {m['id']: m for m in translator.models()}
    assert listing['a']['available'] and listing['a']['scores']['vi-zh']['bleu'] == 30.0
    assert not listing['off']['available'] and listing['off']['reason'] == 'chưa có adapter'


def test_load_scores_reads_metrics(tmp_path):
    (tmp_path / 'baseline_vi-zh').mkdir()
    (tmp_path / 'baseline_vi-zh' / 'metrics.json').write_text('{"bleu": 32.7, "chrf": 31.2, "count": 5000}')
    (tmp_path / 'pivot_zh-vi').mkdir()   # chưa có metrics.json -> bỏ qua
    (tmp_path / 'bad_zh-vi').mkdir()
    (tmp_path / 'bad_zh-vi' / 'metrics.json').write_text('không phải json')
    assert load_scores(tmp_path) == {'baseline': {'vi-zh': {'bleu': 32.7, 'chrf': 31.2}}}


# ---------- app/server.py ----------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.server import create_app

    translator = Translator([FakeEngine('fake'), FakeEngine('off', available=False)])
    return TestClient(create_app(translator))


def test_api_models(client):
    data = client.get('/api/models').json()
    assert [m['id'] for m in data['models']] == ['fake', 'off']
    assert data['limits']['max_chars'] == 5000


def test_api_translate(client):
    r = client.post('/api/translate', json={'text': '你好', 'model': 'fake'})
    assert r.status_code == 200
    assert r.json()['translation'] == 'zh>vi:你好'


@pytest.mark.parametrize('body, status', [
    ({'text': 'x', 'model': 'nope'}, 404),
    ({'text': 'x', 'model': 'off'}, 409),
    ({'text': 'x' * 5001, 'model': 'fake'}, 422),
    ({'text': 'x', 'model': 'fake', 'beam_size': 0}, 422),
    ({'text': 'x', 'model': 'fake', 'source': 'en'}, 422),
])
def test_api_translate_errors(client, body, status):
    assert client.post('/api/translate', json=body).status_code == status


def test_api_compare(client):
    results = client.post('/api/compare', json={'text': 'Xin chào', 'models': ['fake', 'off']}).json()['results']
    assert results[0]['translation'] == 'vi>zh:Xin chào' and 'error' in results[1]


def test_api_batch(client):
    r = client.post('/api/batch', json={'lines': ['Xin chào', '', '你好'], 'model': 'fake'})
    assert r.json()['translations'] == ['vi>zh:Xin chào', '', 'zh>vi:你好']
    assert client.post('/api/batch', json={'lines': ['x'] * 65, 'model': 'fake'}).status_code == 422


def test_index_and_static(client):
    assert 'Dịch Việt ↔ Trung' in client.get('/').text
    assert client.get('/static/app.js').status_code == 200
