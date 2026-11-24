"""Các mô hình dịch, cùng một giao diện: translate(sentences, src, tgt, options) -> list[Result].

Mô hình chỉ được nạp vào bộ nhớ ở lần dùng đầu tiên và giữ lại cho các lần sau.
Thiết lập giải mã mặc định khớp với lúc chấm điểm trong results/matched/:
beam 4, length penalty 1,0 kiểu Hugging Face, tối đa 256 token sinh ra, không cắt câu nguồn.
"""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NLLB_LANG = {'vi': 'vie_Latn', 'zh': 'zho_Hans', 'en': 'eng_Latn'}
DIRECTIONS = (('vi', 'zh'), ('zh', 'vi'))


@dataclass
class Options:
    beam_size: int = 4
    max_new_tokens: int = 256


@dataclass
class Result:
    text: str
    pivot: str | None = None   # câu trung gian (tiếng Anh) của mô hình pivot


class Engine:
    """Lớp cơ sở. Lớp con cài đặt _load() và _translate()."""
    id = ''
    name = ''
    description = ''
    score_prefix = None   # tên thư mục trong results/matched/test/<prefix>_<chiều>/ để lấy điểm

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def availability(self):
        """(dùng được?, lý do nếu không)."""
        return True, None

    def load(self):
        if not self._loaded:
            try:
                self._load()
            except Exception as error:
                raise explain_oom(error) from error
            self._loaded = True

    def translate(self, sentences, src, tgt, options):
        if (src, tgt) not in DIRECTIONS:
            raise ValueError(f'Không hỗ trợ chiều {src}-{tgt}')
        if not sentences:
            return []
        self.load()
        try:
            return self._translate(sentences, src, tgt, options)
        except Exception as error:
            raise explain_oom(error) from error

    def _load(self):
        raise NotImplementedError

    def _translate(self, sentences, src, tgt, options):
        raise NotImplementedError


DEVICE_OVERRIDE = None   # 'cpu' / 'cuda' (tham số --device của server); None = tự chọn


def _device():
    import torch
    if DEVICE_OVERRIDE:
        return torch.device(DEVICE_OVERRIDE)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def explain_oom(error):
    """Đổi lỗi hết VRAM thành thông báo dễ hiểu (GPU thường đang bị tiến trình khác chiếm)."""
    import torch
    if isinstance(error, torch.cuda.OutOfMemoryError):
        free, total = torch.cuda.mem_get_info()
        return RuntimeError(f'GPU hết bộ nhớ (còn trống {free / 2**30:.1f}/{total / 2**30:.1f} GB). '
                            'Hãy tắt bớt tiến trình đang dùng GPU hoặc chạy app với --device cpu.')
    return error


class BaselineEngine(Engine):
    """Transformer hai chiều huấn luyện từ đầu (nmt/), token <2vi>/<2zh> chọn ngôn ngữ đích."""
    id = 'baseline'
    name = 'Transformer baseline'
    description = 'Transformer encoder–decoder huấn luyện từ đầu, một mô hình cho cả hai chiều'
    score_prefix = 'baseline'

    def __init__(self, base_dir=ROOT / 'vi_zh_bidir'):
        super().__init__()
        self.base_dir = Path(base_dir)

    def availability(self):
        ckpt = self.base_dir / 'checkpoints' / 'best_checkpoint.pt'
        return (True, None) if ckpt.is_file() else (False, f'Không thấy checkpoint {ckpt}')

    def _load(self):
        from nmt.config import Config
        from nmt.model import load_model
        from nmt.tokenizer import load_tokenizers

        paths = Config(direction='both', base_dir=str(self.base_dir))
        self.device = _device()
        self.model, self.cfg = load_model(paths.best_ckpt, self.device)
        self.cfg.base_dir = paths.base_dir
        if not self.cfg.bidirectional:
            raise ValueError('Checkpoint baseline phải là mô hình hai chiều (--direction both)')
        self.tokenizers = load_tokenizers(self.cfg, train_if_missing=False)

    def _translate(self, sentences, src, tgt, options):
        import torch
        from nmt.decoding import beam_search
        from nmt.tokenizer import EOS_IDX, lang_tag_id

        tok_src, tok_tgt = self.tokenizers[src], self.tokenizers[tgt]
        tag = lang_tag_id(self.cfg, self.tokenizers, tgt)
        results = []
        for sentence in sentences:
            ids = [tag] + tok_src.encode(sentence).ids + [EOS_IDX]
            src_tensor = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
            out = beam_search(self.model, src_tensor, options.beam_size, max_len=10 ** 6, length_penalty=1.0,
                              length_penalty_style='hf', max_new_tokens=options.max_new_tokens)
            results.append(Result(tok_tgt.decode(out)))
        return results


class NllbModel:
    """Một bản NLLB (gốc hoặc gắn LoRA) đã nạp, dùng chung cho nhiều engine."""

    def __init__(self, loader):
        self._loader = loader
        self.model = None

    def ensure(self):
        if self.model is None:
            import torch
            model, tokenizer = self._loader()
            device = _device()
            if device.type == 'cuda':
                model.half()   # fp16 (~1,2 GB VRAM); lúc đánh giá cũng sinh câu ở fp16 (Accelerate)
            try:
                model.to(device).eval()
            except Exception:
                del model   # nạp dở thì giải phóng VRAM, lần sau thử lại từ đầu
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                raise
            self.model, self.tokenizer, self.device = model, tokenizer, device

    def generate(self, sentences, src, tgt, options, batch_size=8):
        """Chọn ngôn ngữ đích bằng decoder prompt [decoder_start, mã ngôn ngữ đích], như lúc đánh giá."""
        import torch

        self.ensure()
        tokenizer = self.tokenizer
        tokenizer.src_lang = NLLB_LANG[src]
        target_id = tokenizer.convert_tokens_to_ids(NLLB_LANG[tgt])
        start_id = self.model.config.decoder_start_token_id
        outputs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512,
                                pad_to_multiple_of=8).to(self.device)
            prompt = torch.tensor([[start_id, target_id]] * len(batch), device=self.device)
            with torch.inference_mode():
                generated = self.model.generate(**encoded, decoder_input_ids=prompt, num_beams=options.beam_size,
                                                max_new_tokens=options.max_new_tokens, do_sample=False)
            outputs.extend(t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


def _shared_config(path=ROOT / 'configs' / 'nllb_lora_shared.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_base_nllb(config=None, local_files_only=False):
    """NLLB-600M gốc, đúng revision ghim trong config của đồ án."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    spec = (config or _shared_config())['model']
    model = AutoModelForSeq2SeqLM.from_pretrained(spec['name_or_path'], revision=spec['revision'],
                                                  attn_implementation='sdpa', local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(spec['name_or_path'], revision=spec['tokenizer_revision'],
                                              legacy_behaviour=False, local_files_only=local_files_only)
    return model, tokenizer


class NllbDirectEngine(Engine):
    id = 'nllb'
    name = 'NLLB-600M (dịch thẳng)'
    description = 'NLLB-200 distilled 600M chưa tinh chỉnh, dịch thẳng Việt ↔ Trung (zero-shot)'
    score_prefix = 'direct'

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @property
    def loaded(self):
        return self.backend.model is not None   # dịch thẳng và pivot dùng chung một NLLB

    def _load(self):
        self.backend.ensure()

    def _translate(self, sentences, src, tgt, options):
        return [Result(t) for t in self.backend.generate(sentences, src, tgt, options)]


class NllbPivotEngine(NllbDirectEngine):
    id = 'pivot'
    name = 'NLLB-600M (pivot qua tiếng Anh)'
    description = 'Cùng NLLB-600M gốc, dịch nguồn → tiếng Anh → đích'
    score_prefix = 'pivot'

    def _translate(self, sentences, src, tgt, options):
        english = self.backend.generate(sentences, src, 'en', options)
        final = self.backend.generate(english, 'en', tgt, options)
        return [Result(t, pivot=e) for t, e in zip(final, english)]


class NllbLoraEngine(NllbDirectEngine):
    """NLLB-600M + adapter LoRA chung hai chiều (nllb_lora/). Chỉ bật khi có đường dẫn adapter."""
    id = 'lora'
    name = 'NLLB-600M + LoRA'
    description = 'NLLB-600M tinh chỉnh bằng LoRA trên dữ liệu Hoa–Việt của đồ án (một adapter cho cả hai chiều)'
    score_prefix = 'lora'

    def __init__(self, adapter_path=None, config_path=ROOT / 'configs' / 'nllb_lora_shared.json',
                 local_files_only=False):
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.config_path = Path(config_path)
        self.local_files_only = local_files_only
        super().__init__(NllbModel(self._load_adapter))

    def availability(self):
        if self.adapter_path is None:
            return False, 'Chưa có adapter LoRA (chạy app với --lora-adapter <thư mục>)'
        if not (self.adapter_path / 'adapter_config.json').is_file():
            return False, f'Không thấy adapter_config.json trong {self.adapter_path}'
        return True, None

    def _load(self):
        ok, reason = self.availability()
        if not ok:
            raise RuntimeError(reason)
        self.backend.ensure()

    def _load_adapter(self):
        # Dùng đúng hàm nạp của nllb_lora: kiểm adapter khớp config (r, alpha, module...) rồi gắn lên NLLB gốc
        from transformers import AutoTokenizer
        from nllb_lora.train.train_nllb import load_training_model

        config = _shared_config(self.config_path)
        model = load_training_model(config, self.local_files_only, model_path=str(self.adapter_path))
        if (self.adapter_path / 'tokenizer_config.json').is_file():
            tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path), legacy_behaviour=False,
                                                      local_files_only=True)
        else:
            _, tokenizer = load_base_nllb(config, self.local_files_only)
        return model, tokenizer


def default_engines(lora_adapter=None, local_files_only=False):
    base = NllbModel(lambda: load_base_nllb(local_files_only=local_files_only))
    return [BaselineEngine(), NllbDirectEngine(base), NllbPivotEngine(base),
            NllbLoraEngine(lora_adapter, local_files_only=local_files_only)]


def load_scores(results_root=ROOT / 'results' / 'matched' / 'test'):
    """{tiền tố: {chiều: {bleu, chrf, comet?}}} từ metrics.json của các lần chấm trên tập test."""
    scores = {}
    for metrics in Path(results_root).glob('*_*/metrics.json'):
        prefix, direction = metrics.parent.name.rsplit('_', 1)
        try:
            data = json.loads(metrics.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        entry = {key: data[key] for key in ('bleu', 'chrf', 'comet') if isinstance(data.get(key), (int, float))}
        if entry:
            scores.setdefault(prefix, {})[direction] = entry
    return scores
