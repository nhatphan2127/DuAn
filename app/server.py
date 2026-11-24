"""Web app dịch Việt <-> Trung.

Chạy (từ thư mục gốc repo, môi trường .venv-lora):
    .venv-lora/bin/python -m app.server
    .venv-lora/bin/python -m app.server --lora-adapter <thư_mục_adapter> --preload baseline,nllb
rồi mở http://127.0.0.1:8000
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / 'static'
sys.path.insert(0, str(ROOT))   # để import được nmt/ và nllb_lora/ khi chạy từ thư mục khác

from app.engines import Options, default_engines, load_scores   # noqa: E402
from app.service import ModelUnavailable, Translator, UnknownModel   # noqa: E402

MAX_CHARS = 5000
MAX_BATCH_LINES = 64


class DecodeParams(BaseModel):
    source: Literal['auto', 'vi', 'zh'] = 'auto'
    beam_size: int = Field(4, ge=1, le=8)
    max_new_tokens: int = Field(256, ge=16, le=512)

    def options(self):
        return Options(beam_size=self.beam_size, max_new_tokens=self.max_new_tokens)


class TranslateRequest(DecodeParams):
    text: str = Field(max_length=MAX_CHARS)
    model: str = 'baseline'


class CompareRequest(DecodeParams):
    text: str = Field(max_length=MAX_CHARS)
    models: list[str] = Field(min_length=1, max_length=8)


class BatchRequest(DecodeParams):
    lines: list[str] = Field(max_length=MAX_BATCH_LINES)
    model: str = 'baseline'


def create_app(translator):
    app = FastAPI(title='Dịch Việt ↔ Trung', docs_url='/api/docs', redoc_url=None)

    def call(fn, *args):
        try:
            return fn(*args)
        except UnknownModel as error:
            raise HTTPException(404, f'Không có mô hình "{error.args[0]}"')
        except ModelUnavailable as error:
            raise HTTPException(409, str(error))
        except ValueError as error:
            raise HTTPException(400, str(error))
        except Exception as error:
            raise HTTPException(500, f'{type(error).__name__}: {error}')

    @app.get('/api/models')
    def models():
        return {'models': translator.models(), 'limits': {
            'max_chars': MAX_CHARS, 'max_batch_lines': MAX_BATCH_LINES}}

    # Endpoint đồng bộ (def): FastAPI chạy trong threadpool nên dịch lâu không chặn các request khác
    @app.post('/api/translate')
    def translate(req: TranslateRequest):
        return call(translator.translate, req.model, req.text, req.source, req.options())

    @app.post('/api/compare')
    def compare(req: CompareRequest):
        return {'results': translator.compare(req.models, req.text, req.source, req.options())}

    @app.post('/api/batch')
    def batch(req: BatchRequest):
        if any(len(line) > MAX_CHARS for line in req.lines):
            raise HTTPException(422, f'Mỗi dòng tối đa {MAX_CHARS} ký tự')
        return call(translator.batch, req.model, req.lines, req.source, req.options())

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app


def main():
    parser = argparse.ArgumentParser(description='Web app dịch Việt <-> Trung')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--lora-adapter', default=os.environ.get('NMT_LORA_ADAPTER'),
                        help='thư mục adapter LoRA (có adapter_config.json); mặc định lấy từ $NMT_LORA_ADAPTER')
    parser.add_argument('--preload', default='', help='nạp sẵn các mô hình khi khởi động, vd. baseline,nllb')
    parser.add_argument('--local-files-only', action='store_true', help='không tải NLLB từ Hugging Face')
    parser.add_argument('--device', choices=('cuda', 'cpu'), help='mặc định: cuda nếu có')
    args = parser.parse_args()

    import uvicorn
    from app import engines
    engines.DEVICE_OVERRIDE = args.device

    translator = Translator(default_engines(args.lora_adapter, args.local_files_only), load_scores())
    for model_id in filter(None, args.preload.split(',')):
        print(f'Đang nạp {model_id}...', flush=True)
        translator.engine(model_id).load()
    print(f'Mở http://{args.host}:{args.port}', flush=True)
    uvicorn.run(create_app(translator), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
