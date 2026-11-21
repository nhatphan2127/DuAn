"""Shared file contracts for rebuilding the existing dataset in a separate folder."""
import hashlib
import json
import sys
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
if sys.flags.optimize:
    raise RuntimeError('Data verification must run without Python optimization flags')


def config(path='configs/data_freeze.json'):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def json_line(record):
    return json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def pairs(vi_path, zh_path):
    with Path(vi_path).open(encoding='utf-8') as vi, Path(zh_path).open(encoding='utf-8') as zh:
        for line, (v, z) in enumerate(zip_longest(vi, zh), 1):
            if v is None or z is None:
                raise ValueError(f'Unaligned pair at {vi_path}:{line}')
            yield line, v.rstrip('\r\n'), z.rstrip('\r\n')


def fingerprint(path):
    path = Path(path)
    sha = hashlib.sha256()
    lines = 0
    last = b''
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            lines += chunk.count(b'\n')
            last = chunk[-1:]
    stored_path = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    result = {'path': stored_path, 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}
    if path.suffix in {'.vi', '.zh', '.jsonl', '.csv', '.tsv', '.vie_Latn', '.zho_Hans'}:
        result['lines'] = lines + int(bool(last) and last != b'\n')
    return result


def run_paths(cfg, run_id):
    if run_id not in {'run-a', 'run-b'}:
        raise ValueError('run-id must be run-a or run-b')
    base = ROOT / cfg['work_root'] / run_id
    return base / 'processed', base / 'final', ROOT / cfg['evidence_root'] / run_id
