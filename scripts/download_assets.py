"""Tải các file lớn không nằm trong git (vượt giới hạn 100 MB của GitHub) từ Hugging Face Hub.

    python scripts/download_assets.py                  # checkpoint baseline + dữ liệu train
    python scripts/download_assets.py --only checkpoint
    python scripts/download_assets.py --only data

Kết quả:
    vi_zh_bidir/checkpoints/best_checkpoint.pt         checkpoint Transformer baseline (app, dịch, đánh giá)
    data/train.vi, data/train.zh                       dữ liệu train (huấn luyện lại baseline)
    data/rebuild_20261016/frozen/                      bản dữ liệu đóng băng mà configs/nllb_lora_*.json
                                                       và scripts/score_matched.py đọc (final/ + checksums.json)

Mọi file đều được kiểm SHA-256: dữ liệu theo data_pipeline/frozen_manifest/checksums.json, checkpoint
theo hash ghi trong CHECKPOINT bên dưới. Chỉ dùng thư viện chuẩn, chạy được bằng bất kỳ Python >= 3.10 nào.
Nguồn: model https://huggingface.co/nhatpha/vi-zh-transformer-baseline
       dữ liệu https://huggingface.co/datasets/nhatpha/vi-zh-parallel-20261016
Repo private/gated thì cần token: đặt biến môi trường HF_TOKEN, hoặc chạy `hf auth login` một lần.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = 'nhatpha/vi-zh-transformer-baseline'
DATA_REPO = 'nhatpha/vi-zh-parallel-20261016'
REVISION = 'main'
MANIFEST = ROOT / 'data_pipeline' / 'frozen_manifest'
FROZEN = ROOT / 'data' / 'rebuild_20261016' / 'frozen'

CHECKPOINT = {
    'asset': 'best_checkpoint.pt',
    'target': ROOT / 'vi_zh_bidir' / 'checkpoints' / 'best_checkpoint.pt',
    'sha256': '39104ba366a651f65dc62037b46eaa5c4c2737c6c8792830c02ca5452d2a8070',
}
TRAIN_FILES = ('train.vi', 'train.zh')   # asset tương ứng: <tên>.gz
SPLITS = ('train', 'val', 'test')        # val/test đã có sẵn trong data/ của repo


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_data_hashes():
    manifest = MANIFEST / 'checksums.json'
    pinned = (MANIFEST / 'checksums.json.sha256').read_text(encoding='utf-8').split()[0]
    if sha256(manifest) != pinned:
        sys.exit(f'{manifest} không khớp {manifest.name}.sha256: file manifest đã bị sửa')
    files = json.loads(manifest.read_text(encoding='utf-8'))['files']
    return {Path(f['path']).name: f['sha256'] for f in files if f['path'].startswith('final/')}


def hf_token():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    path = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'token'
    if not token and path.is_file():
        token = path.read_text(encoding='utf-8').strip()
    return token or None


def download(filename, dest, repo, kind, revision):
    """Tải một file của repo Hugging Face (kind: 'model' hoặc 'dataset') về dest."""
    prefix = 'datasets/' if kind == 'dataset' else ''
    url = f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}'
    request = urllib.request.Request(url)
    if token := hf_token():
        request.add_header('Authorization', f'Bearer {token}')
    print(f'Tải {filename} từ {url}', flush=True)
    try:
        with urllib.request.urlopen(request) as response, open(dest, 'wb') as out:
            total = int(response.headers.get('Content-Length') or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB', end='', flush=True)
            print()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            sys.exit(f'Không có quyền tải {url} (HTTP {error.code}). Repo private/gated: xin quyền truy cập trên '
                     'trang repo rồi đặt HF_TOKEN hoặc chạy `hf auth login`.')
        sys.exit(f'Không tải được {url}: HTTP {error.code}')
    except OSError as error:
        sys.exit(f'Không tải được {url}: {error}')


def fetch_checkpoint(args):
    target, expected = CHECKPOINT['target'], CHECKPOINT['sha256']
    if target.is_file() and sha256(target) == expected:
        print(f'Đã có {target.relative_to(ROOT)} (khớp SHA-256), bỏ qua')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.part')
    download(CHECKPOINT['asset'], tmp, args.model_repo, 'model', args.revision)
    if sha256(tmp) != expected:
        tmp.unlink()
        sys.exit(f'{CHECKPOINT["asset"]} sai SHA-256, đã xóa file tải về')
    tmp.replace(target)
    print(f'OK {target.relative_to(ROOT)}')


def fetch_data(args):
    hashes = expected_data_hashes()
    data_dir = ROOT / 'data'
    for name in TRAIN_FILES:
        target = data_dir / name
        if target.is_file() and sha256(target) == hashes[name]:
            print(f'Đã có data/{name} (khớp SHA-256), bỏ qua')
            continue
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp_dir:
            archive = Path(tmp_dir) / f'{name}.gz'
            download(f'{name}.gz', archive, args.data_repo, 'dataset', args.revision)
            plain = Path(tmp_dir) / name
            with gzip.open(archive, 'rb') as src, open(plain, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            if sha256(plain) != hashes[name]:
                sys.exit(f'data/{name} giải nén ra sai SHA-256')
            plain.replace(target)
        print(f'OK data/{name}')

    # Dựng data/rebuild_20261016/frozen/: final/ + checksums.json (đủ cho prepare_data --verify-only)
    final = FROZEN / 'final'
    final.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for lang in ('vi', 'zh'):
            name = f'{split}.{lang}'
            src, dst = data_dir / name, final / name
            if sha256(src) != hashes[name]:
                sys.exit(f'data/{name} không khớp checksums.json (file trong repo đã bị sửa?)')
            if dst.is_file() and sha256(dst) == hashes[name]:
                continue
            dst.unlink(missing_ok=True)
            try:
                os.link(src, dst)          # hard link: không tốn thêm dung lượng
            except OSError:
                shutil.copyfile(src, dst)
    for name in ('checksums.json', 'checksums.json.sha256'):
        shutil.copyfile(MANIFEST / name, FROZEN / name)
    print(f'OK {FROZEN.relative_to(ROOT)}/ (final/ + checksums.json)')


def main():
    parser = argparse.ArgumentParser(description='Tải checkpoint baseline và dữ liệu train từ Hugging Face Hub')
    parser.add_argument('--only', choices=('checkpoint', 'data'), help='chỉ tải một loại; mặc định tải cả hai')
    parser.add_argument('--model-repo', default=MODEL_REPO)
    parser.add_argument('--data-repo', default=DATA_REPO)
    parser.add_argument('--revision', default=REVISION, help='nhánh, tag hoặc commit của repo Hugging Face')
    args = parser.parse_args()
    if args.only in (None, 'checkpoint'):
        fetch_checkpoint(args)
    if args.only in (None, 'data'):
        fetch_data(args)


if __name__ == '__main__':
    main()
