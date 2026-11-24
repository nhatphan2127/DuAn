"""Copy and verify inputs before changing scripts or rebuilding data."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from freeze_utils import ROOT, config, fingerprint, write_json


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = config(args.config)
    destination = ROOT / cfg['snapshot_root']
    if destination.exists() and (not args.resume or (destination / 'input_manifest.json').exists()):
        raise FileExistsError(f'Snapshot already exists: {destination}')
    folders = ['data/raw', 'data/processed', 'data/final', 'data/benchmark', 'reports/chapter4', 'data_pipeline']
    files = [p for folder in folders for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [p for p in ROOT.iterdir() if p.is_file()]
    files += list((ROOT / 'configs').glob('*.json'))
    files = sorted(set(files))
    required = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(ROOT).free < required * 2:
        raise OSError('Insufficient free space for snapshot and rebuild')
    destination.mkdir(parents=True, exist_ok=args.resume)
    manifest = []
    for i, source in enumerate(files, 1):
        record = fingerprint(source)
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        copied = fingerprint(target)
        if copied['sha256'] != record['sha256']:
            raise ValueError(f'Snapshot mismatch: {source}')
        record['snapshot_path'] = target.relative_to(ROOT).as_posix()
        manifest.append(record)
        print(f'Snapshot {i}/{len(files)} {record["path"]}', flush=True)
    git = {}
    for name, command in [('commit', ['git', 'rev-parse', 'HEAD']), ('status', ['git', 'status', '--porcelain']), ('diff', ['git', 'diff', '--binary'])]:
        result = subprocess.run(command, cwd=ROOT, text=True, encoding='utf-8', capture_output=True, check=True)
        git[name] = result.stdout
    write_json(destination / 'input_manifest.json', manifest)
    write_json(destination / 'git_state.json', git)
    write_json(ROOT / cfg['evidence_root'] / 'snapshot.json', {'status': 'verified', 'files': len(files), 'bytes': required, 'manifest': str((destination / 'input_manifest.json').relative_to(ROOT))})


if __name__ == '__main__':
    main()
