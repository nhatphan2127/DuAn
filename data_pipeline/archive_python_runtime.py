"""Retain the exact bundled base interpreter independently of installed packages."""
import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from freeze_utils import ROOT, config, fingerprint, write_json


def main():
    cfg = config()
    base = Path(sys.base_prefix)
    target = ROOT / 'data/tools/python-base.zip'
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
        for directory, folders, files in os.walk(base):
            folders[:] = sorted(d for d in folders if d not in {'site-packages', '__pycache__', 'Scripts'})
            for filename in sorted(files):
                source = Path(directory) / filename
                archive.write(source, source.relative_to(base).as_posix())
                count += 1
    write_json(ROOT / cfg['evidence_root'] / 'python_runtime_archive.json', {'archive': fingerprint(target), 'files': count,
               'python_version': sys.version, 'excluded': ['site-packages', '__pycache__', 'Scripts'],
               'packages_source': 'data_pipeline/requirements.lock and data/tools/wheelhouse'})
    print('Archived Python base runtime:', count, 'files')


if __name__ == '__main__':
    main()
