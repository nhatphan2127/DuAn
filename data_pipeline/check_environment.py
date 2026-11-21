"""Lock installed wheel artifacts and verify the data environment."""
import argparse
import hashlib
import importlib.metadata as metadata
import platform
import shutil
import sys
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import fasttext
import numpy as np
import opencc

from freeze_utils import ROOT, config, fingerprint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    cfg = config(args.config)
    assert platform.python_version() == cfg['python_version'], sys.version
    for name, version in cfg['packages'].items():
        assert metadata.version(name) == version, (name, metadata.version(name))
    resource = ROOT / cfg['opencc_resources']
    if args.prepare:
        wheelhouse = ROOT / 'data/tools/wheelhouse'
        locked = []
        for wheel in sorted(wheelhouse.glob('*.whl')):
            with ZipFile(wheel) as archive:
                member = next(x for x in archive.namelist() if x.endswith('.dist-info/METADATA'))
                text = archive.read(member).decode('utf-8')
                name = next(x[6:] for x in text.splitlines() if x.startswith('Name: '))
                version = next(x[9:] for x in text.splitlines() if x.startswith('Version: '))
            locked.append(f'{name}=={version} --hash=sha256:{fingerprint(wheel)["sha256"]}')
        (ROOT / 'data_pipeline/requirements.lock').write_text('\n'.join(locked) + '\n', encoding='utf-8')
        share = Path(opencc.__file__).parent / 'clib/share/opencc'
        resource.mkdir(parents=True, exist_ok=True)
        for name in ['t2s.json', 'TSCharacters.ocd2', 'TSPhrases.ocd2']:
            shutil.copy2(share / name, resource / name)
        write_json(resource / 'checksums.json', [fingerprint(p) for p in sorted(resource.iterdir()) if p.is_file() and p.name != 'checksums.json'])
    import json
    for entry in json.loads((resource / 'checksums.json').read_text(encoding='utf-8')):
        assert fingerprint(ROOT / entry['path'])['sha256'] == entry['sha256']
    model = fasttext.load_model(str(ROOT / cfg['legacy_processed'] / 'lid.176.bin'))
    test_texts = ['Xin chào Việt Nam', '中国和越南的合作', 'a', 'Cảm ơn bạn rất nhiều.', 'SinnFéin，SDLP和联盟党都将失败归咎于DUP。']
    parity = []
    for text in test_texts:
        labels, scores = model.predict(text, k=1)
        score, label = model.f.predict(text + '\n', 1, 0.0, 'strict')[0]
        assert labels[0] == label and abs(float(scores[0]) - score) < 1e-7
        assert (float(scores[0]) >= 0.8) == (score >= 0.8)
        parity.append({'text': text, 'label': label, 'confidence': score})
    interpreter = Path(sys.base_prefix) / 'python.exe'
    report = {'python': sys.version, 'python_executable_sha256': hashlib.sha256(interpreter.read_bytes()).hexdigest(),
              'platform': platform.platform(), 'unicode_version': unicodedata.unidata_version,
              'opencc_native_version': opencc.__version__, 'packages': {d.metadata['Name']: d.version for d in metadata.distributions()},
              'numpy': np.__version__, 'fasttext_api_parity': parity, 'status': 'pass',
              'support_scope': 'CPython 3.12.14 Windows AMD64; other platforms not verified'}
    suffix = args.output or ('environment-replay.json' if 'replay' in sys.prefix else 'environment.json')
    if Path(suffix).name != suffix or not suffix.endswith('.json'):
        raise ValueError('Environment report must be a JSON filename')
    write_json(ROOT / cfg['evidence_root'] / suffix, report)
    print('Environment verified; FastText public/C++ parity passes', flush=True)


if __name__ == '__main__':
    main()
