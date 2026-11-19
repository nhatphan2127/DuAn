"""Confirm both the snapshot and all original corpus inputs remain unchanged."""
import json

from freeze_utils import ROOT, config, fingerprint, write_json


def main():
    cfg = config()
    manifest = json.loads((ROOT / cfg['snapshot_root'] / 'input_manifest.json').read_text(encoding='utf-8'))
    checked = []
    for entry in manifest:
        snapshot = fingerprint(ROOT / entry['snapshot_path'])
        assert snapshot['sha256'] == entry['sha256'], entry['snapshot_path']
        if entry['path'].startswith(('data/raw/', 'data/processed/', 'data/final/', 'data/benchmark/')):
            assert fingerprint(ROOT / entry['path'])['sha256'] == entry['sha256'], entry['path']
            checked.append(entry['path'])
    write_json(ROOT / cfg['evidence_root'] / 'original_data_integrity.json', {'status': 'pass', 'snapshot_files_verified': len(manifest),
               'original_data_files_verified': len(checked), 'original_data_paths': checked, 'original_corpus_overwritten': False})
    print('PASS: snapshot and original data unchanged:', len(manifest), len(checked))


if __name__ == '__main__':
    main()
