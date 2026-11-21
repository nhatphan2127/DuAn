"""Verify checksums relative to a frozen corpus directory, including a copy."""
import argparse
import hashlib
import json
from pathlib import Path

from freeze_utils import ROOT, fingerprint


def verify(path, root=None):
    path = Path(path).resolve()
    root = Path(root).resolve() if root else path.parent
    expected_hash = path.with_suffix(path.suffix + '.sha256').read_text(encoding='ascii').split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, 'Manifest checksum differs'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for entry in manifest['files']:
        target = (root / entry['path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Manifest path escapes dataset directory')
        actual = fingerprint(target)
        assert actual['sha256'] == entry['sha256'] and actual['bytes'] == entry['bytes'], entry['path']
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root')
    args = parser.parse_args()
    manifest = verify(ROOT / args.manifest, ROOT / args.root if args.root else None)
    print('PASS: frozen manifest verified;', len(manifest['files']), 'files;', manifest['status'])


if __name__ == '__main__':
    main()
