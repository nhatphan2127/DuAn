"""Freeze the technically verified dataset under a new directory, same filenames."""
import argparse
import hashlib
import json
import shutil

from freeze_utils import ROOT, config, fingerprint, run_paths, write_json
from verify_manifest import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    args = parser.parse_args()
    cfg = config(args.config)
    _, final, evidence = run_paths(cfg, 'run-a')
    report_root = ROOT / cfg['evidence_root']
    audit = json.loads((evidence / 'audit_summary.json').read_text(encoding='utf-8'))
    replay = json.loads((report_root / 'replay_comparison.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'technical_pass_semantics_not_reviewed' and replay['status'] == 'pass'
    for name in ['run-a', 'run-b']:
        assert (report_root / name / 'source_replay.json').exists()
        other_audit = json.loads((report_root / name / 'audit_summary.json').read_text(encoding='utf-8'))
        assert other_audit['status'] == 'technical_pass_semantics_not_reviewed'
    frozen = ROOT / cfg['work_root'] / 'frozen'
    if (frozen / 'checksums.json').exists():
        raise FileExistsError('Already frozen; do not overwrite')
    frozen.mkdir(parents=True, exist_ok=True)
    sources = [(p, frozen / 'final' / p.name) for p in sorted(final.iterdir())]
    metadata_names = ['split_membership.jsonl', 'split_summary.json', 'quarantine.jsonl', 'variant_groups.jsonl',
                      'T08_decisions.jsonl', 'T09_decisions.jsonl', 'T05_decisions.jsonl', 't05_migration_summary.json', 'audit_summary.json', 'source_replay.json']
    sources += [(evidence / name, frozen / 'metadata' / name) for name in metadata_names]
    sources += [(ROOT / args.config, frozen / 'metadata/config.json'),
                (ROOT / 'data_pipeline/requirements.lock', frozen / 'metadata/requirements-data.lock'),
                (report_root / 'environment.json', frozen / 'metadata/environment.json'),
                (report_root / 'environment-replay.json', frozen / 'metadata/environment-replay.json'),
                (report_root / 'control_filter_summary.json', frozen / 'metadata/control_filter_summary.json'),
                (report_root / 'python_runtime_archive.json', frozen / 'metadata/python_runtime_archive.json'),
                (report_root / 'replay_comparison.json', frozen / 'metadata/replay_comparison.json')]
    sources += [(p, frozen / 'metadata/opencc' / p.name) for p in sorted((ROOT / cfg['opencc_resources']).iterdir()) if p.is_file()]
    reproduction = list((ROOT / 'data_pipeline').glob('*.py')) + list((ROOT / 'data_pipeline').glob('*.ps1'))
    reproduction += list((ROOT / 'reports/chapter4').glob('*_template.md'))
    sources += [(p, frozen / 'reproduction' / p.relative_to(ROOT)) for p in reproduction if p.is_file()]
    sources += [(ROOT / 'data/tools/python-base.zip', frozen / 'reproduction/python-base.zip')]
    sources += [(p, frozen / 'reproduction/wheelhouse' / p.name) for p in sorted((ROOT / 'data/tools/wheelhouse').glob('*.whl'))]
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert fingerprint(source)['sha256'] == fingerprint(target)['sha256']
    code = list((ROOT / 'data_pipeline').glob('*.py'))
    write_json(frozen / 'metadata/code_manifest.json', [fingerprint(p) for p in sorted(code) if p.is_file()])
    write_json(frozen / 'metadata/input_manifest.json', json.loads((ROOT / cfg['snapshot_root'] / 'input_manifest.json').read_text(encoding='utf-8')))
    entries = []
    for path in sorted(frozen.rglob('*')):
        if path.is_file():
            entry = fingerprint(path)
            entry['path'] = path.relative_to(frozen).as_posix()
            entries.append(entry)
    manifest = {'dataset_name': 'Hoa - Việt (ZH-VI)', 'run_label': cfg['run_label'], 'status': 'frozen_technical_only',
                'semantic_review': cfg['semantic_review'], 'encoding': 'UTF-8 without BOM', 'newline': 'LF',
                'limitations': audit['limitations'], 'files': entries}
    manifest_path = frozen / 'checksums.json'
    write_json(manifest_path, manifest)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_path.with_suffix('.json.sha256').write_text(sha + '  checksums.json\n', encoding='ascii')
    verify(manifest_path)
    backup = ROOT / cfg['work_root'] / 'frozen_backup'
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copytree(frozen, backup)
    verify(backup / 'checksums.json')
    write_json(report_root / 'freeze_receipt.json', {'status': manifest['status'], 'dataset_name_unchanged': True,
               'dataset_directory': (frozen / 'final').relative_to(ROOT).as_posix(), 'manifest': manifest_path.relative_to(ROOT).as_posix(),
               'manifest_sha256': sha, 'backup': backup.relative_to(ROOT).as_posix(), 'backup_scope': 'second copy on same local volume; no off-device backup claimed',
               'semantic_review': cfg['semantic_review']})
    print('Frozen with unchanged dataset name:', frozen / 'final')


if __name__ == '__main__':
    main()
