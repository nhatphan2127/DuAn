"""Fail closed on missing technical evidence; disclose unperformed semantic review."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_pipeline'))
from freeze_utils import config, fingerprint, write_json
from verify_manifest import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    parser.add_argument('--include-report', action='store_true')
    args = parser.parse_args()
    cfg = config(args.config)
    report = ROOT / cfg['evidence_root']
    checks = {}
    def load(path):
        return json.loads(path.read_text(encoding='utf-8'))
    for run in ['run-a', 'run-b']:
        audit = load(report / run / 'audit_summary.json')
        assert audit['status'] == 'technical_pass_semantics_not_reviewed'
        assert all(value == 0 for value in audit['pipeline_replay'].values())
        assert set(audit['pipeline_replay']) == {'T04','T05','T06','T07','T08','T09','T10'}
        assert len(audit['flores']) == 12 and all(x['matching_rows'] == 0 for x in audit['flores'])
        assert audit['all_pair_overlap_zero'] and audit['all_single_side_overlap_zero']
        assert load(report / run / 'source_replay.json')['T02_sample_pairs_verified'] == cfg['sample_size']
        split = load(report / run / 'split_summary.json')
        assert split['test'] == cfg['test_size'] and split['val'] == cfg['validation_size']
        assert max(abs(x['delta_percentage_points']) for x in split['distribution']) <= 2
        assert load(report / run / 't05_migration_summary.json')['new_normalization_idempotence_failures'] == 0
        checks[run] = 'pass'
    assert load(report / 'replay_comparison.json')['status'] == 'pass'
    assert load(report / 'environment.json')['status'] == load(report / 'environment-replay.json')['status'] == 'pass'
    checks['environment_and_full_replay'] = 'pass'
    if args.include_report:
        receipt = load(report / 'freeze_receipt.json')
        verify(ROOT / receipt['manifest'])
        original = load(report / 'original_data_integrity.json')
        assert original['status'] == 'pass'
        qa = load(report / 'document_qa.json')
        assert qa['status'] == 'pass' and qa['visual_review'] == 'all_pages_inspected'
        assert fingerprint(ROOT / 'DuAn.docx')['sha256'] == qa['source_docx_sha256']
        changes = load(report / 'run-a/document_changes.json')
        assert changes['output_sha256'] == qa['source_docx_sha256']
        checks['report_and_original_data'] = 'pass'
    write_json(report / 'release_gates.json', {'status': 'pass_technical_only', 'checks': checks,
               'semantic_review': cfg['semantic_review'], 'semantic_gate': 'not_run_by_user_scope',
               'tokenizer_gate': 'pending_model_preparation_not_part_of_raw_corpus_freeze',
               'report_included': args.include_report})
    print('PASS: requested technical gates;', 'report included' if args.include_report else 'data only')


if __name__ == '__main__':
    main()
