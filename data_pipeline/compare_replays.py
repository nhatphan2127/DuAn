"""Compare deterministic artifacts from two independent full raw-data runs."""
import argparse

from freeze_utils import ROOT, config, fingerprint, run_paths, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    args = parser.parse_args()
    cfg = config(args.config)
    checked = []
    for position, label in [(0, 'processed'), (1, 'final')]:
        left, right = run_paths(cfg, 'run-a')[position], run_paths(cfg, 'run-b')[position]
        assert {p.name for p in left.iterdir()} == {p.name for p in right.iterdir()}
        for a in sorted(left.iterdir()):
            b = right / a.name
            ha, hb = fingerprint(a), fingerprint(b)
            assert ha['sha256'] == hb['sha256'], (a, b)
            checked.append({'artifact': label + '/' + a.name, 'sha256': ha['sha256'], 'identical': True})
    ea, eb = run_paths(cfg, 'run-a')[2], run_paths(cfg, 'run-b')[2]
    names = [p.name for p in ea.glob('T*_decisions.jsonl')]
    names += ['ccmatrix_sample_indices.json', 'split_membership.jsonl', 'split_summary.json', 'split_replacements.csv',
              'quarantine.jsonl', 'variant_groups.jsonl', 't05_migration_summary.json']
    for name in sorted(names):
        ha, hb = fingerprint(ea / name), fingerprint(eb / name)
        assert ha['sha256'] == hb['sha256'], name
        checked.append({'artifact': 'evidence/' + name, 'sha256': ha['sha256'], 'identical': True})
    write_json(ROOT / cfg['evidence_root'] / 'replay_comparison.json', {'status': 'pass', 'different_artifacts': 0, 'artifacts': checked,
               'scope': 'Independent raw-data builds using separate isolated environments; Windows AMD64 only'})
    print('PASS: identical deterministic artifacts:', len(checked))


if __name__ == '__main__':
    main()
