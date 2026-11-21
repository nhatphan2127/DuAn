"""Retain failed audit evidence and invalidate generated stage markers only."""
import argparse
import json
import shutil

from freeze_utils import ROOT, config, run_paths, write_json
from rebuild_dataset import STAGES, INVALID_CONTROL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-stage', required=True, choices=STAGES)
    args = parser.parse_args()
    cfg = config()
    report = ROOT / cfg['evidence_root']
    for run in ['run-a', 'run-b']:
        _, final, evidence = run_paths(cfg, run)
        diagnostic = report / 'diagnostics/before-control-filter' / run
        diagnostic.mkdir(parents=True, exist_ok=True)
        controls = []
        for path in sorted(final.glob('*.vi')) + sorted(final.glob('*.zh')):
            with path.open(encoding='utf-8') as stream:
                for line, text in enumerate(stream, 1):
                    if INVALID_CONTROL.search(text):
                        controls.append({'file': path.name, 'line': line, 'text': text.rstrip('\r\n')})
        write_json(diagnostic / 'control_character_failures.json', {'status': 'failed_superseded', 'findings': controls})
        for name in ['split_summary.json', 't05_migration_summary.json', 'variant_summary.json']:
            if (evidence / name).exists():
                shutil.copy2(evidence / name, diagnostic / name)
        for stage in STAGES[STAGES.index(args.from_stage):]:
            path = (evidence / f'{stage}.json').resolve()
            if not path.is_relative_to(evidence.resolve()):
                raise ValueError('Invalid marker path')
            if path.exists():
                shutil.copy2(path, diagnostic / path.name)
                path.unlink()
    if (report / 'replay_comparison.json').exists():
        shutil.copy2(report / 'replay_comparison.json', report / 'diagnostics/before-control-filter/replay_comparison.json')
        write_json(report / 'replay_comparison.json', {'status': 'invalidated_pending_rebuild', 'reason': 'Control characters found by independent audit'})
    print('Generated markers invalidated; original inputs and failing evidence retained')


if __name__ == '__main__':
    main()
