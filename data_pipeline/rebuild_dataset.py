"""Rebuild T01--T10 without overwriting the existing dataset.

Original scripts and corpus are retained in the verified snapshot. Stage outputs
contain stable source IDs; only the six final text files are model inputs.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import fasttext

from freeze_utils import ROOT, config, fingerprint, json_line, pairs, rows, run_paths, write_json
from text_normalization import Normalizer

SOURCES = [('alt', 'alt/ALT.vi-zh'), ('ted2020', 'ted2020/TED2020.vi-zh'),
           ('wikimatrix', 'wikimatrix/WikiMatrix.vi-zh'), ('ccmatrix', 'ccmatrix/CCMatrix.vi-zh')]
STAGES = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10']
FILES = {'T01': 'gold_records.jsonl', 'T02': 'ccmatrix_records.jsonl', 'T03': 'raw_merged.jsonl',
         'T04': 'cleaned_merged.jsonl', 'T05': 'normalized_merged.jsonl', 'T06': 'heuristic_filtered.jsonl',
         'T07': 'lang_filtered.jsonl', 'T08': 'deduped.jsonl', 'T09': 'decontaminated.jsonl'}
PATTERNS = [re.compile(p) for p in [r'<[^>]+>', r'http[s]?://\S+|www\.\S+', r'[\w\.-]+@[\w\.-]+',
            r'[♪♫]', r'[\u200b\u200c\u200d\u200e\u200f\ufeff]', r'^[-—]\s*']]
INVALID_CONTROL = re.compile(r'[\x00-\x08\x0e-\x1f\x7f]')


def invalid_control(rec):
    return any(INVALID_CONTROL.search(rec[lang]) for lang in ['vi', 'zh'])


def clean(text):
    for pattern in PATTERNS:
        text = pattern.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


def raw_records(cfg, source, stem):
    vi_path, zh_path = ROOT / cfg['raw_root'] / (stem + '.vi'), ROOT / cfg['raw_root'] / (stem + '.zh')
    source_key = fingerprint(vi_path)['sha256'] + ':' + fingerprint(zh_path)['sha256']
    for line, vi, zh in pairs(vi_path, zh_path):
        vi, zh = vi.strip(), zh.strip()
        yield {'record_id': hashlib.sha256(f'{source_key}:{line}'.encode()).hexdigest(),
               'source': source, 'source_line': line, 'vi': vi, 'zh': zh}


def write_pairs(records, proc, stem):
    with (proc / (stem + '.vi')).open('w', encoding='utf-8', newline='\n') as vi, \
         (proc / (stem + '.zh')).open('w', encoding='utf-8', newline='\n') as zh:
        for rec in records:
            vi.write(rec['vi'] + '\n')
            zh.write(rec['zh'] + '\n')


def benchmark(cfg, norm):
    indices, mapping = {}, []
    for split, original, expected in [('dev', 'dev', 997), ('test', 'devtest', 1012)]:
        for lang, code in [('vi', 'vie_Latn'), ('zh', 'zho_Hans')]:
            copy = ROOT / cfg['benchmark_root'] / f'{split}.{lang}'
            raw = ROOT / cfg['raw_root'] / 'flores200' / f'{original}.{code}'
            info, source = fingerprint(copy), fingerprint(raw)
            if info['sha256'] != source['sha256'] or info['lines'] != expected:
                raise ValueError(f'Invalid FLORES mapping: {copy}')
            indices[(original, lang)] = {norm.key(s, lang) for s in copy.read_text(encoding='utf-8').splitlines()}
            mapping.append(dict(info, original_path=source['path'], original_split=original, identical_sha256=True))
    return indices, mapping


def source_group(rec):
    return 'ccmatrix' if rec['source'] == 'ccmatrix' else 'gold'


def length_bin(vi):
    n = len(vi.split())
    return '2-10' if n <= 10 else '11-20' if n <= 20 else '21-40' if n <= 40 else '41-150'


def split_groups(cfg, proc, final, evidence, norm):
    records = list(rows(proc / FILES['T09']))
    parent = list(range(len(records)))
    size = [1] * len(records)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]

    vi_index, zh_index = {}, {}
    for i, rec in enumerate(records):
        vi, zh = norm.pair(rec)
        for index, value in [(vi_index, vi), (zh_index, zh)]:
            if value in index:
                union(i, index[value])
            else:
                index[value] = i
    del vi_index, zh_index
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    group_id = {root: min(records[i]['record_id'] for i in members) for root, members in components.items()}
    key_to_index = {norm.pair(rec): i for i, rec in enumerate(records)}
    anchors, matches = {}, {}
    old_exact = {}
    for split in ['test', 'val']:
        anchors[split] = []
        for line, vi, zh in pairs(ROOT / cfg['legacy_final'] / f'{split}.vi', ROOT / cfg['legacy_final'] / f'{split}.zh'):
            key = (norm.key(vi, 'vi'), norm.key(zh, 'zh'))
            i = key_to_index.get(key)
            anchors[split].append({'old_line': line, 'vi': vi, 'zh': zh, 'index': i})
            old_exact[(vi, zh)] = None
        matches[split] = sum(a['index'] is not None for a in anchors[split])
    del key_to_index
    # Infer the historical source group only for the 10k evaluation anchors.
    for rec in rows(ROOT / cfg['legacy_processed'] / 'decontaminated.jsonl'):
        pair = (rec['vi'], rec['zh'])
        if pair in old_exact:
            old_exact[pair] = rec['source']
    goals = {}
    for split in ['test', 'val']:
        goals[split] = Counter()
        for a in anchors[split]:
            source = old_exact[(a['vi'], a['zh'])]
            if source is None:
                raise ValueError('Historical anchor missing from legacy corpus')
            goals[split][(source, length_bin(a['vi']))] += 1
    selected, assigned = {'test': [], 'val': []}, {}
    reasons = {}
    for split in ['test', 'val']:
        for a in anchors[split]:
            i = a['index']
            if i is None:
                reasons[(split, a['old_line'])] = 'no_longer_in_eligible_corpus'
            elif find(i) in assigned:
                reasons[(split, a['old_line'])] = 'component_already_reserved_' + assigned[find(i)]
            else:
                assigned[find(i)] = split
                selected[split].append(i)
                reasons[(split, a['old_line'])] = 'retained_anchor'
    available = sorted((root for root in components if root not in assigned),
                       key=lambda root: hashlib.sha256(f'{cfg["seed"]}:{group_id[root]}'.encode()).digest())
    # One deterministic representative per available component; first prefer
    # source/length deficits relative to the old evaluation distribution.
    buckets = defaultdict(list)
    for root in available:
        representative = min(components[root], key=lambda i: records[i]['record_id'])
        rec = records[representative]
        buckets[(source_group(rec), length_bin(rec['vi']))].append((root, representative))
    cursors = Counter()
    for split, target in [('test', cfg['test_size']), ('val', cfg['validation_size'])]:
        counts = Counter((source_group(records[i]), length_bin(records[i]['vi'])) for i in selected[split])
        while len(selected[split]) < target:
            options = [k for k in buckets if cursors[k] < len(buckets[k])]
            if not options:
                raise ValueError('Not enough independent groups for requested evaluation size')
            bucket = min(options, key=lambda k: (-(goals[split][k] - counts[k]), k))
            root, i = buckets[bucket][cursors[bucket]]
            cursors[bucket] += 1
            assigned[root] = split
            selected[split].append(i)
            counts[bucket] += 1
    chosen = {i: split for split in ['test', 'val'] for i in selected[split]}
    selected['train'] = [i for i in range(len(records)) if find(i) not in assigned]
    random.Random(cfg['seed']).shuffle(selected['train'])
    for i in selected['train']:
        chosen[i] = 'train'
    final.mkdir(parents=True, exist_ok=True)
    memberships = (evidence / 'split_membership.jsonl').open('w', encoding='utf-8', newline='\n')
    with (evidence / 'quarantine.jsonl').open('w', encoding='utf-8', newline='\n') as quarantine:
        for i, rec in enumerate(records):
            if i not in chosen:
                quarantine.write(json_line(dict(rec, group_id=group_id[find(i)], reason='unselected_partner_in_evaluation_component')))
    for split in ['train', 'val', 'test']:
        # Preserve old anchor order; replacements follow deterministic selection.
        write_pairs((records[i] for i in selected[split]), final, split)
        for line, i in enumerate(selected[split], 1):
            rec = records[i]
            memberships.write(json_line({'split': split, 'line': line, 'record_id': rec['record_id'],
                                        'group_id': group_id[find(i)], 'source': rec['source'], 'source_line': rec['source_line']}))
    memberships.close()
    with (evidence / 'split_replacements.csv').open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['split', 'old_line', 'reason'])
        writer.writeheader()
        for (split, line), reason in reasons.items():
            writer.writerow({'split': split, 'old_line': line, 'reason': reason})
    with (evidence / 'variant_groups.jsonl').open('w', encoding='utf-8', newline='\n') as out:
        for root, members in sorted(components.items(), key=lambda item: group_id[item[0]]):
            if len(members) > 1:
                out.write(json_line({'group_id': group_id[root], 'size': len(members),
                                    'split': assigned.get(root, 'train'), 'semantic_review': 'not_performed',
                                    'records': [dict(records[i], disposition=chosen.get(i, 'quarantine')) for i in members]}))
    distributions = []
    for split in ['val', 'test']:
        actual = Counter((source_group(records[i]), length_bin(records[i]['vi'])) for i in selected[split])
        for key in sorted(set(actual) | set(goals[split])):
            distributions.append({'split': split, 'source_group': key[0], 'vi_length_bin': key[1],
                                  'old_count': goals[split][key], 'new_count': actual[key],
                                  'delta_percentage_points': (actual[key] - goals[split][key]) * 100 / len(selected[split])})
    stats = {'eligible': len(records), 'train': len(selected['train']), 'val': len(selected['val']), 'test': len(selected['test']),
             'quarantine': len(records) - len(chosen), 'components': len(components),
             'largest_component': max(map(len, components.values())), 'matched_old_anchors': matches,
             'retained_old_anchors': {split: sum(reason == 'retained_anchor' for (s, _), reason in reasons.items() if s == split) for split in ['test', 'val']},
             'distribution': distributions, 'semantic_review': cfg['semantic_review']}
    assert stats['eligible'] == stats['train'] + stats['val'] + stats['test'] + stats['quarantine']
    write_json(evidence / 'split_summary.json', stats)
    return stats


def execute(stage, cfg, run_id):
    proc, final, evidence = run_paths(cfg, run_id)
    for folder in [proc, final, evidence]:
        folder.mkdir(parents=True, exist_ok=True)
    marker = evidence / f'{stage}.json'
    if marker.exists():
        raise FileExistsError(f'Stage already complete: {marker}')
    start = time.monotonic()
    norm = Normalizer(cfg)
    counts = Counter()
    output = proc / FILES[stage] if stage != 'T10' else None
    log_path = evidence / f'{stage}_decisions.jsonl'
    if stage in ['T01', 'T02']:
        source_counts = []
        with output.open('w', encoding='utf-8', newline='\n') as out:
            if stage == 'T01':
                for source, stem in SOURCES[:3]:
                    total = kept = 0
                    for rec in raw_records(cfg, source, stem):
                        total += 1
                        if rec['vi'] and rec['zh']:
                            out.write(json_line(rec))
                            kept += 1
                    source_counts.append({'corpus': source, 'raw_pairs': total, 'valid_nonempty': kept, 'selected_pairs': kept, 'empty_pairs': total-kept})
                    counts['input'] += total
                    counts['kept'] += kept
            else:
                source, stem = SOURCES[-1]
                total = valid = 0
                for total, vi, zh in pairs(ROOT / cfg['raw_root'] / (stem + '.vi'), ROOT / cfg['raw_root'] / (stem + '.zh')):
                    valid += bool(vi.strip() and zh.strip())
                sample = random.Random(cfg['seed']).sample(range(valid), min(cfg['sample_size'], valid))
                positions = set(sample)
                sampled = {}
                rank = 0
                for rec in raw_records(cfg, source, stem):
                    if rec['vi'] and rec['zh']:
                        if rank in positions:
                            sampled[rank] = rec
                        rank += 1
                for rank in sample:
                    out.write(json_line(sampled[rank]))
                counts.update(input=total, kept=len(sample))
                source_counts.append({'corpus': source, 'raw_pairs': total, 'valid_nonempty': valid, 'selected_pairs': len(sample), 'empty_pairs': total-valid})
                write_json(evidence / 'ccmatrix_sample_indices.json', sample)
        write_json(evidence / f'{stage}_sources.json', source_counts)
        write_pairs(rows(output), proc, 'gold_merged' if stage == 'T01' else 'ccmatrix_1M')
    elif stage == 'T03':
        with output.open('w', encoding='utf-8', newline='\n') as out:
            for previous in ['T01', 'T02']:
                for rec in rows(proc / FILES[previous]):
                    out.write(json_line(rec))
                    counts['input'] += 1
                    counts['kept'] += 1
    elif stage == 'T10':
        stats = split_groups(cfg, proc, final, evidence, norm)
        counts.update(input=stats['eligible'], kept=stats['train'] + stats['val'] + stats['test'], quarantine=stats['quarantine'])
    else:
        previous = STAGES[STAGES.index(stage) - 1]
        seen = {}
        if stage == 'T08':
            rank = {name: i for i, (name, _) in enumerate(SOURCES)}
            for rec in rows(proc / FILES[previous]):
                key = norm.pair(rec)
                priority = (rank[rec['source']], rec['source_line'], rec['record_id'])
                if key not in seen or priority < seen[key]:
                    seen[key] = priority
        if stage == 'T09':
            indices, mapping = benchmark(cfg, norm)
            combined = {lang: indices[('dev', lang)] | indices[('devtest', lang)] for lang in ['vi', 'zh']}
            write_json(evidence / 'benchmark_mapping.json', mapping)
        model = fasttext.load_model(str(ROOT / cfg['legacy_processed'] / 'lid.176.bin')) if stage == 'T07' else None
        legacy_clean = iter(rows(ROOT / cfg['legacy_processed'] / 'cleaned_merged.jsonl')) if stage == 'T05' else None
        legacy_normalized = iter(rows(ROOT / cfg['legacy_processed'] / 'normalized_merged.jsonl')) if stage == 'T05' else None
        old_rule_mismatch = new_rule_mismatch = legacy_line = skipped_legacy_controls = 0
        with output.open('w', encoding='utf-8', newline='\n') as out, log_path.open('w', encoding='utf-8', newline='\n') as decisions:
            for line, original in enumerate(rows(proc / FILES[previous]), 1):
                rec, reason = dict(original), ''
                counts['input'] += 1
                if stage == 'T04':
                    rec.update(vi=clean(rec['vi']), zh=clean(rec['zh']))
                    if invalid_control(rec):
                        reason = 'invalid_control_character'
                    elif not rec['vi'] or not rec['zh']:
                        reason = 'empty_after_cleaning'
                elif stage == 'T05':
                    before, historical = next(legacy_clean), next(legacy_normalized)
                    legacy_line += 1
                    while invalid_control(before):
                        skipped_legacy_controls += 1
                        old_values = {'vi': unicodedata.normalize('NFC', before['vi']),
                                      'zh': unicodedata.normalize('NFC', norm.cc.convert(before['zh']))}
                        new_values = {lang: norm.text(before[lang], lang) for lang in ['vi', 'zh']}
                        old_diff = any(old_values[lang] != historical[lang] for lang in ['vi', 'zh'])
                        new_diff = any(new_values[lang] != historical[lang] for lang in ['vi', 'zh'])
                        old_rule_mismatch += old_diff
                        new_rule_mismatch += new_diff
                        if old_diff or new_diff:
                            decisions.write(json_line({'input_line': legacy_line, 'before': before, 'historical': historical,
                                                       'replayed_old_rule': old_values, 'new': new_values, 'old_rule_differs': old_diff,
                                                       'disposition': 'excluded_by_T04_control_filter'}))
                        before, historical = next(legacy_clean), next(legacy_normalized)
                        legacy_line += 1
                    assert (original['vi'], original['zh']) == (before['vi'], before['zh']), f'Legacy row alignment differs at {line}'
                    old_replay = {'vi': unicodedata.normalize('NFC', before['vi']), 'zh': unicodedata.normalize('NFC', norm.cc.convert(before['zh']))}
                    old_diff = any(old_replay[lang] != historical[lang] for lang in ['vi', 'zh'])
                    old_rule_mismatch += old_diff
                    rec.update(vi=norm.text(rec['vi'], 'vi'), zh=norm.text(rec['zh'], 'zh'))
                    counts['extra_fixed_point_changes'] += rec['zh'] != unicodedata.normalize('NFC', re.sub(r'\s+', ' ', old_replay['zh']).strip())
                    new_diff = any(rec[lang] != historical[lang] for lang in ['vi', 'zh'])
                    new_rule_mismatch += new_diff
                    if old_diff or new_diff:
                        decisions.write(json_line({'record_id': rec['record_id'], 'input_line': legacy_line, 'before': before,
                                                   'historical': historical, 'replayed_old_rule': old_replay,
                                                   'new': {'vi': rec['vi'], 'zh': rec['zh']}, 'old_rule_differs': old_diff}))
                    for lang in ['vi', 'zh']:
                        assert norm.text(rec[lang], lang) == rec[lang], f'Non-idempotent normalization: {line}/{lang}'
                        key = norm.key(rec[lang], lang)
                        assert norm.key(key, lang) == key, f'Non-idempotent comparison key: {line}/{lang}'
                        counts[f'{lang}_changed'] += rec[lang] != original[lang]
                elif stage == 'T06':
                    vl, zl = len(rec['vi'].split()), len(rec['zh'])
                    reason = 'length' if not (2 <= vl <= 150 and 2 <= zl <= 150) else 'ratio' if not 0.3 <= zl / vl <= 3 else ''
                elif stage == 'T07':
                    lv, pv = model.predict(rec['vi'], k=1)
                    lz, pz = model.predict(rec['zh'], k=1)
                    reason = 'wrong_language' if (lv[0] != '__label__vi' or lz[0] != '__label__zh') else 'low_confidence' if min(float(pv[0]), float(pz[0])) < 0.8 else ''
                elif stage == 'T08':
                    representative = seen[norm.pair(rec)][2]
                    if rec['record_id'] != representative:
                        reason = 'normalized_duplicate'
                        rec['representative_id'] = representative
                elif stage == 'T09':
                    matched = [lang for lang in ['vi', 'zh'] if norm.key(rec[lang], lang) in combined[lang]]
                    if matched:
                        reason = 'flores_normalized_match'
                        rec['matched_languages'] = matched
                if reason:
                    counts[reason] += 1
                    decisions.write(json_line(dict(rec, reason=reason)))
                else:
                    out.write(json_line(rec))
                    counts['kept'] += 1
                if line % 200000 == 0:
                    print(stage, line, dict(counts), flush=True)
        if stage == 'T05':
            # Remaining historical records must be control-contaminated rows now excluded at T04.
            for before in legacy_clean:
                historical = next(legacy_normalized)
                assert invalid_control(before), 'Unexpected missing clean row'
                legacy_line += 1
                skipped_legacy_controls += 1
                old_values = {'vi': unicodedata.normalize('NFC', before['vi']), 'zh': unicodedata.normalize('NFC', norm.cc.convert(before['zh']))}
                new_values = {lang: norm.text(before[lang], lang) for lang in ['vi', 'zh']}
                old_rule_mismatch += any(old_values[lang] != historical[lang] for lang in ['vi', 'zh'])
                new_rule_mismatch += any(new_values[lang] != historical[lang] for lang in ['vi', 'zh'])
            assert next(legacy_normalized, None) is None
            write_json(evidence / 't05_migration_summary.json', {'compared': legacy_line, 'new_T05_input': counts['input'],
                       'historical_rows_excluded_for_controls': skipped_legacy_controls, 'legacy_rule_replay_differences': old_rule_mismatch,
                       'new_rule_vs_historical_differences': new_rule_mismatch, 'new_normalization_idempotence_failures': 0,
                       'extra_fixed_point_changes': counts['extra_fixed_point_changes'],
                       'semantic_review': cfg['semantic_review'], 'historical_cause': 'not_proven_missing_original_environment'})
    counts['removed'] = counts['input'] - counts['kept']
    outputs = [output] if output else sorted(final.glob('*')) + [evidence / 'split_membership.jsonl', evidence / 'quarantine.jsonl', evidence / 'variant_groups.jsonl']
    info = {'stage': stage, 'counts': dict(counts), 'outputs': [fingerprint(p) for p in outputs], 'elapsed_seconds': round(time.monotonic() - start, 3)}
    write_json(marker, info)
    print(stage, 'complete', dict(counts), flush=True)


def main(only_stage=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    parser.add_argument('--run-id', default='run-a', choices=['run-a', 'run-b'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=STAGES, help='run only this stage (default: all stages in order)')
    args = parser.parse_args()
    only_stage = only_stage or args.stage
    cfg = config(args.config)
    # Validate pinned environment/resources and original LID model before writing.
    import importlib.metadata as metadata
    import platform
    assert platform.python_version() == cfg['python_version']
    for name, version in cfg['packages'].items():
        assert metadata.version(name) == version
    manifest = json.loads((ROOT / cfg['snapshot_root'] / 'input_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        if entry['path'].startswith(('data/raw/', 'data/benchmark/')) or entry['path'] == cfg['legacy_processed'] + '/lid.176.bin':
            assert fingerprint(ROOT / entry['path'])['sha256'] == entry['sha256'], entry['path']
    for entry in json.loads((ROOT / cfg['opencc_resources'] / 'checksums.json').read_text(encoding='utf-8')):
        assert fingerprint(ROOT / entry['path'])['sha256'] == entry['sha256'], entry['path']
    for stage in [only_stage] if only_stage else STAGES:
        marker = run_paths(cfg, args.run_id)[2] / f'{stage}.json'
        if args.resume and marker.exists():
            data = json.loads(marker.read_text(encoding='utf-8'))
            for entry in data['outputs']:
                assert fingerprint(ROOT / entry['path'])['sha256'] == entry['sha256']
            print('Verified completed stage', stage, flush=True)
        else:
            execute(stage, cfg, args.run_id)


if __name__ == '__main__':
    main()
