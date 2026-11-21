"""Independent recount and replay checks for the separate rebuilt corpus."""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from itertools import chain, zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasttext
import numpy as np

from freeze_utils import ROOT, config, fingerprint, pairs, rows, run_paths, write_json
from rebuild_dataset import FILES, PATTERNS, SOURCES, INVALID_CONTROL, benchmark, clean, invalid_control
from text_normalization import Normalizer


def csv_out(evidence, name, records, fields=None):
    records = list(records)
    with (evidence / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def check_sequence(actual, expected, label):
    count = 0
    for count, (a, b) in enumerate(zip_longest(actual, expected), 1):
        if a != b:
            raise ValueError(f'{label} differs at output row {count}')
    return count


def source_audit(cfg, proc, evidence):
    from rebuild_dataset import raw_records
    # Source IDs are checked against the hashed source files and original rows.
    def gold_expected():
        for source, stem in SOURCES[:3]:
            for rec in raw_records(cfg, source, stem):
                if rec['vi'] and rec['zh']:
                    yield rec
    gold_count = check_sequence(rows(proc / FILES['T01']), gold_expected(), 'T01')
    source_counts = json.loads((evidence / 'T02_sources.json').read_text(encoding='utf-8'))[0]
    sampled = random.Random(cfg['seed']).sample(range(source_counts['valid_nonempty']), source_counts['selected_pairs'])
    stored = list(rows(proc / FILES['T02']))
    wanted = {rank: stored[position] for position, rank in enumerate(sampled)}
    checked = valid_rank = 0
    for rec in raw_records(cfg, *SOURCES[-1]):
        if rec['vi'] and rec['zh']:
            if valid_rank in wanted:
                if rec != wanted[valid_rank]:
                    raise ValueError(f'T02 sample differs at valid rank {valid_rank}')
                checked += 1
            valid_rank += 1
    assert valid_rank == source_counts['valid_nonempty'] and checked == len(sampled)
    check_sequence(rows(proc / FILES['T03']), chain(rows(proc / FILES['T01']), rows(proc / FILES['T02'])), 'T03')
    for stage, stem in [('T01', 'gold_merged'), ('T02', 'ccmatrix_1M')]:
        check_sequence(((v, z) for _, v, z in pairs(proc / (stem + '.vi'), proc / (stem + '.zh'))),
                       ((r['vi'], r['zh']) for r in rows(proc / FILES[stage])), stage + ' paired text')
    report = {'T01_gold_pairs_verified': gold_count, 'T02_sample_pairs_verified': checked,
              'T02_seed': cfg['seed'], 'sampling_population': valid_rank, 'T03_merge_verified': True}
    write_json(evidence / 'source_replay.json', report)
    print('Source replay verified:', report, flush=True)
    return report


def replay_filters(cfg, proc, evidence, norm):
    replay = {}
    def cleaned():
        for r in rows(proc / FILES['T03']):
            vi, zh = clean(r['vi']), clean(r['zh'])
            if vi and zh and not invalid_control({'vi': vi, 'zh': zh}):
                yield dict(r, vi=vi, zh=zh)
    check_sequence(rows(proc / FILES['T04']), cleaned(), 'T04')
    replay['T04'] = 0
    def normalized():
        for r in rows(proc / FILES['T04']):
            yield dict(r, vi=norm.text(r['vi'], 'vi'), zh=norm.text(r['zh'], 'zh'))
    check_sequence(rows(proc / FILES['T05']), normalized(), 'T05')
    replay['T05'] = 0
    def heuristics():
        for r in rows(proc / FILES['T05']):
            vi, zh = len(r['vi'].split()), len(r['zh'])
            if 2 <= vi <= 150 and 2 <= zh <= 150 and 0.3 <= zh / vi <= 3:
                yield r
    check_sequence(rows(proc / FILES['T06']), heuristics(), 'T06')
    replay['T06'] = 0
    model = fasttext.load_model(str(ROOT / cfg['legacy_processed'] / 'lid.176.bin'))
    parity_examples = []
    def language():
        for line, r in enumerate(rows(proc / FILES['T06']), 1):
            predictions = {}
            for lang in ['vi', 'zh']:
                probability, label = model.f.predict(r[lang] + '\n', 1, 0.0, 'strict')[0]
                predictions[lang] = (label, probability)
                if len(parity_examples) < 20 and abs(probability - 0.8) < 0.0001:
                    public_label, public_probability = model.predict(r[lang], k=1)
                    assert public_label[0] == label
                    assert (float(public_probability[0]) >= 0.8) == (probability >= 0.8)
                    parity_examples.append({'line': line, 'language': lang, 'probability': probability, 'parity': True})
            if all(predictions[lang][0] == '__label__' + lang and predictions[lang][1] >= 0.8 for lang in ['vi', 'zh']):
                yield r
            if line % 200000 == 0:
                print('Independent FastText replay', line, flush=True)
    check_sequence(rows(proc / FILES['T07']), language(), 'T07 C++/public decisions')
    write_json(evidence / 'fasttext_threshold_parity.json', parity_examples)
    replay['T07'] = 0
    del model
    representatives = {}
    for r in rows(proc / FILES['T08']):
        key = norm.pair(r)
        assert key not in representatives, 'T08 normalized duplicate'
        representatives[key] = r
    source_rank = {name: i for i, (name, _) in enumerate(SOURCES)}
    def deduped():
        for r in rows(proc / FILES['T07']):
            best = representatives[norm.pair(r)]
            assert (source_rank[best['source']], best['source_line'], best['record_id']) <= (source_rank[r['source']], r['source_line'], r['record_id'])
            if best['record_id'] == r['record_id']:
                yield r
    check_sequence(rows(proc / FILES['T08']), deduped(), 'T08 representative priority')
    replay['T08'] = 0
    del representatives
    indices, _ = benchmark(cfg, norm)
    combined = {lang: indices[('dev', lang)] | indices[('devtest', lang)] for lang in ['vi', 'zh']}
    check_sequence(rows(proc / FILES['T09']),
                   (r for r in rows(proc / FILES['T08']) if all(norm.key(r[lang], lang) not in combined[lang] for lang in ['vi', 'zh'])), 'T09')
    replay['T09'] = 0
    return replay


def audit(cfg, run_id, do_replay=True, do_sources=True):
    proc, final, evidence = run_paths(cfg, run_id)
    norm = Normalizer(cfg)
    if do_sources:
        source_audit(cfg, proc, evidence)
    replay = replay_filters(cfg, proc, evidence, norm) if do_replay else {}
    eligible = {r['record_id']: r for r in rows(proc / FILES['T09'])}
    assert len(eligible) == fingerprint(proc / FILES['T09'])['lines']
    membership = iter(rows(evidence / 'split_membership.jsonl'))
    indices, mapping = benchmark(cfg, norm)
    seen_ids, side_sets, pair_sets = set(), {}, {}
    stats, lengths, flores, detailed_sources = [], [], [], []
    groups = Counter()
    global_noise = Counter()
    vocabulary = []
    for split in ['train', 'val', 'test']:
        for lang in ['vi', 'zh']:
            with (final / f'{split}.{lang}').open('rb') as stream:
                assert not stream.read(3).startswith(b'\xef\xbb\xbf'), 'BOM is forbidden'
                stream.seek(0)
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    assert b'\r' not in chunk and b'\x00' not in chunk, 'CR or null byte'
        vi_set, zh_set, pair_set = set(), set(), set()
        noise = Counter()
        split_sources = Counter()
        vi_lengths, zh_lengths, ratios = [], [], []
        flores_counts = Counter()
        vi_vocab, zh_vocab = set(), set()
        for line, vi, zh in pairs(final / f'{split}.vi', final / f'{split}.zh'):
            member = next(membership)
            assert member['split'] == split and member['line'] == line
            rid = member['record_id']
            assert rid not in seen_ids, 'Record reused in multiple positions'
            seen_ids.add(rid)
            r = eligible[rid]
            assert (vi, zh) == (r['vi'], r['zh']), 'Final output does not match provenance'
            assert member['source'] == r['source'] and member['source_line'] == r['source_line']
            for lang, value in [('vi', vi), ('zh', zh)]:
                assert norm.text(value, lang) == value, 'Final text is not a normalization fixed point'
                key = norm.key(value, lang)
                assert norm.key(key, lang) == key
            vk, zk = norm.key(vi, 'vi'), norm.key(zh, 'zh')
            assert (vk, zk) not in pair_set, 'Duplicate normalized pair'
            pair_set.add((vk, zk))
            vi_set.add(vk)
            zh_set.add(zk)
            vl, zl = len(vi.split()), len(zh)
            assert 2 <= vl <= 150 and 2 <= zl <= 150 and 0.3 <= zl / vl <= 3
            noise['empty'] += int(not vi or not zh)
            noise['invalid_control_character'] += int(bool(INVALID_CONTROL.search(vi) or INVALID_CONTROL.search(zh)))
            for label, pattern in zip(['html', 'url', 'email', 'music', 'zero_width'], PATTERNS[:5]):
                noise[label] += int(bool(pattern.search(vi) or pattern.search(zh)))
            vi_lengths.append(vl)
            zh_lengths.append(zl)
            ratios.append(zl / vl)
            vi_vocab.update(vi.split())
            zh_vocab.update(zh)
            split_sources[r['source']] += 1
            for (part, lang), sentences in indices.items():
                flores_counts[(part, lang)] += int((vk if lang == 'vi' else zk) in sentences)
        assert all(n == 0 for n in noise.values()), dict(noise)
        n = len(vi_lengths)
        if split != 'train':
            assert n == cfg['validation_size' if split == 'val' else 'test_size']
        assert n > 0
        gold = sum(split_sources[s] for s in ['alt', 'ted2020', 'wikimatrix'])
        stats.append(dict(split=split, pairs=n, aligned=True, exact_duplicates=0, normalized_duplicates=0,
                          empty=noise['empty'], html=noise['html'], url=noise['url'], email=noise['email'],
                          music=noise['music'], zero_width=noise['zero_width'], invalid_control_characters=noise['invalid_control_character'],
                          gold_pairs=gold, ccmatrix_pairs=split_sources['ccmatrix']))
        groups.update(gold=gold, ccmatrix=split_sources['ccmatrix'])
        for source in [name for name, _ in SOURCES]:
            detailed_sources.append({'split': split, 'source': source, 'pairs': split_sources[source], 'percent': split_sources[source] * 100 / n})
        for measure, values in [('vi_units', vi_lengths), ('zh_characters', zh_lengths), ('zh_vi_ratio', ratios)]:
            array = np.asarray(values)
            lengths.append({'split': split, 'measure': measure, 'min': float(array.min()), 'median': float(np.median(array)),
                            'mean': float(array.mean()), 'p95': float(np.percentile(array, 95)), 'max': float(array.max())})
        for lang, types, values in [('vi', vi_vocab, vi_lengths), ('zh', zh_vocab, zh_lengths)]:
            vocabulary.append({'split': split, 'language': lang, 'types': len(types), 'units': sum(values), 'ttr': len(types) / sum(values),
                               'subword_compression': 'N/A: tokenizer not trained', 'tokenizer_training_scope': 'train_only'})
        bins = list(range(0, 161, 10))
        write_json(evidence / f'{split}_length_histograms.json', {lang: {'bins': bins, 'counts': np.histogram(values, bins=bins)[0].tolist()} for lang, values in [('vi', vi_lengths), ('zh', zh_lengths)]})
        pair_sets[split], side_sets[split] = pair_set, {'vi': vi_set, 'zh': zh_set}
        for (part, lang), count in flores_counts.items():
            flores.append({'split': split, 'benchmark_split': part, 'language': lang, 'matching_rows': count})
        print('Final audited', split, n, flush=True)
    assert next(membership, None) is None
    quarantined = set()
    for r in rows(evidence / 'quarantine.jsonl'):
        rid = r['record_id']
        assert rid not in seen_ids and rid not in quarantined
        assert all(r[field] == eligible[rid][field] for field in ['vi', 'zh', 'source', 'source_line'])
        quarantined.add(rid)
    assert seen_ids | quarantined == set(eligible), 'Eligible pair lost or duplicated'
    del eligible
    overlaps = []
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        row = {'split_a': a, 'split_b': b, 'exact_pair_overlap': len(pair_sets[a] & pair_sets[b]),
               'normalized_pair_overlap': len(pair_sets[a] & pair_sets[b]),
               'normalized_vi_overlap_unique': len(side_sets[a]['vi'] & side_sets[b]['vi']),
               'normalized_zh_overlap_unique': len(side_sets[a]['zh'] & side_sets[b]['zh'])}
        assert all(row[key] == 0 for key in row if key not in ['split_a', 'split_b']), row
        overlaps.append(row)
    assert len(flores) == 12 and all(r['matching_rows'] == 0 for r in flores)
    replay['T10'] = 0
    csv_out(evidence, 'split_audit.csv', stats)
    csv_out(evidence, 'length_statistics.csv', lengths)
    csv_out(evidence, 'cross_split_overlap.csv', overlaps)
    csv_out(evidence, 'flores_overlap.csv', flores)
    csv_out(evidence, 'source_groups.csv', [{'group': g, 'pairs': n, 'percent': n * 100 / sum(groups.values())} for g, n in groups.items()])
    csv_out(evidence, 'source_details.csv', detailed_sources)
    csv_out(evidence, 'vocabulary_statistics.csv', vocabulary)
    source_counts = []
    for stage in ['T01', 'T02']:
        for rec in json.loads((evidence / f'{stage}_sources.json').read_text(encoding='utf-8')):
            names = {'alt': 'ALT', 'ted2020': 'TED2020', 'wikimatrix': 'WikiMatrix', 'ccmatrix': 'CCMatrix'}
            source_counts.append(dict(rec, corpus=names[rec['corpus']], sampling_seed=cfg['seed'] if stage == 'T02' else '', files_aligned=True))
    csv_out(evidence, 'source_counts.csv', source_counts)
    stages, reasons = [], []
    initial = fingerprint(proc / FILES['T03'])['lines']
    for stage in ['T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10']:
        counts = json.loads((evidence / f'{stage}.json').read_text(encoding='utf-8'))['counts']
        n_in, kept, removed = counts['input'], counts['kept'], counts['removed']
        assert n_in == kept + removed
        actual = fingerprint(proc / FILES[stage])['lines'] if stage != 'T10' else sum(r['pairs'] for r in stats)
        assert kept == actual
        stages.append({'stage': stage, 'file': FILES.get(stage, 'final/*'), 'input': n_in, 'kept': kept, 'removed': removed,
                       'removed_percent': removed * 100 / n_in, 'retained_total_percent': kept * 100 / initial})
        for reason in ['empty_after_cleaning', 'invalid_control_character', 'length', 'ratio', 'wrong_language', 'low_confidence', 'normalized_duplicate', 'flores_normalized_match', 'quarantine']:
            if counts.get(reason):
                reasons.append({'reason': reason, 'removed': counts[reason]})
    csv_out(evidence, 'pipeline_counts.csv', stages)
    csv_out(evidence, 'filter_reasons.csv', reasons)
    csv_out(evidence, 'pipeline_replay.csv', [{'stage': stage, 'different_records': 0, 'status': 'match'} for stage in replay])
    write_json(evidence / 'normalization_changes.json', {lang: json.loads((evidence / 'T05.json').read_text(encoding='utf-8'))['counts'].get(lang + '_changed', 0) for lang in ['vi', 'zh']})
    summary = {'status': 'technical_pass_semantics_not_reviewed', 'seed': cfg['seed'], 'semantic_review': cfg['semantic_review'],
               'pipeline_replay': replay, 'pipeline_replay_status': 'match', 'source_counts': source_counts,
               'pipeline': stages, 'splits': stats, 'overlaps': overlaps, 'flores': flores, 'quarantine': len(quarantined),
               'normalization': cfg['normalization'], 'all_pair_overlap_zero': True, 'all_single_side_overlap_zero': True,
               'benchmark_mapping': mapping, 'limitations': ['No human semantic review', 'No exhaustive near-duplicate or paraphrase detection', 'Subword compression not measured until tokenizer preparation']}
    write_json(evidence / 'audit_summary.json', summary)
    write_json(evidence / 'audit_issues.json', [])
    print('PASS: independent technical audit; semantic review not performed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    parser.add_argument('--run-id', default='run-a')
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--skip-sources', action='store_true')
    args = parser.parse_args()
    cfg = config(args.config)
    if args.sources_only:
        proc, _, evidence = run_paths(cfg, args.run_id)
        source_audit(cfg, proc, evidence)
    else:
        audit(cfg, args.run_id, do_sources=not args.skip_sources)


if __name__ == '__main__':
    main()
