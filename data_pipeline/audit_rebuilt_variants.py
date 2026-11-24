"""Describe structural translation variants without inferring semantic correctness."""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_utils import config, pairs, rows, run_paths, write_json
from text_normalization import Normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    parser.add_argument('--run-id', default='run-a')
    args = parser.parse_args()
    cfg = config(args.config)
    _, final, evidence = run_paths(cfg, args.run_id)
    norm = Normalizer(cfg)
    types, dispositions = Counter(), Counter()
    largest = 0
    with (evidence / 'variant_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group_id', 'record_id', 'source', 'category', 'disposition', 'semantic_review', 'vi', 'zh'])
        writer.writeheader()
        for group in rows(evidence / 'variant_groups.jsonl'):
            vn, zn = Counter(), Counter()
            for rec in group['records']:
                vn[norm.key(rec['vi'], 'vi')] += 1
                zn[norm.key(rec['zh'], 'zh')] += 1
            category = 'both_sides_shared' if max(vn.values()) > 1 and max(zn.values()) > 1 else 'shared_vi' if max(vn.values()) > 1 else 'shared_zh'
            types[category] += 1
            largest = max(largest, group['size'])
            for rec in group['records']:
                short = len(rec['vi'].split()) <= 5 or len(rec['zh']) <= 8
                dispositions[rec['disposition']] += 1
                writer.writerow({**{k: rec[k] for k in ['record_id', 'source', 'vi', 'zh']}, 'group_id': group['group_id'],
                                 'category': category + (';short_surface' if short else ''), 'disposition': rec['disposition'], 'semantic_review': 'not_performed'})
    side_sets, strata = {}, {}
    for split in ['train', 'val', 'test']:
        sides = {'vi': set(), 'zh': set()}
        strata[split] = defaultdict(list)
        member_iter = (m for m in rows(evidence / 'split_membership.jsonl') if m['split'] == split)
        for line, vi, zh in pairs(final / f'{split}.vi', final / f'{split}.zh'):
            sides['vi'].add(norm.key(vi, 'vi'))
            sides['zh'].add(norm.key(zh, 'zh'))
            member = next(member_iter)
            assert member['line'] == line
            n = len(vi.split())
            bucket = '2-10' if n <= 10 else '11-20' if n <= 20 else '21-40' if n <= 40 else '41-150'
            strata[split][(member['source'], bucket)].append(line)
        side_sets[split] = sides
    overlaps = []
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for lang in ['vi', 'zh']:
            shared = len(side_sets[a][lang] & side_sets[b][lang])
            assert shared == 0
            overlaps.append({'split_a': a, 'split_b': b, 'shared_side': lang, 'shared_sentences_unique': shared, 'shared_sentences_with_different_partner': 0})
    with (evidence / 'translation_variants.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overlaps[0]))
        writer.writeheader()
        writer.writerows(overlaps)
    write_json(evidence / 'variant_summary.json', {'groups': sum(types.values()), 'categories': dict(types), 'largest_group': largest,
               'record_disposition': dict(dispositions), 'semantic_review': cfg['semantic_review'],
               'policy': 'Keep valid-by-technical-filters train variants within one component; one representative per evaluation component; other evaluation-component records quarantined'})
    # Reproducible review material is supplied, but no human review is claimed.
    memberships = {'train': [], 'val': [], 'test': []}
    for row in rows(evidence / 'split_membership.jsonl'):
        memberships[row['split']].append(row)
    with (evidence / 'semantic_review_sample.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['split', 'line', 'record_id', 'source', 'vi_length_bin', 'population_in_stratum', 'sample_in_stratum', 'sampling_frame', 'semantic_review', 'vi', 'zh']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, requested in [('train', 600), ('val', 200), ('test', 200)]:
            total = sum(len(v) for v in strata[split].values())
            quotas = {k: int(requested * len(v) / total) for k, v in strata[split].items()}
            remainder_order = sorted(quotas, key=lambda k: (-(requested * len(strata[split][k]) / total - quotas[k]), k))
            for key in remainder_order[:requested - sum(quotas.values())]:
                quotas[key] += 1
            rng = random.Random(cfg['seed'])
            selected = {}
            for key in sorted(quotas):
                for line in rng.sample(strata[split][key], quotas[key]):
                    selected[line] = key
            for line, vi, zh in pairs(final / f'{split}.vi', final / f'{split}.zh'):
                if line in selected:
                    m = memberships[split][line - 1]
                    n = len(vi.split())
                    writer.writerow({'split': split, 'line': line, 'record_id': m['record_id'], 'source': m['source'],
                                     'vi_length_bin': selected[line][1], 'population_in_stratum': len(strata[split][selected[line]]),
                                     'sample_in_stratum': quotas[selected[line]],
                                     'sampling_frame': 'proportional_stratified_source_vi_length_seed_42', 'semantic_review': 'not_performed', 'vi': vi, 'zh': zh})
    print('PASS: zero shared side across splits; variant groups described; semantic review not performed')


if __name__ == '__main__':
    main()
