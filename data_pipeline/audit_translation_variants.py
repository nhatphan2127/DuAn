"""Find shared normalized sentences with different partners across splits."""
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'chapter4' / 'evidence'
CC = OpenCC('t2s')


def normalized(text, lang):
    text = unicodedata.normalize('NFC', text)
    return (CC.convert(text) if lang == 'zh' else text).strip().lower()


def pairs(split):
    with (ROOT / 'data' / 'final' / (split + '.vi')).open(encoding='utf-8') as vi_file, \
         (ROOT / 'data' / 'final' / (split + '.zh')).open(encoding='utf-8') as zh_file:
        for line, (vi, zh) in enumerate(zip(vi_file, zh_file, strict=True), 1):
            yield line, vi.rstrip('\n\r'), zh.rstrip('\n\r')


def main():
    maps = {split: {lang: defaultdict(dict) for lang in ['vi', 'zh']} for split in ['val', 'test', 'train']}
    for split in ['val', 'test']:
        for line, vi, zh in pairs(split):
            vn, zn = normalized(vi, 'vi'), normalized(zh, 'zh')
            record = {'split': split, 'line': line, 'vi': vi, 'zh': zh}
            maps[split]['vi'][vn].setdefault(zn, record)
            maps[split]['zh'][zn].setdefault(vn, record)
    keys = {lang: set(maps['val'][lang]) | set(maps['test'][lang]) for lang in ['vi', 'zh']}
    for line, vi, zh in pairs('train'):
        vn, zn = normalized(vi, 'vi'), normalized(zh, 'zh')
        record = {'split': 'train', 'line': line, 'vi': vi, 'zh': zh}
        if vn in keys['vi']:
            maps['train']['vi'][vn].setdefault(zn, record)
        if zn in keys['zh']:
            maps['train']['zh'][zn].setdefault(vn, record)
        if line % 300000 == 0:
            print('Train rows', line, flush=True)
    rows, examples = [], []
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for lang in ['vi', 'zh']:
            common = sorted(set(maps[a][lang]) & set(maps[b][lang]))
            different = 0
            saved = 0
            for text in common:
                left, right = maps[a][lang][text], maps[b][lang][text]
                variant = next(((lk, rk) for lk in left for rk in right if lk != rk), None)
                if variant is not None:
                    different += 1
                    if saved < 5:
                        lk, rk = variant
                        examples.append({'split_a': a, 'split_b': b, 'shared_side': lang,
                                         'normalized_shared_text': text,
                                         'record_a': left[lk], 'record_b': right[rk]})
                        saved += 1
            rows.append({'split_a': a, 'split_b': b, 'shared_side': lang,
                         'shared_sentences_unique': len(common),
                         'shared_sentences_with_different_partner': different})
    with (OUT / 'translation_variants.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'translation_variant_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    from audit_rebuilt_variants import main as rebuilt_variants
    rebuilt_variants()
