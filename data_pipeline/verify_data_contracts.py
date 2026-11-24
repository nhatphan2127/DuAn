"""Focused regressions for Unicode normalization and transitive split leakage."""
import argparse
import tempfile
from pathlib import Path

from freeze_utils import ROOT, config, json_line, pairs, rows
from rebuild_dataset import split_groups, invalid_control
from text_normalization import Normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_freeze.json')
    args = parser.parse_args()
    cfg = config(args.config)
    norm = Normalizer(cfg)
    assert invalid_control({'vi': 'Tâm sự\x00 chào em', 'zh': '中文句子'})
    assert not invalid_control({'vi': 'Câu có dấu\t bình thường', 'zh': '中文句子'})
    fixtures = [('vi', '  Vie\u0323\u0302t\tNam ', 'Việt Nam'),
                ('vi', 'CẢM ƠN BẠN.', 'CẢM ƠN BẠN.'),
                ('zh', '我愛越南。', '我爱越南。'), ('zh', '\u3000中國\n越南\u00a0', '中国 越南'),
                ('zh', '宜家你哋睇到嘅係 現存於倫敦國家美術館嘅林布藍作品',
                 '宜家你哋睇到嘅系 现存于伦敦国家美术馆嘅林布蓝作品')]
    for lang, text, expected in fixtures:
        assert norm.text(text, lang) == expected, (lang, text, norm.text(text, lang))
        assert norm.text(expected, lang) == expected
        key = norm.key(text, lang)
        assert norm.key(key, lang) == key
    try:
        norm.text(None, 'vi')
    except TypeError:
        pass
    else:
        raise AssertionError('Null sentence accepted')
    temporary_root = ROOT / cfg['work_root']
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contract-', dir=temporary_root) as temporary:
        base = Path(temporary)
        proc, final, evidence, legacy = [base / name for name in ['proc', 'final', 'evidence', 'legacy']]
        for directory in [proc, final, evidence, legacy]:
            directory.mkdir()
        texts = [('câu A', '甲句'), ('câu A', '乙句'), ('câu B', '乙句'),
                 ('câu C', '丙句'), ('câu D', '丁句'), ('câu E', '戊句')]
        records = [{'record_id': f'{i:064x}', 'source': 'alt', 'source_line': i + 1, 'vi': vi, 'zh': zh}
                   for i, (vi, zh) in enumerate(texts)]
        for path in [proc / 'decontaminated.jsonl', legacy / 'decontaminated.jsonl']:
            path.write_text(''.join(json_line(dict(r, source='gold') if path.parent == legacy else r) for r in records), encoding='utf-8')
        for split, i in [('test', 0), ('val', 2)]:
            for lang in ['vi', 'zh']:
                (legacy / f'{split}.{lang}').write_text(records[i][lang] + '\n', encoding='utf-8')
        test_cfg = dict(cfg, legacy_final=legacy.relative_to(ROOT).as_posix(), legacy_processed=legacy.relative_to(ROOT).as_posix(), validation_size=1, test_size=1)
        summary = split_groups(test_cfg, proc, final, evidence, norm)
        assert summary['quarantine'] == 2
        assert summary['retained_old_anchors'] == {'test': 1, 'val': 0}
        sides = {}
        for split in ['train', 'val', 'test']:
            content = list(pairs(final / f'{split}.vi', final / f'{split}.zh'))
            sides[split] = [{norm.key(rec[index], lang) for rec in content} for index, lang in [(1, 'vi'), (2, 'zh')]]
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert all(not (x & y) for x, y in zip(sides[a], sides[b])), (a, b)
        assert len(list(rows(evidence / 'split_membership.jsonl'))) == 4
    print('PASS: fixed normalization outputs, null rejection and transitive bidirectional leakage regression')


if __name__ == '__main__':
    main()
