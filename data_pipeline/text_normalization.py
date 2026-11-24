"""One normalization contract for stored text and comparison keys."""
import re
import unicodedata

import opencc

from freeze_utils import ROOT


class Normalizer:
    def __init__(self, cfg):
        self.cc = opencc.OpenCC(str(ROOT / cfg['opencc_resources'] / 't2s.json'))

    def text(self, text, lang):
        if not isinstance(text, str):
            raise TypeError('Sentence must be a string')
        text = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', text).strip())
        if lang == 'zh':
            visited = {text}
            for _ in range(8):
                converted = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', self.cc.convert(text)).strip())
                if converted == text:
                    return converted
                if converted in visited:
                    raise ValueError('OpenCC conversion cycle detected')
                visited.add(converted)
                text = converted
            raise ValueError('OpenCC did not converge within 8 passes')
        elif lang != 'vi':
            raise ValueError(lang)
        return unicodedata.normalize('NFC', re.sub(r'\s+', ' ', text).strip())

    def key(self, text, lang):
        return unicodedata.normalize('NFC', self.text(text, lang).lower())

    def pair(self, rec):
        return self.key(rec['vi'], 'vi'), self.key(rec['zh'], 'zh')
