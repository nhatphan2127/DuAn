"""beam_search: length penalty kiểu GNMT (mặc định) và kiểu Hugging Face (dùng khi giải mã cùng cấu hình NLLB)."""
import math
import unittest

import torch

from nmt.decoding import beam_search
from nmt.tokenizer import EOS_IDX

A = 4          # một token thường
VOCAB = 5


class FixedModel:
    """Mỗi bước: P(A) = 0,6, P(EOS) = 0,4, các token khác ~ 0."""

    def encode(self, src):
        return torch.zeros(1, src.size(1), 2), torch.ones(1, 1, 1, src.size(1), dtype=torch.bool)

    def decode(self, ys, memory, src_mask):
        logits = torch.full((ys.size(0), ys.size(1), VOCAB), -1e9)
        logits[..., A] = math.log(0.6)
        logits[..., EOS_IDX] = math.log(0.4)
        return logits


class BeamLengthPenaltyTest(unittest.TestCase):
    # Với beam 2, dừng sau 2 bước; ứng viên: [EOS] (-0,92), [A, EOS] (-1,43), [A, A] chưa kết thúc (-1,02)
    src = torch.tensor([[1, 5, 2]])

    def test_default_is_gnmt(self):
        model = FixedModel()
        self.assertEqual(beam_search(model, self.src, 2, 10), beam_search(model, self.src, 2, 10, 0.6, 'gnmt'))
        self.assertEqual(beam_search(model, self.src, 2, 10), [])   # GNMT 0,6 vẫn chọn câu rỗng ở đây

    def test_hf_divides_by_generated_length(self):
        model = FixedModel()
        # hf, α = 1: [A, A] -1,02 / 2 = -0,51 tốt nhất
        self.assertEqual(beam_search(model, self.src, 2, 10, 1.0, 'hf'), [A, A])
        # hf, α = 0: không phạt độ dài -> [EOS] (-0,92)
        self.assertEqual(beam_search(model, self.src, 2, 10, 0.0, 'hf'), [])

    def test_max_new_tokens_limits_steps(self):
        self.assertEqual(beam_search(FixedModel(), self.src, 2, 10, 1.0, 'hf', max_new_tokens=1), [A])

    def test_rejects_unknown_style(self):
        with self.assertRaises(ValueError):
            beam_search(FixedModel(), self.src, 2, 10, 1.0, 'other')


if __name__ == '__main__':
    unittest.main()
