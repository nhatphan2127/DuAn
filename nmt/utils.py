import os
import random

import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_amp_dtype(use_amp):
    """bf16 trên GPU Ampere trở lên (A100, L4), fp16 trên GPU cũ hơn (T4 của Colab free).

    Trả về None nếu tắt AMP hoặc không có GPU.
    """
    if not use_amp or DEVICE.type != 'cuda':
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def atomic_save(obj, path):
    """Ghi ra file tạm rồi đổi tên, để Colab có ngắt giữa chừng cũng không làm hỏng checkpoint cũ."""
    tmp_path = f'{path}.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
