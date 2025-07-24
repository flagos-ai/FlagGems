import importlib
import random

import numpy as np
import torch
from packaging import version


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_torch_version_ge(min_ver: str) -> bool:
    return version.parse(torch.__version__) >= version.parse(min_ver)


def has_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None
