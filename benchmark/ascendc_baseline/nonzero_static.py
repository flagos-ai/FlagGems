import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_nonzero_static():
    source = Path(__file__).with_suffix(".cpp")
    load(
        name="flag_gems_ascendc_nonzero_static",
        sources=[str(source)],
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=os.getenv("FLAGGEMS_ASCENDC_VERBOSE", "0") == "1",
    )
    return torch.ops.flag_gems_ascendc.nonzero_static
