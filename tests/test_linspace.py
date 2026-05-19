import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


@pytest.mark.linspace
@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("end", [256, 2048, 4096])
@pytest.mark.parametrize("steps", [1, 256, 512])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [flag_gems.device, None])
@pytest.mark.parametrize("pin_memory", [False, None])
def test_linspace(start, end, steps, dtype, device, pin_memory):
    ref_out = torch.linspace(
        start,
        end,
        steps,
        dtype=dtype,
        layout=None,
        device="cpu" if cfg.TO_CPU else device,
        pin_memory=pin_memory,
    )
    with flag_gems.use_gems():
        res_out = torch.linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )

    if dtype in [torch.float16, torch.bfloat16, torch.float32, None]:
        utils.gems_assert_close(res_out, ref_out, dtype=dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
