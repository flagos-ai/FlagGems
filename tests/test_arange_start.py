import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.arange_start
@pytest.mark.arange_start_step
@pytest.mark.parametrize("start", utils.ARANGE_START)
@pytest.mark.parametrize("step", [1, 2, 5])
@pytest.mark.parametrize("end", [128, 256, 1024])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [flag_gems.device, None])
# Since triton only target to GPU, pin_memory only used in CPU tensors.
@pytest.mark.parametrize("pin_memory", [False, None])
def test_arange(start, step, end, dtype, device, pin_memory):
    ref_out = torch.arange(
        start,
        end,
        step,
        dtype=dtype,
        device="cpu" if cfg.TO_CPU else device,
        pin_memory=pin_memory,
    )

    with flag_gems.use_gems():
        res_out = torch.arange(
            start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
        )

    utils.gems_assert_equal(res_out, ref_out)
