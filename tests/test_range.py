import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.range
@pytest.mark.parametrize("start", utils.ARANGE_START)
@pytest.mark.parametrize("end", [8, 16, 32])
# torch.range does not support bfloat16 on CUDA
@pytest.mark.parametrize(
    "dtype", utils.PRIMARY_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None]
)
def test_range(start, end, dtype):
    ref_out = utils.to_reference(
        torch.range(
            start,
            end,
            dtype=dtype,
            device="cpu" if cfg.TO_CPU else device,
        )
    )
    with flag_gems.use_gems():
        res_out = torch.range(start, end, dtype=dtype, device=device)
    utils.gems_assert_equal(res_out, ref_out)
