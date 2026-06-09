import random
import time

import pytest
import torch

from flag_gems.ops import normed_cumsum

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NORMED_CUMSUM_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NORMED_CUMSUM_SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]

random.seed(time.time() // 100)


@pytest.mark.normed_cumsum
@pytest.mark.parametrize("shape", NORMED_CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_normed_cumsum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    # Ensure positive values so sum is meaningful for normalization
    inp = inp.abs() + 1e-4
    ref_inp = utils.to_reference(inp, True)

    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1

    # Reference: cumsum normalized by total sum along the dimension
    ref_cumsum = torch.cumsum(ref_inp, dim=dim)
    ref_sum = ref_inp.sum(dim=dim, keepdim=True)
    ref_out = ref_cumsum / ref_sum

    res_out = normed_cumsum(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
