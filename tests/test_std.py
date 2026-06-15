import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    CORRECTION = [1]
    KEEP_DIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    CORRECTION = [0, 1]
    KEEP_DIM = [True, False]

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.std
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST + [None])
@pytest.mark.parametrize("correction", CORRECTION)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_std(shape, dim, correction, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    dims_to_check = []
    if isinstance(dim, int):
        dims_to_check = [dim]
    elif isinstance(dim, (list, tuple)):
        dims_to_check = dim

    if any(d >= len(shape) or d < -len(shape) for d in dims_to_check):
        # skip the test if dimension is out of range for the given shape.
        return

    if correction == 1:
        if dim is not None:
            positive_dims = [d % len(shape) for d in dims_to_check]
            reduction_size = 1
            for d in positive_dims:
                reduction_size *= shape[d]
            if reduction_size < 2:
                # Invalid case: correction=1 requires reduction size of at least 2
                return
        elif inp.numel() < 2:
            # Invalid case: correction=1 requires numel >= 2 for global reduction.
            return

    ref_inp = utils.to_reference(inp)

    with flag_gems.use_gems():
        res_out = torch.std(inp, dim=dim, correction=correction, keepdim=keepdim)

    ref_out = torch.std(ref_inp, dim=dim, correction=correction, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)
