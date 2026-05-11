import pytest
import torch

import flag_gems

from .accuracy_utils import (
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_equal,
    to_reference,
)

NONZERO_SHAPES = REDUCTION_SHAPES


@pytest.mark.nonzero_numpy
@pytest.mark.parametrize("shape", NONZERO_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_nonzero_numpy(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(
            dtype
        )
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)

    ref_out = torch.ops.aten.nonzero_numpy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.nonzero_numpy(inp)

    assert len(res_out) == len(ref_out), "Number of output tensors should match"
    for res_t, ref_t in zip(res_out, ref_out):
        gems_assert_equal(res_t, ref_t)
