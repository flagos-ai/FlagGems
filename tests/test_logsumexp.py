import pytest
import torch

import flag_gems

from .accuracy_utils import (
    DIM_LIST,
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    to_reference,
)


@pytest.mark.logsumexp
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_logsumexp(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.logsumexp(inp, dim=dim, keepdim=keepdim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
