import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_special_logsumexp(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=dim, keepdim=keepdim)

    gems_assert_close(res_out, ref_out, dtype)
