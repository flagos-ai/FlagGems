import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_special_logsumexp(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("shape", [(16, 64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dims", [[0, 1], [0, 2], [1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_special_logsumexp_multi_dim(shape, dtype, dims, keepdim):
    """Dedicated test for the multi-dim (len(dim) > 1) branch."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=dims, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=dims, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)
