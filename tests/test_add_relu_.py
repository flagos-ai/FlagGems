import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.add_relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_relu__tensor(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    # Reference: relu(self + alpha * other), materialized on the reference path.
    ref_out = torch.relu(ref_inp1 + alpha * ref_inp2)
    with flag_gems.use_gems():
        res_out = torch._add_relu_(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)
    # ``inp1`` is the in-place buffer; it must match the computed result too.
    utils.gems_assert_close(inp1, ref_out, dtype)
