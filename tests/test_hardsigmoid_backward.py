import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hardsigmoid_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardsigmoid_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    ref_out = torch.ops.aten.hardsigmoid_backward(ref_grad_output, ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardsigmoid_backward(grad_output, inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
