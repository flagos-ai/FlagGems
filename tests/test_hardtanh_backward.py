import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hardtanh_backward
@pytest.mark.parametrize("min_val, max_val", [(-1.0, 1.0), (0.0, 2.0), (-2.0, 0.0)])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_backward(shape, dtype, min_val, max_val):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    ref_out = torch.ops.aten.hardtanh_backward(
        ref_grad_output, ref_inp, min_val, max_val
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh_backward(grad_output, inp, min_val, max_val)

    utils.gems_assert_close(res_out, ref_out, dtype)
