import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.native_dropout_backward
@pytest.mark.parametrize("shape", [(1024,), (1024, 1024), (1, 8192), (32, 50257)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_dropout_backward(shape, dtype):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    scale = 2.0

    ref_grad_output = utils.to_reference(grad_output)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.native_dropout_backward(ref_grad_output, ref_mask, scale)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.native_dropout_backward(grad_output, mask, scale)

    utils.gems_assert_close(res_out, ref_out, dtype)
