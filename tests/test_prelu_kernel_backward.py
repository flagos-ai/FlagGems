import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.prelu_kernel_backward
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_prelu_kernel_backward(shape, dtype):
    # Test with scalar weight
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor([0.25], dtype=dtype, device=flag_gems.device)

    ref_grad_output = utils.to_reference(grad_output)
    ref_x = utils.to_reference(x)
    ref_weight = utils.to_reference(weight)

    ref_out = torch.ops.aten._prelu_kernel_backward(ref_grad_output, ref_x, ref_weight)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._prelu_kernel_backward(grad_output, x, weight)

    # Check both outputs
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_close(res_out[1], ref_out[1], dtype)


@pytest.mark.prelu_kernel_backward
@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_prelu_kernel_backward_per_channel(shape, dtype):
    # Test with per-channel weight (C channels matching last dimension)
    C = shape[-1]
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(C, dtype=dtype, device=flag_gems.device)

    ref_grad_output = utils.to_reference(grad_output)
    ref_x = utils.to_reference(x)
    ref_weight = utils.to_reference(weight)

    ref_out = torch.ops.aten._prelu_kernel_backward(ref_grad_output, ref_x, ref_weight)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._prelu_kernel_backward(grad_output, x, weight)

    # Check both outputs
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_close(res_out[1], ref_out[1], dtype)
