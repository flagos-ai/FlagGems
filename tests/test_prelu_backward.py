import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.prelu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_prelu_backward_scalar(shape, dtype):
    """Test prelu_backward with scalar weight."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(0.25, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_weight = utils.to_reference(weight)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input, ref_grad_weight = torch.ops.aten.prelu_backward(
        ref_grad_output, ref_x, ref_weight
    )

    with flag_gems.use_gems():
        res_grad_input, res_grad_weight = torch.ops.aten.prelu_backward(
            grad_output, x, weight
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    utils.gems_assert_close(res_grad_weight, ref_grad_weight, dtype)


@pytest.mark.prelu_backward
@pytest.mark.parametrize("shape", [[(16, 3, 32, 32)], (16, 64, 128), (64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_prelu_backward_per_channel(shape, dtype):
    """Test prelu_backward with per-channel weight."""
    # Ensure shape has at least 2 dimensions for per-channel weight
    if len(shape) < 2:
        pytest.skip("Per-channel weight requires at least 2 dimensions")

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    num_channels = shape[1]
    weight = torch.randn(num_channels, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_weight = utils.to_reference(weight)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input, ref_grad_weight = torch.ops.aten.prelu_backward(
        ref_grad_output, ref_x, ref_weight
    )

    with flag_gems.use_gems():
        res_grad_input, res_grad_weight = torch.ops.aten.prelu_backward(
            grad_output, x, weight
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    utils.gems_assert_close(res_grad_weight, ref_grad_weight, dtype)
