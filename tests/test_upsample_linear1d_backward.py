import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_linear1d_backward
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("output_size", [(32,), (64,)])
@pytest.mark.parametrize("input_size", [(1, 3, 16), (2, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_linear1d_backward(output_size, align_corners, input_size, dtype):
    """Test upsample_linear1d_backward with various configurations."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.upsample_linear1d(ref_input, output_size, align_corners)
    ref_grad = torch.randn_like(ref_output)
    ref_output.backward(ref_grad)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.upsample_linear1d(input, output_size, align_corners)
        res_grad = torch.randn_like(res_output)
        res_output.backward(res_grad)

    utils.gems_assert_close(input.grad, ref_input.grad, dtype)


@pytest.mark.upsample_linear1d_backward
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("scales", [2.0, 1.5])
@pytest.mark.parametrize("input_size", [(1, 3, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_linear1d_backward_scales(scales, align_corners, input_size, dtype):
    """Test upsample_linear1d_backward with scales parameter."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.upsample_linear1d(ref_input, None, align_corners, scales)
    ref_grad = torch.randn_like(ref_output)
    ref_output.backward(ref_grad)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.upsample_linear1d(input, None, align_corners, scales)
        res_grad = torch.randn_like(res_output)
        res_output.backward(res_grad)

    utils.gems_assert_close(input.grad, ref_input.grad, dtype)
