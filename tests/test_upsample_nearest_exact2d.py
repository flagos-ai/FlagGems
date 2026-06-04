import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest_exact2d
@pytest.mark.parametrize("output_size", [((32, 32),), ((64, 48),)])
@pytest.mark.parametrize("input_size", [(1, 3, 16, 16), (2, 16, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact2d(output_size, input_size, dtype):
    """Test _upsample_nearest_exact2d with various configurations."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten._upsample_nearest_exact2d(ref_input, output_size[0])
    res_output = torch.ops.aten._upsample_nearest_exact2d(input, output_size[0])

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.upsample_nearest_exact2d
@pytest.mark.parametrize("scales", [(2.0, 2.0), (1.5, 1.5), (2.0, 1.5)])
@pytest.mark.parametrize("input_size", [(1, 3, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact2d_scales(scales, input_size, dtype):
    """Test _upsample_nearest_exact2d with scales parameter."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten._upsample_nearest_exact2d(ref_input, None, scales)
    res_output = torch.ops.aten._upsample_nearest_exact2d(input, None, scales)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.upsample_nearest_exact2d_backward
@pytest.mark.parametrize("output_size", [((32, 32),), ((64, 48),)])
@pytest.mark.parametrize("input_size", [(1, 3, 16, 16), (2, 16, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact2d_backward(output_size, input_size, dtype):
    """Test _upsample_nearest_exact2d_backward with various configurations."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten._upsample_nearest_exact2d(ref_input, output_size[0])
    ref_grad = torch.randn_like(ref_output)
    ref_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward(ref_grad, ref_input.shape, output_size[0])

    with flag_gems.use_gems():
        res_output = torch.ops.aten._upsample_nearest_exact2d(input, output_size[0])
        res_grad = torch.randn_like(res_output)
        res_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward(res_grad, input.shape, output_size[0])

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.upsample_nearest_exact2d_backward
@pytest.mark.parametrize("scales", [(2.0, 2.0), (1.5, 1.5)])
@pytest.mark.parametrize("input_size", [(1, 3, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact2d_backward_scales(scales, input_size, dtype):
    """Test _upsample_nearest_exact2d_backward with scales parameter."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten._upsample_nearest_exact2d(ref_input, None, scales)
    ref_grad = torch.randn_like(ref_output)
    ref_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward(ref_grad, ref_input.shape, None, scales)

    with flag_gems.use_gems():
        res_output = torch.ops.aten._upsample_nearest_exact2d(input, None, scales)
        res_grad = torch.randn_like(res_output)
        res_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward(res_grad, input.shape, None, scales)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
