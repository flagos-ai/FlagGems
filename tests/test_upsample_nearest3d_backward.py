import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest3d_backward
@pytest.mark.parametrize(
    "output_size",
    [((16, 16, 16),), ((32, 32, 32),)],
)
@pytest.mark.parametrize("input_size", [(1, 3, 8, 8, 8), (2, 16, 16, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest3d_backward(output_size, input_size, dtype):
    """Test upsample_nearest3d_backward with various configurations."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.upsample_nearest3d(ref_input, output_size[0])
    ref_grad = torch.randn_like(ref_output)
    ref_output.backward(ref_grad)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.upsample_nearest3d(input, output_size[0])
        res_grad = torch.randn_like(res_output)
        res_output.backward(res_grad)

    utils.gems_assert_close(input.grad, ref_input.grad, dtype)


@pytest.mark.upsample_nearest3d_backward
@pytest.mark.parametrize(
    "scales",
    [(2.0, 2.0, 2.0), (1.5, 1.5, 1.5)],
)
@pytest.mark.parametrize("input_size", [(1, 3, 8, 8, 8)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest3d_backward_scales(scales, input_size, dtype):
    """Test upsample_nearest3d_backward with scales parameter."""
    input = torch.randn(input_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.upsample_nearest3d(ref_input, None, scales[0], scales[1], scales[2])
    ref_grad = torch.randn_like(ref_output)
    ref_output.backward(ref_grad)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.upsample_nearest3d(input, None, scales[0], scales[1], scales[2])
        res_grad = torch.randn_like(res_output)
        res_output.backward(res_grad)

    utils.gems_assert_close(input.grad, ref_input.grad, dtype)
