import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest3d_backward
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest3d_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    out_d = shape[2] * 2
    out_h = shape[3] * 2
    out_w = shape[4] * 2
    output_size = (out_d, out_h, out_w)

    ref_out = torch.ops.aten.upsample_nearest3d(
        ref_x, list(output_size), None, None, None
    )
    grad_output = torch.randn_like(ref_out)

    input_size = tuple(x.shape)  # (N, C, D, H, W)

    ref_grad_input = torch.ops.aten.upsample_nearest3d_backward.default(
        grad_output, output_size, input_size
    )

    res_grad_input = flag_gems.upsample_nearest3d_backward(
        grad_output.to(flag_gems.device), output_size, input_size
    )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.upsample_nearest3d_backward
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest3d_backward_with_scales(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    scale_d = 2.0
    scale_h = 3.0
    scale_w = 2.0
    out_d = int(shape[2] * scale_d)
    out_h = int(shape[3] * scale_h)
    out_w = int(shape[4] * scale_w)
    output_size = (out_d, out_h, out_w)

    ref_out = torch.ops.aten.upsample_nearest3d(
        ref_x, list(output_size), scale_d, scale_h, scale_w
    )
    grad_output = torch.randn_like(ref_out)

    input_size = tuple(x.shape)

    ref_grad_input = torch.ops.aten.upsample_nearest3d_backward.default(
        grad_output, output_size, input_size, scale_d, scale_h, scale_w
    )

    res_grad_input = flag_gems.upsample_nearest3d_backward(
        grad_output.to(flag_gems.device),
        output_size,
        input_size,
        scale_d,
        scale_h,
        scale_w,
    )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@getattr(pytest.mark, "upsample_nearest3d_backward.grad_input")
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest3d_backward_grad_input(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    out_d = shape[2] * 2
    out_h = shape[3] * 2
    out_w = shape[4] * 2
    output_size = (out_d, out_h, out_w)

    out = torch.ops.aten.upsample_nearest3d(x, list(output_size), None, None, None)
    grad_output = torch.randn_like(out)
    ref_grad_output = utils.to_reference(grad_output)

    input_size = tuple(x.shape)

    ref_grad_input = torch.empty(input_size, dtype=dtype, device=ref_grad_output.device)
    ref_grad_input = torch.ops.aten.upsample_nearest3d_backward.grad_input(
        ref_grad_output,
        output_size,
        input_size,
        None,
        None,
        None,
        grad_input=ref_grad_input,
    )

    res_grad_input = torch.empty(input_size, dtype=dtype, device=flag_gems.device)
    res_grad_input = flag_gems.upsample_nearest3d_backward_grad_input(
        grad_output.to(flag_gems.device),
        output_size,
        input_size,
        None,
        None,
        None,
        grad_input=res_grad_input,
    )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
