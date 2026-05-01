import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _nearest2d_backward(grad, input_size, scales_h=None, scales_w=None):
    output_size = [grad.shape[-2], grad.shape[-1]]
    return torch.ops.aten.upsample_nearest2d_backward(
        grad,
        output_size,
        list(input_size),
        scales_h,
        scales_w,
    )


def _make_grad(shape, dtype, layout):
    if layout == "channels_last":
        return torch.randn(shape, dtype=dtype, device=flag_gems.device).contiguous(
            memory_format=torch.channels_last
        )
    if layout == "transpose":
        n, c, h, w = shape
        return torch.randn(
            (n, c, w, h), dtype=dtype, device=flag_gems.device
        ).transpose(2, 3)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "input_size,output_size,layout,scales",
    [
        ((2, 3, 8, 9), (16, 18), "contiguous", (None, None)),
        ((1, 4, 7, 9), (13, 17), "contiguous", (None, None)),
        ((2, 5, 16, 18), (5, 7), "contiguous", (None, None)),
        ((1, 7, 6, 8), (12, 16), "transpose", (None, None)),
        ((2, 16, 8, 8), (16, 16), "channels_last", (None, None)),
        ((1, 64, 5, 6), (10, 12), "channels_last", (None, None)),
        ((1, 3, 5, 6), (11, 14), "contiguous", (2.1, 2.3)),
    ],
)
def test_upsample_nearest2d_backward(dtype, input_size, output_size, layout, scales):
    grad_shape = (input_size[0], input_size[1], output_size[0], output_size[1])
    grad = _make_grad(grad_shape, dtype, layout)
    ref_grad = utils.to_reference(grad).to(torch.float32)
    scales_h, scales_w = scales

    ref_out = _nearest2d_backward(ref_grad, input_size, scales_h, scales_w).to(dtype)
    with flag_gems.use_gems():
        res_out = _nearest2d_backward(grad, input_size, scales_h, scales_w)

    assert res_out.shape == input_size
    if layout == "channels_last":
        assert res_out.is_contiguous(memory_format=torch.channels_last)
    elif layout == "transpose":
        assert res_out.is_contiguous()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=9)


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward_shape_mismatch_rejected():
    grad = torch.randn((1, 2, 5, 6), dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(AssertionError, match="grad_output shape"):
        torch.ops.aten.upsample_nearest2d_backward(
            grad,
            [4, 6],
            [1, 2, 2, 3],
            None,
            None,
        )
