import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize(
    "shape,output_size,scales",
    [
        ((1, 1, 2, 3), (4, 6), (None, None)),
        ((2, 3, 5, 7), (10, 14), (None, None)),
        ((2, 4, 9, 11), (3, 5), (None, None)),
        ((1, 2, 7, 5), (14, 13), (2.0, 2.6)),
        ((3, 2, 8, 6), (5, 4), (0.7, 0.75)),
        ((1, 1, 4, 4), (4, 4), (None, None)),
        ((0, 2, 3, 4), (6, 8), (None, None)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward(dtype, shape, output_size, scales):
    grad_output = torch.randn(
        (*shape[:2], *output_size), dtype=dtype, device=flag_gems.device
    )
    if shape[2] != output_size[0] and output_size[0] > 1:
        grad_output = grad_output.transpose(-1, -2).contiguous().transpose(-1, -2)
    ref_grad = utils.to_reference(grad_output).to(torch.float32)
    scales_h, scales_w = scales

    ref_out = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad,
        list(output_size),
        list(shape),
        scales_h,
        scales_w,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_nearest2d_backward(
            grad_output,
            list(output_size),
            list(shape),
            scales_h,
            scales_w,
        )

    if dtype == torch.float32:
        atol = 1e-4
    elif dtype == torch.float16:
        atol = 3e-3
    else:
        atol = 2e-2
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
