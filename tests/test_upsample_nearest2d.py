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
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward(dtype, shape, scale):
    output_size = [int(shape[i + 2] * scale[i]) for i in range(2)]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp).to(torch.float32)

    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size)
    out_grad = torch.randn_like(ref_out, dtype=dtype, device=flag_gems.device)
    ref_out_grad = utils.to_reference(out_grad).to(torch.float32)

    ref_grad_in = torch.ops.aten.upsample_nearest2d_backward.default(
        ref_out_grad,
        output_size,
        list(shape),
        None,
        None,
    )

    with flag_gems.use_gems():
        gem_grad_in = torch.ops.aten.upsample_nearest2d_backward.default(
            out_grad,
            output_size,
            list(shape),
            None,
            None,
        )

    utils.gems_assert_close(gem_grad_in, ref_grad_in.to(dtype), dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize(
    "scales_h,scales_w", [(None, None), (2.0, 2.0), (1.5, 1.5), (0.5, 0.5)]
)
@pytest.mark.parametrize("shape", [(4, 8, 16, 16), (2, 3, 7, 5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward_with_scales(dtype, shape, scales_h, scales_w):
    N, C, IH, IW = shape
    if scales_h is None:
        OH, OW = IH * 2, IW * 2
    else:
        OH, OW = int(IH * scales_h), int(IW * scales_w)
    if OH == 0 or OW == 0:
        pytest.skip("degenerate output size")

    output_size = [OH, OW]
    input_size = list(shape)

    grad_out = torch.randn(N, C, OH, OW, dtype=dtype, device=flag_gems.device)
    ref_grad_out = utils.to_reference(grad_out).to(torch.float32)

    ref_grad_in = torch.ops.aten.upsample_nearest2d_backward.default(
        ref_grad_out, output_size, input_size, scales_h, scales_w
    )

    with flag_gems.use_gems():
        gem_grad_in = torch.ops.aten.upsample_nearest2d_backward.default(
            grad_out, output_size, input_size, scales_h, scales_w
        )

    utils.gems_assert_close(gem_grad_in, ref_grad_in.to(dtype), dtype)
