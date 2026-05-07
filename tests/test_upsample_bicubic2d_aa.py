import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.upsample_bicubic2d_aa
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.7)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 16, 128, 128),
        (15, 37, 256, 256),
        (3, 5, 127, 127),
        (128, 192, 42, 51),
        (3, 7, 1023, 1025),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_bicubic2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch._C._nn._upsample_bicubic2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn._upsample_bicubic2d_aa(
            input, output_size=output_size, align_corners=align_corners
        )

    def span(scale):
        support = 2 if (scale >= 1.0) else 2.0 / scale
        interpolate_range = int(support + 0.5) * 2 + 1
        return interpolate_range

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    reduce_dim = span(scale[0]) * span(scale[1])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


def upsample_bicubic2d_aa_backward_call(grad, input_size, align_corners):
    orig_shape = tuple(input_size)
    n = 1
    for s in orig_shape[:-2]:
        n *= s
    c = orig_shape[-2] if len(orig_shape) >= 2 else 1
    in_h = orig_shape[-2] if len(orig_shape) >= 3 else 1
    in_w = orig_shape[-1]
    if len(orig_shape) >= 4:
        c = orig_shape[-3]
        in_h = orig_shape[-2]
        in_w = orig_shape[-1]
        n = 1
        for s in orig_shape[:-3]:
            n *= s
    else:
        # For 4D input: (N, C, H, W)
        n, c, in_h, in_w = orig_shape

    shape_4d = (n, c, in_h, in_w)
    out_h = grad.shape[-2]
    out_w = grad.shape[-1]

    grad_4d = grad.reshape(n, c, out_h, out_w)

    out = torch.ops.aten._upsample_bicubic2d_aa_backward(
        grad_4d,
        [out_h, out_w],
        list(shape_4d),
        align_corners,
        None,
        None,
    )

    return out.reshape(orig_shape)


@pytest.mark.upsample_bicubic2d_aa_backward
@pytest.mark.parametrize(
    "N,C,H_in,W_in,H_out,W_out,align_corners",
    [
        (1, 3, 16, 16, 8, 8, False),
        (2, 4, 8, 8, 16, 16, False),
        (1, 3, 32, 32, 10, 10, False),
        (1, 1, 10, 10, 23, 23, False),
        (1, 3, 16, 16, 8, 8, True),
        (1, 3, 8, 8, 16, 16, True),
        (2, 64, 32, 32, 16, 16, False),
        (1, 3, 7, 11, 13, 5, False),
        (1, 1, 4, 4, 4, 4, False),
        (1, 1, 8, 8, 1, 1, True),
        # Extra cases
        (1, 1, 64, 64, 16, 16, False),
        (1, 1, 64, 64, 128, 128, False),
        (512, 1024, 32, 32, 8, 8, False),
        (256, 512, 64, 64, 16, 16, False),
        (4, 16, 16, 16, 4, 4, False),
        (4, 16, 4, 4, 16, 16, False),
        (4, 16, 64, 128, 32, 64, False),
        (4, 16, 64, 128, 128, 256, True),
        (1, 1, 4096, 4096, 1024, 1024, False),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_bicubic2d_aa_backward(
    N, C, H_in, W_in, H_out, W_out, align_corners, dtype
):
    shape = (N, C, H_in, W_in)

    grad_shape = (N, C, H_out, W_out)

    res_grad = torch.randn(
        grad_shape,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_grad = utils.to_reference(res_grad)

    ref_out = upsample_bicubic2d_aa_backward_call(
        ref_grad,
        shape,
        align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = upsample_bicubic2d_aa_backward_call(
            res_grad.to(dtype),
            shape,
            align_corners,
        )

    assert res_out.shape == shape

    # dtype-specific tolerance
    if dtype == torch.float32:
        atol = 1e-4
    elif dtype == torch.float16:
        atol = 3e-3
    else:  # bfloat16
        atol = 2e-2

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
