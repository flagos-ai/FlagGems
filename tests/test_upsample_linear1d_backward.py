import os
import sys

import pytest
import torch

import flag_gems

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tests.accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
    to_reference,
)


def normalize_1d_shape(shape):
    if len(shape) == 1:
        return (1, 1, shape[0])
    elif len(shape) == 2:
        return (shape[0], 1, shape[1])
    elif len(shape) == 3:
        return shape
    else:
        n = 1
        for s in shape[:-2]:
            n *= s
        return (n, shape[-2], shape[-1])


def upsample_linear1d_backward_call(grad, input_size, align_corners):
    orig_shape = tuple(input_size)
    shape_3d = normalize_1d_shape(orig_shape)

    out_w = grad.shape[-1]

    grad_3d = grad.reshape(*shape_3d[:-1], out_w)

    out = torch.ops.aten.upsample_linear1d_backward(
        grad_3d,
        [out_w],
        list(shape_3d),
        align_corners,
        None,
    )

    return out.reshape(orig_shape)


@pytest.mark.upsample_linear1d_backward
@pytest.mark.parametrize(
    "shape",
    [
        (16,),
        (4, 8),
        (4, 8, 16),
        (2, 3, 64),
        (8, 16, 128),
        (3, 7, 33),
        (2, 1, 17),
        (32, 256),
        (16, 64, 128),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("scale_factor", [0.5, 1.5, 2.0])
@pytest.mark.parametrize("align_corners", [False, True])
def test_accuracy_upsample_linear1d_backward(
    shape, dtype, scale_factor, align_corners
):
    device = flag_gems.device

    in_w = shape[-1]
    out_w = max(1, int(in_w * scale_factor))

    grad_shape = list(shape)
    grad_shape[-1] = out_w

    grad = torch.randn(
        grad_shape,
        device=device,
        dtype=dtype,
    )

    ref = to_reference(
        upsample_linear1d_backward_call(
            grad,
            shape,
            align_corners,
        )
    )

    with flag_gems.use_gems():
        res = upsample_linear1d_backward_call(
            grad,
            shape,
            align_corners,
        )

    assert res.shape == tuple(shape)
    assert res.dtype == grad.dtype

    atol=1e-4 if dtype == torch.float32 else 2e-2
    gems_assert_close(res, ref, dtype, atol=atol)


@pytest.mark.upsample_linear1d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_linear1d_backward_non_contiguous(dtype):
    device = flag_gems.device

    base_shape = (8, 16, 64)
    x = torch.randn(base_shape, device=device, dtype=dtype)

    x = x.transpose(0, 1)  # non-contiguous

    shape = x.shape
    in_w = shape[-1]
    out_w = in_w * 2

    grad_shape = list(shape)
    grad_shape[-1] = out_w

    grad = torch.randn(
        grad_shape,
        device=device,
        dtype=dtype,
    )

    ref = to_reference(
        upsample_linear1d_backward_call(
            grad,
            shape,
            False,
        )
    )

    with flag_gems.use_gems():
        res = upsample_linear1d_backward_call(
            grad,
            shape,
            False,
        )

    atol=1e-4 if dtype == torch.float32 else 2e-2
    gems_assert_close(res, ref, dtype, atol=atol)


@pytest.mark.upsample_linear1d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_linear1d_backward_small_and_edge(dtype):
    device = flag_gems.device

    shape = (1, 1, 1)
    out_w = 1

    grad = torch.randn(
        (1, 1, out_w),
        device=device,
        dtype=dtype,
    )

    ref = to_reference(
        upsample_linear1d_backward_call(
            grad,
            shape,
            False,
        )
    )

    with flag_gems.use_gems():
        res = upsample_linear1d_backward_call(
            grad,
            shape,
            False,
        )

    atol=1e-4 if dtype == torch.float32 else 2e-2
    gems_assert_close(res, ref, dtype, atol=atol)
