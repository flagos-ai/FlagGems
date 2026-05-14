import random
import time

import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

random.seed(time.time() // 100)


def normalize_1d_shape(shape):
    if len(shape) == 1:
        return (1, 1, shape[0])
    if len(shape) == 2:
        return (shape[0], 1, shape[1])
    if len(shape) == 3:
        return shape

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
        (1, 1, 2),
        (1, 1, 3),
        (1, 3, 4),
        (2, 1, 5),
        (2, 3, 33),
        (3, 7, 17),
        (2, 3, 64),
        (4, 8, 16),
        (8, 16, 128),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("scale_factor", [0.5, 1.5, 2.0])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "non_contiguous"])
@pytest.mark.parametrize("edge_case", [False, True])
def test_upsample_linear1d_backward(
    shape, dtype, scale_factor, align_corners, layout, edge_case
):
    if edge_case:
        shape = (1, 1, 1)
        align_corners = False
        out_w = 1
    else:
        if layout == "non_contiguous":
            base_shape = (8, 16, 64)
            res_x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
            res_x = res_x.transpose(0, 1)
            shape = res_x.shape

        in_w = shape[-1]
        out_w = max(1, int(in_w * scale_factor))

    grad_shape = list(shape)
    grad_shape[-1] = out_w

    res_grad = torch.randn(
        grad_shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_grad = to_reference(res_grad)

    ref_out = upsample_linear1d_backward_call(
        ref_grad,
        shape,
        align_corners,
    )

    with flag_gems.use_gems():
        res_out = upsample_linear1d_backward_call(
            res_grad,
            shape,
            align_corners,
        )

    assert res_out.shape == tuple(shape)
    assert res_out.dtype == res_grad.dtype

    if dtype == torch.float32:
        atol = 1e-4
    elif dtype == torch.float16:
        atol = 1e-2
    else:
        atol = 2e-2

    gems_assert_close(res_out, ref_out, dtype, atol=atol)
