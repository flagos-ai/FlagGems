import random
import time

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    UPSAMPLE_SHAPES_1D,
    gems_assert_close,
    to_reference,
)

random.seed(time.time() // 100)

BOUNDARY_CASES = [
    ("W_in_1_upsample", (2, 3, 1), [5], True, None),
    ("W_in_1_upsample", (2, 3, 1), [5], False, None),
    ("W_out_1", (1, 1, 10), [1], False, None),
    ("identity_scale_ac", (2, 2, 100), [100], True, None),
    ("identity_scale_nc", (2, 2, 100), [100], False, None),
    ("value_nan", (1, 1, 10), [20], False, "nan"),
    ("value_inf", (1, 1, 10), [20], False, "inf"),
    ("non_contiguous", (2, 4, 10), [15], True, "non_contiguous"),
    ("non_contiguous", (2, 4, 10), [15], False, "non_contiguous"),
]


@pytest.mark.upsample_linear1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("case", BOUNDARY_CASES, ids=lambda x: x[0])
def test_upsample_linear1d_boundaries(dtype, case):
    _, shape, output_size, align_corners, special_cfg = case

    if special_cfg == "nan":
        input_tensor = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        input_tensor.fill_(float("nan"))
    elif special_cfg == "inf":
        input_tensor = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        input_tensor.fill_(float("inf"))
    else:
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if special_cfg == "non_contiguous":
        if shape[2] > 2:
            input_tensor = input_tensor[:, :, :-2]

            input_tensor = input_tensor.transpose(0, 2)
            input_tensor = input_tensor.transpose(0, 2)
    ref_i = to_reference(input_tensor).to(torch.float32)

    ref_out = torch._C._nn.upsample_linear1d(
        ref_i,
        output_size=output_size,
        align_corners=align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_linear1d(
            input_tensor,
            output_size=output_size,
            align_corners=align_corners,
        )
    if special_cfg == "nan":
        assert torch.isnan(res_out).all(), "Output should be all NaN"
        assert torch.isnan(ref_out).all(), "Reference should be all NaN"
    elif special_cfg == "inf":

        def is_inf_or_nan(x):
            return torch.isinf(x) | torch.isnan(x)

        assert is_inf_or_nan(res_out).all(), "Output should be all inf or nan"
        assert is_inf_or_nan(ref_out).all(), "Reference should be all inf or nan"
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_linear1d
@pytest.mark.skip(reason="Issue #2498: Result not close.")
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [2, 2.5, 0.3, 0.7])
@pytest.mark.parametrize("shape", UPSAMPLE_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_linear1d(dtype, shape, scale, align_corners):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = to_reference(input).to(torch.float32)
    output_size = [int(ref_i.shape[i + 2] * scale) for i in range(1)]

    ref_out = torch._C._nn.upsample_linear1d(
        ref_i,
        output_size=output_size,
        align_corners=align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_linear1d(
            input,
            output_size=output_size,
            align_corners=align_corners,
        )

    gems_assert_close(res_out, ref_out, dtype)


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
