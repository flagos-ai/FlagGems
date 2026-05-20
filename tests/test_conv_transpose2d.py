import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

# --- 4.1.4 coverage matrix ---------------------------------------------------
# Input regimes
#   small    : 1x1 / 8x8 spatial
#   regular  : 32x32, 64x64 spatial
#   large    : 256x256 spatial (kept to one shape to bound test runtime)
# Each entry is ((N, C_in, H_in, W_in), (C_in, C_out_per_group, K_h, K_w)).
# ---------------------------------------------------------------------------
SMALL_SHAPES = [
    ((1, 4, 1, 1), (4, 8, 3, 3)),
    ((2, 4, 8, 8), (4, 8, 3, 3)),
]
REGULAR_SHAPES = [
    ((2, 8, 32, 32), (8, 16, 3, 3)),
    ((2, 16, 64, 64), (16, 16, 5, 5)),
]
LARGE_SHAPES = [
    ((1, 8, 256, 256), (8, 8, 3, 3)),
]
ALL_SHAPES = SMALL_SHAPES + REGULAR_SHAPES + LARGE_SHAPES


def _reduce_dim(kernel_shape, groups):
    in_channels = kernel_shape[0]
    kh = kernel_shape[2]
    kw = kernel_shape[3]
    return (in_channels // groups) * kh * kw


# ---------------------------------------------------------------------------
# Test 1 — golden path: vary stride / padding / size / dtype.
# ---------------------------------------------------------------------------
@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel", ALL_SHAPES)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_transpose2d_basic(shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_reduce_dim(kernel, 1))


# ---------------------------------------------------------------------------
# Test 2 — bias path.
# ---------------------------------------------------------------------------
@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel", SMALL_SHAPES + REGULAR_SHAPES)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_transpose2d_bias(shape, kernel, stride, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    out_channels = kernel[1]  # groups == 1, so per-group == total
    bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp, ref_weight, bias=ref_bias, stride=stride, padding=0, dilation=1
    )
    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias, stride=stride, padding=0, dilation=1
    )

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_reduce_dim(kernel, 1))


# ---------------------------------------------------------------------------
# Test 3 — asymmetric (stride, padding, dilation, kernel) and output_padding.
# Covers parameter-mode completeness from 4.1.4.
# ---------------------------------------------------------------------------
@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "shape, kernel, stride, padding, output_padding, dilation",
    [
        # asymmetric stride
        ((2, 4, 8, 8), (4, 8, 3, 3), (2, 1), 0, 0, 1),
        # asymmetric padding
        ((2, 4, 8, 8), (4, 8, 3, 3), 1, (1, 0), 0, 1),
        # output_padding < stride
        ((2, 4, 8, 8), (4, 8, 3, 3), 2, 1, 1, 1),
        # dilation > 1
        ((2, 4, 8, 8), (4, 8, 3, 3), 1, 0, 0, 2),
        # asymmetric kernel
        ((2, 4, 8, 8), (4, 8, 3, 5), 1, (1, 2), 0, 1),
        # asymmetric dilation
        ((2, 4, 12, 12), (4, 8, 3, 3), 1, 0, 0, (2, 3)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_asymmetric(
    shape, kernel, stride, padding, output_padding, dilation, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )
    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_reduce_dim(kernel, 1))


# ---------------------------------------------------------------------------
# Test 4 — grouped convolution.
# ---------------------------------------------------------------------------
@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 8, 16, 16), (8, 4, 3, 3), 2),
        ((2, 12, 16, 16), (12, 4, 3, 3), 3),
        ((2, 16, 16, 16), (16, 1, 3, 3), 16),  # depthwise
    ],
)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_transpose2d_groups(shape, kernel, groups, stride, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=0,
        dilation=1,
        groups=groups,
    )
    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=0,
        dilation=1,
        groups=groups,
    )

    gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(kernel, groups)
    )


# ---------------------------------------------------------------------------
# Test 5 — invalid input shapes raise.
# ---------------------------------------------------------------------------
@pytest.mark.conv_transpose2d
def test_conv_transpose2d_rejects_invalid_rank():
    inp = torch.randn(2, 4, 8, device=flag_gems.device)  # 3D
    weight = torch.randn(4, 8, 3, 3, device=flag_gems.device)
    with pytest.raises(AssertionError):
        flag_gems.conv_transpose2d(inp, weight)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_rejects_channel_mismatch():
    inp = torch.randn(2, 4, 8, 8, device=flag_gems.device)
    weight = torch.randn(5, 8, 3, 3, device=flag_gems.device)
    with pytest.raises(AssertionError):
        flag_gems.conv_transpose2d(inp, weight)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_rejects_bad_output_padding():
    inp = torch.randn(2, 4, 8, 8, device=flag_gems.device)
    weight = torch.randn(4, 8, 3, 3, device=flag_gems.device)
    with pytest.raises(AssertionError):
        flag_gems.conv_transpose2d(inp, weight, stride=2, output_padding=2)
