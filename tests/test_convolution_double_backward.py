import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

CONV_CASES = [
    # non-transposed
    ((1, 2, 5, 5), (1, 2, 3, 3), 1, 1, 1, 1, False, (0, 0)),
    ((1, 2, 5, 5), (1, 2, 3, 3), 1, 2, 1, 1, False, (0, 0)),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1, 2, 1, 1, False, (0, 0)),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1, 1, 0, 1, False, (0, 0)),
    ((2, 4, 8, 8), (4, 2, 3, 3), 2, 2, 1, 1, False, (0, 0)),
    ((1, 3, 7, 7), (2, 3, 3, 3), 1, 1, 1, 2, False, (0, 0)),
    # transposed (weight shape is (in_channels, out_channels, kH, kW))
    ((1, 2, 5, 5), (2, 3, 3, 3), 1, 1, 1, 1, True, (0, 0)),
    ((2, 4, 5, 5), (4, 3, 3, 3), 1, 2, 1, 1, True, (0, 0)),
    ((2, 4, 5, 5), (4, 3, 3, 3), 1, 2, 1, 1, True, (1, 1)),
    ((2, 4, 8, 8), (4, 2, 3, 3), 2, 2, 1, 1, True, (0, 0)),
    ((1, 3, 7, 7), (3, 2, 3, 3), 1, 3, 1, 1, True, (2, 2)),
]

# 1D convolution cases: shapes are (N, C, L); weights are
# (out_channels, in_channels/groups, kL) for non-transposed and
# (in_channels, out_channels/groups, kL) for transposed. Stride / padding /
# dilation / output_padding are scalars (length-1 spatial).
CONV1D_CASES = [
    # non-transposed
    ((1, 2, 9), (4, 2, 3), 1, 2, 1, 1, False, 0),
    ((1, 2, 9), (4, 2, 3), 1, 1, 1, 1, False, 0),
    ((2, 3, 9), (4, 3, 3), 1, 2, 1, 1, False, 0),
    ((2, 4, 8), (4, 2, 3), 2, 2, 1, 1, False, 0),
    ((1, 3, 7), (2, 3, 3), 1, 1, 1, 2, False, 0),
    # transposed (weight shape is (in_channels, out_channels, kL))
    ((1, 2, 5), (2, 3, 3), 1, 1, 1, 1, True, 0),
    ((2, 4, 5), (4, 3, 3), 1, 2, 1, 1, True, 0),
    ((2, 4, 5), (4, 3, 3), 1, 2, 1, 1, True, 1),
    ((2, 4, 8), (4, 2, 3), 2, 2, 1, 1, True, 0),
    ((1, 3, 7), (3, 2, 3), 1, 3, 1, 1, True, 2),
]


# Restricted dtype list: the generator limits the tested precisions for numerical-stability reasons (see worktree).
FLOAT_DTYPES = [torch.float16, torch.float32]


# (has_ggI, has_ggW, has_ggb): controls which grad-output inputs are passed.
# Note ``has_ggb`` is only meaningful when the forward convolution has a bias.
GRAD_INPUT_PRESENCE = [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
]


def _forward(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    if x.ndim == 3:
        if not transposed:
            return torch.nn.functional.conv1d(
                x,
                w,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        return torch.nn.functional.conv_transpose1d(
            x,
            w,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
        )
    if not transposed:
        return torch.nn.functional.conv2d(
            x,
            w,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    return torch.nn.functional.conv_transpose2d(
        x,
        w,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        groups=groups,
    )


@pytest.mark.convolution_double_backward
@pytest.mark.parametrize(
    "shape, weight_shape, groups, stride, padding, dilation, transposed, output_padding",
    CONV_CASES + CONV1D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize(
    "has_ggI, has_ggW, has_ggb",
    GRAD_INPUT_PRESENCE,
)
def test_convolution_double_backward(
    shape,
    weight_shape,
    groups,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    dtype,
    has_bias,
    has_ggI,
    has_ggW,
    has_ggb,
):
    # Single-int params are broadcast to per-axis lists by the operator; build the
    # per-axis lists here for the forward / reference calls, sized to the spatial
    # arity (1D -> length-1, 2D -> length-2).
    spatial_ndim = len(shape) - 2
    if isinstance(stride, int):
        stride = (stride,) * spatial_ndim
    if isinstance(padding, int):
        padding = (padding,) * spatial_ndim
    if isinstance(dilation, int):
        dilation = (dilation,) * spatial_ndim
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * spatial_ndim

    torch.backends.cudnn.allow_tf32 = False

    # ggb is only meaningful for a biased convolution.
    if not has_bias and has_ggb:
        pytest.skip("ggb requires a forward bias")

    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    w = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    if has_bias:
        # Bias has one entry per total output channel: for a non-transposed
        # conv the weight is (out_c, in_c/groups, kH, kW); for a transposed conv
        # it is (in_c, out_c/groups, kH, kW), so out_c = w.shape[1] * groups.
        out_channels = weight_shape[0] if not transposed else weight_shape[1] * groups
        bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
    else:
        bias = None

    y = _forward(
        x, w, bias, stride, padding, dilation, transposed, output_padding, groups
    )
    gO = torch.randn_like(y)
    ggI = torch.randn_like(x) if has_ggI else None
    ggW = torch.randn_like(w) if has_ggW else None
    ggb = torch.randn_like(bias) if (has_bias and has_ggb) else None

    # fp64 CPU reference (upcasted), computed through aten. output_mask is
    # ignored by aten for its outputs, so pass all-True.
    output_mask = (True, True, True)
    ref_x = utils.to_reference(x, True)
    ref_w = utils.to_reference(w, True)
    ref_gO = utils.to_reference(gO, True)
    ref_ggI = utils.to_reference(ggI, True)
    ref_ggW = utils.to_reference(ggW, True)
    ref_ggb = utils.to_reference(ggb, True)

    ref = torch.ops.aten._convolution_double_backward(
        ref_ggI,
        ref_ggW,
        ref_ggb,
        ref_gO,
        ref_w,
        ref_x,
        list(stride),
        list(padding),
        list(dilation),
        transposed,
        list(output_padding),
        groups,
        output_mask,
    )

    # GEMS computation on GPU.
    with flag_gems.use_gems():
        res = torch.ops.aten._convolution_double_backward(
            ggI,
            ggW,
            ggb,
            gO,
            w,
            x,
            list(stride),
            list(padding),
            list(dilation),
            transposed,
            list(output_padding),
            groups,
            output_mask,
        )

    # fp32 keeps the default 1e-4 atol; fp16 accumulates across several summed
    # convolution contributions in grad_ggO, so loosen the absolute tolerance.
    atol = 2e-2 if dtype == torch.float16 else 1e-4

    names = ["grad_ggO", "grad_self", "grad_weight"]
    for r, ref_r, name in zip(res, ref, names):
        # A None grad-output input makes the corresponding dependency output
        # None on both sides (ggI=None -> grad_weight None; ggW=None -> grad_self
        # None); otherwise compare values.
        if ref_r is None:
            assert (
                r is None
            ), f"{name} should be None when its grad-output input is missing"
            continue
        assert r is not None, f"{name} should not be None"
        utils.gems_assert_close(r, ref_r, dtype, atol=atol)
