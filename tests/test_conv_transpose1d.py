import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

# conv_transpose1d test shapes: (input_shape, weight_shape)
# input: (N, in_channels, L_in)
# weight: (in_channels, out_channels/groups, kernel_width)
SHAPE_CONV_TRANSPOSE1D = [
    ((2, 4, 8), (4, 8, 3)),
    ((4, 8, 16), (8, 16, 3)),
    ((2, 16, 32), (16, 32, 5)),
]


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_transpose1d(shape, kernel, stride, padding, dtype, monkeypatch):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv_transpose1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    in_channels = kernel[0]
    out_channels = kernel[1]
    kernel_width = kernel[2]
    gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_width * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_transpose1d_bias(shape, kernel, stride, padding, dtype, monkeypatch):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    out_channels = kernel[1]
    kernel_width = kernel[2]
    bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
    ref_bias = to_reference(bias, True)

    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, bias=ref_bias, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv_transpose1d(
        inp, weight, bias=bias, stride=stride, padding=padding, dilation=1
    )
    in_channels = kernel[0]
    out_channels = kernel[1]
    gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_width * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 8, 16), (8, 4, 3), 2),
        ((4, 12, 32), (12, 4, 3), 3),
    ],
)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_transpose1d_groups(
    shape, kernel, groups, stride, padding, dtype, monkeypatch
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )

    res_out = flag_gems.conv_transpose1d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )
    in_channels_per_group = kernel[0] // groups
    kernel_width = kernel[2]
    gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=kernel_width * in_channels_per_group
    )
