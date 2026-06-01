import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

# Test shapes for _convolution (using 2D shapes)
SHAPE_CONVOLUTION = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONVOLUTION)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("bias", [True, False])
def test_convolution(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    if bias is True:
        bias_tensor = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device
        )
        bias_ref = to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        bias_ref = None

    ref_weight = to_reference(weight, True)

    # Convert stride/padding/dilation to lists as expected by _convolution
    stride_list = [stride, stride]
    padding_list = [padding, padding]
    dilation_list = [dilation, dilation]
    output_padding_list = [0, 0]

    ref_out = torch._convolution(
        ref_inp,
        ref_weight,
        bias_ref,
        stride_list,
        padding_list,
        dilation_list,
        False,  # transposed
        output_padding_list,
        groups,
        False,  # benchmark
        False,  # deterministic
        True,  # cudnn_enabled
        True,  # allow_tf32
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch._convolution(
            inp,
            weight,
            bias_tensor,
            stride_list,
            padding_list,
            dilation_list,
            False,  # transposed
            output_padding_list,
            groups,
            False,  # benchmark
            False,  # deterministic
            True,  # cudnn_enabled
            True,  # allow_tf32
        )

    gems_assert_close(res_out, ref_out, dtype)


# Test shapes for _convolution 1D
SHAPE_CONVOLUTION_1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel", SHAPE_CONVOLUTION_1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_convolution_1d(monkeypatch, shape, kernel, stride, padding, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)

    stride_list = [stride]
    padding_list = [padding]
    dilation_list = [1]
    output_padding_list = [0]

    ref_out = torch._convolution(
        ref_inp,
        ref_weight,
        None,
        stride_list,
        padding_list,
        dilation_list,
        False,  # transposed
        output_padding_list,
        1,  # groups
        False,  # benchmark
        False,  # deterministic
        True,  # cudnn_enabled
        True,  # allow_tf32
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch._convolution(
            inp,
            weight,
            None,
            stride_list,
            padding_list,
            dilation_list,
            False,  # transposed
            output_padding_list,
            1,  # groups
            False,  # benchmark
            False,  # deterministic
            True,  # cudnn_enabled
            True,  # allow_tf32
        )

    gems_assert_close(res_out, ref_out, dtype)


# Test shapes for _convolution 3D
SHAPE_CONVOLUTION_3D = [
    ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
]


@pytest.mark.convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONVOLUTION_3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_convolution_3d(monkeypatch, shape, kernel, stride, padding, groups, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)

    stride_list = [stride, stride, stride]
    padding_list = [padding, padding, padding]
    dilation_list = [1, 1, 1]
    output_padding_list = [0, 0, 0]

    ref_out = torch._convolution(
        ref_inp,
        ref_weight,
        None,
        stride_list,
        padding_list,
        dilation_list,
        False,  # transposed
        output_padding_list,
        groups,
        False,  # benchmark
        False,  # deterministic
        True,  # cudnn_enabled
        True,  # allow_tf32
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch._convolution(
            inp,
            weight,
            None,
            stride_list,
            padding_list,
            dilation_list,
            False,  # transposed
            output_padding_list,
            groups,
            False,  # benchmark
            False,  # deterministic
            True,  # cudnn_enabled
            True,  # allow_tf32
        )

    gems_assert_close(res_out, ref_out, dtype)
