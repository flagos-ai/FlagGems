import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SUPPORTED_CONV_TRANSPOSE2D_CASES = [
    pytest.param(
        (16, 32, 8, 8),
        (32, 24, 5, 5),
        False,
        2,
        2,
        0,
        1,
        1,
        torch.float16,
        id="fp16_direct_16x32x8",
    ),
    pytest.param(
        (32, 64, 16, 16),
        (64, 32, 3, 3),
        False,
        2,
        1,
        0,
        1,
        1,
        torch.float32,
        id="fp32_direct_32x64x16",
    ),
    pytest.param(
        (32, 64, 32, 32),
        (64, 32, 3, 3),
        False,
        1,
        0,
        0,
        1,
        1,
        torch.float32,
        id="fp32_direct_benchmark_stride1",
    ),
    pytest.param(
        (16, 32, 32, 32),
        (32, 64, 3, 3),
        False,
        2,
        1,
        0,
        1,
        1,
        torch.bfloat16,
        id="bf16_direct_16x32x32",
    ),
    pytest.param(
        (1, 2, 4, 4),
        (2, 3, 3, 3),
        True,
        1,
        0,
        0,
        1,
        1,
        torch.float32,
        id="fp32_general_bias_shape_previously_unsupported",
    ),
    pytest.param(
        (2, 4, 5, 4),
        (4, 3, 3, 2),
        True,
        (2, 1),
        (1, 0),
        (1, 0),
        2,
        1,
        torch.float16,
        id="fp16_general_groups_asymmetric_stride_output_padding",
    ),
    pytest.param(
        (1, 2, 4, 4),
        (2, 3, 2, 3),
        False,
        1,
        (2, 1),
        (1, 0),
        1,
        (2, 1),
        torch.float32,
        id="fp32_general_dilation_output_padding",
    ),
    pytest.param(
        (1, 3, 3, 5),
        (3, 2, 2, 2),
        False,
        (2, 2),
        (0, 1),
        (0, 1),
        1,
        1,
        torch.bfloat16,
        id="bf16_general_non_tuned",
    ),
]


def _skip_if_unsupported_test_device(dtype):
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("conv_transpose2d Triton kernels require CUDA")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 conv_transpose2d requires CUDA BF16 support")


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "input_shape, weight_shape, use_bias, stride, padding, output_padding, groups, "
    "dilation, dtype",
    SUPPORTED_CONV_TRANSPOSE2D_CASES,
)
def test_accuracy_conv_transpose2d_supported(
    monkeypatch,
    input_shape,
    weight_shape,
    use_bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
    dtype,
):
    _skip_if_unsupported_test_device(dtype)
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    out_channels = weight_shape[1] * groups
    bias = None
    ref_bias = None
    if use_bias:
        bias = torch.randn((out_channels,), dtype=dtype, device=flag_gems.device)
        ref_bias = utils.to_reference(bias, True)

    torch.backends.cudnn.allow_tf32 = False
    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=ref_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_invalid_groups_raise():
    _skip_if_unsupported_test_device(torch.float32)
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 1, 3, 3), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="divisible by groups"):
        flag_gems.conv_transpose2d(inp, weight, groups=3)
