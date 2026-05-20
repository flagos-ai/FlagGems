"""Accuracy tests for ``torch.nn.functional.conv_transpose2d``.

Coverage matrix (FlagGems competition rubric §4.1.4):
    sizes      small (1x1, 8x8) / regular (32x32, 64x64) / large (128x128)
    dims       4D only (operator definition); empty batch is tested separately
    params     stride, padding, output_padding, groups, dilation, bias each
               toggled across default / scalar / per-axis-tuple modes
    dtypes     torch.float16, torch.bfloat16, torch.float32
    branches   fp32 tf32 tile path, fp16/bf16 mma path, scalar fallback
"""

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

CONV_TRANSPOSE2D_FUNCTIONAL_CASES = [
    pytest.param(
        (1, 4, 1, 1),
        (4, 6, 1, 1),
        False,
        1,
        0,
        0,
        1,
        1,
        torch.float32,
        id="fp32_small_1x1_kernel",
    ),
    pytest.param(
        (1, 4, 8, 8),
        (4, 6, 3, 3),
        True,
        1,
        1,
        0,
        1,
        1,
        torch.float32,
        id="fp32_small_8x8_bias",
    ),
    pytest.param(
        (4, 16, 16, 16),
        (16, 32, 3, 3),
        False,
        2,
        1,
        0,
        1,
        1,
        torch.float16,
        id="fp16_stride2_padding1",
    ),
    pytest.param(
        (2, 8, 32, 32),
        (8, 16, 4, 4),
        True,
        2,
        1,
        1,
        1,
        1,
        torch.float16,
        id="fp16_stride2_kernel4_outpad1_bias",
    ),
    pytest.param(
        (2, 8, 16, 16),
        (8, 16, 3, 3),
        False,
        1,
        0,
        0,
        1,
        1,
        torch.bfloat16,
        id="bf16_default_args",
    ),
    pytest.param(
        (2, 8, 24, 24),
        (8, 16, 5, 5),
        True,
        2,
        2,
        1,
        1,
        1,
        torch.bfloat16,
        id="bf16_kernel5_padding2",
    ),
    pytest.param(
        (1, 64, 64, 64),
        (64, 32, 3, 3),
        False,
        2,
        1,
        0,
        1,
        1,
        torch.float32,
        id="fp32_64x64_stride2",
    ),
    pytest.param(
        (4, 32, 32, 32),
        (32, 32, 3, 3),
        False,
        2,
        1,
        0,
        1,
        1,
        torch.float16,
        id="fp16_4x32x32_stride2",
    ),
    pytest.param(
        (1, 32, 128, 128),
        (32, 16, 3, 3),
        False,
        1,
        1,
        0,
        1,
        1,
        torch.float32,
        id="fp32_128x128_stride1",
    ),
    pytest.param(
        (2, 8, 12, 12),
        (8, 4, 3, 3),
        False,
        1,
        0,
        0,
        4,
        1,
        torch.float32,
        id="fp32_groups4",
    ),
    pytest.param(
        (2, 6, 9, 11),
        (6, 4, 3, 2),
        True,
        (2, 1),
        (1, 0),
        (1, 0),
        2,
        1,
        torch.float16,
        id="fp16_groups2_asymmetric_stride_outpad",
    ),
    pytest.param(
        (1, 4, 8, 8),
        (4, 6, 3, 3),
        False,
        1,
        (2, 1),
        (0, 0),
        1,
        (2, 1),
        torch.float32,
        id="fp32_dilation_asymmetric_padding",
    ),
    pytest.param(
        (1, 6, 5, 5),
        (6, 3, 2, 2),
        True,
        2,
        0,
        0,
        1,
        2,
        torch.float32,
        id="fp32_dilation2_outpad0",
    ),
    pytest.param(
        (0, 4, 8, 8),
        (4, 6, 3, 3),
        False,
        1,
        1,
        0,
        1,
        1,
        torch.float32,
        id="fp32_empty_batch",
    ),
]


def _skip_if_unsupported(dtype):
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("conv_transpose2d Triton kernels require CUDA")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 conv_transpose2d needs a CUDA BF16 capable GPU")


def _run_conv_transpose2d_accuracy(
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
    _skip_if_unsupported(dtype)

    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    out_channels = weight_shape[1] * groups
    bias = (
        torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
        if use_bias
        else None
    )

    ref_inp = to_reference(inp, upcast=True)
    ref_weight = to_reference(weight, upcast=True)
    ref_bias = to_reference(bias, upcast=True) if bias is not None else None

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

    in_channels_per_group = weight_shape[0] // max(groups, 1)
    reduce_dim = in_channels_per_group * weight_shape[2] * weight_shape[3]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(reduce_dim, 1))


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "input_shape, weight_shape, use_bias, stride, padding, output_padding,"
    " groups, dilation, dtype",
    CONV_TRANSPOSE2D_FUNCTIONAL_CASES,
)
def test_conv_transpose2d_functional(
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
    _run_conv_transpose2d_accuracy(
        input_shape,
        weight_shape,
        use_bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        dtype,
    )


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_invalid_groups():
    _skip_if_unsupported(torch.float32)
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 1, 3, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.conv_transpose2d(inp, weight, groups=3)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_channel_mismatch():
    _skip_if_unsupported(torch.float32)
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((3, 3, 3, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.conv_transpose2d(inp, weight)
