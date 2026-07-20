# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

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


ADDITIONAL_CONV_TRANSPOSE2D_CASES = [
    pytest.param(
        (2, 4, 4, 5),
        (4, 2, 2, 3),
        True,
        (3, 2),
        (1, 2),
        (2, 1),
        2,
        1,
        torch.float32,
        id="fp32_groups2_bias_stride3_asymmetric_output_padding",
    ),
    pytest.param(
        (2, 4, 4, 5),
        (4, 2, 3, 3),
        True,
        (4, 4),
        (1, 2),
        (3, 1),
        2,
        1,
        torch.float16,
        id="fp16_scatter_groups_bias_stride4_output_padding",
    ),
    pytest.param(
        (1, 8, 5, 4),
        (8, 4, 3, 3),
        False,
        (3, 4),
        (1, 0),
        (2, 3),
        1,
        1,
        torch.bfloat16,
        id="bf16_scatter_stride3x4_output_padding",
    ),
    pytest.param(
        (1, 3, 5, 5),
        (3, 4, 3, 2),
        True,
        2,
        (2, 1),
        1,
        1,
        (2, 1),
        torch.float16,
        id="fp16_bias_tuple_padding_dilation",
    ),
    pytest.param(
        (2, 1, 1, 1),
        (1, 3, 1, 1),
        True,
        1,
        0,
        0,
        1,
        1,
        torch.float32,
        id="fp32_1x1_kernel_single_pixel",
    ),
    pytest.param(
        (0, 2, 4, 4),
        (2, 3, 3, 3),
        False,
        1,
        1,
        0,
        1,
        1,
        torch.float32,
        id="fp32_empty_batch",
    ),
    pytest.param(
        (1, 4, 4, 4),
        (4, 2, 2, 2),
        False,
        (1, 2),
        (1, 0),
        (0, 1),
        2,
        (2, 2),
        torch.bfloat16,
        id="bf16_grouped_dilation_output_padding",
    ),
]


def _skip_if_unsupported_test_device(dtype):
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("conv_transpose2d Triton kernels require CUDA")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 conv_transpose2d requires CUDA BF16 support")


def _assert_conv_transpose2d_matches(
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

    reduce_dim = max(
        (weight_shape[0] // max(groups, 1)) * weight_shape[2] * weight_shape[3],
        1,
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
    return res_out, ref_out


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "input_shape, weight_shape, use_bias, stride, padding, output_padding, groups, "
    "dilation, dtype",
    (
        SUPPORTED_CONV_TRANSPOSE2D_CASES[:2]
        if cfg.QUICK_MODE
        else SUPPORTED_CONV_TRANSPOSE2D_CASES
    ),
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
    _assert_conv_transpose2d_matches(
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
    )


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "input_shape, weight_shape, use_bias, stride, padding, output_padding, groups, "
    "dilation, dtype",
    (
        ADDITIONAL_CONV_TRANSPOSE2D_CASES[:2]
        if cfg.QUICK_MODE
        else ADDITIONAL_CONV_TRANSPOSE2D_CASES
    ),
)
def test_accuracy_conv_transpose2d_extended_parameters(
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
    _assert_conv_transpose2d_matches(
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
    )


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_invalid_groups_raise():
    _skip_if_unsupported_test_device(torch.float32)
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 1, 3, 3), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="divisible by groups"):
        flag_gems.conv_transpose2d(inp, weight, groups=3)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "case, match",
    [
        ("groups_zero", "positive integer"),
        ("channel_mismatch", "match weight input channels"),
        ("bias_size", "one element per output channel"),
        ("stride_zero", "non-positive stride"),
        ("negative_padding", "negative padding"),
        ("invalid_output_padding", "smaller than either stride or dilation"),
        ("dilation_zero", "greater than zero"),
        ("output_too_small", "output size is too small"),
    ],
)
def test_conv_transpose2d_invalid_arguments_raise(case, match):
    _skip_if_unsupported_test_device(torch.float32)
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 3, 3, 3), dtype=torch.float32, device=flag_gems.device)
    bias = torch.randn((3,), dtype=torch.float32, device=flag_gems.device)
    kwargs = {
        "bias": bias,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "dilation": 1,
    }

    if case == "groups_zero":
        kwargs["groups"] = 0
    elif case == "channel_mismatch":
        weight = torch.randn((3, 3, 3, 3), dtype=torch.float32, device=flag_gems.device)
    elif case == "bias_size":
        kwargs["bias"] = torch.randn((4,), dtype=torch.float32, device=flag_gems.device)
    elif case == "stride_zero":
        kwargs["stride"] = 0
    elif case == "negative_padding":
        kwargs["padding"] = -1
    elif case == "invalid_output_padding":
        kwargs["stride"] = 2
        kwargs["dilation"] = 2
        kwargs["output_padding"] = 2
    elif case == "dilation_zero":
        kwargs["dilation"] = 0
    elif case == "output_too_small":
        kwargs["padding"] = 8

    with pytest.raises(RuntimeError, match=match):
        flag_gems.conv_transpose2d(inp, weight, **kwargs)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "argument", ["stride", "padding", "output_padding", "dilation"]
)
def test_conv_transpose2d_rejects_triplet_hyperparameters(argument):
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32)
    weight = torch.randn((2, 3, 3, 3), dtype=torch.float32)
    kwargs = {
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "dilation": 1,
    }
    kwargs[argument] = (1, 1, 1)

    with pytest.raises(RuntimeError, match="single int or a pair"):
        flag_gems.conv_transpose2d(inp, weight, **kwargs)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_unsupported_dtype_raise():
    _skip_if_unsupported_test_device(torch.float32)
    inp = torch.randn((1, 2, 4, 5), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 3, 3, 3), dtype=torch.float64, device=flag_gems.device)
    bias = torch.randn((3,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="dtype"):
        flag_gems.conv_transpose2d(inp, weight, bias=bias)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_unbatched_3d_matches_pytorch(monkeypatch):
    res_out, ref_out = _assert_conv_transpose2d_matches(
        monkeypatch,
        (2, 4, 5),
        (2, 3, 3, 2),
        True,
        (2, 1),
        (1, 0),
        (1, 0),
        1,
        1,
        torch.float32,
    )

    assert res_out.dim() == ref_out.dim() == 3


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("noncontiguous", ["input", "weight", "bias", "all"])
def test_conv_transpose2d_noncontiguous_tensors_match_pytorch(
    monkeypatch, noncontiguous
):
    _skip_if_unsupported_test_device(torch.float32)
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    inp = torch.randn((1, 2, 4, 5), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 3, 3, 2), dtype=torch.float32, device=flag_gems.device)
    bias = torch.randn((3,), dtype=torch.float32, device=flag_gems.device)

    if noncontiguous in ("input", "all"):
        inp = torch.randn((1, 2, 4, 10), dtype=torch.float32, device=flag_gems.device)[
            :, :, :, ::2
        ]
    if noncontiguous in ("weight", "all"):
        weight = torch.randn(
            (2, 3, 3, 4), dtype=torch.float32, device=flag_gems.device
        )[:, :, :, ::2]
    if noncontiguous in ("bias", "all"):
        bias = torch.randn((6,), dtype=torch.float32, device=flag_gems.device)[::2]

    if noncontiguous in ("input", "all"):
        assert not inp.is_contiguous()
    if noncontiguous in ("weight", "all"):
        assert not weight.is_contiguous()
    if noncontiguous in ("bias", "all"):
        assert not bias.is_contiguous()

    ref_out = torch.nn.functional.conv_transpose2d(
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        bias=utils.to_reference(bias, True),
        stride=(2, 1),
        padding=(1, 0),
        output_padding=(1, 0),
    ).to(torch.float32)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias,
        stride=(2, 1),
        padding=(1, 0),
        output_padding=(1, 0),
    )

    reduce_dim = max(weight.shape[0] * weight.shape[2] * weight.shape[3], 1)
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=reduce_dim)
