import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_CONV_TRANSPOSE2D = [
    # (input_shape, weight_shape [C_in, C_out/groups, kH, kW], groups)
    ((1, 2, 4, 4), (2, 1, 3, 3), 1),
    ((2, 3, 6, 6), (3, 1, 3, 3), 1),
    ((1, 4, 8, 8), (4, 1, 3, 5), 1),
    ((1, 4, 8, 8), (4, 1, 5, 3), 1),
    ((1, 3, 8, 16), (3, 2, 3, 3), 1),
    ((1, 4, 8, 8), (4, 2, 3, 3), 2),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("output_padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d(
    monkeypatch,
    shape,
    kernel,
    groups,
    stride,
    padding,
    output_padding,
    dilation,
    dtype,
    bias,
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.env("MUSA_ENABLE_SQMMA", "1")

    if flag_gems.vendor_name == "hygon":
        monkeypatch.env("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    if output_padding >= stride:
        pytest.skip("output_padding must be < stride")
    kH, kW = kernel[2], kernel[3]
    if dilation * (kH - 1) < padding or dilation * (kW - 1) < padding:
        pytest.skip("effective padding would be negative")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    if bias:
        b = torch.randn(kernel[1] * groups, dtype=dtype, device=flag_gems.device)
        ref_b = utils.to_reference(b, True)
    else:
        b = None
        ref_b = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=ref_b,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=b,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_conv_transpose2d_lowp_bias_dtype_mismatch_raises(dtype):
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("lowp conv_transpose2d fallback path requires CUDA")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("native BF16 conv_transpose2d is unsupported on this CUDA device")

    inp = torch.randn((1, 2, 4, 4), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((2, 1, 3, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((1,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.conv_transpose2d(inp, weight, bias=bias)
