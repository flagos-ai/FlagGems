import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SUPPORTED_CONV_TRANSPOSE2D_CASES = [
    pytest.param(
        (16, 32, 8, 8),
        (32, 24, 5, 5),
        2,
        2,
        torch.float16,
        id="fp16_direct_16x32x8",
    ),
    pytest.param(
        (32, 64, 16, 16),
        (64, 32, 3, 3),
        2,
        1,
        torch.float32,
        id="fp32_direct_32x64x16",
    ),
    pytest.param(
        (16, 32, 32, 32),
        (32, 64, 3, 3),
        2,
        1,
        torch.bfloat16,
        id="bf16_direct_16x32x32",
    ),
]


def _skip_if_unsupported_test_device(dtype):
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("conv_transpose2d Triton kernels require CUDA")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 conv_transpose2d requires CUDA BF16 support")


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "input_shape, weight_shape, stride, padding, dtype",
    SUPPORTED_CONV_TRANSPOSE2D_CASES,
)
def test_accuracy_conv_transpose2d_supported(
    monkeypatch,
    input_shape,
    weight_shape,
    stride,
    padding,
    dtype,
):
    _skip_if_unsupported_test_device(dtype)
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    torch.backends.cudnn.allow_tf32 = False
    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=0,
        groups=1,
        dilation=1,
    ).to(dtype)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=0,
        groups=1,
        dilation=1,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_unsupported_shape_raises():
    inp = torch.randn((1, 2, 4, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((2, 1, 3, 3), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="supports only tuned Triton cases"):
        flag_gems.conv_transpose2d(inp, weight)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"bias": "tensor"}, id="bias"),
        pytest.param({"output_padding": 1}, id="output_padding"),
        pytest.param({"dilation": 2}, id="dilation"),
        pytest.param({"groups": 2}, id="groups"),
    ],
)
def test_conv_transpose2d_unsupported_schema_variants_raise(kwargs):
    kwargs = dict(kwargs)
    inp = torch.randn((16, 32, 8, 8), dtype=torch.float16, device=flag_gems.device)
    weight_shape = (32, 12, 5, 5) if kwargs.get("groups") == 2 else (32, 24, 5, 5)
    weight = torch.randn(weight_shape, dtype=torch.float16, device=flag_gems.device)
    bias = kwargs.pop("bias", None)
    if bias == "tensor":
        bias = torch.randn((24,), dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="supports only tuned Triton cases"):
        flag_gems.conv_transpose2d(
            inp,
            weight,
            bias=bias,
            stride=2,
            padding=2,
            **kwargs,
        )
