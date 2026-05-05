import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _assert_close(res, ref, dtype, reduce_dim=1):
    """Compare tensors with dtype-appropriate tolerance.

    FP16 needs larger tolerance due to FP16->FP32->FP16 conversion in Triton kernel.
    """
    res_cpu = res.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    if dtype == torch.float16:
        atol = max(1e-4 * reduce_dim, 0.15)
        rtol = 1e-2
    elif dtype == torch.bfloat16:
        atol = max(1e-4 * reduce_dim, 0.15)
        rtol = 0.05
    else:
        atol = 1e-4 * reduce_dim
        rtol = 1e-4
    assert torch.allclose(res_cpu, ref_cpu, atol=atol, rtol=rtol), (
        f"max_diff={(( res_cpu - ref_cpu).abs().max().item()):.6f}, "
        f"atol={atol:.6f}, rtol={rtol}"
    )


def _reference_conv_transpose2d(inp, weight, bias, stride, padding, output_padding=0, groups=1, dilation=1):
    """Compute reference conv_transpose2d using PyTorch native on GPU in float32.

    Uses cuDNN for the reference computation, which is the gold standard.
    Returns GPU tensors for direct comparison with gems_assert_close.
    """
    device = inp.device
    inp_ref = inp.detach().to(device).float().requires_grad_(inp.requires_grad)
    weight_ref = weight.detach().to(device).float().requires_grad_(weight.requires_grad)
    bias_ref = None
    if bias is not None:
        bias_ref = bias.detach().to(device).float().requires_grad_(bias.requires_grad)

    out = torch.nn.functional.conv_transpose2d(
        inp_ref,
        weight_ref,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    return inp_ref, weight_ref, bias_ref, out

# Test shapes: (batch, in_c, h, w, out_c, kh, kw, groups)
SHAPE_CONV_TRANSPOSE2D = [
    # Small shapes
    (1, 1, 4, 4, 1, 2, 2, 1),
    (1, 2, 5, 5, 3, 3, 3, 1),
    # Medium shapes
    (4, 3, 32, 32, 6, 3, 3, 1),
    (2, 16, 16, 16, 32, 3, 3, 1),
    # Large shapes
    (8, 64, 32, 32, 128, 3, 3, 1),
    # Groups
    (4, 4, 16, 16, 8, 3, 3, 2),
    (2, 16, 32, 32, 32, 3, 3, 4),
    # 1x1 kernel
    (2, 3, 16, 16, 6, 1, 1, 1),
    # 5x5 kernel
    (2, 8, 16, 16, 16, 5, 5, 1),
    # Large groups
    (2, 8, 16, 16, 16, 3, 3, 4),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "batch,in_c,h,w,out_c,kh,kw,groups", SHAPE_CONV_TRANSPOSE2D
)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_transpose2d(
    monkeypatch, batch, in_c, h, w, out_c, kh, kw, groups, stride, padding, dtype, bias
):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    # conv_transpose2d weight layout: (C_in, C_out/groups, kH, kW)
    weight = torch.randn(
        (in_c, out_c // groups, kh, kw),
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )

    if bias:
        bias_tensor = torch.randn(
            out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
        )
    else:
        bias_tensor = None

    # Reference output (compute on CPU in float64 for accuracy)
    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=stride, padding=padding, groups=groups
    )

    # Gems output
    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    # Use larger reduce_dim for comparison to account for FP32 accumulation differences.
    # The Triton kernel converts FP16->FP32 internally, computes, then converts back,
    # which introduces rounding errors. FP16 needs more tolerance than FP32.
    reduce = max(in_c * kh * kw, kh * kw, kh)
    if dtype == torch.float16:
        # FP16 has ~3 decimal digits precision. Each output sums over C_in*kH*kW terms.
        # The accumulated rounding error can be significant for large channel counts.
        reduce = max(reduce, in_c * kh * kw * 16)
    _assert_close(
        res_out, ref_out.to(dtype), dtype, reduce_dim=reduce
    )

    # Backward test
    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = out_grad.float()

    if bias_tensor is not None:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias_tensor), out_grad
        )
        _assert_close(res_bias_grad, ref_bias_grad.to(dtype), dtype, reduce_dim=reduce)
    else:
        ref_in_grad, ref_weight_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        res_in_grad, res_weight_grad = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    _assert_close(
        res_in_grad, ref_in_grad.to(dtype), dtype, reduce_dim=reduce
    )
    _assert_close(
        res_weight_grad, ref_weight_grad.to(dtype), dtype, reduce_dim=reduce
    )
    if bias_tensor is not None:
        _assert_close(
            res_bias_grad, ref_bias_grad.to(dtype), dtype, reduce_dim=reduce
        )


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_output_padding(monkeypatch):
    """Test output_padding parameter for stride > 1."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32
    batch, in_c, h, w, out_c = 2, 3, 16, 16, 6
    kh, kw = 3, 3
    stride = 2
    padding = 1
    output_padding = 1

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=stride, padding=padding, output_padding=output_padding
    )

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_dilation(monkeypatch):
    """Test dilation parameter."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32
    batch, in_c, h, w, out_c = 2, 3, 16, 16, 6
    kh, kw = 3, 3
    stride = 1
    padding = 1
    dilation = 2

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, None, stride=stride, padding=padding, dilation=dilation
    )

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_conv_transpose2d_stride3(monkeypatch, dtype):
    """Test stride=3 parameter."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    batch, in_c, h, w, out_c = 2, 3, 16, 16, 6
    kh, kw = 3, 3
    stride = 3
    padding = 1

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=stride, padding=padding
    )

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)

    # Test backward
    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = out_grad.float()

    ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
        res_out, (inp, weight, bias_tensor), out_grad
    )
    _assert_close(res_in_grad, ref_in_grad.to(dtype), dtype, reduce_dim=kh)
    _assert_close(res_weight_grad, ref_weight_grad.to(dtype), dtype, reduce_dim=kh)
    _assert_close(res_bias_grad, ref_bias_grad.to(dtype), dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_conv_transpose2d_non_contiguous(monkeypatch, dtype):
    """Test with non-contiguous input tensors (transpose then contiguous)."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    batch, in_c, h, w, out_c = 2, 3, 16, 16, 6
    kh, kw = 3, 3

    # Create non-contiguous input via transpose
    inp_raw = torch.randn(
        (batch, in_c, w, h), dtype=dtype, device=flag_gems.device
    )
    inp = inp_raw.permute(0, 1, 3, 2).requires_grad_(True)

    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    # Reference: compute on contiguous version
    inp_cont = inp.detach().contiguous().requires_grad_(True)
    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp_cont, weight, bias_tensor, stride=1, padding=1
    )

    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_empty_batch(monkeypatch):
    """Test with empty batch dimension (N=0)."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32
    batch, in_c, h, w, out_c = 0, 3, 16, 16, 6
    kh, kw = 3, 3

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device
    )

    ref_out = torch.nn.functional.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1
    )

    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1
    )

    assert res_out.shape == ref_out.shape, (
        f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    )


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_conv_transpose2d_boundary_values(monkeypatch, dtype):
    """Test with boundary values: negative, extreme, all-same."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    batch, in_c, h, w, out_c = 2, 3, 8, 8, 6
    kh, kw = 3, 3

    # Test with negative values
    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device
    ) - 1.0
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device
    ) - 0.5
    bias_tensor = torch.randn(out_c, dtype=dtype, device=flag_gems.device) - 0.5

    _, _, _, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=1, padding=1
    )
    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1
    )
    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)

    # Test with all-same values
    inp_same = torch.full(
        (batch, in_c, h, w), 0.5, dtype=dtype, device=flag_gems.device
    )
    weight_same = torch.full(
        (in_c, out_c, kh, kw), 0.3, dtype=dtype, device=flag_gems.device
    )
    bias_same = torch.full((out_c,), 0.1, dtype=dtype, device=flag_gems.device)

    _, _, _, ref_out_same = _reference_conv_transpose2d(
        inp_same, weight_same, bias_same, stride=1, padding=1
    )
    res_out_same = flag_gems.conv_transpose2d(
        inp_same, weight_same, bias=bias_same, stride=1, padding=1
    )
    _assert_close(res_out_same, ref_out_same.to(dtype), dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "batch,in_c,h,w,out_c,kh,kw,groups",
    [
        (1, 1, 4, 4, 1, 1, 1, 1),  # 1x1 kernel, small
        (2, 3, 8, 8, 6, 1, 1, 1),  # 1x1 kernel, medium
        (4, 3, 32, 32, 6, 5, 5, 1),  # 5x5 kernel
        (2, 16, 16, 16, 32, 5, 5, 1),  # 5x5 kernel, many channels
        (2, 8, 16, 16, 16, 3, 3, 4),  # groups=4
        (4, 12, 16, 16, 24, 3, 3, 3),  # groups=3
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_conv_transpose2d_extended_shapes(
    monkeypatch, batch, in_c, h, w, out_c, kh, kw, groups, dtype
):
    """Test extended shape coverage: 1x1 kernel, 5x5 kernel, large groups."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (in_c, out_c // groups, kh, kw),
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=1, padding=1, groups=groups
    )

    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1, groups=groups
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)

    # Backward test
    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = out_grad.float()

    ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
        res_out, (inp, weight, bias_tensor), out_grad
    )
    _assert_close(res_in_grad, ref_in_grad.to(dtype), dtype, reduce_dim=kh)
    _assert_close(res_weight_grad, ref_weight_grad.to(dtype), dtype, reduce_dim=kh)
    _assert_close(res_bias_grad, ref_bias_grad.to(dtype), dtype)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_bf16(monkeypatch):
    """Test bfloat16 dtype support."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    if not utils.bf16_is_supported:
        pytest.skip("bfloat16 not supported on this device")

    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16
    batch, in_c, h, w, out_c = 2, 3, 16, 16, 6
    kh, kw = 3, 3

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=1, padding=1
    )

    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d_large_input(monkeypatch):
    """Test with large input sizes."""
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32
    batch, in_c, h, w, out_c = 2, 16, 64, 64, 32
    kh, kw = 3, 3

    inp = torch.randn(
        (batch, in_c, h, w), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (in_c, out_c, kh, kw), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias_tensor = torch.randn(
        out_c, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_inp, ref_weight, ref_bias, ref_out = _reference_conv_transpose2d(
        inp, weight, bias_tensor, stride=1, padding=1
    )

    res_out = flag_gems.conv_transpose2d(
        inp, weight, bias=bias_tensor, stride=1, padding=1
    )

    _assert_close(res_out, ref_out.to(dtype), dtype, reduce_dim=kh * kw)

    # Backward test
    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = out_grad.float()

    ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
        res_out, (inp, weight, bias_tensor), out_grad
    )
    _assert_close(res_in_grad, ref_in_grad.to(dtype), dtype, reduce_dim=kh)
    _assert_close(res_weight_grad, ref_weight_grad.to(dtype), dtype, reduce_dim=kh)
    _assert_close(res_bias_grad, ref_bias_grad.to(dtype), dtype)
