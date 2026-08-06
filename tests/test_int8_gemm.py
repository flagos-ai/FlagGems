# tests/test_int8_gemm.py
import pytest
import torch

import flag_gems


def to_reference(x, keep_device: bool = True):
    """
    Keep tensors on the same device by default.
    IMPORTANT: do NOT upcast int8 inputs to float64 here, otherwise reference matmul
    can get dtype-mismatched when mixed with float32 scales.
    """
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x

    if keep_device:
        return x

    # If you ever want CPU reference, keep dtype unchanged as well.
    return x.detach().cpu()


def int8_gemm_reference(
    a_int8: torch.Tensor,
    w_int8: torch.Tensor,
    a_scale,
    w_scale,
    bias=None,
    out_dtype=torch.float32,
):
    """
    Reference: dequant -> fp32 compute -> optional bias -> cast to out_dtype

    a_int8: (M, K) int8
    w_int8: (K, N) int8
    a_scale: scalar (python float or 0-d tensor)
    w_scale: scalar or (N,) tensor
    bias: None or (N,) tensor
    """
    # Always compute in float32 to avoid dtype pollution (e.g., float64 w_scale).
    compute_dtype = torch.float32

    # a_scale: allow python float or tensor
    if torch.is_tensor(a_scale):
        a_scale_t = a_scale.to(device=a_int8.device, dtype=compute_dtype)
    else:
        a_scale_t = torch.tensor(a_scale, device=a_int8.device, dtype=compute_dtype)

    # w_scale: allow python float or tensor (scalar or per-channel (N,))
    if torch.is_tensor(w_scale):
        w_scale_t = w_scale.to(device=w_int8.device, dtype=compute_dtype)
    else:
        w_scale_t = torch.tensor(w_scale, device=w_int8.device, dtype=compute_dtype)

    a = a_int8.to(dtype=compute_dtype) * a_scale_t  # (M, K) fp32

    w = w_int8.to(dtype=compute_dtype)
    # Broadcast if per-channel scale (N,)
    w = w * w_scale_t  # (K, N) fp32

    out = a @ w  # (M, N) fp32

    if bias is not None:
        out = out + bias.to(dtype=compute_dtype)

    return out.to(out_dtype)


MNK_SHAPES = [
    (16, 32, 64),
    (33, 65, 127),
    (256, 512, 1024),
]


@pytest.mark.int8_gemm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("w_scale_mode", ["scalar", "per_channel"])
@pytest.mark.parametrize("bias_mode", ["none", "fp16", "fp32"])
def test_accuracy_int8_gemm(M, N, K, out_dtype, w_scale_mode, bias_mode):
    # Inputs
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=flag_gems.device)
    w = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=flag_gems.device)

    # Scales
    a_scale = 0.02
    if w_scale_mode == "scalar":
        w_scale = 0.03
    else:
        w_scale = (
            torch.rand((N,), device=flag_gems.device, dtype=torch.float32) * 0.05
            + 0.001
        )

    # Bias
    if bias_mode == "none":
        bias = None
    elif bias_mode == "fp16":
        bias = torch.randn((N,), device=flag_gems.device, dtype=torch.float16)
    else:
        bias = torch.randn((N,), device=flag_gems.device, dtype=torch.float32)

    # Reference (keep device; do NOT upcast inputs)
    ref_a = to_reference(a, True)
    ref_w = to_reference(w, True)
    ref_bias = to_reference(bias, True) if bias is not None else None
    ref_w_scale = to_reference(w_scale, True) if torch.is_tensor(w_scale) else w_scale

    ref_out = int8_gemm_reference(
        ref_a,
        ref_w,
        a_scale,
        ref_w_scale,
        bias=ref_bias,
        out_dtype=out_dtype,
    )

    # DUT
    with flag_gems.use_gems():
        out = flag_gems.ops.int8_gemm(
            a,
            w,
            a_scale=a_scale,
            w_scale=w_scale,
            bias=bias,
            out_dtype=out_dtype,
        )

    # Tolerances
    # int8 dequant + matmul + cast fp16: allow slightly looser tolerance
    if out_dtype == torch.float16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 2e-2, 2e-2

    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.int8_gemm
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_accuracy_int8_gemm_noncontiguous_inputs(out_dtype):
    # Create non-contiguous a and w by slicing/transpose patterns
    M, N, K = (33, 127, 65)
    a_base = torch.randint(
        -128, 127, (M, K * 2), dtype=torch.int8, device=flag_gems.device
    )
    a = a_base[:, ::2]  # (M, K), non-contiguous stride

    w_base = torch.randint(
        -128, 127, (N, K), dtype=torch.int8, device=flag_gems.device
    )
    w = w_base.t()  # (K, N), likely non-contiguous

    a_scale = 0.02
    w_scale = torch.rand((N,), device=flag_gems.device, dtype=torch.float32) * 0.05 + 0.001
    bias = torch.randn((N,), device=flag_gems.device, dtype=torch.float32)

    ref_a = to_reference(a, True)
    ref_w = to_reference(w, True)
    ref_w_scale = to_reference(w_scale, True)
    ref_bias = to_reference(bias, True)

    ref_out = int8_gemm_reference(
        ref_a,
        ref_w,
        a_scale,
        ref_w_scale,
        bias=ref_bias,
        out_dtype=out_dtype,
    )

    with flag_gems.use_gems():
        out = flag_gems.ops.int8_gemm(
            a,
            w,
            a_scale=a_scale,
            w_scale=w_scale,
            bias=bias,
            out_dtype=out_dtype,
        )

    if out_dtype == torch.float16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 2e-2, 2e-2

    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)