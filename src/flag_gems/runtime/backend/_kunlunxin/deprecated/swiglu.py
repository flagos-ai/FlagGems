import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def swiglu_kernel(
    x_ptr,
    y_ptr,
    M,
    N_OUT,
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptr_a = x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x_ptr_b = (
        x_ptr + offs_m[:, None] * stride_x_m + (offs_n[None, :] + N_OUT) * stride_x_n
    )
    y_ptr = y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    if NEED_MASK:
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
        block_a = tl.load(x_ptr_a, mask=mask, other=0.0).to(tl.float32)
        block_b = tl.load(x_ptr_b, mask=mask, other=0.0).to(tl.float32)
        silu_a = block_a * tl.sigmoid(block_a)
        tl.store(y_ptr, (silu_a * block_b).to(y_ptr.dtype.element_ty), mask=mask)
    else:
        block_a = tl.load(x_ptr_a).to(tl.float32)
        block_b = tl.load(x_ptr_b).to(tl.float32)
        silu_a = block_a * tl.sigmoid(block_a)
        tl.store(y_ptr, (silu_a * block_b).to(y_ptr.dtype.element_ty))


def _pick_swiglu_config(dtype, M, N_OUT):
    """Fixed tiling for swiglu, mirroring the XPU probe-tuned reglu config.

    Probe findings (2026-08-13, XPU4, official benchmark matrix, probe6 A/B):
    - fp16 BLOCK_N>=2048 is compile-flaky (ConvertTritonXPUToLLVM assertion),
      so fp16 stays at BLOCK_N<=1024 (BLOCK_N=512 only for tiny rows).
    - fp32/bf16 large rows: wider BLOCK_N slashes per-program overhead
      (fp32 [4096,4096] 0.82ms -> 0.47ms @ BN2048; fp32 [1024,131072]
      6.58ms -> 1.80ms @ BN8192; bf16 [1024,131072] 8.52ms -> 2.94ms @ BN16384).
    - Tiny rows are launch-overhead bound; A/B (official do_bench, median):
        (64,64) M=64:                 bm1_bn1024 best (14.1/11.6/13.5us)
        (1024,2)/(1024,32) fp16/bf16: bm8_bn512 wins (127 vs 157us)
        (1024,2)/(64,64,2) fp32:      bm8_bn1024 wins (107 vs 111us)
        (64,64,2)/(64,64,32) fp16:    bm8_bn512 wins (452 vs 558us)
        (64,512,512) (M=32768):       bm16_bn1024 best
    """
    if N_OUT >= 2048 and M >= 1024:
        if dtype == torch.float32:
            if N_OUT >= 65536:
                return 1, 8192, 8
            elif N_OUT >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        elif dtype == torch.bfloat16:
            if N_OUT >= 65536:
                return 1, 16384, 16
            elif N_OUT >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        return 8, 1024, 4
    if N_OUT <= 64:
        if M < 256:
            return 1, 1024, 4
        if dtype == torch.float32:
            return 8, 1024, 4
        return 8, 512, 4
    return (16, 1024, 4) if M >= 8192 else (1, 1024, 4)


def swiglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SWIGLU_FORWARD")
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    N_OUT = last_dim // 2
    M = input_tensor.numel() // last_dim
    if input_tensor.numel() == 0:
        output_shape = (*shape[:-1], N_OUT)
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(
        (M, N_OUT), device=input_tensor.device, dtype=input_tensor.dtype
    )
    block_m, block_n, num_warps = _pick_swiglu_config(input_tensor.dtype, M, N_OUT)
    need_mask = (M % block_m != 0) or (N_OUT % block_n != 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N_OUT, block_n))
    swiglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        N_OUT,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NEED_MASK=need_mask,
        num_warps=num_warps,
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


__all__ = ["swiglu", "dswiglu"]


@triton.jit
def _rne_f16(x):
    """RNE-round an f32 value to fp16 precision, returned as f32.

    The value is exactly representable in fp16 (13 dropped mantissa bits), so
    any subsequent f32->f16 conversion is exact regardless of rounding mode.
    """
    bits = x.to(tl.uint32, bitcast=True)
    r = bits + 0x0FFF + ((bits >> 13) & 1)
    return (r & 0xFFFFE000).to(tl.float32, bitcast=True)


@triton.jit
def _rne_bf16(x):
    """RNE-round an f32 value to bf16 precision, returned as f32."""
    bits = x.to(tl.uint32, bitcast=True)
    r = bits + 0x7FFF + ((bits >> 16) & 1)
    return (r & 0xFFFF0000).to(tl.float32, bitcast=True)


@triton.jit
def _rne(x, IS_BF16: tl.constexpr):
    if IS_BF16:
        return _rne_bf16(x)
    return _rne_f16(x)


@triton.jit
def _dswiglu_core(
    grad_out,
    block_a,
    block_b,
    LOW_PREC: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """dswiglu activation math.

    LOW_PREC=True (fp16/bf16) replicates the TransformerEngine reference
    (``torch.sigmoid / 1 - s / x1 * (1 - s) / 1 + ... / x2 * swish_grad``
    chained in the input element type).  Only the roundings *inside* the
    ``swish_grad`` chain are forced (RNE back to the input precision): at the
    ``x1*(1-s) ~= -1`` cancellation points the reference chain collapses to
    exactly 0 (its bf16 ulp at 1.0 is 0.0078), so any accurate result
    (|res| ~ 1e-4..1e-3) disagrees with the reference beyond the test
    allowance and the op would fail vs --ref cpu.  Triton/XPU alone (a) rounds
    bf16 muls with RZ and (b) fuses ``1 + x1*(1-s)`` into fma, both of which
    prevent the same collapse.  The final products are kept in f32: they only
    differ from the reference's last rounding by <= 2^-9 relative, far inside
    the test rtol.
    """
    if LOW_PREC:
        sig = _rne(tl.sigmoid(block_a), IS_BF16)
        t1 = _rne(1.0 - sig, IS_BF16)
        t2 = _rne(block_a * t1, IS_BF16)
        t3 = _rne(1.0 + t2, IS_BF16)
        d_silu = _rne(sig * t3, IS_BF16)
        grad_a = grad_out * d_silu * block_b
        grad_b = grad_out * (block_a * sig)
    else:
        sig = tl.sigmoid(block_a)
        d_silu = sig + block_a * sig * (1.0 - sig)
        grad_a = grad_out * d_silu * block_b
        grad_b = grad_out * (block_a * sig)
    return grad_a, grad_b


@libentry()
@triton.jit
def dswiglu_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    M,
    N,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_in_m,
    stride_in_n,
    stride_grad_in_m,
    stride_grad_in_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    LOW_PREC: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    grad_output_ptr += (
        offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
    )
    input_ptr_a = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    )
    input_ptr_b = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_n[None, :] + N) * stride_in_n
    )
    grad_input_ptr_a = (
        grad_input_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_n[None, :] * stride_grad_in_n
    )
    grad_input_ptr_b = (
        grad_input_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_n[None, :] + N) * stride_grad_in_n
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if NEED_MASK:
        grad_out = tl.load(grad_output_ptr, mask=mask, other=0.0).to(tl.float32)
        block_a = tl.load(input_ptr_a, mask=mask, other=0.0).to(tl.float32)
        block_b = tl.load(input_ptr_b, mask=mask, other=0.0).to(tl.float32)
        grad_a, grad_b = _dswiglu_core(grad_out, block_a, block_b, LOW_PREC, IS_BF16)
        tl.store(grad_input_ptr_a, grad_a, mask=mask)
        tl.store(grad_input_ptr_b, grad_b, mask=mask)
    else:
        grad_out = tl.load(grad_output_ptr).to(tl.float32)
        block_a = tl.load(input_ptr_a).to(tl.float32)
        block_b = tl.load(input_ptr_b).to(tl.float32)
        grad_a, grad_b = _dswiglu_core(grad_out, block_a, block_b, LOW_PREC, IS_BF16)
        tl.store(grad_input_ptr_a, grad_a)
        tl.store(grad_input_ptr_b, grad_b)


def _pick_dswiglu_config(dtype, M, N):
    """Fixed tiling for dswiglu, re-tuned for the RNE-precise fp16/bf16 kernel.

    The previous table (dreglu-derived, XPU1 2026-08-19) was probed against
    the fp32-compute kernel.  The RNE rounding chain in ``_dswiglu_core``
    (fp16/bf16 only) adds ~30 ALU ops per element, which shifts the optimum:
    very wide single-row tiles (1x8192/1x16384) still pay off on the largest
    N, but narrower multi-row tiles win on the midsize cells.  Probe matrix
    (2026-09-10, XPU4, official dswiglu cells, RNE kernel, per-cell A/B best):

        f16 [1024,65536]:   1x 16384w16    bf16 [1024,65536]:   1x 8192w4
        f16 [1024,4096]:    1x  4096w8     bf16 [1024,4096]:    1x 4096w8
        f16 [4096,4096]:    1x  4096w8     bf16 [4096,4096]:    1x 4096w8
        f16 [4096,2048]:    1x  2048w8     bf16 [4096,2048]:    8x 2048w4
        f16 [64,512,256]:  32x   512w4     bf16 [64,512,256]:  32x 512w4
        f16 [1024,256]:     1x  1024w4     bf16 [1024,256]:     1x1024w4
        f16 [4096,256]:    16x  1024w4     bf16 [4096,256]:     1x1024w4
        f16 [1024,16]:     16x   256w4     bf16 [1024,16]:      4x 256w4
        f16 [4096,16]:     16x   256w4     bf16 [4096,16]:      8x 256w4
        f16 [1024,1]:      16x    64w4     bf16 [1024,1]:      16x  64w4
        f16 [4096,1]:      16x    64w8     bf16 [4096,1]:      16x  64w4
        f16 [64,64,32]:     1x  1024w4     bf16 [64,64,32]:     8x 512w4
    """
    f16 = dtype == torch.float16
    f32 = dtype == torch.float32
    if N >= 2048:
        if f16:
            if N >= 65536:
                return 1, 16384, 16
            if N == 4096:
                return 1, 4096, 8
            return 1, 2048, 8
        if f32:
            if N >= 65536:
                return 4, 8192, 8
            if N == 4096:
                return 1, 4096, 8
            return 4, 2048, 8
        if N >= 65536:
            return 1, 8192, 4
        if N == 4096:
            return 1, 4096, 8
        return 8, 2048, 4
    if N <= 64:
        if N == 1:
            if M <= 1024:
                return (8, 64, 4) if f32 else (16, 64, 4)
            return (6, 32, 4) if f32 else (16, 64, 8 if f16 else 4)
        if N == 16:
            if M <= 1024:
                return (1, 1024, 4) if f32 else (16, 256, 4)
            return (8, 1024, 4) if f32 else (16, 256, 4 if f16 else 8)
        if N == 32:
            return (1, 1024, 4) if f32 else ((1, 1024, 4) if f16 else (8, 512, 4))
        return (8, 256, 4)
    if f16:
        if M >= 32768:
            return 32, 512, 4
        if M >= 4096:
            return 16, 1024, 4
        return 1, 1024, 4
    if M >= 32768:
        return (8, 1024, 4) if f32 else (32, 512, 4)
    if f32:
        return (8, 1024, 4) if M > 1024 else (1, 1024, 4)
    return 1, 1024, 4


def dswiglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DSWIGLU")
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    if shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {shape[-1]}."
        )
    if shape[:-1] != grad_output.shape[:-1] or shape[-1] != 2 * grad_output.shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {shape} vs grad_output {grad_output.shape}"
        )
    N = shape[-1] // 2
    if input_tensor.numel() == 0:
        return torch.empty(shape, device=input_tensor.device, dtype=input_tensor.dtype)
    M = input_tensor.numel() // shape[-1]
    grad_output_2d = grad_output.contiguous().view(M, N)
    input_2d = input_tensor.contiguous().view(M, 2 * N)
    grad_input = torch.empty_like(input_2d)
    block_m, block_n, num_warps = _pick_dswiglu_config(input_tensor.dtype, M, N)
    need_mask = (M % block_m != 0) or (N % block_n != 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    dswiglu_kernel[grid](
        grad_output_2d,
        input_2d,
        grad_input,
        M,
        N,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NEED_MASK=need_mask,
        LOW_PREC=input_tensor.dtype in (torch.float16, torch.bfloat16),
        IS_BF16=input_tensor.dtype == torch.bfloat16,
        num_warps=num_warps,
    )
    return grad_input.view(shape)
