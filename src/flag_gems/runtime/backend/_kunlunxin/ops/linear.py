import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_FAST_MODE_ENV = "XMLIR_MATMUL_FAST_MODE"


def _set_matmul_fast_mode(a_dtype, M, N, K):
    """Mirror of mm.py: XMLIR_MATMUL_FAST_MODE=1 speeds the bf16 tl.dot
    lowering for large-K GEMMs (K >= 2048, M/N >= 128); small bf16 shapes
    regress, so apply it selectively."""
    if a_dtype == torch.bfloat16 and K >= 2048 and M >= 128 and N >= 128:
        saved = os.environ.get(_FAST_MODE_ENV)
        os.environ[_FAST_MODE_ENV] = "1"
        return saved
    return None


def _restore_matmul_fast_mode(saved):
    if saved is None:
        os.environ.pop(_FAST_MODE_ENV, None)
    else:
        os.environ[_FAST_MODE_ENV] = saved


@libentry()
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bn,
    BIAS: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear kernel: y = x @ W^T + b
    - input: (M, K) where M is batch size (flattened), K is in_features
    - weight: (N, K) where N is out_features
    - bias: (N,) optional
    - output: (M, N)

    W is row-major (N, K): load the (BLOCK_N, BLOCK_K) tile in its natural
    (coalesced) order and transpose in registers for the dot.  Loading the
    (BLOCK_K, BLOCK_N) tile directly (the pre-20260909 indexing) is one scalar
    load per lane on this backend (inner stride K) and costs ~3x.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)

    weight_ptrs = weight_ptr + (
        offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            a = tl.load(input_ptrs)
            w = tl.load(weight_ptrs)
        else:
            input_mask_m = offs_m < M
            input_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            input_mask = input_mask_m[:, None] & input_mask_k[None, :]

            a = tl.load(input_ptrs, mask=input_mask, other=0.0)

            weight_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            weight_mask_n = offs_n < N
            weight_mask = weight_mask_n[:, None] & weight_mask_k[None, :]

            w = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        b = tl.trans(w)

        accumulator += tl.dot(a, b, allow_tf32=False)

        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (
        offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    )

    if BIAS:
        bias_ptrs = bias_ptr + offs_on
        if EVEN:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator = accumulator + bias

    output = accumulator.to(output_ptr.dtype.element_ty)
    if EVEN:
        tl.store(output_ptrs, output)
    else:
        output_mask_m = offs_m < M
        output_mask_n = offs_n < N
        output_mask = output_mask_m[:, None] & output_mask_n[None, :]
        tl.store(output_ptrs, output, mask=output_mask)


def _block_m(M):
    return 128 if M <= 512 else 256


def _block_n(N):
    return 128 if N <= 512 else 256


def _block_k(M, N):
    if M <= 512 and N <= 512:
        return 128
    return 256


def _num_warps(M, N):
    return 4 if (M <= 512 and N <= 512) else 8


def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Args:
        input: Input tensor of shape (*, in_features) where * means any number of
               additional dimensions, including none.
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (*, out_features)
    """
    logger.debug("GEMS_KUNLUNXIN LINEAR")

    if (input.dtype == torch.bfloat16 and input.shape[-1] >= 2048) or (
        input.dtype == torch.float32
        and (input.shape[-1] >= 32768 or input.numel() >= 2**25)
    ):
        return _linear_legacy_kernel(input, weight, bias)

    input_dim = input.dim()
    if input_dim == 1:
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]
    N = weight.shape[0]

    input_flat = input.view(M, K)

    weight = weight.contiguous()

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    blk_m = _block_m(M)
    blk_n = _block_n(N)
    blk_k = _block_k(M, N)
    num_warps = _num_warps(M, N)
    grid = lambda META: (
        triton.cdiv(M, blk_m),
        triton.cdiv(N, blk_n),
    )

    saved = _set_matmul_fast_mode(input.dtype, M, N, K)
    try:
        with torch_device_fn.device(input.device):
            linear_kernel[grid](
                input_flat,
                weight,
                bias if bias is not None else weight,
                output,
                M,
                N,
                K,
                input_flat.stride(0),
                input_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                output.stride(0),
                output.stride(1),
                bias.stride(0) if bias is not None else 0,
                BIAS=bias is not None,
                EVEN=((M % blk_m == 0) and (N % blk_n == 0) and (K % blk_k == 0)),
                BLOCK_SIZE_M=blk_m,
                BLOCK_SIZE_N=blk_n,
                BLOCK_SIZE_K=blk_k,
                num_warps=num_warps,
            )
    finally:
        _restore_matmul_fast_mode(saved)

    output = output.view(*batch_dims, N)

    if single_1d:
        output = output.squeeze(0)

    return output


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("linear"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def linear_kernel_legacy(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bn,
    BIAS: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)

    weight_ptrs = weight_ptr + (
        offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            a = tl.load(input_ptrs)
            b = tl.load(weight_ptrs)
        else:
            input_mask_m = offs_m < M
            input_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            input_mask = input_mask_m[:, None] & input_mask_k[None, :]

            a = tl.load(input_ptrs, mask=input_mask, other=0.0)

            weight_mask_k = offs_k < (K - k * BLOCK_SIZE_K)
            weight_mask_n = offs_n < N
            weight_mask = weight_mask_k[:, None] & weight_mask_n[None, :]

            b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=False)

        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (
        offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    )

    if BIAS:
        bias_ptrs = bias_ptr + offs_on
        if EVEN:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator = accumulator + bias

    output = accumulator.to(output_ptr.dtype.element_ty)
    if EVEN:
        tl.store(output_ptrs, output)
    else:
        output_mask_m = offs_m < M
        output_mask_n = offs_n < N
        output_mask = output_mask_m[:, None] & output_mask_n[None, :]
        tl.store(output_ptrs, output, mask=output_mask)


def _even(matrix, block):
    return (matrix % block) == 0


_EVEN_M = 128
_EVEN_N = 256
_EVEN_K = 64


def _linear_legacy_kernel(input, weight, bias=None):
    """Verbatim batch-20260908 vendor wrapper (autotuned trans-free kernel)."""
    input_dim = input.dim()
    if input_dim == 1:
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]
    N = weight.shape[0]

    input_flat = input.view(M, K)
    weight = weight.contiguous()

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    with torch_device_fn.device(input.device):
        linear_kernel_legacy[grid](
            input_flat,
            weight,
            bias if bias is not None else weight,
            output,
            M,
            N,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            bias.stride(0) if bias is not None else 0,
            BIAS=bias is not None,
            EVEN=(_even(M, _EVEN_M) and _even(N, _EVEN_N) and _even(K, _EVEN_K)),
        )

    output = output.view(*batch_dims, N)

    if single_1d:
        output = output.squeeze(0)

    return output
