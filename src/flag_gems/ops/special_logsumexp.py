import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def logsumexp_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    Kernel for logsumexp when K > 1 (non-inner dimension case).
    Computes log(sum(exp(x - max(x))), dim=dim) with numerical stability.
    Input shape: (M, N, K), Output shape: (M, K)
    """
    pid_k = tle.program_id(1)
    pid_m = tle.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        # Input offset: (M, N, K) flattened
        inp_offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Numerical stability: subtract max (across N dimension, axis=0)
        m = tl.max(inp, axis=0)  # shape: (TILE_K,)
        e = tl.exp(inp - m)
        z = tl.sum(e, axis=0)  # shape: (TILE_K,)
        out = m + tl.log(z)  # shape: (TILE_K,)
        # Output offset: (M, K) - reduced along N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        mask_k = k_offsets < K
        tl.store(output_ptrs, out, mask=mask_k)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, axis=0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), axis=0)  # (TILE_K, )
        m = m_reduced

        # Compute final logsumexp
        out = m + tl.log(z)  # shape: (TILE_K,)
        # Output offset: (M, K) - reduced along N
        out_offset = pid_m * K + k_offsets
        mask_k = k_offsets < K
        tl.store(output_ptr + out_offset, out, mask=mask_k)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def logsumexp_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    Kernel for logsumexp when K == 1 (inner dimension case).
    Computes log(sum(exp(x - max(x))), dim=dim) with numerical stability.
    Input shape: (M, N), Output shape: (M,) or (M, 1) when keepdim=True
    """
    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        # Input offset: (M, N) flattened
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = m + tl.log(z)
        # Output offset: (M,) - reduced along N
        output_ptrs = output_ptr + pid_m
        tl.store(output_ptrs, out)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        # Don't advance output_ptr - each program writes to output[pid_m]

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        o = m + tl.log(z)
        # Output: write to output[pid_m]
        tl.store(output_ptr + pid_m, o)


def logsumexp(inp, dim, keepdim=False):
    """
    Computes log(sum(exp(x), dim=dim)) with numerical stability.

    Args:
        inp: Input tensor
        dim: Dimension along which to compute logsumexp (int or list of ints)
        keepdim: Whether to keep the reduced dimension

    Returns:
        Tensor with logsumexp computed along the specified dimension
    """
    logger.debug("GEMS LOGSUMEXP")

    # Handle dim as int or list
    if isinstance(dim, (list, tuple)):
        dim = list(dim)
    else:
        dim = [dim]

    # Validate all dimensions
    for d in dim:
        assert d >= -inp.ndim and d < inp.ndim, "Invalid dim"

    # Handle empty tensor
    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            for d in dim:
                out_shape[d % inp.ndim] = 1
        else:
            for d in sorted(dim, reverse=True):
                out_shape.pop(d % inp.ndim)
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        zero_(out)
        return out

    # Handle single dimension case (most common)
    if len(dim) == 1:
        dim = dim[0] % inp.ndim
        shape = list(inp.shape)
        M = 1
        N = inp.shape[dim]
        for i in range(dim):
            M *= inp.shape[i]
        inp = inp.contiguous()
        dtype = inp.dtype
        # Compute output shape (reduced along dim)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        out = torch.empty(shape, dtype=dtype, device=inp.device)
        K = inp.numel() // M // N

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                logsumexp_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                grid = (M, 1, 1)
                logsumexp_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        return out

    # Multiple dimensions - delegate to PyTorch for now
    # Convert dim indices to positive if needed
    dim = [d % inp.ndim for d in dim]
    inp = inp.contiguous()
    dtype = inp.dtype
    out = torch.special.logsumexp(inp, dim=dim, keepdim=keepdim)
    return out
    inp = inp.contiguous()
    dtype = inp.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            logsumexp_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            grid = (M, 1, 1)
            logsumexp_kernel_inner[grid](
                out,
                inp,
                M,
                N,
            )

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
