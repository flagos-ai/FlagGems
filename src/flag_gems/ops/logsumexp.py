import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

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
    """Kernel for logsumexp when reduction dimension is not the innermost."""
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, axis=0, keep_dims=True)
        # Handle case where entire column is -inf
        safe_m = tl.where(m == float("-inf"), tl.zeros_like(m), m)
        e = tl.exp(inp - safe_m)
        z = tl.sum(e, axis=0, keep_dims=True)
        out = safe_m + tl.log(z)
        # If all inputs were -inf, result should be -inf
        out = tl.where(m == float("-inf"), m, out)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, axis=0, keep_dims=True)
        z = tl.sum(z * tl.exp(m - m_reduced), axis=0, keep_dims=True)
        m = m_reduced
        # Handle case where all inputs were -inf
        out = tl.where(m == float("-inf"), m, m + tl.log(z))
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


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
    """Kernel for logsumexp when reduction dimension is the innermost."""
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, axis=0)
        # Handle case where all inputs are -inf
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        e = tl.exp(inp - safe_m)
        z = tl.sum(e, axis=0)
        out = safe_m + tl.log(z)
        # If all inputs were -inf, result should be -inf
        out = tl.where(m == float("-inf"), m, out)
        output_ptrs = output_ptr + pid_m
        tl.store(output_ptrs, out)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, axis=0)
        z = tl.sum(z * tl.exp(m - m_reduced), axis=0)
        m = m_reduced
        # Handle case where all inputs were -inf
        out = tl.where(m == float("-inf"), m, m + tl.log(z))
        output_ptrs = output_ptr + pid_m
        tl.store(output_ptrs, out)


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        # Handle multi-dimensional reduction
        if len(dim) == 0:
            # Empty dim list means no reduction, just return the input
            return inp.clone()
        if len(dim) == 1:
            dim = dim[0]
        else:
            # For multiple dims, reduce sequentially
            # Sort dims in descending order to handle dimension shifts correctly
            sorted_dims = sorted([d % inp.ndim for d in dim], reverse=True)
            result = inp
            for d in sorted_dims:
                result = logsumexp(result, d, keepdim=True)
            if not keepdim:
                # Remove the reduced dimensions
                for d in sorted(sorted_dims, reverse=True):
                    result = result.squeeze(d)
            return result

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    M = 1
    N = inp.shape[dim]
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    # Output shape with reduction dimension set to 1
    shape = list(inp.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)

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
