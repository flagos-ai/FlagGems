import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def median_gather_kernel(
    sorted_vals,
    sorted_idx,
    out_vals,
    out_idx,
    N,
    K,
    median_pos,
    BLOCK_K: tl.constexpr,
):
    """Gather element at median_pos along dim N from a (M, N, K) layout tensor.

    Grid: (M, cdiv(K, BLOCK_K))
    """
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k_off < K

    src_off = pid_m * N * K + median_pos * K + k_off
    dst_off = pid_m * K + k_off

    vals = tl.load(sorted_vals + src_off, mask=mask, other=0)
    idxs = tl.load(sorted_idx + src_off, mask=mask, other=0)

    tl.store(out_vals + dst_off, vals, mask=mask)
    tl.store(out_idx + dst_off, idxs, mask=mask)


def median(inp, dim=None, keepdim=False):
    """torch.median(input) or torch.median(input, dim, keepdim)."""
    logger.debug("GEMS MEDIAN")
    from flag_gems.ops.sort import sort_stable

    if dim is None:
        n = inp.numel()
        if n == 0:
            raise RuntimeError(
                "median() is not defined for empty tensors"
            )
        flat = inp.contiguous().reshape(-1)
        sorted_vals, _ = sort_stable(flat, stable=True, dim=0)
        return sorted_vals[(n - 1) // 2]

    assert dim >= -inp.ndim and dim < inp.ndim, (
        f"Dimension out of range (expected to be in range of "
        f"[-{inp.ndim}, {inp.ndim - 1}], but got {dim})"
    )
    dim = dim % inp.ndim

    shape = inp.shape
    N_dim = shape[dim]
    M = math.prod(shape[:dim]) if dim > 0 else 1
    K = math.prod(shape[dim + 1:]) if dim + 1 < len(shape) else 1
    median_pos = (N_dim - 1) // 2

    # Handle empty tensors
    if inp.numel() == 0:
        out_shape = list(shape)
        del out_shape[dim]
        if keepdim:
            out_shape.insert(dim, 1)
        out_vals = inp.new_empty(out_shape)
        out_idx = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        return torch.return_types.median(out_vals, out_idx)

    # Sort along dim using FlagGems Triton radix sort
    sorted_vals, sorted_indices = sort_stable(inp, stable=True, dim=dim)

    # Contiguous (M, N, K) layout for the gather kernel
    sv = sorted_vals.contiguous().view(M, N_dim, K)
    si = sorted_indices.contiguous().view(M, N_dim, K)

    out_vals = torch.empty((M, K), dtype=inp.dtype, device=inp.device)
    out_idx = torch.empty((M, K), dtype=torch.int64, device=inp.device)

    BLOCK_K = min(triton.next_power_of_2(K), 1024)
    grid = (M, triton.cdiv(K, BLOCK_K))

    with torch_device_fn.device(inp.device):
        median_gather_kernel[grid](
            sv, si,
            out_vals, out_idx,
            N_dim, K,
            median_pos,
            BLOCK_K=BLOCK_K,
        )

    # Reshape to expected output dimensions
    out_shape = list(shape)
    del out_shape[dim]
    out_vals = out_vals.view(out_shape) if out_shape else out_vals.view([])
    out_idx = out_idx.view(out_shape) if out_shape else out_idx.view([])

    if keepdim:
        out_vals = out_vals.unsqueeze(dim)
        out_idx = out_idx.unsqueeze(dim)

    return torch.return_types.median(out_vals, out_idx)


def median_dim(inp, dim, keepdim=False):
    """aten.median.dim — returns (values, indices) named tuple."""
    logger.debug("GEMS MEDIAN_DIM")
    return median(inp, dim=dim, keepdim=keepdim)


def median_dim_values(inp, dim, keepdim=False, *, values, indices):
    """aten.median.dim_values — fills pre-allocated output tensors."""
    logger.debug("GEMS MEDIAN_DIM_VALUES")
    result = median(inp, dim=dim, keepdim=keepdim)
    values.copy_(result.values)
    indices.copy_(result.indices)
    return result


def median_out(inp, *, out):
    """aten.median.out — stores flat median into out tensor."""
    logger.debug("GEMS MEDIAN_OUT")
    result = median(inp)
    out.copy_(result)
    return out
