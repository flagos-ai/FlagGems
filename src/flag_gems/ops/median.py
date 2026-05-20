import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

from .topk import argsort

logger = logging.getLogger(__name__)

# In-block bitonic sort caps at this size; beyond it we fall back to
# `torch.sort` + a tiny gather kernel.
MAX_BITONIC_N = 1024


@libentry()
@triton.jit
def median_bitonic_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    median_pos,
    BLOCK_N: tl.constexpr,
):
    """One program per row. Bitonic-sort the row together with its original
    indices, then pick position `median_pos` (lower median)."""
    pid = ext.program_id(0)
    if pid >= M:
        return

    dtype = inp.type.element_ty
    pad = get_dtype_max(dtype)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_ptr = inp + pid * N
    vals = tl.load(row_ptr + cols, mask=mask, other=pad)
    ids = cols.to(tl.int64)

    sorted_vals, sorted_ids = argsort(vals, ids, 0, descending=False)

    # Select column `median_pos` via masked reduction. Using `tl.where`
    # (rather than multiplication) preserves +inf and other non-finite
    # values at non-selected positions without producing NaN.
    pick_mask = cols == median_pos
    zero_val = tl.zeros([BLOCK_N], dtype=sorted_vals.dtype)
    zero_idx = tl.zeros([BLOCK_N], dtype=tl.int64)
    median_value = tl.sum(tl.where(pick_mask, sorted_vals, zero_val), axis=0)
    median_index = tl.sum(tl.where(pick_mask, sorted_ids, zero_idx), axis=0)

    tl.store(out_value + pid, median_value)
    tl.store(out_index + pid, median_index)


@libentry()
@triton.jit
def median_gather_kernel(
    sorted_vals,
    sorted_ids,
    out_value,
    out_index,
    M,
    N,
    median_pos,
    BLOCK_M: tl.constexpr,
):
    """Pick column `median_pos` from a pre-sorted (M, N) value/index pair."""
    pid = ext.program_id(0)
    m_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m_offset < M

    val = tl.load(sorted_vals + m_offset * N + median_pos, mask=mask)
    idx = tl.load(sorted_ids + m_offset * N + median_pos, mask=mask)
    tl.store(out_value + m_offset, val, mask=mask)
    tl.store(out_index + m_offset, idx, mask=mask)


def _median_dim_impl(inp: torch.Tensor, dim: int):
    """Core (values, indices) computation along `dim`. Caller handles
    keepdim / squeezing / NaN propagation."""
    N = inp.shape[dim]
    if N == 0:
        raise RuntimeError(
            "median(): Expected reduction dim to have non-zero size."
        )

    inp = dim_compress(inp, dim)  # reduction dim → last
    M = inp.numel() // N
    median_pos = (N - 1) // 2

    flat_shape = (M,)
    out_value = torch.empty(flat_shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(flat_shape, dtype=torch.int64, device=inp.device)

    with torch_device_fn.device(inp.device):
        if N <= MAX_BITONIC_N:
            block_n = triton.next_power_of_2(N)
            median_bitonic_kernel[(M,)](
                inp, out_value, out_index, M, N, median_pos, BLOCK_N=block_n
            )
        else:
            inp_2d = inp.reshape(M, N)
            sorted_vals, sorted_ids = inp_2d.sort(dim=-1)
            block_m = 256
            grid = (triton.cdiv(M, block_m),)
            median_gather_kernel[grid](
                sorted_vals,
                sorted_ids,
                out_value,
                out_index,
                M,
                N,
                median_pos,
                BLOCK_M=block_m,
            )

    out_shape = list(inp.shape[:-1])  # everything except the now-last red. dim
    return out_value.reshape(out_shape), out_index.reshape(out_shape)


def _propagate_nan(inp: torch.Tensor, dim: int, values: torch.Tensor) -> torch.Tensor:
    """Match PyTorch semantics: if any NaN exists in the reduction slice,
    the median is NaN."""
    if not inp.is_floating_point():
        return values
    has_nan = torch.isnan(inp).any(dim=dim)
    if has_nan.any():
        nan_val = torch.tensor(float("nan"), dtype=values.dtype, device=values.device)
        values = torch.where(has_nan, nan_val, values)
    return values


Median_out = namedtuple("median_out", ["values", "indices"])


def median(inp: torch.Tensor) -> torch.Tensor:
    """Global median over all elements. Returns a 0-dim tensor."""
    logger.debug("GEMS MEDIAN")
    flat = inp.contiguous().view(-1)
    values, _ = _median_dim_impl(flat, dim=0)
    values = _propagate_nan(flat, dim=0, values=values)
    return values.reshape(())


def median_dim(inp: torch.Tensor, dim: int, keepdim: bool = False):
    """median.dim: (values, indices) along `dim`."""
    logger.debug("GEMS MEDIAN DIM")
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    values, indices = _median_dim_impl(inp, dim)
    values = _propagate_nan(inp, dim=dim, values=values)

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return Median_out(values=values, indices=indices)


