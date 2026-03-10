import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.sort import convert_to_uint_preverse_order, sort_stable
from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_TRITON_MEDIAN_COLS = 8192*2
_RADIX_BITS_PER_PASS = 4


@libentry()
@triton.jit
def median_lastdim_kernel(
    out_ptr,
    out_idx_ptr,
    inp_ptr,
    n_cols,
    kth,
    TOTAL_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BITS_PER_PASS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_ptr = inp_ptr + row * n_cols

    if row_ptr.dtype.element_ty.is_floating():
        other = _get_finfo_val(row_ptr.dtype.element_ty, return_max=True)
    else:
        other = _get_iinfo_val(row_ptr.dtype.element_ty, return_max=True)

    vals = tl.load(row_ptr + cols, mask=mask, other=other)
    if row_ptr.dtype.element_ty.is_floating():
        nan_mask = mask & (vals != vals)
        nan_mask_i32 = nan_mask.to(tl.int32)
        has_nan = tl.max(nan_mask_i32, axis=0) != 0
        first_nan = tl.argmax(nan_mask_i32, axis=0)
        vals_for_key = tl.where(nan_mask, tl.zeros_like(vals), vals)
    else:
        vals_for_key = vals

    keys = convert_to_uint_preverse_order(vals_for_key, descending=False)
    active = mask
    remaining = kth

    num_buckets: tl.constexpr = 1 << BITS_PER_PASS
    bucket_mask: tl.constexpr = num_buckets - 1
    num_passes: tl.constexpr = TOTAL_BITS // BITS_PER_PASS
    hist_bins: tl.constexpr = num_buckets * 2

    for pass_idx in tl.static_range(0, num_passes):
        shift = TOTAL_BITS - (pass_idx + 1) * BITS_PER_PASS
        bucket = (keys >> shift) & bucket_mask
        masked_bucket = tl.where(active, bucket, num_buckets).to(tl.int32)
        hist_all = masked_bucket.histogram(hist_bins)
        bucket_ids = tl.arange(0, hist_bins)
        prefix = hist_all.cumsum(0)
        prefix_before_all = prefix - hist_all
        selected_mask = (
            (bucket_ids < num_buckets)
            & (prefix_before_all <= remaining)
            & (remaining < prefix)
        )
        selected_bucket = selected_mask.to(tl.int32).argmax(0)
        prefix_before = tl.where(
            bucket_ids == selected_bucket, prefix_before_all, tl.zeros_like(prefix_before_all)
        ).max(0)

        remaining = remaining - prefix_before
        active = active & (bucket.to(tl.int32) == selected_bucket)

    first_active = tl.argmax(active.to(tl.int32), axis=0)
    first_mask = cols == first_active
    selected_val = tl.sum(tl.where(first_mask, vals, tl.zeros_like(vals)), axis=0)
    selected_idx = first_active.to(tl.int64)

    if row_ptr.dtype.element_ty.is_floating():
        first_nan_mask = cols == first_nan
        first_nan_val = tl.sum(tl.where(first_nan_mask, vals, tl.zeros_like(vals)), axis=0)
        selected_val = tl.where(has_nan, first_nan_val, selected_val)
        selected_idx = tl.where(has_nan, first_nan.to(tl.int64), selected_idx)

    tl.store(out_ptr + row, selected_val)
    tl.store(out_idx_ptr + row, selected_idx)


def _median_dim_sort_fallback(moved):
    sorted_vals, _ = sort_stable(moved, stable=True, dim=-1, descending=False)
    median_pos = (moved.shape[-1] - 1) // 2
    values = torch.select(sorted_vals, -1, median_pos)
    matches = moved == values.unsqueeze(-1)
    indices = matches.to(torch.int64).argmax(dim=-1)
    return values, indices


def median_dim(inp, dim, keepdim=False):
    logger.debug("GEMS MEDIAN.DIM")

    if inp.dtype == torch.bool:
        raise RuntimeError('"median_out" not implemented for \'Bool\'')

    if inp.ndim == 0:
        raise IndexError("Dimension specified as 0 but tensor has no dimensions")

    dim = dim if dim >= 0 else dim + inp.ndim
    if dim < 0 or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )

    if inp.shape[dim] == 0:
        raise IndexError("median(): Expected reduction dim to be non-zero.")

    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]

    if inp.numel() == 0:
        values = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        return values, indices

    moved = torch.movedim(inp, dim, -1).contiguous()
    n_cols = moved.shape[-1]

    use_triton = n_cols <= _MAX_TRITON_MEDIAN_COLS
    if use_triton:
        n_rows = moved.numel() // n_cols
        flat = moved.reshape(n_rows, n_cols)
        values = torch.empty((n_rows,), dtype=inp.dtype, device=inp.device)
        indices = torch.empty((n_rows,), dtype=torch.int64, device=inp.device)
        median_pos = (n_cols - 1) // 2
        total_bits = inp.element_size() * 8
        block_size = triton.next_power_of_2(n_cols)
        median_lastdim_kernel[(n_rows,)](
            values,
            indices,
            flat,
            n_cols,
            median_pos,
            TOTAL_BITS=total_bits,
            BLOCK_SIZE=block_size,
            BITS_PER_PASS=_RADIX_BITS_PER_PASS,
        )
        values = values.reshape(moved.shape[:-1])
        indices = indices.reshape(moved.shape[:-1])
    else:
        values, indices = _median_dim_sort_fallback(moved)
        if moved.dtype.is_floating_point:
            nan_mask = torch.isnan(moved)
            if torch.any(nan_mask):
                has_nan = nan_mask.any(dim=-1)
                if torch.any(has_nan):
                    first_nan_idx = nan_mask[has_nan].to(torch.int64).argmax(dim=-1)
                    values = values.clone()
                    indices = indices.clone()
                    values[has_nan] = float("nan")
                    indices[has_nan] = first_nan_idx

    if keepdim:
        values = torch.movedim(values.unsqueeze(-1), -1, dim)
        indices = torch.movedim(indices.unsqueeze(-1), -1, dim)

    return values, indices
