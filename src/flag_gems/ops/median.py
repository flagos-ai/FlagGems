import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Reuse the namedtuple type across calls (avoid recreating it inside median_dim).
Median_out = namedtuple("median", ["values", "indices"])


@triton.jit
def _radix_convert(v, elem_ty: tl.constexpr):
    if elem_ty is tl.float32:
        x = v.to(tl.uint32, bitcast=True)
        sign = x & 0x8000_0000
        mask = tl.where(sign != 0, 0xFFFF_FFFF, 0x8000_0000).to(tl.uint32)
        y = x ^ mask
        y = tl.where(v == v, y, 0xFFFF_FFFF).to(tl.uint32)
        return y
    if elem_ty is tl.float16:
        x16 = v.to(tl.uint16, bitcast=True)
        x = x16.to(tl.uint32)
        sign = x & 0x0000_8000
        mask = tl.where(sign != 0, 0x0000_FFFF, 0x0000_8000).to(tl.uint32)
        y = x ^ mask
        y = tl.where(v == v, y, 0x0000_FFFF).to(tl.uint32)
        return y
    if elem_ty is tl.bfloat16:
        x16 = v.to(tl.uint16, bitcast=True)
        x = x16.to(tl.uint32)
        sign = x & 0x0000_8000
        mask = tl.where(sign != 0, 0x0000_FFFF, 0x0000_8000).to(tl.uint32)
        y = x ^ mask
        y = tl.where(v == v, y, 0x0000_FFFF).to(tl.uint32)
        return y
    if elem_ty is tl.int32:
        x = v.to(tl.uint32, bitcast=True)
        return x ^ 0x8000_0000
    if elem_ty is tl.int16:
        x16 = v.to(tl.uint16, bitcast=True)
        x = x16.to(tl.uint32)
        return x ^ 0x0000_8000
    return v.to(tl.uint32)


@libentry()
@triton.jit
def _radix_count_kernel(
    inp_ptr,
    counts_ptr,
    desired_ptr,
    desired_mask_ptr,
    stride_sm,
    stride_sn,
    stride_counts_m,
    M,
    N,
    BIT_OFFSET: tl.constexpr,
    K_BITS: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    row_mask = row < M

    desired = tl.load(desired_ptr + row, mask=row_mask, other=0).to(tl.uint32)
    desired_mask = tl.load(desired_mask_ptr + row, mask=row_mask, other=0).to(tl.uint32)

    counts = tl.zeros([NUM_BINS], dtype=tl.int32)

    for c in range(0, N, BLOCK_N):
        cols = c + tl.arange(0, BLOCK_N)
        m = row_mask & (cols < N)
        ptr = inp_ptr + row * stride_sm + cols * stride_sn
        vals = tl.load(ptr, mask=m, other=0)
        key = _radix_convert(vals, inp_ptr.dtype.element_ty)

        ok = m & ((key & desired_mask) == desired)
        digit = (key >> BIT_OFFSET) & (NUM_BINS - 1)

        for b in tl.static_range(NUM_BINS):
            counts_b = tl.sum(ok & (digit == b), axis=0)
            counts = tl.where(
                tl.arange(0, NUM_BINS) == b,
                counts + tl.cast(counts_b, tl.int32),
                counts,
            )

    base = counts_ptr + row * stride_counts_m
    tl.store(base + tl.arange(0, NUM_BINS), counts, mask=row_mask)


@libentry()
@triton.jit
def _find_pattern_kernel(
    inp_ptr,
    out_val_ptr,
    out_idx_ptr,
    desired_ptr,
    desired_mask_ptr,
    stride_sm,
    stride_sn,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    row_mask = row < M

    desired = tl.load(desired_ptr + row, mask=row_mask, other=0).to(tl.uint32)
    desired_mask = tl.load(desired_mask_ptr + row, mask=row_mask, other=0).to(tl.uint32)

    found = tl.zeros((), dtype=tl.int1)
    first_idx = tl.zeros((), dtype=tl.int32)
    first_val = tl.zeros((), dtype=inp_ptr.dtype.element_ty)

    for c in range(0, N, BLOCK_N):
        cols = c + tl.arange(0, BLOCK_N)
        m = row_mask & (cols < N)
        ptr = inp_ptr + row * stride_sm + cols * stride_sn
        vals = tl.load(ptr, mask=m, other=0)
        key = _radix_convert(vals, inp_ptr.dtype.element_ty)
        match = m & ((key & desired_mask) == desired)
        has = tl.sum(match, axis=0) > 0
        idx_in = tl.argmax(match, axis=0).to(tl.int32)
        cand_idx = (c + idx_in).to(tl.int32)
        cand_val = tl.load(
            inp_ptr + row * stride_sm + cand_idx * stride_sn, mask=row_mask
        )

        do_set = (~found) & has
        first_idx = tl.where(do_set, cand_idx, first_idx)
        first_val = tl.where(do_set, cand_val, first_val)
        found = found | has

    tl.store(out_val_ptr + row, first_val, mask=row_mask)
    tl.store(out_idx_ptr + row, first_idx.to(tl.int64), mask=row_mask)


def _radix_kthvalue_lastdim(inp_2d: torch.Tensor, k: int, *, largest: bool):
    assert inp_2d.dim() == 2
    M, N = inp_2d.shape
    device = inp_2d.device

    if inp_2d.dtype in (torch.float16, torch.bfloat16, torch.int16):
        num_bits = 16
    elif inp_2d.dtype in (torch.float32, torch.int32):
        num_bits = 32
    else:
        raise RuntimeError(f"radix_select: unsupported dtype {inp_2d.dtype}")

    K_BITS = 4
    NUM_BINS = 1 << K_BITS
    BLOCK_N = 256

    desired = torch.zeros((M,), device=device, dtype=torch.int32)
    desired_mask = torch.zeros((M,), device=device, dtype=torch.int32)
    k_to_find = torch.full((M,), int(k), device=device, dtype=torch.int32)

    counts = torch.empty((M, NUM_BINS), device=device, dtype=torch.int32)

    grid = (M,)
    for bit_offset in range(num_bits - K_BITS, -1, -K_BITS):
        with torch_device_fn.device(device):
            _radix_count_kernel[grid](
                inp_2d,
                counts,
                desired,
                desired_mask,
                inp_2d.stride(0),
                inp_2d.stride(1),
                counts.stride(0),
                M,
                N,
                BIT_OFFSET=bit_offset,
                K_BITS=K_BITS,
                NUM_BINS=NUM_BINS,
                BLOCK_N=BLOCK_N,
            )

        if largest:
            counts_work = counts.flip(-1)
        else:
            counts_work = counts

        prefix = torch.cumsum(counts_work, dim=-1)
        ge = prefix >= k_to_find.unsqueeze(-1)
        bin_idx_work = ge.int().argmax(dim=-1)

        prev = torch.zeros_like(k_to_find)
        has_prev = bin_idx_work > 0
        if has_prev.any():
            prev_vals = prefix.gather(
                -1, (bin_idx_work - 1).clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            prev = torch.where(has_prev, prev_vals, prev)

        k_to_find = k_to_find - prev

        if largest:
            bin_idx = (NUM_BINS - 1) - bin_idx_work
        else:
            bin_idx = bin_idx_work

        desired = desired | (bin_idx << bit_offset)
        desired_mask = desired_mask | ((NUM_BINS - 1) << bit_offset)

    out_val = torch.empty((M,), device=device, dtype=inp_2d.dtype)
    out_idx = torch.empty((M,), device=device, dtype=torch.int64)
    with torch_device_fn.device(device):
        _find_pattern_kernel[grid](
            inp_2d,
            out_val,
            out_idx,
            desired,
            desired_mask,
            inp_2d.stride(0),
            inp_2d.stride(1),
            M,
            N,
            BLOCK_N=BLOCK_N,
        )
    return out_val, out_idx


def median_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    shape = list(inp.shape)
    N = shape[dim]
    if N == 0:
        raise RuntimeError("median: dimension is empty")

    median_pos = (N - 1) // 2

    inp_c = dim_compress(inp, dim)
    M = inp_c.numel() // N
    inp_2d = inp_c.reshape(M, N)

    out_value = torch.empty((M,), device=inp.device, dtype=inp.dtype)
    out_index = torch.empty((M,), device=inp.device, dtype=torch.int64)

    has_nan = None
    if inp_2d.is_floating_point() or inp_2d.is_complex():
        nan_mask = torch.isnan(inp_2d)
        has_nan = nan_mask.any(dim=-1)
        first_nan = nan_mask.int().argmax(dim=-1)

    if has_nan is not None and has_nan.any():
        nan_val = torch.tensor(
            float("nan"), device=out_value.device, dtype=out_value.dtype
        )
        out_value = torch.where(has_nan, nan_val, out_value)
        out_index[has_nan] = first_nan[has_nan]

        no_nan = ~has_nan
        if no_nan.any():
            v, idx = _radix_kthvalue_lastdim(
                inp_2d[no_nan], median_pos + 1, largest=False
            )
            out_value[no_nan] = v
            out_index[no_nan] = idx
    else:
        out_value, out_index = _radix_kthvalue_lastdim(
            inp_2d, median_pos + 1, largest=False
        )

    shape[dim] = 1
    out_value = out_value.reshape(shape)
    out_index = out_index.reshape(shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    return Median_out(values=out_value, indices=out_index)
