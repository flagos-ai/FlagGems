import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _min_value(dtype):
    """Return a finite sentinel strictly below all values of the dtype.

    This is a local copy of flag_gems.utils.limits.get_dtype_min, kept here so
    that the max-reduction kernels do not depend on the float64 ``inf`` sentinel
    (which triggers a "Device do not support double dtype" downcast warning on
    Ascend).  ``-FLT_MAX`` is a safe sentinel for max reductions: no legitimate
    float value falls below it."""
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = -3.4028234663852886e38
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = -1 * 2 ** (width - 1)
        return value
    if dtype_.is_int_unsigned():
        value: tl.constexpr = 0
        return value


@triton.jit
def _max_nan(a, b):
    """NaN-aware element-wise max (torch.maximum semantics).

    ``tl.maximum`` on Ascend drops NaN: ``tl.maximum(1.0, nan) == 1.0``, whereas
    torch's max-pooling propagates a NaN in the window to the output.  ``b``
    wins when ``b > a`` OR ``b`` is NaN (so a NaN propagates regardless of which
    operand holds it)."""
    return tl.where((b > a) | (b != b), b, a)


# ==============================================================================
# Kernel 1 — Direct scan  (M, K)
#
#   M = OUT_PER_BLOCK   — output elements per block
#   K = CHAN_PER_BLOCK  — channels per block (1 = standard, >1 = chunked)
#
# (M, K) are chosen by the manual heuristic _select_kernel_direct below
# (autotune is too expensive to run for every shape on Ascend).
# ==============================================================================


def _select_kernel_direct(
    max_win_d, max_win_h, max_win_w, in_c, in_n, out_d, out_h, out_w
):
    """Manual kernel config selection (replaces @triton.autotune)."""
    total_output = in_n * in_c * out_d * out_h * out_w
    win_size = max_win_d * max_win_h * max_win_w
    if in_c >= 8 and win_size <= 64 and total_output <= 4096:
        return 64, 8, 8, 3
    elif in_c >= 4 and win_size <= 256 and total_output <= 16384:
        return 64, 4, 4, 4
    elif total_output >= 65536:
        return 256, 1, 8, 3
    elif total_output <= 1024:
        return 64, 1, 2, 5
    else:
        return 256, 1, 4, 4


@libentry()
@triton.jit
def _kernel_direct(
    in_ptr,
    out_ptr,
    idx_ptr,
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    OUT_PER_BLOCK: tl.constexpr,
    CHAN_PER_BLOCK: tl.constexpr,
    MAX_WIN_D: tl.constexpr,
    MAX_WIN_H: tl.constexpr,
    MAX_WIN_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """Direct-scan: each thread independently scans one output element."""
    pid = tl.program_id(0)
    block_start = pid * OUT_PER_BLOCK
    c_groups = (in_c + CHAN_PER_BLOCK - 1) // CHAN_PER_BLOCK
    flat_elems = in_n * c_groups * out_d * out_h * out_w

    tid = tl.arange(0, OUT_PER_BLOCK)
    flat_idx = block_start + tid
    valid = flat_idx < flat_elems

    # int64 division (int32 `//` is broken for dividends >= 2**24 on Ascend),
    # but int64 is a slow software divide — use int32 when flat indices stay
    # below 2**24.
    tmp = flat_idx.to(tl.int64) if USE_INT64 else flat_idx
    w_out_pos = tmp % out_w
    tmp //= out_w
    h_out_pos = tmp % out_h
    tmp //= out_h
    d_out_pos = tmp % out_d
    tmp //= out_d
    c_group = tmp % c_groups
    tmp //= c_groups
    n_idx = tmp

    n_idx = tl.where(valid, n_idx, 0)
    c_group = tl.where(valid, c_group, 0)
    d_out_pos = tl.where(valid, d_out_pos, 0)
    h_out_pos = tl.where(valid, h_out_pos, 0)
    w_out_pos = tl.where(valid, w_out_pos, 0)
    c_base = c_group * CHAN_PER_BLOCK

    d_start = d_out_pos * in_d // out_d
    win_d = ((d_out_pos + 1) * in_d + out_d - 1) // out_d - d_start
    h_start = h_out_pos * in_h // out_h
    win_h = ((h_out_pos + 1) * in_h + out_h - 1) // out_h - h_start
    w_start = w_out_pos * in_w // out_w
    win_w = ((w_out_pos + 1) * in_w + out_w - 1) // out_w - w_start

    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    out_dhw = out_d * out_h * out_w

    for c_off in range(CHAN_PER_BLOCK):
        c_idx = c_base + c_off
        chan_valid = (c_idx < in_c) & valid
        in_base = (
            in_ptr + n_idx * in_c * in_d * in_h * in_w + c_idx * in_d * in_h * in_w
        )

        acc_val = tl.full((OUT_PER_BLOCK,), min_val, dtype=dtype)
        acc_idx = (
            tl.full((OUT_PER_BLOCK,), -1, dtype=tl.int64) if RETURN_INDICES else acc_val
        )

        for kd in range(MAX_WIN_D):
            d_in_raw = d_start + kd
            d_valid = (kd < win_d) & (d_in_raw < in_d) & chan_valid
            d_s = tl.where(d_valid, d_in_raw, 0)
            d_off = d_s * in_h * in_w
            for kh in range(MAX_WIN_H):
                h_in_raw = h_start + kh
                h_valid = (kh < win_h) & (h_in_raw < in_h) & chan_valid
                h_s = tl.where(h_valid, h_in_raw, 0)
                dh_off = d_off + h_s * in_w
                for kw in range(MAX_WIN_W):
                    w_in_raw = w_start + kw
                    w_valid = (kw < win_w) & (w_in_raw < in_w) & chan_valid
                    in_mask = d_valid & h_valid & w_valid
                    w_s = tl.where(w_valid, w_in_raw, 0)
                    off = dh_off + w_s
                    cv = tl.load(
                        in_base + off, mask=in_mask, other=min_val, cache_modifier=".ca"
                    )
                    is_new = (cv > acc_val) | (cv != cv)
                    acc_val = tl.where(is_new, cv, acc_val)
                    if RETURN_INDICES:
                        ci = d_s * in_h * in_w + h_s * in_w + w_s
                        acc_idx = tl.where(is_new & in_mask, ci, acc_idx)

        out_off = (
            n_idx * in_c * out_dhw
            + c_idx * out_dhw
            + d_out_pos * out_h * out_w
            + h_out_pos * out_w
            + w_out_pos
        )
        tl.store(out_ptr + out_off, acc_val, mask=valid & chan_valid)
        if RETURN_INDICES:
            tl.store(idx_ptr + out_off, acc_idx, mask=valid & chan_valid)


# ==============================================================================
# Helpers
# ==============================================================================


@triton.jit
def _merge_outd1_indices_kernel(
    spatial_indices_ptr,
    d_argmax_ptr,
    output_indices_ptr,
    n_elements,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    out_spatial = out_h * out_w
    # int64 division (int32 `//` is broken for dividends >= 2**24 on Ascend).
    off64 = offsets.to(tl.int64)
    nc_idx = off64 // out_spatial
    rem = off64 % out_spatial
    h_out_pos = rem // out_w
    w_out_pos = rem % out_w
    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    n_idx = tl.where(mask, n_idx, 0)
    c_idx = tl.where(mask, c_idx, 0)
    h_out_pos = tl.where(mask, h_out_pos, 0)
    w_out_pos = tl.where(mask, w_out_pos, 0)
    spatial_idx = tl.load(spatial_indices_ptr + offsets, mask=mask, other=0)
    h_best = spatial_idx // in_w
    w_best = spatial_idx % in_w
    d_off = n_idx * in_c * in_h * in_w + c_idx * in_h * in_w + h_best * in_w + w_best
    d_best = tl.load(d_argmax_ptr + d_off, mask=mask, other=0)
    tl.store(
        output_indices_ptr + offsets, d_best * in_h * in_w + spatial_idx, mask=mask
    )


@triton.jit
def _fill_identity_indices_kernel(
    indices_ptr,
    spatial_total,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill indices for the identity path: each element repeats the pattern
    [0, 1, ..., spatial_total-1] according to its location in the spatial dims.
    Much faster than expand+contiguous for large N*C."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    # int64 modulo + store is ~2.5x faster on Ascend than a 2D-grid variant;
    # keep the store int64 to match the ATen index dtype.
    tl.store(indices_ptr + offsets, (offsets % spatial_total).to(tl.int64), mask=mask)


@triton.jit
def _merge_d2h_indices_kernel(
    d_argmax_ptr,
    spatial_idx_ptr,
    out_idx_ptr,
    hw_in,
    out_hw,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge D indices with spatial indices.
    For each output element:
      sidx = spatial_idx[pos]           # flattened h*W + w from 2D pool
      d_idx = d_argmax[batch, sidx]     # D index at winning spatial position
      full_idx = d_idx * hw_in + sidx
    This replaces a slow torch.gather on NPU with a fused kernel."""
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute batch and position within batch (int64 division — int32 `//` is
    # broken for dividends >= 2**24 on Ascend).
    batch = offsets.to(tl.int64) // out_hw

    # Load spatial index
    sidx = tl.load(spatial_idx_ptr + offsets, mask=mask, other=0)

    # Load D index from d_argmax at (batch, sidx)
    d_arg_pos = batch * hw_in + sidx
    d_val = tl.load(d_argmax_ptr + d_arg_pos, mask=mask, other=0)

    # Full 3D index
    tl.store(out_idx_ptr + offsets, d_val * hw_in + sidx, mask=mask)


# ==============================================================================
# Kernel 3 — Unified D-reduction
#
#   Merges the old _d_reduce_kernel and _d_reduce_tiled_kernel into a single
#   kernel with two strategies selected by STRATEGY: tl.constexpr.
#
#   STRATEGY=0 (simple): each block processes BLOCK_M positions, simple D loop.
#       Grid: ceil(N*C*H*W / BLOCK_M).  Supports values + indices.
#   STRATEGY=1 (tiled): each block handles TILE_SIZE contiguous HW positions.
#       Grid: (N*C, ceil(H*W / TILE_SIZE)), optionally 2D for the NPU 65535
#       per-dimension grid limit.  Values-only (no indices).
#
#   Both strategies bound-check pid (MAX_GRID_BLOCKS / TOTAL_BLOCKS), so the
#   grid can be shared across configs safely.
#
#   Selection is done in _d_reduce_unified (manual heuristics, no autotune).
# ==============================================================================


@libentry()
@triton.jit
def _unified_d_reduce_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    total_positions,
    MAX_GRID_BLOCKS,
    STRATEGY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    N_TILES: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """Unified D-reduction kernel with two strategies.

    STRATEGY=0 (simple): each block processes BLOCK_M (n,c,h,w) positions.
        Handles both values and indices output.
    STRATEGY=1 (tiled): each block loads TILE_SIZE contiguous HW elements.
        Values-only output.  Supports 1D or 2D grid via pid combining.
    """
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    if pid >= MAX_GRID_BLOCKS:
        return

    if STRATEGY == 0:
        # ---- Simple D-reduce: like old _d_reduce_kernel ----
        # Early return for blocks beyond this config's work range.
        # max_grid across all configs may exceed the grid needed by the
        # selected config; these extra blocks would otherwise execute the
        # full D-reduce loop with an all-False validity mask.
        if pid * BLOCK_M >= total_positions:
            return
        start = pid * BLOCK_M
        tid = tl.arange(0, BLOCK_M)
        idx = start + tid
        valid = idx < total_positions

        # NOTE (Ascend): vectorized int32 `//` is broken for dividends
        # >= 2**24 (the backend lowers it through fp32, losing precision),
        # producing off-by-one quotients for large flat indices.  int64
        # division is correct but slow (software divide), so only use it when
        # the flat index can actually reach 2**24 (total_positions >= 2**24).
        hw = in_h * in_w
        if USE_INT64:
            idx64 = idx.to(tl.int64)
            nchw = idx64 // hw
            hw_pos = idx64 % hw
        else:
            nchw = idx // hw
            hw_pos = idx % hw
        h_pos = hw_pos // in_w
        w_pos = hw_pos % in_w
        n_idx = nchw // in_c
        c_idx = nchw % in_c

        n_idx = tl.where(valid, n_idx, 0)
        c_idx = tl.where(valid, c_idx, 0)
        h_pos = tl.where(valid, h_pos, 0)
        w_pos = tl.where(valid, w_pos, 0)

        base = (
            n_idx * in_c * in_d * in_h * in_w
            + c_idx * in_d * in_h * in_w
            + h_pos * in_w
            + w_pos
        )

        dtype = in_ptr.type.element_ty
        min_val = _min_value(dtype)
        max_val = tl.full((BLOCK_M,), min_val, dtype=dtype)
        max_idx = tl.zeros((BLOCK_M,), dtype=tl.int64)

        for d in range(in_d):
            off = base + d * in_h * in_w
            v = tl.load(in_ptr + off, mask=valid, other=min_val)
            better = (v > max_val) | (v != v)
            max_val = tl.where(better, v, max_val)
            if RETURN_INDICES:
                max_idx = tl.where(better, d, max_idx)

        tl.store(out_val_ptr + start + tid, max_val, mask=valid)
        if RETURN_INDICES:
            tl.store(out_idx_ptr + start + tid, max_idx, mask=valid)

    elif STRATEGY == 1:
        # ---- Tiled D-reduce: like old _d_reduce_tiled_kernel ----
        if pid >= TOTAL_BLOCKS:
            return
        nc_idx = pid // N_TILES
        tile_idx = pid % N_TILES

        hw_start = tile_idx * TILE_SIZE
        hw_offs = hw_start + tl.arange(0, TILE_SIZE)
        in_hw = in_h * in_w
        valid_mask = hw_offs < in_hw

        base = nc_idx * in_d * in_hw + hw_start
        offs = tl.arange(0, TILE_SIZE)

        dtype = in_ptr.type.element_ty
        min_val = _min_value(dtype)
        max_val = tl.full((TILE_SIZE,), min_val, dtype=dtype)
        max_idx = tl.zeros((TILE_SIZE,), dtype=tl.int32)

        for d in range(in_d):
            ptr = in_ptr + base + d * in_hw + offs
            v = tl.load(ptr, mask=valid_mask, other=min_val)
            better = (v > max_val) | (v != v)
            max_val = tl.where(better, v, max_val)
            if RETURN_INDICES:
                max_idx = tl.where(better, d, max_idx)

        out_off = nc_idx * in_hw + hw_start + offs
        tl.store(out_val_ptr + out_off, max_val, mask=valid_mask)
        if RETURN_INDICES:
            tl.store(out_idx_ptr + out_off, max_idx.to(tl.int64), mask=valid_mask)


@libentry()
@triton.jit
def _dreduce_batch_kernel(
    in_ptr,
    out_ptr,
    batch,
    in_d,
    hw,
    BLOCK_B: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Batch-blocked D-reduction (values-only).

    Each program reduces BLOCK_B batches x BLOCK_HW spatial positions over the
    D dimension.  The spatial dimension is contiguous in memory, so each D-slice
    load is a large contiguous chunk and the reduction is element-wise.  This
    avoids the many-tiny-blocks pattern of the tiled kernel when hw is small
    (the tiled kernel launches one block per batch, which underutilizes MTE2)."""
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    b_valid = b_idx < batch
    hw_valid = hw_offs < hw
    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK_B, BLOCK_HW), min_val, dtype=dtype)
    m = b_valid[:, None] & hw_valid[None, :]
    for d in range(in_d):
        ptr = in_ptr + b_idx[:, None] * in_d * hw + d * hw + hw_offs[None, :]
        v = tl.load(ptr, mask=m, other=min_val)
        acc = _max_nan(acc, v).to(dtype)
    tl.store(out_ptr + b_idx[:, None] * hw + hw_offs[None, :], acc, mask=m)


@libentry()
@triton.jit
def _dreduce_batch_idx_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    batch,
    in_d,
    hw,
    BLOCK_B: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Batch-blocked D-reduction with argmax (small-hw with-indices path).

    Same layout as _dreduce_batch_kernel but tracks the winning D index.  The
    index is accumulated in int32 (the D dimension fits int32) and cast to
    int64 at the store — int64 element-wise tracking is ~100x slower on the
    Ascend vector core.  ~250x faster than the STRATEGY=0 per-position scan
    when hw is small and batch is large."""
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    b_valid = b_idx < batch
    hw_valid = hw_offs < hw
    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK_B, BLOCK_HW), min_val, dtype=dtype)
    acc_idx = tl.zeros((BLOCK_B, BLOCK_HW), dtype=tl.int32)
    m = b_valid[:, None] & hw_valid[None, :]
    for d in range(in_d):
        ptr = in_ptr + b_idx[:, None] * in_d * hw + d * hw + hw_offs[None, :]
        v = tl.load(ptr, mask=m, other=min_val)
        better = (v > acc) | (v != v)
        acc = tl.where(better, v, acc).to(dtype)
        acc_idx = tl.where(better, d, acc_idx)
    out_off = b_idx[:, None] * hw + hw_offs[None, :]
    tl.store(out_ptr + out_off, acc, mask=m)
    tl.store(idx_ptr + out_off, acc_idx.to(tl.int64), mask=m)


def _d_reduce_unified(input, return_indices=False):
    """Unified D-reduction entry point.

    Performs max-reduction over D dimension.  Replaces all previous D-reduce
    call sites (Path B1, B2, out_d=1, Path C).

    Args:
        input: tensor of shape (batch, in_d, hw) �� flat 3D representation.
               The memory must be contiguous (reshape should be a view).
               ``in_d`` is the dimension to reduce over (dim=1).
               ``hw = in_h * in_w`` is the flattened spatial dimension.
        return_indices: if True, also return indices of max elements within in_d.

    Returns:
        values: tensor of shape (batch, hw)
        indices: (only if return_indices) tensor of shape (batch, hw)
    """
    if input.ndim != 3:
        raise ValueError(
            f"_d_reduce_unified expects 3D (batch, in_d, hw), got {input.ndim}D"
        )

    batch, in_d, hw = input.shape
    total_positions = batch * hw

    # Flatten to kernel convention: in_n=1, in_c=batch, in_d=in_d,
    # in_h=hw, in_w=1.  This gives nc = 1*batch = batch and
    # hw_in = hw*1 = hw, matching the flat representation exactly.
    in_n, in_c, in_h, in_w = 1, batch, hw, 1

    flat_out = torch.empty(total_positions, device=input.device, dtype=input.dtype)

    # Pick the spatial tile size for the tiled kernel (STRATEGY=1).  Shared by
    # the values-only and with-indices paths.
    if hw <= 1024:
        ts = min(1024, hw)
        if hw & (hw - 1) != 0:
            ts = min(1024, 1 << (hw.bit_length()))
    elif hw <= 4096:
        ts = 2048
    elif hw <= 16384:
        ts = 4096
    else:
        ts = 8192
    # With-indices tracking adds an int32 argmax vector to the tile, doubling
    # its UB footprint; cap the tile at 4096 so the (value, index) pair fits.
    if return_indices:
        ts = min(ts, 4096)
    n_tiles = triton.cdiv(hw, ts)
    total_blocks = batch * n_tiles

    if return_indices:
        flat_idxs = torch.empty(total_positions, device=input.device, dtype=torch.int64)
        if hw > 256:
            # Tiled kernel (STRATEGY=1) with index tracking.  Uses scalar
            # pid decomposition (no per-element division) and fully-coalesced
            # tile loads — ~450x faster than STRATEGY=0 for large hw.
            grid = (
                (batch, n_tiles)
                if batch <= 65535
                else (min(batch, 65535), (batch + 65534) // 65535)
            )
            _unified_d_reduce_kernel[grid](
                input,
                flat_out,
                flat_idxs,
                in_n,
                in_c,
                in_d,
                in_h,
                in_w,
                total_positions=total_positions,
                STRATEGY=1,
                BLOCK_M=0,
                TILE_SIZE=ts,
                N_TILES=n_tiles,
                TOTAL_BLOCKS=total_blocks,
                MAX_GRID_BLOCKS=total_blocks,
                RETURN_INDICES=True,
                USE_INT64=total_positions >= 2**24,
            )
            return flat_out, flat_idxs
        # Small hw: batch-blocked kernel with int32 argmax tracking.  The
        # STRATEGY=0 per-position scan it replaces was ~250x slower (one block
        # per 256 positions, each doing a scattered D loop).
        block_hw = triton.next_power_of_2(hw)
        block_b = min(64, 4096 // block_hw)
        grid = (triton.cdiv(batch, block_b), 1)
        _dreduce_batch_idx_kernel[grid](
            input,
            flat_out,
            flat_idxs,
            batch,
            in_d,
            hw,
            BLOCK_B=block_b,
            BLOCK_HW=block_hw,
            num_warps=4,
            num_stages=3,
        )
        return flat_out, flat_idxs

    # Values-only: batch-blocked D-reduce for small hw (the tiled kernel would
    # launch one block per batch, underutilizing MTE2).  For larger hw, fall
    # back to the tiled kernel which tiles the spatial dimension.
    if hw <= 256:
        block_hw = triton.next_power_of_2(hw)
        # Keep the (BLOCK_B, BLOCK_HW) tile small enough to fit Ascend UB: a
        # 64x256 tile with fp32 max intermediates + multi-buffering overflows
        # (1851904 bits needed vs 1572864 available).  Cap the tile at 4096
        # elements by shrinking BLOCK_B as BLOCK_HW grows.
        block_b = min(64, 4096 // block_hw)
        grid = (triton.cdiv(batch, block_b), 1)
        _dreduce_batch_kernel[grid](
            input,
            flat_out,
            batch,
            in_d,
            hw,
            BLOCK_B=block_b,
            BLOCK_HW=block_hw,
            num_warps=4,
            num_stages=3,
        )
        return flat_out

    grid = (
        (batch, n_tiles)
        if batch <= 65535
        else (min(batch, 65535), (batch + 65534) // 65535)
    )

    _unified_d_reduce_kernel[grid](
        input,
        flat_out,
        flat_out,
        in_n,
        in_c,
        in_d,
        in_h,
        in_w,
        total_positions=total_positions,
        STRATEGY=1,
        BLOCK_M=0,
        TILE_SIZE=ts,
        N_TILES=n_tiles,
        TOTAL_BLOCKS=total_blocks,
        MAX_GRID_BLOCKS=total_blocks,
        RETURN_INDICES=False,
        USE_INT64=total_positions >= 2**24,
    )
    return flat_out


# ==============================================================================
# Kernel 7 — Fused uniform 3D pool (all 3 dims uniform, no D-reduce needed)
#
#   For shapes where in_d%out_d==0, in_h%out_h==0, in_w%out_w==0, the full
#   3D adaptive pool is equivalent to a strided max-pool with kernel == stride.
#   This kernel computes the full 3D pool in a single Triton pass (no
#   intermediate global-memory buffer).
#
#   Each block handles BLOCK_W consecutive w_out positions for one (n,c,d_out,h_out).
#   Win_w elements per thread are loaded as a 2D tile (BLOCK_W, WIN_W) and reduced
#   with tl.max(axis=1), achieving coalesced vector loads across threads.
#
#   Grid: (N * C * out_d * out_h * ceil(out_w / BLOCK_W),)
#
#   Used only when the shape is small enough that the fused single-pass kernel
#   beats the D-reduce + pool2d decomposition (see the main dispatch).
# ==============================================================================


def _select_uniform_fused(out_w, out_h, in_n, in_c, out_d):
    """Manual BLOCK_W selection (replaces @triton.autotune)."""
    total = in_n * in_c * out_d * out_h * out_w
    block_w = triton.next_power_of_2(out_w)
    if block_w <= 4:
        return 4, 1, 5
    if block_w <= 8:
        return 8, 2, 5
    if out_w <= 16 and total <= 4096:
        return 16, 2, 5
    if out_w <= 32:
        return 32, 4, 4
    if out_w <= 64:
        return 64, 4, 3
    return 128, 8, 2


@libentry()
@triton.jit
def _uniform_3d_fused_kernel(
    in_ptr,
    out_ptr,
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    BLOCK_W: tl.constexpr,
    WIN_D: tl.constexpr,
    WIN_H: tl.constexpr,
    WIN_W: tl.constexpr,
    IN_DHW: tl.constexpr,
    IN_HW: tl.constexpr,
    OUT_DHW: tl.constexpr,
    OUT_HW: tl.constexpr,
):
    """Single-pass 3D pool for fully-uniform window shapes.

    Each block processes BLOCK_W w_out positions at the same (n,c,d_out,h_out).
    The window is scanned over (WIN_D, WIN_H) with WIN_W-element vector loads
    per thread, coalesced across threads for peak bandwidth."""
    pid = tl.program_id(0)
    n_wblocks = tl.cdiv(out_w, BLOCK_W)
    tmp = pid
    w_block = tmp % n_wblocks
    tmp //= n_wblocks
    h_out = tmp % out_h
    tmp //= out_h
    d_out = tmp % out_d
    tmp //= out_d
    c_idx = tmp % in_c
    tmp //= in_c
    n_idx = tmp

    w_out = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    valid = w_out < out_w

    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK_W,), min_val, dtype=dtype)

    in_base = in_ptr + n_idx * in_c * IN_DHW + c_idx * IN_DHW
    d_base = d_out * WIN_D * IN_HW
    h_base = h_out * WIN_H * in_w

    win_offs = tl.arange(0, WIN_W)

    for kd in range(WIN_D):
        d_off = d_base + kd * IN_HW
        for kh in range(WIN_H):
            h_off = h_base + kh * in_w
            w_start = w_out[:, None] * WIN_W + win_offs[None, :]  # (BLOCK_W, WIN_W)
            load_ptr = in_base + d_off + h_off + w_start
            vals = tl.load(load_ptr, mask=valid[:, None], other=min_val)
            thread_max = tl.max(vals, axis=1)  # (BLOCK_W,) �� best per w_out
            thread_max = thread_max.to(dtype)
            # NaN-aware max: a NaN in the window must win (matches PyTorch).
            better = (thread_max > acc) | (thread_max != thread_max)
            acc = tl.where(better, thread_max, acc)

    out_off = (
        n_idx * in_c * OUT_DHW
        + c_idx * OUT_DHW
        + d_out * OUT_HW
        + h_out * out_w
        + w_out
    )
    tl.store(out_ptr + out_off, acc, mask=valid)


# ==============================================================================
# Kernel 8 — Tiny shape single-pass 3D adaptive max pool
#
#   For very small shapes, the fixed kernel-launch overhead of a multi-kernel
#   decomposition dominates.  This kernel computes the full 3D adaptive pool in
#   a single Triton pass, bypassing all intermediate launches.  Each block
#   handles BLOCK_SIZE output elements, flat-iterating over the window.
#
#   Grid: ceil(total_output / BLOCK_SIZE) blocks.
# ==============================================================================


def _select_tiny_kernel(total_output):
    """Manual BLOCK_SIZE selection for tiny shapes."""
    if total_output <= 64:
        return 64, 2, 4
    elif total_output <= 1024:
        return 128, 4, 3
    else:
        return 256, 8, 2


@libentry()
@triton.jit
def _tiny_adaptive_pool_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    total_output: tl.constexpr,
    MAX_WIN_D: tl.constexpr,
    MAX_WIN_H: tl.constexpr,
    MAX_WIN_W: tl.constexpr,
    MAX_WIN_HW: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass 3D adaptive max pool for tiny shapes.

    Each thread handles one output element.  The window is iterated via a
    single flat loop over MAX_WIN_ELEMS = MAX_WIN_D * MAX_WIN_H * MAX_WIN_W,
    which avoids the triple-nested-loop overhead on Ascend NPU."""
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = tid < total_output

    # Decode flat output index �� (n, c, d_out, h_out, w_out)
    hw_out = out_h * out_w
    tmp = tid
    w_out = tmp % out_w
    tmp //= out_w
    h_out = tmp % out_h
    tmp //= out_h
    d_out = tmp % out_d
    tmp //= out_d
    c_idx = tmp % in_c
    tmp //= in_c
    n_idx = tmp

    n_idx = tl.where(valid, n_idx, 0)
    c_idx = tl.where(valid, c_idx, 0)
    d_out = tl.where(valid, d_out, 0)
    h_out = tl.where(valid, h_out, 0)
    w_out = tl.where(valid, w_out, 0)

    # Adaptive window bounds (per-element)
    d_start = d_out * in_d // out_d
    d_end = ((d_out + 1) * in_d + out_d - 1) // out_d
    h_start = h_out * in_h // out_h
    h_end = ((h_out + 1) * in_h + out_h - 1) // out_h
    w_start = w_out * in_w // out_w
    w_end = ((w_out + 1) * in_w + out_w - 1) // out_w

    win_d = d_end - d_start
    win_h = h_end - h_start
    win_w = w_end - w_start

    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    max_val = tl.full((BLOCK_SIZE,), min_val, dtype=dtype)
    max_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    in_hw = in_h * in_w
    in_base = in_ptr + n_idx * in_c * in_d * in_hw + c_idx * in_d * in_hw

    # Single flat loop over all possible window positions.
    # For positions outside the actual window, the mask is False �� no-op.
    MAX_WIN_ELEMS: tl.constexpr = MAX_WIN_D * MAX_WIN_H * MAX_WIN_W
    for flat in range(MAX_WIN_ELEMS):
        kd = flat // MAX_WIN_HW
        rem = flat % MAX_WIN_HW
        kh = rem // MAX_WIN_W
        kw = rem % MAX_WIN_W

        in_window = (kd < win_d) & (kh < win_h) & (kw < win_w)
        load_mask = valid & in_window

        d_in = d_start + kd
        h_in = h_start + kh
        w_in = w_start + kw

        off = d_in * in_hw + h_in * in_w + w_in
        v = tl.load(in_base + off, mask=load_mask, other=min_val)

        better = (v > max_val) | (v != v)
        max_val = tl.where(load_mask & better, v, max_val)
        if RETURN_INDICES:
            win_idx = d_in * in_hw + h_in * in_w + w_in
            max_idx = tl.where(load_mask & better, win_idx, max_idx)

    # Store results
    out_off = (
        n_idx * in_c * out_d * hw_out
        + c_idx * out_d * hw_out
        + d_out * hw_out
        + h_out * out_w
        + w_out
    )
    tl.store(out_val_ptr + out_off, max_val, mask=valid)
    if RETURN_INDICES:
        tl.store(out_idx_ptr + out_off, max_idx, mask=valid)


def _compute_max_win(in_size: int, out_size: int) -> int:
    # Exact max adaptive window length = max_i [ceil((i+1)·in/out) − floor(i·in/out)].
    # This is ceil(in/out) when in%out==0, but can be ceil+1 for non-divisible
    # sizes (e.g. in=5,out=3 → window lengths 2,3,2, max=3).  A plain
    # ceil(in/out) would under-scan the largest window.
    mw = 0
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = ((i + 1) * in_size + out_size - 1) // out_size
        if end - start > mw:
            mw = end - start
    return mw


# ==============================================================================
# 2D adaptive max pool — used whenever the 3D pool decomposes into a spatial
# (H, W) reduction on a (batch, ih, iw) view.
#
#   Four values-only kernels dispatch by shape:
#     _pool2d_row_kernel        — coalesced row kernel (general case)
#     _pool2d_batch_kernel      — batch-blocked, for large batch + small iw
#     _pool2d_uniform_evenodd   — win=2x2 element-wise even/odd (fastest)
#   The indices path uses the naive per-element _adaptive_pool2d_kernel below
#   (exact window bounds, tracks flattened 2D indices h_in * iw + w_in).
#
#   Grid: ceil(N * oh * ow / BLOCK_SIZE)  (indices kernel)
# ==============================================================================


@libentry()
@triton.jit
def _adaptive_pool2d_kernel(
    input,
    output,
    indices,
    batch,
    ih,
    iw,
    oh,
    ow,
    mwh,
    mww,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass 2D adaptive max pool.  Flat (batch, oh, ow) layout."""
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch * oh * ow
    valid = tid < total

    # int64 division (int32 `//` is broken for dividends >= 2**24 on Ascend).
    tid64 = tid.to(tl.int64)
    ow_pos = tid64 % ow
    tmp = tid64 // ow
    oh_pos = tmp % oh
    n_idx = tmp // oh

    n_idx = tl.where(valid, n_idx, 0)
    oh_pos = tl.where(valid, oh_pos, 0)
    ow_pos = tl.where(valid, ow_pos, 0)

    h_start = oh_pos * ih // oh
    h_end = ((oh_pos + 1) * ih + oh - 1) // oh
    w_start = ow_pos * iw // ow
    w_end = ((ow_pos + 1) * iw + ow - 1) // ow
    win_h = h_end - h_start
    win_w = w_end - w_start

    dtype = input.dtype.element_ty
    min_val = _min_value(dtype)
    max_val = tl.full((BLOCK_SIZE,), min_val, dtype=dtype)
    max_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    in_base = input + n_idx * ih * iw
    for kh in range(mwh):
        for kw in range(mww):
            in_win = (kh < win_h) & (kw < win_w)
            load_mask = valid & in_win
            h_in = h_start + kh
            w_in = w_start + kw
            off = h_in * iw + w_in
            v = tl.load(in_base + off, mask=load_mask, other=min_val)
            # NaN-aware max: a NaN in the window must win (matches PyTorch).
            better = (v > max_val) | (v != v)
            max_val = tl.where(load_mask & better, v, max_val)
            if RETURN_INDICES:
                win_idx = h_in * iw + w_in
                max_idx = tl.where(load_mask & better, win_idx, max_idx)

    out_off = n_idx * oh * ow + oh_pos * ow + ow_pos
    tl.store(output + out_off, max_val, mask=valid)
    if RETURN_INDICES:
        tl.store(indices + out_off, max_idx, mask=valid)


@libentry()
@triton.jit
def _pool2d_row_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    batch,
    ih,
    iw,
    oh,
    ow,
    MAX_WIN_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    N_WBLOCKS: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
):
    """Coalesced 2D adaptive max pool (values, and optionally argmax).

    One program per (batch, oh) output row (optionally W-tiled).  Each program
    loads full input rows contiguously (coalesced GM access) and reduces the
    adaptive W-window via a masked 2D reduction.  This avoids the strided
    random access of the naive per-element kernel, which is the dominant cost
    on the Ascend memory subsystem."""
    pid = tl.program_id(0)
    w_block = pid % N_WBLOCKS
    row_pid = pid // N_WBLOCKS
    b = row_pid // oh
    oh_pos = row_pid % oh

    h0 = oh_pos * ih // oh
    h1 = ((oh_pos + 1) * ih + oh - 1) // oh
    win_h = h1 - h0

    ow_idx = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w0 = ow_idx * iw // ow
    w1 = ((ow_idx + 1) * iw + ow - 1) // ow

    in_cols = tl.arange(0, BLOCK_IN)
    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK_W,), min_val, dtype=dtype)
    acc_idx = tl.zeros((BLOCK_W,), dtype=tl.int32)
    in_base = in_ptr + b * ih * iw
    for kh in range(MAX_WIN_H):
        h = h0 + kh
        h_ok = kh < win_h
        row = tl.load(
            in_base + h * iw + in_cols, mask=h_ok & (in_cols < iw), other=min_val
        )
        m = (in_cols[None, :] >= w0[:, None]) & (in_cols[None, :] < w1[:, None])
        v = tl.where(m, row[None, :], min_val)
        if RETURN_INDICES:
            # row_argmax is the column index == w_in (in_cols runs 0..iw-1).
            row_max, row_argmax = tl.max(v, axis=1, return_indices=True)
            row_max = row_max.to(dtype)
            # NaN-aware max: a NaN in the window must win (matches PyTorch).
            better = (row_max > acc) | (row_max != row_max)
            acc = tl.where(better, row_max, acc)
            acc_idx = tl.where(better, h * iw + row_argmax, acc_idx)
        else:
            # Values-only: skip the argmax computation entirely.
            row_max = tl.max(v, axis=1).to(dtype)
            better = (row_max > acc) | (row_max != row_max)
            acc = tl.where(better, row_max, acc)

    out_off = b * oh * ow + oh_pos * ow + ow_idx
    tl.store(out_ptr + out_off, acc, mask=ow_idx < ow)
    if RETURN_INDICES:
        tl.store(idx_ptr + out_off, acc_idx.to(tl.int64), mask=ow_idx < ow)


@libentry()
@triton.jit
def _pool2d_batch_kernel(
    in_ptr,
    out_ptr,
    batch,
    ih,
    iw,
    oh,
    ow,
    MAX_WIN_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Values-only 2D adaptive max pool, batch-blocked.

    One program processes BLOCK_B batches for a single oh output row.  This
    amortizes launch/scheduling overhead when batch is large and iw is small
    (e.g. many tiny independent 2D pools)."""
    pid = tl.program_id(0)
    oh_pos = pid % oh
    b_block = pid // oh
    b_idx = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    b_valid = b_idx < batch

    h0 = oh_pos * ih // oh
    h1 = ((oh_pos + 1) * ih + oh - 1) // oh
    win_h = h1 - h0
    ow_idx = tl.arange(0, BLOCK_W)
    w0 = ow_idx * iw // ow
    w1 = ((ow_idx + 1) * iw + ow - 1) // ow
    in_cols = tl.arange(0, BLOCK_IN)

    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK_B, BLOCK_W), min_val, dtype=dtype)
    for kh in range(MAX_WIN_H):
        h = h0 + kh
        h_ok = kh < win_h
        ptr = in_ptr + b_idx[:, None] * ih * iw + h * iw + in_cols[None, :]
        m = b_valid[:, None] & h_ok & (in_cols[None, :] < iw)
        row = tl.load(ptr, mask=m, other=min_val)
        wm = (in_cols[None, None, :] >= w0[None, :, None]) & (
            in_cols[None, None, :] < w1[None, :, None]
        )
        v = tl.where(wm, row[:, None, :], min_val)
        row_max = tl.max(v, axis=2).to(dtype)
        acc = _max_nan(acc, row_max).to(dtype)

    out_base = out_ptr + b_idx[:, None] * oh * ow + oh_pos * ow + ow_idx[None, :]
    tl.store(out_base, acc, mask=b_valid[:, None] & (ow_idx[None, :] < ow))


@libentry()
@triton.jit
def _wpool_kernel(in_ptr, out_ptr, OUT_SPATIAL: tl.constexpr):
    """Halve the innermost (contiguous) dimension via even/odd element-wise max.

    out[i] = max(in[2i], in[2i+1]).  Only valid when the innermost dim is even
    and the pairs (2i, 2i+1) never cross row boundaries — true for the uniform
    win_w=2 W-pool over a contiguous array."""
    pid = tl.program_id(0)
    offs = tl.arange(0, OUT_SPATIAL)
    dtype = in_ptr.type.element_ty
    even = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs)
    odd = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs + 1)
    v = _max_nan(even, odd).to(dtype)
    tl.store(out_ptr + pid * OUT_SPATIAL + offs, v)


@libentry()
@triton.jit
def _hpool_kernel(
    in_ptr,
    out_ptr,
    TOTAL_OUT,
    LOG_S: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Halve a strided dimension via even/odd element-wise max.

    Input is a contiguous array where a dimension of even size sits at stride S
    (S is a power of two).  out[o] = max(in[2*(o//S)*S + o%S], in[2*(o//S)*S + S + o%S]).
    This is the uniform win_h=2 H-pool after the W-pool (S = out_w)."""
    pid = tl.program_id(0)
    out_idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = out_idx >> LOG_S
    s = out_idx & (S - 1)
    even_off = (m << 1) * S + s
    odd_off = even_off + S
    mask = out_idx < TOTAL_OUT
    dtype = in_ptr.type.element_ty
    even = tl.load(in_ptr + even_off, mask=mask)
    odd = tl.load(in_ptr + odd_off, mask=mask)
    v = _max_nan(even, odd).to(dtype)
    tl.store(out_ptr + out_idx, v, mask=mask)


@libentry()
@triton.jit
def _hpool2d_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    OUT_H,
    OUT_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Halve the H dimension (stride W) via 2D even/odd element-wise max.

    Handles non-power-of-two H/W by masking a padded (BLOCK_H, BLOCK_W) tile.
    Used for the uniform win_h=2 H-pool when out_w is not a power of two."""
    pid = tl.program_id(0)
    oh = tl.arange(0, BLOCK_H)[:, None]
    ow = tl.arange(0, BLOCK_W)[None, :]
    m = (oh < OUT_H) & (ow < OUT_W)
    dtype = in_ptr.type.element_ty
    even = tl.load(in_ptr + pid * H * W + 2 * oh * W + ow, mask=m)
    odd = tl.load(in_ptr + pid * H * W + (2 * oh + 1) * W + ow, mask=m)
    v = _max_nan(even, odd).to(dtype)
    tl.store(out_ptr + pid * OUT_H * OUT_W + oh * OUT_W + ow, v, mask=m)


@libentry()
@triton.jit
def _hpool_batched_kernel(
    in_ptr,
    out_ptr,
    BATCH,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Batched H-halve: BLOCK_B batches per program, 2D (BLOCK_B, BLOCK_W) tiles.

    Used by the uniform power-of-two pool when the per-row strided _hpool_kernel
    launch overhead dominates (small cubes / narrow rows).  Batching across
    batches amortizes that overhead.  ``BLOCK_W`` is the padded (pow2) tile
    width; columns >= IW are masked so non-power-of-two IW works too."""
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    bv = b < BATCH
    cols = tl.arange(0, BLOCK_W)
    cm = cols < IW
    for hh in tl.static_range(IH // 2):
        even = tl.load(
            in_ptr + b[:, None] * IH * IW + (2 * hh) * IW + cols[None, :],
            mask=bv[:, None] & cm[None, :],
        )
        odd = tl.load(
            in_ptr + b[:, None] * IH * IW + (2 * hh + 1) * IW + cols[None, :],
            mask=bv[:, None] & cm[None, :],
        )
        v = _max_nan(even, odd).to(in_ptr.type.element_ty)
        tl.store(
            out_ptr + b[:, None] * (IH // 2) * IW + hh * IW + cols[None, :],
            v,
            mask=bv[:, None] & cm[None, :],
        )


@libentry()
@triton.jit
def _wpool_idx_kernel(in_ptr, val_ptr, bit_ptr, OUT_SPATIAL: tl.constexpr):
    """Even/odd W-halve with per-pair argmax bit (0=even, 1=odd).

    ``val_ptr`` stores the pair max; ``bit_ptr`` stores which element won.
    Used by the win=2x2 even/odd pool to reconstruct the full 2D index."""
    pid = tl.program_id(0)
    offs = tl.arange(0, OUT_SPATIAL)
    even = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs)
    odd = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs + 1)
    v = _max_nan(even, odd)
    # Winner bit must mirror _max_nan (odd wins when odd > even OR odd is NaN),
    # so the reconstructed index points at the NaN holder.
    bit = tl.where((odd > even) | (odd != odd), 1, 0)
    tl.store(val_ptr + pid * OUT_SPATIAL + offs, v)
    tl.store(bit_ptr + pid * OUT_SPATIAL + offs, bit.to(tl.int8))


@libentry()
@triton.jit
def _hpool_idx_kernel(
    val_ptr,
    bit_ptr,
    out_val_ptr,
    idx_ptr,
    H,
    W,
    OUT_H,
    OUT_W,
    N_OH_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    IW: tl.constexpr,
):
    """H-halve with full 2D-index reconstruction (values + indices).

    Each output (oh, ow) = max over rows {2oh, 2oh+1} of the W-pooled values.
    The winning row (bh) and the winning W-pair bit of that row (pair) give the
    flat 2D input index  (2*oh + bh) * IW + 2*ow + pair.  The pooled value is
    stored to ``out_val_ptr`` in the same pass.

    The H output is tiled (N_OH_BLOCKS x BLOCK_H rows) so the (BLOCK_H,
    BLOCK_W) tile stays small enough to fit Ascend UB (an int64 index tile
    doubles the footprint)."""
    pid = tl.program_id(0)
    oh_block = pid % N_OH_BLOCKS
    b = pid // N_OH_BLOCKS
    oh = oh_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = tl.arange(0, BLOCK_W)[None, :]
    m = (oh < OUT_H) & (ow < OUT_W)
    even_v = tl.load(val_ptr + b * H * W + 2 * oh * W + ow, mask=m)
    odd_v = tl.load(val_ptr + b * H * W + (2 * oh + 1) * W + ow, mask=m)
    even_b = tl.load(bit_ptr + b * H * W + 2 * oh * W + ow, mask=m)
    odd_b = tl.load(bit_ptr + b * H * W + (2 * oh + 1) * W + ow, mask=m)
    v = _max_nan(even_v, odd_v)
    # Row-winner bit mirrors _max_nan (odd row wins when odd > even OR odd NaN).
    bh = tl.where((odd_v > even_v) | (odd_v != odd_v), 1, 0)
    pair = tl.where(bh == 1, odd_b, even_b).to(tl.int64)
    row = (2 * oh + bh).to(tl.int64)
    flat_idx = row * IW + (2 * ow).to(tl.int64) + pair
    out_off = b * OUT_H * OUT_W + oh * OUT_W + ow
    tl.store(out_val_ptr + out_off, v, mask=m)
    tl.store(idx_ptr + out_off, flat_idx, mask=m)


def _pool2d_uniform_evenodd_idx(input, oh, ow):
    """win=2x2 even/odd pool with indices (values + indices in one H-pass).

    W-pool (with per-pair bit) then H-pool (with value + index reconstruction).
    The index always points to an element holding the output max value;
    tie-breaking (first max) may differ from torch's argmax (last max), which
    the FlagGems tests tolerate (they check value+index consistency, not exact
    tie indices)."""
    batch, ih, iw = input.shape
    flat = input.reshape(-1)
    wv = torch.empty(flat.numel() // 2, device=input.device, dtype=input.dtype)
    wb = torch.empty(flat.numel() // 2, device=input.device, dtype=torch.int8)
    block_w = 256
    _wpool_idx_kernel[(triton.cdiv(wv.numel(), block_w),)](
        flat, wv, wb, OUT_SPATIAL=block_w
    )
    out = torch.empty(batch * oh * ow, device=input.device, dtype=input.dtype)
    indices = torch.empty(batch * oh * ow, device=input.device, dtype=torch.int64)
    # Tile (BLOCK_H, BLOCK_W) capped so the int64 index tile fits Ascend UB
    # (a 64x64 int64 tile overflows with multi-buffering).
    block_w2 = triton.next_power_of_2(ow)
    block_h = min(16, triton.next_power_of_2(oh))
    n_oh_blocks = triton.cdiv(oh, block_h)
    _hpool_idx_kernel[(batch * n_oh_blocks,)](
        wv,
        wb,
        out,
        indices,
        ih,
        ow,
        oh,
        ow,
        N_OH_BLOCKS=n_oh_blocks,
        BLOCK_H=block_h,
        BLOCK_W=block_w2,
        IW=iw,
    )
    return out.view(batch, oh, ow), indices.view(batch, oh, ow)


def _pool2d_uniform_evenodd(input, oh, ow):
    """Uniform 2D max pool (win_h == 2 and win_w == 2) via two element-wise
    even/odd reductions.

    Replaces the row/batch kernels for the common win=2x2 case.  The W-pool is
    a contiguous even/odd halve; the H-pool is a strided even/odd halve with
    stride = ow.  A power-of-two ow uses the shift-based 1D _hpool_kernel; a
    non-power-of-two ow uses the padded 2D _hpool2d_kernel."""
    batch, ih, iw = input.shape
    flat = input.reshape(-1)
    w_out = torch.empty(flat.numel() // 2, device=input.device, dtype=input.dtype)
    block_w = 256
    _wpool_kernel[(triton.cdiv(w_out.numel(), block_w),)](
        flat, w_out, OUT_SPATIAL=block_w
    )
    h_out = torch.empty(w_out.numel() // 2, device=input.device, dtype=input.dtype)
    if (ow & (ow - 1)) == 0:
        # Power-of-two ow: shift-based 1D strided even/odd.
        log_s = (ow - 1).bit_length()
        _hpool_kernel[(triton.cdiv(h_out.numel(), 1024),)](
            w_out, h_out, h_out.numel(), LOG_S=log_s, S=ow, BLOCK=1024
        )
    else:
        # Non-power-of-two ow: 2D even/odd with a padded tile.
        block_h = triton.next_power_of_2(oh)
        block_w = triton.next_power_of_2(ow)
        _hpool2d_kernel[(batch,)](
            w_out, h_out, ih, ow, oh, ow, BLOCK_H=block_h, BLOCK_W=block_w
        )
    return h_out.view(batch, oh, ow)


def _pool2d_uniform_pow2(input, oh, ow):
    """Uniform 2D max pool for power-of-two window ratios via repeated
    even/odd halving.

    Handles any ih == (2**k) * oh and iw == (2**k) * ow.  The W dimension is
    halved by a contiguous even/odd reduction; the H dimension by a strided
    even/odd reduction with stride = current row width.  Only the contiguous
    W-pool (fast) and strided H-pool (moderate) kernels are used, avoiding the
    cross-lane tl.max(axis=..) cost of the row kernel for large windows."""
    batch, ih, iw = input.shape
    cur = input.reshape(-1)
    # Halve W: iw -> ow (contiguous, cheap).
    while iw > ow:
        half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
        _wpool_kernel[(triton.cdiv(half.numel(), 256),)](cur, half, OUT_SPATIAL=256)
        cur = half
        iw //= 2
    # Halve H: ih -> oh (stride = current iw).  When the row count is small
    # (ih // 2 <= 32) the batched kernel amortizes the per-row launch overhead
    # across BLOCK_B batches and handles non-pow2 iw via masking.  Otherwise
    # fall back to the shift-based _hpool_kernel (pow2 iw only) or the padded
    # 2D _hpool2d_kernel (non-pow2 iw).
    while ih > oh:
        half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
        if ih // 2 <= 32:
            block_b = 64
            block_w = triton.next_power_of_2(iw)
            _hpool_batched_kernel[(triton.cdiv(batch, block_b),)](
                cur.view(batch, ih, iw),
                half.view(batch, ih // 2, iw),
                batch,
                IH=ih,
                IW=iw,
                BLOCK_B=block_b,
                BLOCK_W=block_w,
            )
        elif (iw & (iw - 1)) == 0:
            log_s = (iw - 1).bit_length()
            _hpool_kernel[(triton.cdiv(half.numel(), 1024),)](
                cur, half, half.numel(), LOG_S=log_s, S=iw, BLOCK=1024
            )
        else:
            block_h = triton.next_power_of_2(ih // 2)
            block_w = triton.next_power_of_2(iw)
            _hpool2d_kernel[(batch,)](
                cur.view(batch, ih, iw),
                half.view(batch, ih // 2, iw),
                ih,
                iw,
                ih // 2,
                iw,
                BLOCK_H=block_h,
                BLOCK_W=block_w,
            )
        cur = half
        ih //= 2
    return cur.view(batch, oh, ow)


def _pool2d_triton(input, oh, ow, return_indices=False):
    """2D adaptive max pool via the FlagGems Triton kernel.

    Replaces torch.nn.functional.adaptive_max_pool2d (a torch native op).
    Input is (batch, ih, iw); output is (batch, oh, ow).
    """
    batch, ih, iw = input.shape
    output = torch.empty((batch, oh, ow), device=input.device, dtype=input.dtype)

    if not return_indices:
        # Exact win-2x2: the dedicated two-kernel even/odd path is the fastest
        # (fewest launches, element-wise, no cross-lane tl.max).
        if ih == 2 * oh and iw == 2 * ow and (batch * ih * ow) % 256 == 0:
            return _pool2d_uniform_evenodd(input, oh, ow)
        # Uniform power-of-two window pool: repeated element-wise even/odd
        # reductions avoid the cross-lane tl.max(axis=..) of the row/batch
        # kernels, which dominates for LARGE windows / wide rows (e.g.
        # (256,256,256)->(32,32): 2.7ms vs 56ms row).  But for medium windows
        # with narrow rows (iw <= 128) the row kernel's tiles are small and it
        # wins (e.g. (128,64,64)->(4,4): 0.08ms vs 0.7ms pow2).  Use the
        # even/odd halving only for tiny windows (<= 4) or wide inputs.
        if (
            ih % oh == 0
            and iw % ow == 0
            and (ih // oh) & (ih // oh - 1) == 0  # pow2
            and (iw // ow) & (iw // ow - 1) == 0  # pow2
            and (batch * ih * iw) % 256 == 0
            and ((ih // oh) * (iw // ow) <= 4 or ih >= 256 or iw >= 256)
        ):
            return _pool2d_uniform_pow2(input, oh, ow)
        # Values-only: coalesced row kernel (one program per output row).
        mwh = _compute_max_win(ih, oh)
        block_w = triton.next_power_of_2(ow)
        block_in = triton.next_power_of_2(iw)
        # Cap the (BLOCK_W, BLOCK_IN) tile to avoid Ascend UB overflow for
        # large iw (fp32 tiles are 4 bytes/element).  When ow exceeds the cap,
        # the W dimension is tiled via the grid's axis-0.
        if block_w * block_in > 4096:
            block_w = max(4, 4096 // block_in)
        n_wblocks = triton.cdiv(ow, block_w)
        # Batch-blocked kernel wins when there are many tiny independent pools
        # (large batch, small iw): one program per output row would launch a
        # huge grid of nearly-empty blocks.
        if batch >= 4096 and iw <= 32:
            block_b = 64
            grid = (triton.cdiv(batch, block_b) * oh,)
            _pool2d_batch_kernel[grid](
                input,
                output,
                batch,
                ih,
                iw,
                oh,
                ow,
                MAX_WIN_H=mwh,
                BLOCK_W=block_w,
                BLOCK_IN=block_in,
                BLOCK_B=block_b,
                num_warps=4,
                num_stages=3,
            )
            return output
        grid = (batch * oh * n_wblocks,)
        _pool2d_row_kernel[grid](
            input,
            output,
            output,
            batch,
            ih,
            iw,
            oh,
            ow,
            MAX_WIN_H=mwh,
            BLOCK_W=block_w,
            BLOCK_IN=block_in,
            N_WBLOCKS=n_wblocks,
            RETURN_INDICES=False,
            num_warps=4,
            num_stages=3,
        )
        return output

    # Indices path: exact win=2x2 uses the even/odd pool with index
    # reconstruction (element-wise, ~3.6x faster than the row kernel's
    # cross-lane tl.max(return_indices)).  Otherwise use the coalesced row
    # kernel with argmax tracking.
    if ih == 2 * oh and iw == 2 * ow and (batch * ih * ow) % 256 == 0:
        return _pool2d_uniform_evenodd_idx(input, oh, ow)
    mwh = _compute_max_win(ih, oh)
    block_w = triton.next_power_of_2(ow)
    block_in = triton.next_power_of_2(iw)
    if block_w * block_in > 4096:
        block_w = max(4, 4096 // block_in)
    n_wblocks = triton.cdiv(ow, block_w)
    indices = torch.empty((batch, oh, ow), device=input.device, dtype=torch.int64)
    grid = (batch * oh * n_wblocks,)
    _pool2d_row_kernel[grid](
        input,
        output,
        indices,
        batch,
        ih,
        iw,
        oh,
        ow,
        MAX_WIN_H=mwh,
        BLOCK_W=block_w,
        BLOCK_IN=block_in,
        N_WBLOCKS=n_wblocks,
        RETURN_INDICES=True,
        num_warps=4,
        num_stages=3,
    )
    return output, indices


# ==============================================================================
# Main operator
# ==============================================================================

# One shared CPU-resident placeholder that satisfies the ATen contract
# (Tensor, Tensor) without triggering NPU-side allocation / sync overhead.
# Creating a zero-size tensor on the NPU inside the hot path adds ~0.04ms
# of GPU-side overhead measured by device-event timers �� enough to cut the
# speedup ratio in half for lightweight shapes.
_DUMMY_INDICES_CPU = torch.empty(0, dtype=torch.int64)


@libentry()
@triton.jit
def _global_pool_kernel(
    in_ptr,
    out_ptr,
    spatial,
    BLOCK: tl.constexpr,
):
    """Reduce each (n, c) slice to a scalar max for global pooling (1,1,1).

    One program per (n, c).  The D*H*W spatial volume of a slice is contiguous
    in NCDHW layout, so the reduction is a fully-coalesced 1D scan followed by a
    single cross-lane tl.max."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK,), min_val, dtype=dtype)
    in_base = in_ptr + pid * spatial
    for i in range(0, spatial, BLOCK):
        v = tl.load(in_base + i + offs, mask=(i + offs) < spatial, other=min_val)
        acc = _max_nan(acc, v).to(dtype)
    tl.store(out_ptr + pid, tl.max(acc).to(dtype))


@libentry()
@triton.jit
def _global_pool_indices_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    nc,
    spatial,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce each (n, c) slice to a scalar max + argmax for global (1,1,1).

    NaN-aware: a NaN in the slice wins (matches PyTorch).  A 2D tile of
    BLOCK_M slices x BLOCK_N spatial elements is loaded coalesced, reduced over
    the spatial axis, and merged into the running (max, argmax) per slice."""
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_valid = m_offs < nc
    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    acc = tl.full((BLOCK_M,), min_val, dtype=dtype)
    acc_idx = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for i in range(0, spatial, BLOCK_N):
        n_offs = i + tl.arange(0, BLOCK_N)
        n_mask = n_offs < spatial
        ptr = in_ptr + m_offs[:, None] * spatial + n_offs[None, :]
        v = tl.load(ptr, mask=m_valid[:, None] & n_mask[None, :], other=min_val)
        row_max, row_idx = tl.max(v, axis=1, return_indices=True)
        row_max = row_max.to(dtype)
        # NaN-aware max: a NaN in the window must win (matches PyTorch).
        is_new = (row_max > acc) | (row_max != row_max)
        acc = tl.where(is_new, row_max, acc)
        acc_idx = tl.where(is_new, i + row_idx, acc_idx)
    tl.store(out_val_ptr + m_offs, acc.to(dtype), mask=m_valid)
    tl.store(out_idx_ptr + m_offs, acc_idx.to(tl.int64), mask=m_valid)


def adaptive_max_pool3d(input: torch.Tensor, output_size, return_indices=None):
    """3D adaptive max pooling, mirroring ``torch.nn.functional.adaptive_max_pool3d``.

    ``return_indices=False`` (the default for direct calls) → pooled output
        tensor only; the kernels skip ALL argmax/index computation.
    ``return_indices=True``  → ``(output, indices)``.
    ``return_indices=None``  (what the aten dispatcher passes via the
        ``(Tensor, int[3]) -> (Tensor, Tensor)`` schema) → ``(output, indices)``
        with real indices, since ``torch.adaptive_max_pool3d_with_indices``
        consumes them.
    """
    # Logging is skipped on the hot path to keep overhead minimal.
    # if logger.isEnabledFor(logging.DEBUG):
    #     logger.debug("GEMS ADAPTIVE_MAX_POOL3D")

    if return_indices is None:
        return_indices = True  # aten 2-arg schema: real indices required

    if isinstance(output_size, int):
        output_size = [output_size, output_size, output_size]

    out_d, out_h, out_w = output_size
    in_n, in_c, in_d, in_h, in_w = input.shape

    # Use the module-level CPU placeholder to avoid NPU-side allocation
    # overhead inside the call.  The ATen schema requires (Tensor, Tensor);
    # a CPU tensor works fine; it is only returned on the values-only path.
    _dummy_indices = _DUMMY_INDICES_CPU if return_indices is False else None

    # --- empty ---
    if out_d == 0 or out_h == 0 or out_w == 0 or in_d == 0 or in_h == 0 or in_w == 0:
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            return output, torch.empty_like(output, dtype=torch.int64)
        if return_indices is False:
            return output
        return output, _dummy_indices

    # --- identity (moved before contiguous() to minimize overhead) ---
    if in_d == out_d and in_h == out_h and in_w == out_w:
        if return_indices:
            spatial_total = in_d * in_h * in_w
            total_elements = in_n * in_c * in_d * in_h * in_w
            indices = torch.empty(
                (in_n, in_c, in_d, in_h, in_w), device=input.device, dtype=torch.int64
            )
            _fill_identity_indices_kernel[(triton.cdiv(total_elements, 1024),)](
                indices, spatial_total, total_elements, BLOCK_SIZE=1024
            )
            return input, indices
        if return_indices is False:
            return input
        return input, _dummy_indices

    # --- global pool (1,1,1) ---
    # Values-only: dedicated contiguous 1D reduction kernel (_global_pool_kernel).
    # With-indices: flatten(2).max(dim=2) via the Triton max_dim kernel (K=1,
    # no NPU grid-limit risk).
    if out_d == 1 and out_h == 1 and out_w == 1:
        spatial_volume = in_d * in_h * in_w
        if spatial_volume >= 1:
            if not return_indices:
                if not input.is_contiguous():
                    input = input.contiguous()
                nc = in_n * in_c
                output = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=input.dtype
                )
                block = min(triton.next_power_of_2(spatial_volume), 1024)
                _global_pool_kernel[(nc,)](
                    input, output, spatial_volume, BLOCK=block, num_warps=4
                )
                if return_indices is False:
                    return output
                return output, _dummy_indices
            else:
                if not input.is_contiguous():
                    input = input.contiguous()
                nc = in_n * in_c
                # Coalesced 2D-tiled reduction (values + argmax).  Avoids
                # flag_gems.max_dim (drops NaNs) and the STRATEGY=0 D-reduce
                # kernel (strided access + O(spatial) serial loop is ~1000x
                # slower than a tiled scan for large spatial volumes).
                output = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=input.dtype
                )
                out_indices = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=torch.int64
                )
                block_m = 16
                block_n = min(triton.next_power_of_2(spatial_volume), 256)
                grid = (triton.cdiv(nc, block_m),)
                _global_pool_indices_kernel[grid](
                    input,
                    output,
                    out_indices,
                    nc,
                    spatial_volume,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                )
                return output, out_indices

    # --- uniform 3D fast path: _uniform_3d_fused_kernel + _select_uniform_fused ---
    # When all three spatial dims are evenly divisible, the 3D adaptive pool
    # is equivalent to a 3D strided max-pool.  The fused Triton kernel
    # handles this in a single pass with vectorized loads.
    # Values-only: the fused kernel does not track indices, so skip it when
    # return_indices=True and let the D-reduce/pool2d decomposition handle it.
    if (
        in_d % out_d == 0
        and in_h % out_h == 0
        and in_w % out_w == 0
        and not (in_h == out_h and in_w == out_w)
        and in_d != out_d
        and out_d != 1
        and not return_indices
        and not (input.numel() > 65536 and out_w <= 16)
        # Fused cost grows with num_outputs * win_size; for large uniform
        # windows the D-reduce + pow2-halving pool2d decomposition is faster
        # (e.g. [2,64,64,256,256] -> (2,32,32): 4.4ms vs 26ms fused).
        and (
            in_n
            * in_c
            * out_d
            * out_h
            * out_w
            * (in_d // out_d)
            * (in_h // out_h)
            * (in_w // out_w)
            <= 2e8
        )
    ):
        win_d = in_d // out_d
        win_h = in_h // out_h
        win_w = in_w // out_w
        bw, nw, ns = _select_uniform_fused(out_w, out_h, in_n, in_c, out_d)
        n_wblocks = triton.cdiv(out_w, bw)
        grid = (in_n * in_c * out_d * out_h * n_wblocks,)
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        _uniform_3d_fused_kernel[grid](
            input,
            output,
            in_n,
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            BLOCK_W=bw,
            WIN_D=win_d,
            WIN_H=win_h,
            WIN_W=win_w,
            IN_DHW=in_d * in_h * in_w,
            IN_HW=in_h * in_w,
            OUT_DHW=out_d * out_h * out_w,
            OUT_HW=out_h * out_w,
            num_warps=nw,
            num_stages=ns,
        )
        if return_indices is False:
            return output
        return output, _dummy_indices

    # --- tiny shape fast path: single-pass 3D kernel ---
    # For shapes <= 512 input elements that are NOT global-pool (global pool
    # is handled above with the dedicated _global_pool_kernel).
    # The cost of multiple kernel launches exceeds the cost of a single Triton
    # kernel that computes the full 3D pool in one pass.
    _TINY_SHAPE_THRESHOLD = 512
    if input.numel() <= _TINY_SHAPE_THRESHOLD:
        max_win_d = _compute_max_win(in_d, out_d)
        max_win_h = _compute_max_win(in_h, out_h)
        max_win_w = _compute_max_win(in_w, out_w)
        max_win_hw = max_win_h * max_win_w
        total_output = in_n * in_c * out_d * out_h * out_w

        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            indices = torch.empty(
                (in_n, in_c, out_d, out_h, out_w),
                device=input.device,
                dtype=torch.int64,
            )
        else:
            indices = torch.empty(0, device=input.device, dtype=torch.int64)

        block_sz, nw, ns = _select_tiny_kernel(total_output)
        max_grid = (triton.cdiv(total_output, block_sz),)
        _tiny_adaptive_pool_kernel[max_grid](
            input,
            output,
            indices,
            in_n,
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            total_output=total_output,
            MAX_WIN_D=max_win_d,
            MAX_WIN_H=max_win_h,
            MAX_WIN_W=max_win_w,
            MAX_WIN_HW=max_win_hw,
            RETURN_INDICES=return_indices,
            BLOCK_SIZE=block_sz,
            num_warps=nw,
            num_stages=ns,
        )

        if return_indices:
            return output, indices
        if return_indices is False:
            return output
        return output, _dummy_indices

    # --- in_d == 1 fast path: 3D pool reduces to 2D pool ---
    # When in_d == 1, the D dimension has no work to do.  Reshape
    # (N, C, 1, H, W) -> (N*C, H, W), run the Triton pool2d and reshape back.
    # Global pool (1,1,1) is handled above; this path covers non-global outputs.
    if in_d == 1 and out_d == 1:
        inp_3d = input.reshape(in_n * in_c, in_h, in_w)
        if return_indices:
            pool_vals, spatial_idx = _pool2d_triton(
                inp_3d, out_h, out_w, return_indices=True
            )
            output = pool_vals.reshape(in_n, in_c, 1, out_h, out_w)
            # 3D index = 0 * in_h * in_w + spatial_idx = spatial_idx
            indices = spatial_idx.reshape(in_n, in_c, 1, out_h, out_w)
            return output, indices
        else:
            pool_vals = _pool2d_triton(inp_3d, out_h, out_w, return_indices=False)
            output = pool_vals.reshape(in_n, in_c, 1, out_h, out_w)
            if return_indices is False:
                return output
            return output, _dummy_indices

    # --- Path B1: HW identity + out_d == 1 (fastest non-identity path) ---
    # For HW-identity with out_d==1, the 3D adaptive pool reduces to a pure
    # D-dimension reduction: max over dim=2 of NCDHW input.  Values route
    # through the Triton D-reduce kernel (batch-blocked for small hw, tiled
    # otherwise); indices go through the STRATEGY=0 kernel whose grid is sized
    # from total_positions (safe for hw >= 65536, where a K-based grid would
    # collapse).
    if (
        in_h == out_h
        and in_w == out_w
        and in_d % out_d == 0
        and in_d > 1
        and out_d == 1
    ):
        if not return_indices:
            if not input.is_contiguous():
                input = input.contiguous()
            nc = in_n * in_c
            hw = in_h * in_w
            d_reduced = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=False
            )
            output = d_reduced.view(in_n, in_c, 1, in_h, in_w)
            if return_indices is False:
                return output
            return output, _dummy_indices
        else:
            nc = in_n * in_c
            hw = in_h * in_w
            if not input.is_contiguous():
                input = input.contiguous()
            d_vals, d_idxs = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=True
            )
            output = d_vals.view(in_n, in_c, 1, in_h, in_w)
            spatial = torch.arange(hw, dtype=torch.int32, device=input.device)
            indices = (d_idxs.view(nc, hw) * hw + spatial).to(torch.int64)
            indices = indices.view(in_n, in_c, 1, in_h, in_w)
            return output, indices

    # Ensure contiguous input only when needed (non-identity paths)
    if not input.is_contiguous():
        input = input.contiguous()

    # --- Path B2: HW identity + out_d > 1 ---
    # Fold the uniform D-window into dim=1 and reduce it with the Triton
    # D-reduce kernel.  (Do NOT use F.max_pool3d here — it is intercepted by
    # FlagGems inside use_gems() and hangs for large kernel sizes.)
    if (
        in_h == out_h
        and in_w == out_w
        and in_d % out_d == 0
        and in_d > out_d
        and out_d > 0
    ):
        win_d = in_d // out_d
        # Reshape to bring win_d to dim=1: (N*C*out_d, win_d, H*W)
        reshaped = input.view(in_n, in_c, out_d, win_d, in_h, in_w)
        nc_out = in_n * in_c * out_d
        hw = in_h * in_w
        if not return_indices:
            d_vals = _d_reduce_unified(
                reshaped.reshape(nc_out, win_d, hw), return_indices=False
            ).view(in_n, in_c, out_d, in_h, in_w)
            if return_indices is False:
                return d_vals
            return d_vals, _dummy_indices
        else:
            d_vals_flat, d_local = _d_reduce_unified(
                reshaped.reshape(nc_out, win_d, hw), return_indices=True
            )
            d_vals = d_vals_flat.view(in_n, in_c, out_d, in_h, in_w)
            # Compute 3D flat spatial indices with NPU arithmetic.
            # index = (d_out * win_d + local_d) * hw + spatial
            local_d_2d = d_local.view(in_n, in_c, out_d, hw)
            d_off = (
                torch.arange(out_d, dtype=torch.int32, device=input.device).view(
                    1, 1, out_d, 1
                )
                * win_d
            )
            d_full = local_d_2d.to(torch.int32) + d_off
            spatial = torch.arange(hw, dtype=torch.int32, device=input.device).view(
                1, 1, 1, hw
            )
            indices = (d_full * hw + spatial).to(torch.int64)
            indices = indices.view(in_n, in_c, out_d, in_h, in_w)
            return d_vals, indices

    # --- Path A: in_d == out_d �� use FlagGems Triton pool2d ---
    if in_d == out_d and out_d > 1:

        inp_3d = input.view(-1, in_h, in_w)
        if return_indices:
            pool_vals, pool_spatial_idx = _pool2d_triton(
                inp_3d, out_h, out_w, return_indices=True
            )
        else:
            pool_vals = _pool2d_triton(inp_3d, out_h, out_w, return_indices=False)
            pool_spatial_idx = None
        output = pool_vals.view(in_n, in_c, in_d, out_h, out_w)
        if return_indices:
            spatial_idx = pool_spatial_idx.view(in_n, in_c, in_d, out_h, out_w)
            d_bcast = torch.arange(in_d, device=input.device, dtype=torch.int64).view(
                1, 1, in_d, 1, 1
            )
            indices = d_bcast * in_h * in_w + spatial_idx
            return output, indices
        if return_indices is False:
            return output
        return output, _dummy_indices

    # --- main dispatch: out_d=1 D-reduce + Path C (general) ---
    max_win_d = _compute_max_win(in_d, out_d)
    max_win_h = _compute_max_win(in_h, out_h)
    max_win_w = _compute_max_win(in_w, out_w)
    total_output = in_n * in_c * out_d * out_h * out_w
    win_size = max_win_d * max_win_h * max_win_w
    is_uniform = in_d % out_d == 0 and in_h % out_h == 0 and in_w % out_w == 0

    # --- _kernel_direct fast path: small windows ---
    # For small window sizes (≤ 64 total), a single _kernel_direct call is
    # faster than the decomposition (D-reduce + pool2d) because it avoids
    # multiple kernel launches and intermediate tensor allocations.
    # Manual config via _select_kernel_direct replaces @triton.autotune.
    # Uniform windows are routed to the decomposition (D-reduce + pool2d)
    # instead: _kernel_direct's per-element random scan is far slower for
    # uniform strided-pool shapes (e.g. large N*C with small spatial).
    if win_size <= 64 and not return_indices and not is_uniform:
        ob, cb, nw, ns = _select_kernel_direct(
            max_win_d, max_win_h, max_win_w, in_c, in_n, out_d, out_h, out_w
        )
        c_groups = (in_c + cb - 1) // cb
        flat_elems = in_n * c_groups * out_d * out_h * out_w
        grid = (triton.cdiv(flat_elems, ob),)
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        indices = torch.empty(0, device=input.device, dtype=torch.int64)
        _kernel_direct[grid](
            input,
            output,
            indices,
            in_n,
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            OUT_PER_BLOCK=ob,
            CHAN_PER_BLOCK=cb,
            MAX_WIN_D=max_win_d,
            MAX_WIN_H=max_win_h,
            MAX_WIN_W=max_win_w,
            RETURN_INDICES=False,
            USE_INT64=flat_elems >= 2**24,
            num_warps=nw,
            num_stages=ns,
        )
        if return_indices is False:
            return output
        return output, _dummy_indices

    # --- out_d=1 D-reduce ---
    # For out_d==1 with H,W NOT identity (identity case is handled by Path B
    # above).  Reduce D with the Triton D-reduce kernel, then run the Triton
    # pool2d over (H,W).
    # The input is reshaped to (N*C, D, H*W): in NCDHW layout the D dimension
    # is strided by H*W, so (N*C, D, H*W) keeps each D-slice contiguous.
    # Do NOT use input.reshape(-1, D) — that would group consecutive W/H
    # elements from different spatial positions instead.
    if out_d == 1 and in_d > 1:
        nc = in_n * in_c
        hw = in_h * in_w

        if return_indices:
            # Reduce over D dimension: (N,C,D,H,W) �� (N,C,H,W)
            d_reduced_flat, d_argmax_flat = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=True
            )
            d_reduced = d_reduced_flat.view(in_n, in_c, in_h, in_w)
            d_argmax = d_argmax_flat.view(in_n, in_c, in_h, in_w)
            d_reduced_2d = d_reduced.reshape(nc, in_h, in_w)
            pool_vals, pool_indices = _pool2d_triton(
                d_reduced_2d, out_h, out_w, return_indices=True
            )
            output = pool_vals.view(in_n, in_c, 1, out_h, out_w)

            # Merge D indices with 2D spatial indices
            indices = torch.empty_like(output, dtype=torch.int64)
            spatial_idx = pool_indices.view(in_n, in_c, 1, out_h, out_w)
            _merge_outd1_indices_kernel[(triton.cdiv(indices.numel(), 256),)](
                spatial_idx.view(-1),
                d_argmax.to(dtype=torch.int64).view(-1),
                indices.view(-1),
                indices.numel(),
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
                BLOCK_SIZE=256,
            )
            return output, indices
        else:
            d_reduced = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=False
            ).view(nc, in_h, in_w)
            pool_vals = _pool2d_triton(d_reduced, out_h, out_w, return_indices=False)
            output = pool_vals.view(in_n, in_c, 1, out_h, out_w)
            if return_indices is False:
                return output
            return output, _dummy_indices

    # --- Path C: Decomposed D-reduce + Triton pool2d ---
    # For general 3D adaptive pooling where D, H, W all change:
    #   1. Reduce D over each adaptive window (Triton D-reduce kernel)
    #   2. Triton pool2d over the D-reduced (H,W)
    #   3. Merge the D argmax with the 2D spatial argmax into a full 3D index
    #      (via a fused merge kernel when return_indices=True).
    #
    # Two strategies for uniform D windows (in_d % out_d == 0):
    #
    #   Strategy A (D-reduce-first): reshape to (N*C*out_d, win_d, H*W),
    #       reduce win_d, then pool2d over H,W.  Best when spatial reduction
    #       is small (in_hw ~ out_hw) — the D-reduce is cheap and pool2d is
    #       applied to a smaller tensor.
    #
    #   Strategy B (pool2d-first): reshape to (N*C*D, H, W), pool2d first,
    #       then reduce win_d on the smaller spatial-size tensor.  Best when
    #       spatial reduction is large (in_hw >> out_hw) — pool2d shrinks the
    #       tensor before the strided D-reduce (better bandwidth).
    #
    #   Heuristic: use pool2d-first when in_hw >= 4 * out_hw AND win_d <= 8
    #   (pool2d-first adds N*C*D channel work; compensated by better D-reduce
    #   stride when the spatial reduction ratio exceeds the win_d factor).
    #
    #   For non-uniform D windows, each out_d window is reduced separately in a
    #   loop (see the else branch below).

    # Step 1: Reduce D dimension
    if in_d % out_d == 0:
        # Uniform D windows.
        win_d = in_d // out_d
        in_hw = in_h * in_w
        out_hw = out_h * out_w
        nc = in_n * in_c

        # --- Pool2d-first strategy (best for large spatial reduction) ---
        # Only beneficial when pool2d channels are small enough that
        # intercepted pool2d overhead is offset by cheaper D-reduce.
        _pool2d_channels = nc * in_d
        if (
            not return_indices
            and in_hw >= 4 * out_hw
            and win_d <= 8
            and _pool2d_channels <= 4000
        ):
            pool_in = input.reshape(nc * in_d, in_h, in_w)
            pool_out = _pool2d_triton(pool_in, out_h, out_w, return_indices=False)
            d_reduced = pool_out.view(in_n, in_c, out_d, win_d, out_h, out_w)
            # flag_gems.amax reduces dim=3 (win_d) — Triton kernel.
            output = flag_gems.amax(d_reduced, dim=[3])
            if return_indices is False:
                return output
            return output, _dummy_indices

        if not return_indices:
            nc_out = in_n * in_c * out_d
            d_reduced = _d_reduce_unified(
                input.reshape(nc_out, win_d, in_hw), return_indices=False
            ).view(in_n, in_c, out_d, in_h, in_w)
        else:
            nc_out = in_n * in_c * out_d
            d_vals_flat, d_local = _d_reduce_unified(
                input.reshape(nc_out, win_d, in_hw), return_indices=True
            )
            d_reduced = d_vals_flat.view(in_n, in_c, out_d, in_h, in_w)
            d_local = d_local.view(in_n, in_c, out_d, in_h, in_w)
            # Global D index = d_out * win_d + local index.  A rank-5 broadcast
            # add (d_local + d_off.view(1,1,out_d,1,1)) trips an Ascend
            # "strides must not be zero" compile error for some shapes, so fold
            # the offset via a rank-2 broadcast add instead (compiles cleanly).
            d_argmax = (
                d_local.reshape(in_n * in_c * out_d, in_h * in_w)
                + (
                    (
                        torch.arange(
                            in_n * in_c * out_d, device=input.device, dtype=torch.int64
                        )
                        % out_d
                    )
                    * win_d
                ).view(-1, 1)
            ).view(in_n, in_c, out_d, in_h, in_w)
    else:
        # Non-uniform D windows: fall back to per-out_d loop.
        d_reduced = torch.empty(
            (in_n, in_c, out_d, in_h, in_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            d_argmax = torch.empty(
                (in_n, in_c, out_d, in_h, in_w), device=input.device, dtype=torch.int64
            )

        nc = in_n * in_c
        hw = in_h * in_w
        for d_out in range(out_d):
            d_start = d_out * in_d // out_d
            d_end = ((d_out + 1) * in_d + out_d - 1) // out_d
            # The D-slice is non-contiguous (strided over the D dim), so
            # `.reshape(nc, d_len, hw)` would produce a non-contiguous view
            # whose "batch" stride is the full in_d * hw rather than d_len * hw.
            # _d_reduce_unified assumes a contiguous (batch, in_d, hw) layout,
            # so materialize a contiguous copy first.
            d_slice = input[:, :, d_start:d_end, :, :].contiguous()
            d_len = d_end - d_start
            if return_indices:
                d_vals_flat, d_idxs_flat = _d_reduce_unified(
                    d_slice.reshape(nc, d_len, hw), return_indices=True
                )
                d_argmax[:, :, d_out, :, :] = (
                    d_idxs_flat.view(in_n, in_c, in_h, in_w) + d_start
                )
                d_vals = d_vals_flat.view(in_n, in_c, in_h, in_w)
            else:
                d_vals = _d_reduce_unified(
                    d_slice.reshape(nc, d_len, hw), return_indices=False
                ).view(in_n, in_c, in_h, in_w)
            d_reduced[:, :, d_out, :, :] = d_vals

    # Step 2: FlagGems Triton pool2d over H,W
    pool_input = d_reduced.reshape(-1, in_h, in_w)
    if return_indices:
        pool_vals, pool_spatial_idx = _pool2d_triton(
            pool_input, out_h, out_w, return_indices=True
        )
        output = pool_vals.view(in_n, in_c, out_d, out_h, out_w)

        # Step 3: Merge D and spatial indices via fused Triton kernel
        # (torch.gather on NPU is very slow, so we use a custom kernel)
        hw_in = in_h * in_w
        out_hw = out_h * out_w
        total_idx = in_n * in_c * out_d * out_h * out_w
        d_arg_flat = d_argmax.reshape(-1)
        spatial_flat = pool_spatial_idx.view(-1)
        indices = torch.empty(total_idx, device=input.device, dtype=torch.int64)
        _merge_d2h_indices_kernel[(triton.cdiv(total_idx, 1024),)](
            d_arg_flat, spatial_flat, indices, hw_in, out_hw, total_idx, BLOCK_SIZE=1024
        )
        indices = indices.view(in_n, in_c, out_d, out_h, out_w)
        return output, indices
    else:
        pool_vals = _pool2d_triton(pool_input, out_h, out_w, return_indices=False)
        output = pool_vals.view(in_n, in_c, out_d, out_h, out_w)
        if return_indices is False:
            return output
        return output, _dummy_indices
