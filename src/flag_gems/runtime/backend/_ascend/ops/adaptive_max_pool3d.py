import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _min_value(dtype):
    """Finite sentinel below every value of ``dtype`` -- a fork of
    ``flag_gems.utils.limits.get_dtype_min``, whose float64 ``inf`` raises on Ascend."""
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
    """NaN-aware element-wise max: ``tl.maximum`` drops NaN, max-pooling must not."""
    return tl.where((b > a) | (b != b), b, a)


# Kernel 1 -- direct scan: M = OUT_PER_BLOCK outputs, K = CHAN_PER_BLOCK channels.


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

    # int32 ``//`` is wrong past 2**24 on Ascend; int64 is a slow software divide.
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


# --- Helpers ---


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
    base,
    BLOCK_SIZE: tl.constexpr,
):
    # Two offsets: the decode needs absolute ``offsets``, the sliced pointers need ``rel``.
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    rel = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = base + rel
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
    spatial_idx = tl.load(spatial_indices_ptr + rel, mask=mask, other=0)
    h_best = spatial_idx // in_w
    w_best = spatial_idx % in_w
    d_off = n_idx * in_c * in_h * in_w + c_idx * in_h * in_w + h_best * in_w + w_best
    d_best = tl.load(d_argmax_ptr + d_off, mask=mask, other=0)
    tl.store(output_indices_ptr + rel, d_best * in_h * in_w + spatial_idx, mask=mask)


# The identity index fill is a torch broadcast, not a kernel -- do not reintroduce.


@triton.jit
def _merge_d2h_indices_kernel(
    d_local_ptr,
    spatial_idx_ptr,
    out_idx_ptr,
    hw_in,
    out_hw,
    win_d,
    in_d,
    out_d,
    n_rows,
    row_base,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    UNIFORM_D: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Merge the D index with the 2-D spatial index into the final flat index.

    One program owns BLOCK_R whole output rows, so the per-row decode is scalar per step
    and the gather is fused away.  Every load stays 1-D on purpose: the 2-D-tile form is
    faster but silently wrong for some ``out_hw``.  See notes.md.
    """
    # ``row`` is absolute (it selects the d_local row); the sliced accesses use ``rl``.
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < out_hw

    for i in tl.static_range(BLOCK_R):
        rl = pid * BLOCK_R + i
        if rl < n_rows:
            off = rl * out_hw + cols
            row = row_base + rl
            if WIDE:
                # int64 throughout: the index space can exceed 2**31 -- host picks this only then.
                row64 = row.to(tl.int64)
                sidx = tl.load(spatial_idx_ptr + off, mask=col_mask, other=0)
                d_local = tl.load(
                    d_local_ptr + row64 * hw_in + sidx, mask=col_mask, other=0
                )
                d_out = row64 % out_d
                d_global = d_local + tl.where(
                    UNIFORM_D, d_out * win_d, d_out * in_d // out_d
                )
                tl.store(out_idx_ptr + off, d_global * hw_in + sidx, mask=col_mask)
            else:
                # int32: int64 element-wise ops are ~100x slower here.  Every address stays below
                # 2**31 (host-asserted); ATen's int64 index is materialized by the final cast.
                sidx = tl.load(spatial_idx_ptr + off, mask=col_mask, other=0).to(
                    tl.int32
                )
                d_local = tl.load(
                    d_local_ptr + row * hw_in + sidx, mask=col_mask, other=0
                ).to(tl.int32)
                d_out = row % out_d
                d_global = d_local + tl.where(
                    UNIFORM_D, d_out * win_d, d_out * in_d // out_d
                )
                tl.store(
                    out_idx_ptr + off,
                    (d_global * hw_in + sidx).to(tl.int64),
                    mask=col_mask,
                )


# --- Kernel 3 -- tiled D-reduction: one block per (row, tile) of contiguous HW ---


@libentry()
@triton.jit
def _unified_d_reduce_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    in_d,
    in_hw,
    TILE_SIZE: tl.constexpr,
    N_TILES: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    IDX32: tl.constexpr,
):
    """Tiled D-reduction: one block per (row, tile); pid flattens the (batch, N_TILES) grid."""
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    if pid >= TOTAL_BLOCKS:
        return
    nc_idx = pid // N_TILES
    tile_idx = pid % N_TILES

    hw_start = tile_idx * TILE_SIZE
    hw_offs = hw_start + tl.arange(0, TILE_SIZE)
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
        # int64 is the ATen *return* contract; an in-file intermediate stays int32.
        if IDX32:
            tl.store(out_idx_ptr + out_off, max_idx, mask=valid_mask)
        else:
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
    """Batch-blocked D-reduction (values-only), BLOCK_B batches x BLOCK_HW positions.
    The spatial axis is contiguous, so every D-slice load is one long run.
    """
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
    IDX32: tl.constexpr,
):
    """Batch-blocked D-reduction with argmax (small-hw with-indices path).
    Same layout as _dreduce_batch_kernel; the winning D index accumulates in int32.
    """
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
    if IDX32:
        tl.store(idx_ptr + out_off, acc_idx, mask=m)
    else:
        tl.store(idx_ptr + out_off, acc_idx.to(tl.int64), mask=m)


# The launch cap is blockDim <= 65535 on the *product* gridX*gridY*gridZ (the Ascend
# launcher flattens a triton grid into one value), so a (32768, 2) grid is over it.
_MAX_GRID_BLOCKS = 65535

# Largest integer f32 represents exactly: an index in f32 (or int32, whose ``//`` is
# only correct below this) is safe while the whole index space stays under it.
_F32_EXACT_MAX = 2**24

# Programs the global-pool index path aims for -- its grid comes only from (n, c).
_GPOOL_TARGET_PROGRAMS = 32

# Programs ``_idx_combine_hw_kernel`` aims for: the grid matters more than the tile.
_COMBINE_MIN_PROGRAMS = 64

# Elements in the ``_hpool_idx_kernel`` tile: ~22 B/element of UB; 4096 fails to build.
_INT_IDX_TILE_ELEMS = 2048

# Elements per program in the two W-halve kernels.  Both are unmasked, so OUT_SPATIAL
# must divide the flat length (callers give a multiple of 256); the ceiling is UB.
_WPOOL_BLOCK_MAX_F32 = 8192
_WPOOL_BLOCK_MAX_F16 = 16384
_WPOOL_IDX_BLOCK_MAX = 8192


def _wpool_block(total, cap):
    """Largest power of two <= ``cap`` that divides ``total``.  ``total`` is NOT always a
    multiple of 256 (a halve runs on half a flat length, and the reordered path halves W
    after H), and the W-halve kernels are unmasked -- a block that does not divide it walks
    ``pid * 2 * OUT_SPATIAL`` off the end of the buffer.  ``total & -total`` is the largest
    power of two that divides it, so the fallback is always in range.
    """
    block = 256
    while block * 2 <= cap and total % (block * 2) == 0:
        block *= 2
    if total % block:
        # max(): an empty buffer would otherwise give block 0 and divide by zero.
        block = max(total & -total, 1)
    return block


# UB ceiling for the index kernels' tiles: 8192 builds, 16384 fails in the wide ones.
_HPOOL_IDX_TILE_ELEMS = 8192


# Tiling -> the largest BLOCK_B that built; a UB refusal at launch must not be retried.
_HPOOL_IDX_B_FAILED: dict = {}


def _hpool_idx_block_b(batch, tile_elems):
    """How many whole images one _hpool_idx_kernel program folds in; its batch axis is
    unmasked, so this must divide the batch and be a power of two.
    """
    cap = max(1, _HPOOL_IDX_TILE_ELEMS // tile_elems)
    block_b = 1
    while block_b * 2 <= cap and batch % (block_b * 2) == 0:
        block_b *= 2
    return block_b


def _spans(total, block, align=1):
    """Element spans of the kernels whose grid is ``cdiv(total, block)``: one span
    ``(0, total)`` unless the grid would exceed the launch cap, then whole ``align`` rows.
    """
    step = (block * _MAX_GRID_BLOCKS // align) * align
    if step < max(block, align) or total <= step:
        return ((0, total),)
    return tuple((s, min(s + step, total)) for s in range(0, total, step))


def _row_spans(rows, block, units=1):
    """Row spans of the kernels that put one *image* on the grid: whole ``block``-rows,
    because they re-derive the row from the pid (``units`` = programs one row needs).
    """
    per_row = _MAX_GRID_BLOCKS // units
    if per_row < block:
        return ((0, rows),)
    step = per_row * block
    if rows <= step:
        return ((0, rows),)
    return tuple((s, min(s + step, rows)) for s in range(0, rows, step))


# Elements in one (BLOCK_B, BLOCK_HW) batch-D-reduce tile; UB is the binding limit.
_BATCH_TILE_ELEMS = 4096

# Lanes in one _pool2d_batch_kernel (BLOCK_B, BLOCK_W, BLOCK_IN) select tile.  The kernel
# keeps *two* full tiles live -- ``v`` and the fp32 ``tl.where(v != v, 1.0, 0.0)`` that
# feeds the NaN test -- so 16384 lanes (2 x 64 KB fp32) builds and 32768 (2 x 128 KB) is
# refused by UB.  Measured, not derived; the fp32 literal makes the second tile 4 B/lane
# whatever the input dtype, so the budget is in lanes and not in bytes.
_BATCH_TILE_LANES = 16384


def _batch_block(hw, batch):
    """(BLOCK_B, BLOCK_HW) for a batch-blocked D-reduce with a legal grid: UB caps
    ``BLOCK_B * BLOCK_HW``, and the search runs only when the grid would exceed the cap.
    """
    block_hw = min(_BATCH_TILE_ELEMS, triton.next_power_of_2(hw))
    block_b = min(64, _BATCH_TILE_ELEMS // block_hw)
    if triton.cdiv(batch, block_b) * triton.cdiv(hw, block_hw) <= _MAX_GRID_BLOCKS:
        return block_b, block_hw
    best = None
    bh = 1
    while bh <= _BATCH_TILE_ELEMS:
        bb = min(64, _BATCH_TILE_ELEMS // bh)
        blocks = triton.cdiv(batch, bb) * triton.cdiv(hw, bh)
        if best is None or (blocks, -bh) < best[0]:
            best = ((blocks, -bh), bb, bh)
        bh *= 2
    return best[1], best[2]


def _launch_batch_dreduce(
    kernel, input, out, batch, in_d, hw, block_b, block_hw, idx=None, idx32=False
):
    """Launch a batch-blocked D-reduce, splitting the rows if one grid is too big.
    ``b_idx`` is relative to the ``batch`` handed in, so a row chunk is exact.
    """
    tiles_hw = triton.cdiv(hw, block_hw)
    rows_per_launch = max(block_b, (_MAX_GRID_BLOCKS // tiles_hw) * block_b)
    for start in range(0, batch, rows_per_launch):
        rows = min(rows_per_launch, batch - start)
        stop = start + rows
        args = [input[start:stop], out[start * hw : stop * hw]]
        kwargs = {}
        if idx is not None:
            # Only the index variant declares IDX32; hand it out exactly when an index buffer is
            # passed, which keeps the two in step.
            args.append(idx[start * hw : stop * hw])
            kwargs["IDX32"] = idx32
        kernel[(triton.cdiv(rows, block_b), tiles_hw)](
            *args,
            rows,
            in_d,
            hw,
            BLOCK_B=block_b,
            BLOCK_HW=block_hw,
            num_warps=4,
            num_stages=3,
            **kwargs,
        )


def _d_reduce_unified(input, return_indices=False, idx32=False):
    """Unified D-reduction entry point: max over D (dim=1) of a (batch, in_d, hw) tensor;
    replaces every earlier D-reduce call site (B1, B2, out_d=1, Path C).  ``input`` must
    be contiguous (``hw = in_h*in_w``).  ``idx32`` emits int32 indices for an in-file
    consumer (half the store traffic, and the pool kernel can load them).  Returns
    ``values`` of shape (batch, hw), plus ``indices`` when ``return_indices``.
    """
    if input.ndim != 3:
        raise ValueError(
            f"_d_reduce_unified expects 3D (batch, in_d, hw), got {input.ndim}D"
        )

    batch, in_d, hw = input.shape
    total_positions = batch * hw

    flat_out = torch.empty(total_positions, device=input.device, dtype=input.dtype)

    # Tile = smallest power of two covering hw (no wasted lanes, one program per row).
    ts = min(triton.next_power_of_2(hw), 8192)
    # With-indices doubles the tile's UB footprint, so the cap drops to 4096.
    if return_indices:
        ts = min(ts, 4096)
    n_tiles = triton.cdiv(hw, ts)
    total_blocks = batch * n_tiles

    # The tiled grid cannot shrink, so a shape over the cap goes to the batch-blocked
    # kernels, which fold rows into a program.
    tiled = hw > 256 and total_blocks <= _MAX_GRID_BLOCKS

    if return_indices:
        flat_idxs = torch.empty(
            total_positions,
            device=input.device,
            dtype=torch.int32 if idx32 else torch.int64,
        )
        if tiled:
            # Scalar pid decomposition (no per-element division), coalesced loads.
            _unified_d_reduce_kernel[(batch, n_tiles)](
                input,
                flat_out,
                flat_idxs,
                in_d,
                hw,
                TILE_SIZE=ts,
                N_TILES=n_tiles,
                TOTAL_BLOCKS=total_blocks,
                RETURN_INDICES=True,
                IDX32=idx32,
            )
            return flat_out, flat_idxs
        # Folding rows into a program is what makes index tracking affordable.
        block_b, block_hw = _batch_block(hw, batch)
        _launch_batch_dreduce(
            _dreduce_batch_idx_kernel,
            input,
            flat_out,
            batch,
            in_d,
            hw,
            block_b,
            block_hw,
            flat_idxs,
            idx32=idx32,
        )
        return flat_out, flat_idxs

    # Values-only.  Same split as above, for the same reason.
    if not tiled:
        block_b, block_hw = _batch_block(hw, batch)
        _launch_batch_dreduce(
            _dreduce_batch_kernel, input, flat_out, batch, in_d, hw, block_b, block_hw
        )
        return flat_out

    _unified_d_reduce_kernel[(batch, n_tiles)](
        input,
        flat_out,
        flat_out,
        in_d,
        hw,
        TILE_SIZE=ts,
        N_TILES=n_tiles,
        TOTAL_BLOCKS=total_blocks,
        RETURN_INDICES=False,
        IDX32=idx32,  # unread (the store is inside `if RETURN_INDICES`) but required
    )
    return flat_out


# Kernel 7 -- fused uniform 3D pool: all three dims divisible makes the pool a strided
# max-pool, done in one pass; used only where it beats the D-reduce + pool2d
# decomposition.


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
    """Single-pass 3D pool for fully-uniform windows: BLOCK_W w_out positions per block."""
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
            thread_max = tl.max(vals, axis=1)  # (BLOCK_W,) - best per w_out
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


# Tiny shapes -- single-pass 3D pool: the multi-kernel decomposition's launch
# overhead dominates on very small shapes.  Grid: cdiv(total_output, BLOCK_SIZE).


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
    """Single-pass 3D adaptive max pool for tiny shapes: one output element per thread,
    the window a single flat loop (a triple nesting pays nested-loop overhead on Ascend).
    """
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = tid < total_output

    # Decode flat output index to (n, c, d_out, h_out, w_out)
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

    # Flat loop over all window positions; outside the real window the mask is False.
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
    # Exact max adaptive window length = max_i [ceil((i+1)*in/out) - floor(i*in/out)]:
    # ceil(in/out) when in%out==0, but ceil+1 for some sizes (in=5, out=3 -> 2,3,2).
    mw = 0
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = ((i + 1) * in_size + out_size - 1) // out_size
        if end - start > mw:
            mw = end - start
    return mw


# 2D adaptive max pool, run on the (batch, ih, iw) view the 3D pool decomposes to.
# Values-only dispatch by shape: _pool2d_uniform_evenodd (win=2x2, fastest) |
# _pool2d_uniform_pow2 (pow2 ratios) | _pool2d_batch_kernel (big batch, small iw) |
# _pool2d_row_kernel (general case, and the indices / D-index-folding paths).


# _pool2d_row_kernel costs ~49 ns fixed per program plus ~15.7 ns per wide-square
# iteration, so adding programs loses and removing squares wins.  Numbers in notes.md.


def _row_block_oh(
    oh: int,
    batch: int,
    block_w: int,
    block_hw: int,
    block_in: int,
    n_wblocks: int,
    dtype,
    return_indices: bool = False,
) -> tuple[int, int]:
    """(BLOCK_OH, N_OH_BLOCKS) for ``_pool2d_row_kernel``: BLOCK_OH output rows per
    program, or 1, amortizing the per-program fixed cost over rows of identical work.
    All four gates are empirical (sweep in notes.md): BLOCK_IN <= 128, at least 2048
    programs, ``BLOCK_OH | oh``, and UB.
    """
    if block_in > 128 or batch * oh * n_wblocks < 2048:
        return 1, oh
    h_lanes = block_hw * block_in
    if return_indices:
        h_cap = 14336 if dtype == torch.float32 else 8192
    else:
        h_cap = 8192 if dtype == torch.bfloat16 else 14336
    for cand in (7, 4, 2):
        if (
            oh % cand == 0
            and cand * block_w * block_in <= 16384
            and cand * h_lanes <= h_cap
        ):
            return cand, oh // cand
    return 1, oh


def _row_hw_tiling(max_win_h: int, block_in: int) -> tuple[int, int]:
    """(BLOCK_HW, N_HCHUNKS) for ``_pool2d_row_kernel``'s H-window pass: BLOCK_HW
    shrinks as BLOCK_IN grows to stay inside UB; N_HCHUNKS covers ``max_win_h``.
    """
    block_hw = max(
        1, min(16, triton.next_power_of_2(max(max_win_h, 1)), max(1, 4096 // block_in))
    )
    return block_hw, -(-max_win_h // block_hw)


@libentry()
@triton.jit
def _pool2d_row_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    d_ptr,
    d_shift_ptr,
    batch,
    ih,
    iw,
    oh,
    ow,
    BLOCK_HW: tl.constexpr,
    N_HCHUNKS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    N_OH_BLOCKS: tl.constexpr,
    N_WBLOCKS: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    IDX_FP32: tl.constexpr,
    FOLD_D: tl.constexpr,
    D_PACK: tl.constexpr,
):
    """Coalesced 2D adaptive max pool: values, argmax, optionally the D index.

    One program per (batch, BLOCK_OH output rows), optionally W-tiled; full input rows
    load contiguously and the adaptive W-window is a masked 2D reduction.  The H window
    collapses first (a column-max pass per chunk), so the wide square runs once, not once
    per H row -- that square is the expensive part.  BLOCK_OH amortizes the per-program
    fixed cost (gates in _row_block_oh).

    RETURN_INDICES emulates the H-pass argmax (tl.max(return_indices) on a 3-D tile is
    what makes BLOCK_OH>1 a 0.2x loss) and keeps the bookkeeping in f32, which IDX_FP32
    gates on the index fitting f32.  FOLD_D emits the *final 3-D* index, packing the
    winner's ``h_off * D_PACK + d_local`` into one f32 lane so a single ``tl.min`` returns
    both, which removes the merge gather.  The D window start comes from the host
    ``d_shift_ptr`` table, never an inline ``(b % out_d) * step`` (wrong whenever
    BLOCK_OH < oh).
    """
    pid = tl.program_id(0)
    w_block = pid % N_WBLOCKS
    row_pid = pid // N_WBLOCKS
    oh_block = row_pid % N_OH_BLOCKS
    b = row_pid // N_OH_BLOCKS

    oh_pos = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    oh_ok = oh_pos < oh
    h0 = oh_pos * ih // oh
    h1 = ((oh_pos + 1) * ih + oh - 1) // oh
    win_h = h1 - h0

    ow_idx = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w0 = ow_idx * iw // ow
    w1 = ((ow_idx + 1) * iw + ow - 1) // ow

    in_cols = tl.arange(0, BLOCK_IN)
    dtype = in_ptr.type.element_ty
    min_val = _min_value(dtype)
    hw_in = ih * iw
    in_base = in_ptr + b * hw_in
    if FOLD_D:
        d_base = d_ptr + b * hw_in
        d_shift = tl.load(d_shift_ptr + b).to(tl.float32)

    # Per-column max (and winning H offset) over the whole H window, chunked at BLOCK_HW
    # rows to stay in UB.  f32, not int32: the int32 form costs 1.28x-1.62x more here.
    col_max = tl.full((BLOCK_OH, BLOCK_IN), min_val, dtype=dtype)
    if RETURN_INDICES:
        col_hd = tl.zeros((BLOCK_OH, BLOCK_IN), dtype=tl.float32)
    for hc in range(N_HCHUNKS):
        hh = hc * BLOCK_HW + tl.arange(0, BLOCK_HW)
        off3 = (h0[:, None, None] + hh[None, :, None]) * iw + in_cols[None, None, :]
        hm = (
            (hh[None, :, None] < win_h[:, None, None])
            & oh_ok[:, None, None]
            & (in_cols[None, None, :] < iw)
        )
        t = tl.load(in_base + off3, mask=hm, other=min_val)
        # NaN in the window must win: tl.max drops NaN, so detect it and force the column.
        nan_c = tl.max(tl.where(t != t, 1.0, 0.0), axis=1)
        cm = tl.max(t, axis=1)
        cm = tl.where(nan_c > 0, float("nan"), cm)
        better = (cm > col_max) | (cm != cm)
        col_max = tl.where(better, cm, col_max).to(dtype)
        if RETURN_INDICES:
            # Emulated argmax over the H rows: the FIRST row matching the column max, as
            # tl.max(return_indices) gives.  ``hh`` is already window-relative -- do NOT add
            # ``hc * BLOCK_HW`` on top.
            if FOLD_D:
                # The d plane through the same addresses: coalesced, and cached from the load above.
                dt = tl.load(d_base + off3, mask=hm, other=0).to(tl.float32)
                packed = tl.min(
                    tl.where(
                        t == cm[:, None, :],
                        hh[None, :, None].to(tl.float32) * D_PACK + dt,
                        float(N_HCHUNKS * BLOCK_HW * D_PACK),
                    ),
                    axis=1,
                )
            else:
                packed = tl.min(
                    tl.where(
                        t == cm[:, None, :],
                        hh[None, :, None],
                        N_HCHUNKS * BLOCK_HW,  # > every hh, i.e. "no row matched"
                    ),
                    axis=1,
                ).to(tl.float32)
            col_hd = tl.where(better, packed, col_hd)
    if RETURN_INDICES:
        # Only an all-NaN column reaches "no row matched"; report row 0 (and d 0), as before.
        if FOLD_D:
            col_hd = tl.where(
                col_hd < float(N_HCHUNKS * BLOCK_HW * D_PACK), col_hd, 0.0
            )
        else:
            col_hd = tl.where(col_hd < float(N_HCHUNKS * BLOCK_HW), col_hd, 0.0)

    # W window: the one wide square, over the per-column maxima.
    m = (in_cols[None, None, :] >= w0[None, :, None]) & (
        in_cols[None, None, :] < w1[None, :, None]
    )
    v = tl.where(m, col_max[:, None, :], min_val)
    has_nan = tl.max(tl.where(v != v, 1.0, 0.0), axis=2)
    if RETURN_INDICES:
        # row_argmax is the column index == w_in (in_cols runs 0..iw-1).
        row_max, row_argmax = tl.max(v, axis=2, return_indices=True)
        row_max = tl.where(has_nan > 0, float("nan"), row_max).to(dtype)
        # The winning column's H offset (and d, with FOLD_D): Triton has no register gather.
        sel = tl.sum(
            tl.where(
                in_cols[None, None, :] == row_argmax[:, :, None],
                col_hd[:, None, :],
                0.0,
            ),
            axis=2,
        )
        if FOLD_D:
            col_hit = tl.floor(sel / D_PACK)
            d_hit = sel - col_hit * D_PACK
        else:
            col_hit = sel
        # f32 index arithmetic throughout: exact while the whole 3-D index fits f32's integer
        # range (IDX_FP32 asserts it), so the int64 cast at the store is exact.
        if IDX_FP32:
            acc_idx = (h0[:, None] + col_hit) * iw + row_argmax
        else:
            acc_idx = (h0[:, None] + col_hit.to(tl.int32)) * iw + row_argmax
        if FOLD_D:
            acc_idx = (d_hit + d_shift) * hw_in + acc_idx
    else:
        # Values-only: skip the argmax computation entirely.
        row_max = tl.where(has_nan > 0, float("nan"), tl.max(v, axis=2)).to(dtype)

    out_off = b * oh * ow + oh_pos[:, None] * ow + ow_idx[None, :]
    store_mask = oh_ok[:, None] & (ow_idx[None, :] < ow)
    tl.store(out_ptr + out_off, row_max, mask=store_mask)
    if RETURN_INDICES:
        tl.store(idx_ptr + out_off, acc_idx.to(tl.int64), mask=store_mask)


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
    """Values-only 2D adaptive max pool, batch-blocked: BLOCK_B batches for one oh row,
    amortizing launch overhead when batch is large and iw is small.
    """
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
        # tl.max drops NaN; detect it and force row_max to NaN.
        has_nan = tl.max(tl.where(v != v, 1.0, 0.0), axis=2)
        row_max = tl.max(v, axis=2).to(dtype)
        row_max = tl.where(has_nan > 0, float("nan"), row_max)
        acc = _max_nan(acc, row_max).to(dtype)

    out_base = out_ptr + b_idx[:, None] * oh * ow + oh_pos * ow + ow_idx[None, :]
    tl.store(out_base, acc, mask=b_valid[:, None] & (ow_idx[None, :] < ow))


@libentry()
@triton.jit
def _wpool_kernel(in_ptr, out_ptr, OUT_SPATIAL: tl.constexpr):
    """Halve the innermost (contiguous) dim: out[i] = max(in[2i], in[2i+1]) (even, row-aligned)."""
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
    """Halve a strided dimension via even/odd max:
    out[o] = max(in[2*(o//S)*S + o%S], in[2*(o//S)*S + S + o%S]) at stride S (a power of two).
    """
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
    """Halve the H dim (stride W) via 2D even/odd max over a padded tile, so non-pow2 H/W work."""
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
    """Batched H-halve: BLOCK_B batches per program, amortizing the strided launch cost."""
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
    """Even/odd W-halve with a per-pair argmax bit (0=even, 1=odd), to rebuild the 2D index."""
    pid = tl.program_id(0)
    offs = tl.arange(0, OUT_SPATIAL)
    even = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs)
    odd = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs + 1)
    v = _max_nan(even, odd)
    # The bit must mirror _max_nan (odd wins on a tie OR a NaN) so the index points at
    # the NaN holder.
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
    drow_ptr,
    H,
    W,
    OUT_H,
    OUT_W,
    N_OH_BLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    IW: tl.constexpr,
    ADD_D: tl.constexpr,
):
    """H-halve with full 2-D index reconstruction (values + indices).

    Each output (oh, ow) is the max over rows {2oh, 2oh+1} of the W-pooled values; the
    winning row (bh) and its W-pair bit give ``(2*oh + bh) * IW + 2*ow + pair``.  The H
    output is tiled (N_OH_BLOCKS x BLOCK_H rows) so the tile and its int64 index tile fit
    UB.  BLOCK_B folds whole images into one program -- a large win on a wide trailing
    axis, a loss at BLOCK_W=4, which routes to ``_hpool_idx_flat_kernel`` (notes.md).
    The batch axis is deliberately *not* masked: a runtime 3-D mask costs more than the
    batching saves.
    """
    pid = tl.program_id(0)
    oh_block = pid % N_OH_BLOCKS
    # The batch axis stays 1-D: ADD_D loads the D term with it and (BLOCK_B, 1, 1) is slow.
    b1 = (pid // N_OH_BLOCKS) * BLOCK_B + tl.arange(0, BLOCK_B)
    b = b1[:, None, None]
    oh = oh_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :, None]
    ow = tl.arange(0, BLOCK_W)[None, None, :]
    m = (oh < OUT_H) & (ow < OUT_W)
    base = b * H * W
    even_v = tl.load(val_ptr + base + 2 * oh * W + ow, mask=m)
    odd_v = tl.load(val_ptr + base + (2 * oh + 1) * W + ow, mask=m)
    even_b = tl.load(bit_ptr + base + 2 * oh * W + ow, mask=m, other=0)
    odd_b = tl.load(bit_ptr + base + (2 * oh + 1) * W + ow, mask=m, other=0)
    v = _max_nan(even_v, odd_v)
    # Row-winner bit mirrors _max_nan (odd row wins when odd > even OR odd NaN).
    bh = tl.where((odd_v > even_v) | (odd_v != odd_v), 1, 0)
    # int32 arithmetic, int64 only at the store: int64 element-wise ops are ~100x slower
    # here, and they dominated this kernel (640 -> 190 us on (2,128,4,28,28)).
    pair = tl.where(bh == 1, odd_b, even_b).to(tl.int32)
    row = 2 * oh + bh
    flat_idx = row * IW + 2 * ow + pair
    if ADD_D:
        # One scalar per image, not per lane: a 1-D (BLOCK_B,) read broadcast over the tile.
        # Added *before* the int64 cast -- that add costs 180 us against the int32 one's ~0.
        flat_idx += tl.load(drow_ptr + b1)[:, None, None]
    flat_idx = flat_idx.to(tl.int64)
    out_off = b * OUT_H * OUT_W + oh * OUT_W + ow
    tl.store(out_val_ptr + out_off, v, mask=m)
    tl.store(idx_ptr + out_off, flat_idx, mask=m)


@libentry()
@triton.jit
def _hpool_idx_flat_kernel(
    val_ptr,
    bit_ptr,
    out_val_ptr,
    idx_ptr,
    drow_ptr,
    NROWS,
    OW,
    OH,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
    IW: tl.constexpr,
    ADD_D: tl.constexpr,
):
    """H-halve with index reconstruction, tiled by W-halved *row*.

    Same pass as ``_hpool_idx_kernel``, indexed differently: the W-halved plane is read
    as consecutive rows of length ``2 * OW``, so output row ``r = b * OH + oh`` takes
    ``2 * OW * r + ow`` and ``+ OW`` -- affine in (r, ow), no per-row decode, many images
    per program.  It exists for the narrow tiles where the image-tiled form is one
    per-program-cost program per image (above all oh == ow == 4).  Only the index carries
    a per-row term, ``oh = r % OH`` -- one modulo per tile row, not per lane.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    cols = tl.arange(0, BLOCK_W)[None, :]
    m = (r < NROWS) & (cols < OW)
    even_v = tl.load(val_ptr + r * (2 * OW) + cols, mask=m)
    odd_v = tl.load(val_ptr + r * (2 * OW) + OW + cols, mask=m)
    even_b = tl.load(bit_ptr + r * (2 * OW) + cols, mask=m, other=0)
    odd_b = tl.load(bit_ptr + r * (2 * OW) + OW + cols, mask=m, other=0)
    v = _max_nan(even_v, odd_v)
    bh = tl.where((odd_v > even_v) | (odd_v != odd_v), 1, 0)
    pair = tl.where(bh == 1, odd_b, even_b).to(tl.int32)
    oh = r % OH
    off = r * OW + cols
    tl.store(out_val_ptr + off, v, mask=m)
    flat_idx = (2 * oh + bh) * IW + 2 * cols + pair
    if ADD_D:
        # One D term per *row*, not per lane; both slice ends stay relative to the span.
        flat_idx += tl.load(drow_ptr + r, mask=r < NROWS, other=0)
    tl.store(idx_ptr + off, flat_idx.to(tl.int64), mask=m)


# (rows, oh, in_d, hw, device) -> per-row D term.  Host-built and cached like the other
# tables: a CPU arange costs no kernel, and it must not be rebuilt per call.
_D_OFFSET_CACHE: dict = {}


def _d_offset_rows(rows, oh, in_d, hw, device):
    """``((row // oh) % in_d) * hw`` -- the D half of the 3-D flat index, with the plane
    factor folded in here; for the image-tiled kernel the rows are whole images (``oh = 1``).
    """
    key = (rows, oh, in_d, hw, str(device))
    table = _D_OFFSET_CACHE.get(key)
    if table is None:
        ids = torch.arange(rows, dtype=torch.int64)
        table = (((ids // oh) % in_d) * hw).to(device, torch.int32)
        _D_OFFSET_CACHE[key] = table
    return table


def _pool2d_uniform_evenodd_idx(input, oh, ow, d_span=None):
    """win=2x2 even/odd pool with indices: W-pool (per-pair bit) then H-pool (value +
    index).  Tie-breaking (first max) may differ from torch's argmax (last max), which
    the FlagGems tests tolerate.
    """
    batch, ih, iw = input.shape
    # ``d_span`` is set only by Path A (in_d == out_d), which runs this on flattened
    # (n, c, in_d) planes and wants the 3-D index; folding the plane's D term in saved a
    # whole output pass (27.7 of 63.6 us).  Both H-pass kernels slice their table per span.
    add_d = d_span is not None
    _empty = torch.empty(0, dtype=torch.int32, device=input.device)
    drow_rows = (
        _d_offset_rows(batch * oh, oh, d_span[0], d_span[1], input.device)
        if add_d
        else _empty
    )
    drow_img = (
        _d_offset_rows(batch, 1, d_span[0], d_span[1], input.device)
        if add_d
        else _empty
    )
    flat = input.reshape(-1)
    wv = torch.empty(flat.numel() // 2, device=input.device, dtype=input.dtype)
    wb = torch.empty(flat.numel() // 2, device=input.device, dtype=torch.int8)
    block_w = _wpool_block(wv.numel(), _WPOOL_IDX_BLOCK_MAX)
    for start, stop in _spans(wv.numel(), block_w):
        _wpool_idx_kernel[(triton.cdiv(stop - start, block_w),)](
            flat[2 * start : 2 * stop],
            wv[start:stop],
            wb[start:stop],
            OUT_SPATIAL=block_w,
        )
    out = torch.empty(batch * oh * ow, device=input.device, dtype=input.dtype)
    indices = torch.empty(batch * oh * ow, device=input.device, dtype=torch.int64)
    # Tile capped so the int64 index tile fits UB (64x64 overflows with multi-buffering).
    block_w2 = triton.next_power_of_2(ow)
    block_h = min(16, triton.next_power_of_2(oh))
    n_oh_blocks = triton.cdiv(oh, block_h)
    in_per_img = ih * ow
    out_per_img = oh * ow
    if block_w2 <= 4:
        # Narrow trailing axis: the image-tiled kernel degenerates (its (4,4) tiles lose at
        # every BLOCK_B), so index the plane by row instead -- rows span images here.
        block_r = max(16, min(256, _HPOOL_IDX_TILE_ELEMS // block_w2))
        nrows = batch * oh
        for start, stop in _row_spans(nrows, block_r):
            _hpool_idx_flat_kernel[(triton.cdiv(stop - start, block_r),)](
                wv[start * 2 * ow : stop * 2 * ow],
                wb[start * 2 * ow : stop * 2 * ow],
                out[start * ow : stop * ow],
                indices[start * ow : stop * ow],
                drow_rows[start:stop],
                stop - start,
                ow,
                oh,
                BLOCK_R=block_r,
                BLOCK_W=block_w2,
                IW=iw,
                ADD_D=add_d,
            )
        return out.view(batch, oh, ow), indices.view(batch, oh, ow)
    # BLOCK_B is the largest power of two dividing the batch (the kernel has no batch mask).
    tiling = (batch, block_h, block_w2)
    block_b = min(
        _hpool_idx_block_b(batch, block_h * block_w2),
        _HPOOL_IDX_B_FAILED.get(tiling, 1 << 30),
    )
    while True:
        try:
            # The kernel re-derives the image from the pid, so whole images per launch are exact.
            for start, stop in _row_spans(batch, block_b, units=n_oh_blocks):
                _hpool_idx_kernel[((stop - start) // block_b * n_oh_blocks,)](
                    wv[start * in_per_img : stop * in_per_img],
                    wb[start * in_per_img : stop * in_per_img],
                    out[start * out_per_img : stop * out_per_img],
                    indices[start * out_per_img : stop * out_per_img],
                    drow_img[start:stop],
                    ih,
                    ow,
                    oh,
                    ow,
                    N_OH_BLOCKS=n_oh_blocks,
                    BLOCK_B=block_b,
                    BLOCK_H=block_h,
                    BLOCK_W=block_w2,
                    IW=iw,
                    ADD_D=add_d,
                )
            break
        except Exception:  # noqa: BLE001 -- UB/build refusal: step down, once
            # No arithmetic cap covers this (8192 elements is fine at (16,16), overflows at (8,8)
            # with BLOCK_B=128); halving on refusal does, and every halving divides the batch.
            if block_b == 1:
                raise
            block_b //= 2
            _HPOOL_IDX_B_FAILED[tiling] = block_b
    return out.view(batch, oh, ow), indices.view(batch, oh, ow)


@libentry()
@triton.jit
def _evenodd2x2_kernel(
    in_ptr,
    out_ptr,
    Q,
    BLOCK_Q: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    IW: tl.constexpr,
    OW: tl.constexpr,
):
    """Fused win=2x2 even/odd pool, one output row per grid slot: ``out[q, ow]`` is the
    max of the 2x2 block at rows {2*oh, 2*oh+1}, columns {2*ow, 2*ow+1}, ``q`` across all
    images.  Because ``IH == 2*OH`` the row pair starts at ``2*IW*q``, so all four loads
    are affine -- unlike the two-pass form it replaced, whose computed H-pool address
    behind a runtime mask ran at 17 GB/s against 230 GB/s for the W half.

    The odd-row base is its own expression on purpose: ``r0 + IW`` fails to lower
    (ConvertTritonIRToLinalgIR), while ``(2*q + 1) * IW + col`` compiles.
    """
    pid = tl.program_id(0)
    q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
    ow = tl.arange(0, BLOCK_OW)[None, :]
    m = (q < Q) & (ow < OW)
    col = 2 * ow
    r0 = 2 * IW * q + col
    r1 = (2 * q + 1) * IW + col
    even = tl.load(in_ptr + r0, mask=m)
    even1 = tl.load(in_ptr + r0 + 1, mask=m)
    odd = tl.load(in_ptr + r1, mask=m)
    odd1 = tl.load(in_ptr + r1 + 1, mask=m)
    v = _max_nan(_max_nan(even, even1), _max_nan(odd, odd1)).to(in_ptr.type.element_ty)
    tl.store(out_ptr + q * OW + ow, v, mask=m)


# Lanes in one _evenodd2x2_kernel tile: five live multi-buffered tiles at ~4 B.
_EVENODD_TILE_ELEMS = 4096


def _pool2d_uniform_evenodd(input, oh, ow):
    """Uniform 2D max pool (win_h == win_w == 2) in one fused pass, fp32 and power-of-two
    ``ow`` only; fp16/bf16 go to _pool2d_uniform_pow2 (this select chain is ~6x slower).
    """
    batch, ih, iw = input.shape
    flat = input.reshape(-1)
    out = torch.empty(batch * oh * ow, device=input.device, dtype=input.dtype)
    block_ow = triton.next_power_of_2(ow)
    block_q = max(16, min(128, _EVENODD_TILE_ELEMS // block_ow))
    q = batch * oh
    # One output row owns 2*IW input elements (its two input rows).
    in_per_q = 2 * iw
    for start, stop in _row_spans(q, block_q):
        _evenodd2x2_kernel[(triton.cdiv(stop - start, block_q),)](
            flat[start * in_per_q : stop * in_per_q],
            out[start * ow : stop * ow],
            stop - start,
            BLOCK_Q=block_q,
            BLOCK_OW=block_ow,
            IW=iw,
            OW=ow,
            num_warps=4,
            num_stages=3,
        )
    return out.view(batch, oh, ow)


# Lanes one batched H-pass tile aims to cover; the row width is iw (the reorder), not ow.
_HPOOL_TILE_LANES = 4096

# Run length per row below which the wide H-pass's extra IW of read traffic pays.
_HPOOL_MIN_RUN_BYTES = 32


@libentry()
@triton.jit
def _hpool_wide_kernel(
    in_ptr,
    out_ptr,
    NPLANE,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Halve H over a (NPLANE, IH, IW) plane with full-width contiguous reads.

    Reads the two source rows as one contiguous 2*IW run and maxes it against itself
    shifted by IW; only the store narrows back.  Against the batched (BLOCK_B, IW) tile it
    trades an extra IW of read traffic per row for a run twice as long, and wins exactly
    while IW is short (see _HPOOL_MIN_RUN_BYTES).  ``IW`` must be a power of two, and the
    shifted load reads past its run -- inside the buffer for every plane but the last, so
    the caller must not hand it the final plane of a tensor.
    """
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    bv = b < NPLANE
    cols = tl.arange(0, 2 * IW)
    for hh in tl.static_range(IH // 2):
        base = b[:, None] * (IH * IW) + (2 * hh) * IW + cols[None, :]
        a = tl.load(in_ptr + base, mask=bv[:, None])
        c = tl.load(in_ptr + base + IW, mask=bv[:, None])
        v = _max_nan(a, c).to(in_ptr.type.element_ty)
        tl.store(
            out_ptr + b[:, None] * (IH // 2) * IW + hh * IW + cols[None, :],
            v,
            mask=bv[:, None] & (cols < IW)[None, :],
        )


def _pow2_reorders(ih, iw):
    """Whether _pool2d_uniform_pow2 halving H first pays: the H pass rows are iw wide when
    H goes first and ow when W does, so the reorder wins only while that width is short.
    """
    return ih // 2 <= 32 and iw <= 32


def _pool2d_uniform_pow2(input, oh, ow):
    """Uniform 2D max pool for power-of-two window ratios via repeated even/odd halving:
    W contiguous, H at stride = the current row width, so the row kernel's cross-lane
    tl.max(axis=..) is avoided.  The halving *order* sets the H pass's row width (ow vs
    iw), which is the whole cost on iw-short shapes -- those run H first (_pow2_reorders).
    """
    batch, ih, iw = input.shape

    def halve_w(cur):
        """Halve W over the whole buffer; contiguous even/odd, so dense."""
        half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
        block_w = _wpool_block(
            half.numel(),
            (
                _WPOOL_BLOCK_MAX_F32
                if input.dtype == torch.float32
                else _WPOOL_BLOCK_MAX_F16
            ),
        )
        for start, stop in _spans(half.numel(), block_w):
            _wpool_kernel[(triton.cdiv(stop - start, block_w),)](
                cur[2 * start : 2 * stop],
                half[start:stop],
                OUT_SPATIAL=block_w,
            )
        return half

    # Halve H first, so its rows are the unhalved iw wide -- see _pow2_reorders.
    reorder = _pow2_reorders(ih, iw)
    cur = input.reshape(-1)
    if reorder:
        esz = input.element_size()
        while ih > oh:
            half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
            block_w = triton.next_power_of_2(iw)
            block_b = max(1, min(64, _HPOOL_TILE_LANES // block_w))
            wide = (iw & (iw - 1)) == 0 and iw * esz < _HPOOL_MIN_RUN_BYTES
            in_stride, out_stride = ih * iw, (ih // 2) * iw
            spans = list(_spans(batch, block_b))
            for pos, (start, stop) in enumerate(spans):
                # Only the span ending the buffer can be overrun by the wide kernel's shifted load.
                tail = wide and pos == len(spans) - 1 and stop == batch
                n = stop - start - 1 if tail else stop - start
                if n > 0:
                    args = (
                        cur[start * in_stride : (start + n) * in_stride],
                        half[start * out_stride : (start + n) * out_stride],
                    )
                    if wide:
                        _hpool_wide_kernel[(triton.cdiv(n, block_b),)](
                            *args, n, IH=ih, IW=iw, BLOCK_B=block_b
                        )
                    else:
                        _hpool_batched_kernel[(triton.cdiv(n, block_b),)](
                            *args,
                            n,
                            IH=ih,
                            IW=iw,
                            BLOCK_B=block_b,
                            BLOCK_W=block_w,
                        )
                if tail:
                    _hpool_batched_kernel[(1,)](
                        cur[(stop - 1) * in_stride : stop * in_stride],
                        half[(stop - 1) * out_stride : stop * out_stride],
                        1,
                        IH=ih,
                        IW=iw,
                        BLOCK_B=1,
                        BLOCK_W=block_w,
                    )
            cur = half
            ih //= 2
        while iw > ow:
            cur = halve_w(cur)
            iw //= 2
        return cur.view(batch, oh, ow)
    # Halve W: iw -> ow (contiguous, cheap).
    while iw > ow:
        cur = halve_w(cur)
        iw //= 2
    # Halve H: ih -> oh (stride = current iw).  Small row counts go to the batched kernel,
    # pow2 iw to the shift-based one, else the padded 2D _hpool2d_kernel.
    while ih > oh:
        half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
        if ih // 2 <= 32:
            block_b = 64
            block_w = triton.next_power_of_2(iw)
            for start, stop in _spans(batch, block_b):
                _hpool_batched_kernel[(triton.cdiv(stop - start, block_b),)](
                    cur[start * ih * iw : stop * ih * iw],
                    half[start * (ih // 2) * iw : stop * (ih // 2) * iw],
                    stop - start,
                    IH=ih,
                    IW=iw,
                    BLOCK_B=block_b,
                    BLOCK_W=block_w,
                )
        elif (iw & (iw - 1)) == 0:
            log_s = (iw - 1).bit_length()
            for start, stop in _spans(half.numel(), 1024, align=iw):
                _hpool_kernel[(triton.cdiv(stop - start, 1024),)](
                    cur[2 * start : 2 * stop],
                    half[start:stop],
                    stop - start,
                    LOG_S=log_s,
                    S=iw,
                    BLOCK=1024,
                )
        else:
            block_h = triton.next_power_of_2(ih // 2)
            block_w = triton.next_power_of_2(iw)
            in_per_img = ih * iw
            out_per_img = (ih // 2) * iw
            for start, stop in _spans(batch, 1):
                _hpool2d_kernel[(stop - start,)](
                    cur[start * in_per_img : stop * in_per_img],
                    half[start * out_per_img : stop * out_per_img],
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


@libentry()
@triton.jit
def _wpool_flatidx_kernel(
    in_ptr,
    fidx_ptr,
    out_ptr,
    out_fidx_ptr,
    OUT_SPATIAL: tl.constexpr,
):
    """Even/odd W-halve that also carries the winning element's flat index, propagated
    through each halve so the index array ends up pointing at the pooled input element.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, OUT_SPATIAL)
    even_v = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs)
    odd_v = tl.load(in_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs + 1)
    even_i = tl.load(fidx_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs)
    odd_i = tl.load(fidx_ptr + pid * (2 * OUT_SPATIAL) + 2 * offs + 1)
    v = _max_nan(even_v, odd_v)
    better = (odd_v > even_v) | (odd_v != odd_v)
    tl.store(out_ptr + pid * OUT_SPATIAL + offs, v)
    tl.store(out_fidx_ptr + pid * OUT_SPATIAL + offs, tl.where(better, odd_i, even_i))


@libentry()
@triton.jit
def _hpool_flatidx_kernel(
    in_ptr,
    fidx_ptr,
    out_ptr,
    out_fidx_ptr,
    TOTAL_OUT,
    LOG_S: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Strided H-halve that also carries the winning row's flat index."""
    pid = tl.program_id(0)
    out_idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = out_idx < TOTAL_OUT
    m = out_idx >> LOG_S
    s = out_idx & (S - 1)
    even_off = (m << 1) * S + s
    even_v = tl.load(in_ptr + even_off, mask=mask)
    odd_v = tl.load(in_ptr + even_off + S, mask=mask)
    even_i = tl.load(fidx_ptr + even_off, mask=mask)
    odd_i = tl.load(fidx_ptr + even_off + S, mask=mask)
    v = _max_nan(even_v, odd_v)
    better = (odd_v > even_v) | (odd_v != odd_v)
    tl.store(out_ptr + out_idx, v, mask=mask)
    tl.store(out_fidx_ptr + out_idx, tl.where(better, odd_i, even_i), mask=mask)


def _pool2d_uniform_pow2_idx(input, oh, ow):
    """Uniform power-of-two window pool WITH indices: the same halving, propagating the
    winner's flat index, so the result matches torch.max's even/earlier-wins tie-breaking.
    """
    batch, ih, iw = input.shape
    flat = input.reshape(-1)
    # Per-batch flat index, like torch's adaptive_max_pool2d indices.
    fidx = torch.arange(ih * iw, device=input.device, dtype=torch.int32).repeat(batch)
    cur = flat
    # Halve W: iw -> ow.
    while iw > ow:
        half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
        half_fidx = torch.empty(
            cur.numel() // 2, device=input.device, dtype=torch.int32
        )
        block_w = _wpool_block(half.numel(), 256)
        for start, stop in _spans(half.numel(), block_w):
            _wpool_flatidx_kernel[(triton.cdiv(stop - start, block_w),)](
                cur[2 * start : 2 * stop],
                fidx[2 * start : 2 * stop],
                half[start:stop],
                half_fidx[start:stop],
                OUT_SPATIAL=block_w,
            )
        cur = half
        fidx = half_fidx
        iw //= 2
    # Halve H: ih -> oh (stride = current iw; pow2 for this path).
    while ih > oh:
        half = torch.empty(cur.numel() // 2, device=input.device, dtype=input.dtype)
        half_fidx = torch.empty(
            cur.numel() // 2, device=input.device, dtype=torch.int32
        )
        log_s = (iw - 1).bit_length()
        for start, stop in _spans(half.numel(), 1024, align=iw):
            _hpool_flatidx_kernel[(triton.cdiv(stop - start, 1024),)](
                cur[2 * start : 2 * stop],
                fidx[2 * start : 2 * stop],
                half[start:stop],
                half_fidx[start:stop],
                stop - start,
                LOG_S=log_s,
                S=iw,
                BLOCK=1024,
            )
        cur = half
        fidx = half_fidx
        ih //= 2
    return cur.view(batch, oh, ow), fidx.to(torch.int64).view(batch, oh, ow)


def _evenodd_idx_wins(batch: int, ih: int, iw: int, oh: int, ow: int) -> bool:
    """Would ``_pool2d_triton``'s indices path end at the even/odd kernel?

    Asked before stealing the shape for the fused kernel.  The clauses mirror *both* of
    that dispatch's entry points -- keep them in step.
    """
    if ih != 2 * oh or iw != 2 * ow or (batch * ih * ow) % 256 != 0:
        return False
    if (
        min(16, triton.next_power_of_2(oh)) * triton.next_power_of_2(ow)
        <= _INT_IDX_TILE_ELEMS
    ):
        return True
    return (ow & (ow - 1)) == 0 and (batch * ih * iw) % 256 == 0


# Tilings whose fused D-index kernel failed to build: a launch-time UB error, not retried.
_ROW_D_FAILED: set = set()

# (rows, out_d, in_d, device) -> per-plane D window start; host-built and cached.
_D_SHIFT_CACHE: dict = {}


def _d_window_shifts(rows: int, out_d: int, in_d: int, device):
    """Per-plane D window start, as int32: ``d_out * in_d // out_d`` (the uniform-window form)."""
    key = (rows, out_d, in_d, str(device))
    table = _D_SHIFT_CACHE.get(key)
    if table is None:
        ids = torch.arange(rows, dtype=torch.int32)
        table = ((ids % out_d) * in_d // out_d).to(device, torch.int32)
        _D_SHIFT_CACHE[key] = table
    return table


def _pool2d_triton_d_indices(input, d_local, out_d, in_d, oh, ow):
    """Pool + fold the D argmax into the final 3-D index, or None to merge after.

    ``input`` is the D-reduced plane (rows, ih, iw), ``d_local`` the D-window argmax of the
    same positions.  The packed lane's ceiling is ``_compute_max_win(in_d, out_d)`` -- the
    *ceiling* the caller could pass (``cdiv(in_d, out_d)``) is not the max window length, and
    a d_local past it collides with the next h in ``hh * D_PACK + d``.
    Returns ``(values, indices)`` in absolute 3-D index space, or None to let the caller
    run the pool and ``_merge_d2h_indices_kernel``.  Faster than a merge removal alone
    (10.1 ms vs 2.31 ms on (2,64,64,256,256)) but a 0.38x *loss* on small planes like
    (32,512,8,8,8), which is why the even/odd shapes bail out above.
    """
    rows, ih, iw = input.shape
    # f32 index arithmetic must be exact end to end: past _F32_EXACT_MAX the packed lane rounds.
    if in_d * ih * iw > _F32_EXACT_MAX:
        return None
    if _evenodd_idx_wins(rows, ih, iw, oh, ow):
        return None
    mwh = _compute_max_win(ih, oh)
    block_w = triton.next_power_of_2(ow)
    block_in = triton.next_power_of_2(iw)
    if block_w * block_in > 4096:  # UB cap; same trade as the plain row kernel
        block_w = max(4, 4096 // block_in)
    n_wblocks = triton.cdiv(ow, block_w)
    block_hw, n_hchunks = _row_hw_tiling(mwh, block_in)
    block_oh, n_oh_blocks = _row_block_oh(
        oh,
        rows,
        block_w,
        block_hw,
        block_in,
        n_wblocks,
        input.dtype,
        return_indices=True,
    )
    dpack = triton.next_power_of_2(max(_compute_max_win(in_d, out_d), 1))
    if (n_hchunks * block_hw) * dpack >= _F32_EXACT_MAX:
        return None
    key = (
        ih,
        iw,
        oh,
        ow,
        input.dtype,
        block_w,
        block_in,
        block_hw,
        n_hchunks,
        block_oh,
        n_wblocks,
    )
    if key in _ROW_D_FAILED:
        return None

    # The D window's start, one int32 per plane from a cached host table: an inline
    # ``(b % out_d) * step`` was wrong on every BLOCK_OH < oh tiling (see that kernel).
    d_shift = _d_window_shifts(rows, out_d, in_d, input.device)

    output = torch.empty((rows, oh, ow), device=input.device, dtype=input.dtype)
    indices = torch.empty((rows, oh, ow), device=input.device, dtype=torch.int64)
    units = n_oh_blocks * n_wblocks
    try:
        # One [start:stop] slice applied to all four per-plane tensors: an
        # absolute index with a sliced pointer would double-add the base.
        for start, stop in _row_spans(rows, 1, units=units):
            _pool2d_row_kernel[((stop - start) * units,)](
                input[start:stop],
                output[start:stop],
                indices[start:stop],
                d_local[start:stop],
                d_shift[start:stop],
                stop - start,
                ih,
                iw,
                oh,
                ow,
                BLOCK_HW=block_hw,
                N_HCHUNKS=n_hchunks,
                BLOCK_W=block_w,
                BLOCK_IN=block_in,
                BLOCK_OH=block_oh,
                N_OH_BLOCKS=n_oh_blocks,
                N_WBLOCKS=n_wblocks,
                RETURN_INDICES=True,
                IDX_FP32=True,  # gated on in_d*ih*iw <= _F32_EXACT_MAX above
                FOLD_D=True,
                D_PACK=dpack,
                num_warps=4,
                num_stages=3,
            )
    except Exception:  # noqa: BLE001 -- UB/grid refusal: fall back, once
        _ROW_D_FAILED.add(key)
        return None
    return output, indices


def _pool2d_triton(input, oh, ow, return_indices=False, d_span=None):
    """2D adaptive max pool via the FlagGems Triton kernel.

    ``d_span`` = ``(in_d, in_h * in_w)`` asks for the *3-D* index: ``input`` is then the
    flattened (n, c, in_d) plane stack of an in_d == out_d pool, and the D term is folded
    in as the index is written (Path A only, even/odd index kernels only).
    """
    batch, ih, iw = input.shape
    output = torch.empty((batch, oh, ow), device=input.device, dtype=input.dtype)

    if not return_indices:
        # Exact win=2x2, fp32, pow2 ow: the single-pass path.  fp16/bf16 must not see it (its
        # select lowering is ~4x slower) and neither must a non-pow2 ow (the column mask stops
        # folding away).  Where _pow2_reorders holds the two-pass wins in fp32 too, so evenodd
        # is held back to exactly the shapes the reorder does not cover.  Both sweeps in notes.md.
        if ih == 2 * oh and iw == 2 * ow and (batch * ih * ow) % 256 == 0:
            if (
                input.dtype == torch.float32
                and (ow & (ow - 1)) == 0
                and not _pow2_reorders(ih, iw)
            ):
                return _pool2d_uniform_evenodd(input, oh, ow)
            return _pool2d_uniform_pow2(input, oh, ow)
        # Uniform pow2 windows: even/odd reductions avoid the cross-lane tl.max that dominates
        # for large windows / wide rows, but the row kernel wins medium windows with narrow rows.
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
        # Cap the (BLOCK_W, BLOCK_IN) tile for UB; past the cap W is tiled along grid axis 0.
        #
        # Do NOT shrink BLOCK_W to track the window: W-tiling the row kernel is a 0.23x-0.66x
        # *loss* on every shape that reaches it, since a program costs ~49 ns fixed against
        # 15.7 ns per wide-square iteration.
        if block_w * block_in > 4096:
            block_w = max(4, 4096 // block_in)
        n_wblocks = triton.cdiv(ow, block_w)
        block_hw, n_hchunks = _row_hw_tiling(mwh, block_in)
        # Many tiny independent pools (large batch, small iw): one program per
        # output row would launch a huge grid of nearly-empty blocks.
        if batch >= 4096 and iw <= 32:
            # The kernel's select builds a full BLOCK_B x BLOCK_W x BLOCK_IN tile, so BLOCK_B
            # follows the widths instead of being fixed at 64 (see _BATCH_TILE_LANES).
            block_b = max(1, min(64, _BATCH_TILE_LANES // (block_w * block_in)))
            for start, stop in _row_spans(batch, block_b, units=oh):
                _pool2d_batch_kernel[(triton.cdiv(stop - start, block_b) * oh,)](
                    input[start:stop],
                    output[start:stop],
                    stop - start,
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
        block_oh, n_oh_blocks = _row_block_oh(
            oh, batch, block_w, block_hw, block_in, n_wblocks, input.dtype
        )
        units = n_oh_blocks * n_wblocks
        for start, stop in _row_spans(batch, 1, units=units):
            _pool2d_row_kernel[((stop - start) * units,)](
                input[start:stop],
                output[start:stop],
                output[start:stop],  # idx/d/d_shift are unused with the flags off
                output[start:stop],
                output[start:stop],
                stop - start,
                ih,
                iw,
                oh,
                ow,
                BLOCK_HW=block_hw,
                N_HCHUNKS=n_hchunks,
                BLOCK_W=block_w,
                BLOCK_IN=block_in,
                BLOCK_OH=block_oh,
                N_OH_BLOCKS=n_oh_blocks,
                N_WBLOCKS=n_wblocks,
                RETURN_INDICES=False,
                IDX_FP32=True,  # unused on the values path
                FOLD_D=False,
                D_PACK=1,
                num_warps=4,
                num_stages=3,
            )
        return output

    # Indices path: uniform pow2 windows use the even/odd halving with index tracking
    # (element-wise, far faster than the row kernel's cross-lane tl.max(return_indices));
    # exact win=2x2 keeps its own kernel; otherwise the coalesced row kernel.
    #
    # A pow2 ow is NOT required here -- that test belongs to _pool2d_uniform_pow2_idx's
    # strided H-halve; evenodd_idx masks a padded tile, and testing the width here sent
    # (2,128,4,28,28) to the row kernel, ~20x slower.
    #
    # The %256 test guards ``wv.numel() % 256 == 0`` with ``wv.numel() == batch*ih*ow``;
    # testing ``batch*ih*iw`` is half as strong.  (_wpool_block now clamps the block to a
    # divisor of the buffer as well, so this is a tiling choice rather than the only guard.)
    if (
        ih == 2 * oh
        and iw == 2 * ow
        and (batch * ih * ow) % 256 == 0
        and min(16, triton.next_power_of_2(oh)) * triton.next_power_of_2(ow)
        <= _INT_IDX_TILE_ELEMS
    ):
        return _pool2d_uniform_evenodd_idx(input, oh, ow, d_span=d_span)
    if (
        ih % oh == 0
        and iw % ow == 0
        and (ih // oh) & (ih // oh - 1) == 0  # pow2
        and (iw // ow) & (iw // ow - 1) == 0  # pow2
        and (ow & (ow - 1)) == 0  # pow2 row width for the strided H-halve
        and (batch * ih * iw) % 256 == 0
    ):
        if ih == 2 * oh and iw == 2 * ow:
            return _pool2d_uniform_evenodd_idx(input, oh, ow, d_span=d_span)
        return _pool2d_uniform_pow2_idx(input, oh, ow)
    mwh = _compute_max_win(ih, oh)
    block_w = triton.next_power_of_2(ow)
    block_in = triton.next_power_of_2(iw)
    if block_w * block_in > 4096:  # UB cap; see the values path for the note on
        block_w = max(4, 4096 // block_in)  # why W-tiling is NOT the answer
    n_wblocks = triton.cdiv(ow, block_w)
    block_hw, n_hchunks = _row_hw_tiling(mwh, block_in)
    indices = torch.empty((batch, oh, ow), device=input.device, dtype=torch.int64)
    # The index path blocks too: tl.max(return_indices) on the H pass' 3-D tile collapses
    # BLOCK_OH>1 to 0.2x, so emulating it (see the kernel) is a large win.  The tail's stays.
    block_oh, n_oh_blocks = _row_block_oh(
        oh,
        batch,
        block_w,
        block_hw,
        block_in,
        n_wblocks,
        input.dtype,
        return_indices=True,
    )
    units = n_oh_blocks * n_wblocks
    for start, stop in _row_spans(batch, 1, units=units):
        _pool2d_row_kernel[((stop - start) * units,)](
            input[start:stop],
            output[start:stop],
            indices[start:stop],
            output[start:stop],  # d/d_shift are unused with FOLD_D off
            output[start:stop],
            stop - start,
            ih,
            iw,
            oh,
            ow,
            BLOCK_HW=block_hw,
            N_HCHUNKS=n_hchunks,
            BLOCK_W=block_w,
            BLOCK_IN=block_in,
            BLOCK_OH=block_oh,
            N_OH_BLOCKS=n_oh_blocks,
            N_WBLOCKS=n_wblocks,
            RETURN_INDICES=True,
            # f32 index arithmetic is exact while the whole flat index fits f32; else the int32 form.
            IDX_FP32=ih * iw <= _F32_EXACT_MAX,
            FOLD_D=False,
            D_PACK=1,
            num_warps=4,
            num_stages=3,
        )
    return output, indices


# --- Main operator ---

# Per-row ``(row % out_d) * win_d``: the D half of a 3-D flat *window* offset, host-built
# and cached.  The caller's ``* hw`` supplies the plane factor -- do NOT fold it in here.
_D_ROW_BASE_CACHE: dict = {}


def _d_row_base(rows, out_d, win_d, device):
    key = (rows, out_d, win_d)
    t = _D_ROW_BASE_CACHE.get(key)
    if t is None:
        t = ((torch.arange(rows, dtype=torch.int32) % out_d) * win_d).to(
            device, torch.int32
        )
        _D_ROW_BASE_CACHE[key] = t
    return t


@libentry()
@triton.jit
def _idx_combine_hw_kernel(
    d_ptr,
    base_ptr,
    out_ptr,
    rows,
    hw,
    WITH_BASE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    N_HWB: tl.constexpr,
):
    """``out[r, c] = (base[r] + d[r, c]) * hw + c`` -- the final 3-D flat index.

    The tail of the HW-identity paths, where the pooled (h, w) index *is* the column
    position, so one affine pass replaces a torch chain that was most of the cost there.
    ``d`` stays int32; ``base`` is the per-row *window* offset ``d_out * win_d``, zero when
    out_d == 1 (hence WITH_BASE) -- the ``* hw`` here is the only plane factor applied.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    rmask = r < rows
    base = tl.load(base_ptr + r, mask=rmask, other=0) if WITH_BASE else 0
    # A column loop, not a second grid axis: a (rows/BR) x (hw/256) grid product can pass
    # the 65535 cap only by luck, and without the loop hw > 256 silently lost columns.
    cols = tl.arange(0, BLOCK_HW)[None, :]
    for b in range(N_HWB):
        c = b * BLOCK_HW + cols
        m = rmask & (c < hw)
        off = r * hw + c
        d = tl.load(d_ptr + off, mask=m, other=0)
        tl.store(out_ptr + off, ((base + d) * hw + c).to(tl.int64), mask=m)


def _idx_combine_hw(d_idxs, out_shape, rows, hw, out_d, win_d, device):
    """Run _idx_combine_hw_kernel and return the int64 index in out_shape.

    ``d_idxs`` is the int32 (rows, hw) D-window argmax plane; the index is rebuilt in
    int32 and widened at the store (every touched shape stays below 2**31, re-checked).
    """
    if out_d * win_d * hw >= 2**31:
        return None  # index space wider than int32: not worth a WIDE branch
    block_hw = min(triton.next_power_of_2(hw), 256)
    # Worth at least _COMBINE_MIN_PROGRAMS programs, and never more than the element budget.
    block_r = max(
        1,
        min(
            4096 // block_hw,
            triton.next_power_of_2(triton.cdiv(rows, _COMBINE_MIN_PROGRAMS)),
        ),
    )
    n_hwb = triton.cdiv(hw, block_hw)
    indices = torch.empty((rows, hw), device=device, dtype=torch.int64)
    with_base = out_d != 1
    base = (
        _d_row_base(rows, out_d, win_d, device)
        if with_base
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    _idx_combine_hw_kernel[(triton.cdiv(rows, block_r),)](
        d_idxs,
        base,
        indices,
        rows,
        hw,
        WITH_BASE=with_base,
        BLOCK_R=block_r,
        BLOCK_HW=block_hw,
        N_HWB=n_hwb,
        num_warps=4,
        num_stages=3,
    )
    return indices.view(out_shape)


@libentry()
@triton.jit
def _global_pool_kernel(
    in_ptr,
    out_ptr,
    spatial,
    BLOCK: tl.constexpr,
):
    """Reduce each (n, c) slice to a scalar max for global pooling (1,1,1): the D*H*W
    volume is contiguous in NCDHW, so it is a coalesced 1D scan plus one cross-lane tl.max.
    """
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
    """Reduce each (n, c) slice to a scalar max + argmax for global (1,1,1).  NaN-aware:
    a NaN wins (matches PyTorch).  A BLOCK_M x BLOCK_N tile is loaded coalesced, reduced
    over the spatial axis and merged into the running (max, argmax) per slice.
    """
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
    ``return_indices=False`` -> pooled output only (the kernels skip all index work);
    anything else -> ``(output, indices)``, including the ``None`` the aten 2-arg schema passes.
    """
    logger.debug("GEMS_ASCEND ADAPTIVE_MAX_POOL3D")

    if return_indices is None:
        return_indices = True  # aten 2-arg schema: real indices required
    # Normalise to a strict bool so every exit below has exactly two cases.
    # torch itself treats a falsy non-bool (e.g. 0) as "no indices".
    return_indices = bool(return_indices)

    if isinstance(output_size, int):
        output_size = [output_size, output_size, output_size]

    out_d, out_h, out_w = output_size
    in_n, in_c, in_d, in_h, in_w = input.shape

    # --- empty ---
    if out_d == 0 or out_h == 0 or out_w == 0 or in_d == 0 or in_h == 0 or in_w == 0:
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            return output, torch.empty_like(output, dtype=torch.int64)
        return output

    # --- identity (moved before contiguous() to minimize overhead) ---
    if in_d == out_d and in_h == out_h and in_w == out_w:
        if return_indices:
            spatial_total = in_d * in_h * in_w
            # No kernel: a broadcast, and torch's fill beats the per-lane int64 modulo it replaces
            # (a software divide scaling with element count).  ``repeat`` ties or beats expand().
            indices = (
                torch.arange(spatial_total, dtype=torch.int64, device=input.device)
                .repeat(in_n * in_c)
                .view(in_n, in_c, in_d, in_h, in_w)
            )
            return input, indices
        return input

    # One restride for every kernel path below: they all index ``input`` as flat NCDHW,
    # including the uniform-3d and tiny paths, which dispatch *before* Path B1/B2.
    if not input.is_contiguous():
        input = input.contiguous()

    # --- global pool (1,1,1): one (n, c) slice on the grid, values-only via
    # _global_pool_kernel and values+indices via _global_pool_indices_kernel ---
    if out_d == 1 and out_h == 1 and out_w == 1:
        spatial_volume = in_d * in_h * in_w
        if spatial_volume >= 1:
            nc = in_n * in_c
            if not return_indices:
                output = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=input.dtype
                )
                block = min(triton.next_power_of_2(spatial_volume), 1024)
                in_slices = input.reshape(nc, spatial_volume)
                out_slices = output.reshape(nc)
                for start, stop in _row_spans(nc, 1):
                    _global_pool_kernel[(stop - start,)](
                        in_slices[start:stop],
                        out_slices[start:stop],
                        spatial_volume,
                        BLOCK=block,
                        num_warps=4,
                    )
                return output
            else:
                # Coalesced 2D-tiled reduction (values + argmax); flag_gems.max_dim drops NaNs.
                output = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=input.dtype
                )
                out_indices = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=torch.int64
                )
                # Tile shape: BLOCK_M is the only parallelism (grid = cdiv(nc, BLOCK_M)) and BLOCK_N
                # trades tile size against SEQUENTIAL tiles; the per-tile cross-lane
                # tl.max(axis=1, return_indices=True) is what costs.
                #
                # Do NOT raise BLOCK_N to 2048 or BLOCK_M to 16 at BLOCK_N=1024: both compile and then
                # produce silently WRONG indices with no UB error.
                block_m = max(
                    1, min(8, triton.next_power_of_2(nc) // _GPOOL_TARGET_PROGRAMS)
                )
                block_n = min(triton.next_power_of_2(spatial_volume), 1024)
                in_slices = input.reshape(nc, spatial_volume)
                out_slices = output.reshape(nc)
                idx_slices = out_indices.reshape(nc)
                for start, stop in _row_spans(nc, block_m):
                    rows = stop - start
                    _global_pool_indices_kernel[(triton.cdiv(rows, block_m),)](
                        in_slices[start:stop],
                        out_slices[start:stop],
                        idx_slices[start:stop],
                        rows,
                        spatial_volume,
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        num_warps=4,
                    )
                return output, out_indices

    # --- uniform 3D fast path: _uniform_3d_fused_kernel + _select_uniform_fused ---
    # All three dims divisible -> a 3D strided max-pool in one pass.  Values-only, so
    # return_indices=True lets the D-reduce/pool2d decomposition handle it.
    if (
        in_d % out_d == 0
        and in_h % out_h == 0
        and in_w % out_w == 0
        and not (in_h == out_h and in_w == out_w)
        and in_d != out_d
        and out_d != 1
        and not return_indices
        and not (input.numel() > 65536 and out_w <= 16)
        # Fused cost grows with num_outputs * win_size; large uniform windows go to the
        # decomposition ([2,64,64,256,256] -> (2,32,32): 4.4ms vs 26ms fused).
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
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        # The pid decodes to (n, c, d_out, h_out, w_block) with n outermost: slicing n re-bases it.
        units = in_c * out_d * out_h * n_wblocks
        for start, stop in _row_spans(in_n, 1, units=units):
            _uniform_3d_fused_kernel[((stop - start) * units,)](
                input[start:stop],
                output[start:stop],
                stop - start,
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
        return output

    # --- tiny shape fast path: one pass beats several launches under ~512 input elements ---
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
        return output

    # --- in_d == 1: reshape (N,C,1,H,W) -> (N*C,H,W), pool2d, reshape back ---
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
            return output

    # --- Path B1: HW identity + out_d == 1 (fastest non-identity path) ---
    # A pure max over dim=2; values and indices both go through _d_reduce_unified.
    if (
        in_h == out_h
        and in_w == out_w
        and in_d % out_d == 0
        and in_d > 1
        and out_d == 1
    ):
        nc = in_n * in_c
        hw = in_h * in_w
        if not return_indices:
            d_reduced = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=False
            )
            output = d_reduced.view(in_n, in_c, 1, in_h, in_w)
            return output
        else:
            d_vals, d_idxs = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=True, idx32=True
            )
            output = d_vals.view(in_n, in_c, 1, in_h, in_w)
            out_shape = (in_n, in_c, 1, in_h, in_w)
            indices = _idx_combine_hw(d_idxs, out_shape, nc, hw, 1, in_d, input.device)
            if indices is None:  # index space past int32: keep the torch chain
                spatial = torch.arange(hw, dtype=torch.int32, device=input.device)
                indices = (d_idxs.view(nc, hw) * hw + spatial).to(torch.int64)
                indices = indices.view(out_shape)
            return output, indices

    # --- Path B2: HW identity + out_d > 1 ---
    # Fold the uniform D-window into dim=1 and D-reduce it.  Do NOT use F.max_pool3d:
    # FlagGems intercepts it inside use_gems() and it hangs for large kernel sizes.
    if (
        in_h == out_h
        and in_w == out_w
        and in_d % out_d == 0
        and in_d > out_d
        and out_d > 0
    ):
        win_d = in_d // out_d
        # (N*C*out_d, win_d, H*W) puts the window on dim=1.
        reshaped = input.view(in_n, in_c, out_d, win_d, in_h, in_w)
        nc_out = in_n * in_c * out_d
        hw = in_h * in_w
        if not return_indices:
            d_vals = _d_reduce_unified(
                reshaped.reshape(nc_out, win_d, hw), return_indices=False
            ).view(in_n, in_c, out_d, in_h, in_w)
            return d_vals
        else:
            d_vals_flat, d_local = _d_reduce_unified(
                reshaped.reshape(nc_out, win_d, hw), return_indices=True, idx32=True
            )
            d_vals = d_vals_flat.view(in_n, in_c, out_d, in_h, in_w)
            # index = (d_out * win_d + local_d) * hw + spatial, all in one pass
            out_shape = (in_n, in_c, out_d, in_h, in_w)
            indices = _idx_combine_hw(
                d_local, out_shape, nc_out, hw, out_d, win_d, input.device
            )
            if indices is None:  # index space past int32: keep the torch chain
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
                indices = indices.view(out_shape)
            return d_vals, indices

    # --- Path A: in_d == out_d -> use FlagGems Triton pool2d ---
    if in_d == out_d and out_d > 1:

        inp_3d = input.view(-1, in_h, in_w)
        # Ask the pool to write the 3-D index directly (its 2-D spatial index plus ``d * hw``),
        # removing an extra output pass.  _evenodd_idx_wins says whether this shape reaches it;
        # a shape it declines keeps the torch chain below.
        fuse_d = return_indices and _evenodd_idx_wins(
            inp_3d.shape[0], in_h, in_w, out_h, out_w
        )
        if return_indices:
            pool_vals, pool_spatial_idx = _pool2d_triton(
                inp_3d,
                out_h,
                out_w,
                return_indices=True,
                d_span=(in_d, in_h * in_w) if fuse_d else None,
            )
        else:
            pool_vals = _pool2d_triton(inp_3d, out_h, out_w, return_indices=False)
            pool_spatial_idx = None
        output = pool_vals.view(in_n, in_c, in_d, out_h, out_w)
        if return_indices:
            spatial_idx = pool_spatial_idx.view(in_n, in_c, in_d, out_h, out_w)
            if fuse_d:
                return output, spatial_idx
            d_bcast = torch.arange(in_d, device=input.device, dtype=torch.int64).view(
                1, 1, in_d, 1, 1
            )
            indices = d_bcast * in_h * in_w + spatial_idx
            return output, indices
        return output

    # --- main dispatch: out_d=1 D-reduce + Path C (general) ---
    max_win_d = _compute_max_win(in_d, out_d)
    max_win_h = _compute_max_win(in_h, out_h)
    max_win_w = _compute_max_win(in_w, out_w)
    total_output = in_n * in_c * out_d * out_h * out_w
    win_size = max_win_d * max_win_h * max_win_w
    is_uniform = in_d % out_d == 0 and in_h % out_h == 0 and in_w % out_w == 0

    # --- _kernel_direct fast path: small windows get one call (no extra launches, no
    # intermediates); uniform windows are excluded, its per-element scan is far slower ---
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
            USE_INT64=flat_elems >= _F32_EXACT_MAX,
            num_warps=nw,
            num_stages=ns,
        )
        return output

    # --- out_d=1 D-reduce: reduce D then pool2d; (N*C, D, H*W) keeps D-slices contiguous.
    # Do NOT use input.reshape(-1, D) -- that groups consecutive W/H elements instead. ---
    if out_d == 1 and in_d > 1:
        nc = in_n * in_c
        hw = in_h * in_w

        if return_indices:
            # (N,C,D,H,W) -> (N,C,H,W)
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
            total_out = indices.numel()
            spatial_flat = spatial_idx.view(-1)
            d_argmax_flat = d_argmax.to(dtype=torch.int64).view(-1)
            out_flat = indices.view(-1)
            for start, stop in _spans(total_out, 256):
                _merge_outd1_indices_kernel[(triton.cdiv(stop - start, 256),)](
                    spatial_flat[start:stop],
                    d_argmax_flat,
                    out_flat[start:stop],
                    total_out,
                    in_c,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    base=start,
                    BLOCK_SIZE=256,
                )
            return output, indices
        else:
            d_reduced = _d_reduce_unified(
                input.reshape(nc, in_d, hw), return_indices=False
            ).view(nc, in_h, in_w)
            pool_vals = _pool2d_triton(d_reduced, out_h, out_w, return_indices=False)
            output = pool_vals.view(in_n, in_c, 1, out_h, out_w)
            return output

    # --- Path C: decomposed D-reduce + Triton pool2d (D, H and W all change) ---
    #   1. reduce D over each adaptive window, 2. pool2d over the D-reduced (H, W),
    #   3. with return_indices, fuse the D argmax with the 2-D spatial argmax.
    #
    # Uniform D windows (in_d % out_d == 0) pick between D-first (reshape to
    # (N*C*out_d, win_d, H*W): best when the spatial reduction is small) and pool2d-first
    # (reshape to (N*C*D, H, W): best when it is large; gated on in_hw >= 4 * out_hw and
    # win_d <= 8 because of the extra N*C*D of channel work).  Non-uniform windows loop.

    # Step 1: Reduce D dimension
    if in_d % out_d == 0:
        # Uniform D windows.
        win_d = in_d // out_d
        in_hw = in_h * in_w
        out_hw = out_h * out_w
        nc = in_n * in_c

        # --- Pool2d-first strategy (gated above; see the Path C banner) ---
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
            # NaN-aware D-reduce over win_d via the in-file Triton kernel.
            d_vals = _d_reduce_unified(
                d_reduced.reshape(nc * out_d, win_d, out_h * out_w),
                return_indices=False,
            ).view(in_n, in_c, out_d, out_h, out_w)
            return d_vals

        if not return_indices:
            nc_out = in_n * in_c * out_d
            d_reduced = _d_reduce_unified(
                input.reshape(nc_out, win_d, in_hw), return_indices=False
            ).view(in_n, in_c, out_d, in_h, in_w)
        else:
            nc_out = in_n * in_c * out_d
            # int32: this plane is consumed below, never returned, and an int64
            # one costs the writer 2x bandwidth and the reader 7x.
            d_vals_flat, d_local = _d_reduce_unified(
                input.reshape(nc_out, win_d, in_hw), return_indices=True, idx32=True
            )
            d_reduced = d_vals_flat.view(in_n, in_c, out_d, in_h, in_w)
            # Keep the LOCAL D index; the merge kernel below folds in the
            # per-out_d offset, so no host-side broadcast add over (nc*out_d, hw).
            d_argmax = d_local.view(in_n, in_c, out_d, in_h, in_w)
    else:
        # Non-uniform D windows: fall back to per-out_d loop.
        d_reduced = torch.empty(
            (in_n, in_c, out_d, in_h, in_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            d_argmax = torch.empty(
                (in_n, in_c, out_d, in_h, in_w), device=input.device, dtype=torch.int32
            )

        nc = in_n * in_c
        hw = in_h * in_w
        for d_out in range(out_d):
            d_start = d_out * in_d // out_d
            d_end = ((d_out + 1) * in_d + out_d - 1) // out_d
            # The D-slice is strided, so reshape would give a view whose batch stride is in_d * hw;
            # _d_reduce_unified wants a contiguous (batch, in_d, hw) layout.
            d_slice = input[:, :, d_start:d_end, :, :].contiguous()
            d_len = d_end - d_start
            if return_indices:
                d_vals_flat, d_idxs_flat = _d_reduce_unified(
                    d_slice.reshape(nc, d_len, hw), return_indices=True, idx32=True
                )
                # LOCAL D index; the merge kernel adds the window start.
                d_argmax[:, :, d_out, :, :] = d_idxs_flat.view(in_n, in_c, in_h, in_w)
                d_vals = d_vals_flat.view(in_n, in_c, in_h, in_w)
            else:
                d_vals = _d_reduce_unified(
                    d_slice.reshape(nc, d_len, hw), return_indices=False
                ).view(in_n, in_c, in_h, in_w)
            d_reduced[:, :, d_out, :, :] = d_vals

    # Step 2: FlagGems Triton pool2d over H,W
    pool_input = d_reduced.reshape(-1, in_h, in_w)
    if return_indices:
        # Fold the D argmax into the pool when it pays: ``(d + d_window_start) * hw_in +
        # spatial``, the D half already known per position, so the pool carries it through the
        # same affine load as the values and the merge's gather disappears.  None -> the old route.
        fused = _pool2d_triton_d_indices(
            pool_input,
            d_argmax.reshape(-1, in_h, in_w),
            out_d,
            in_d,
            out_h,
            out_w,
        )
        if fused is not None:
            output, indices = fused
            return (
                output.view(in_n, in_c, out_d, out_h, out_w),
                indices.view(in_n, in_c, out_d, out_h, out_w),
            )
        pool_vals, pool_spatial_idx = _pool2d_triton(
            pool_input, out_h, out_w, return_indices=True
        )
        output = pool_vals.view(in_n, in_c, out_d, out_h, out_w)

        # Step 3: merge D and spatial indices (torch.gather is slow on NPU)
        hw_in = in_h * in_w
        out_hw = out_h * out_w
        total_idx = in_n * in_c * out_d * out_h * out_w
        d_arg_flat = d_argmax.reshape(-1)
        spatial_flat = pool_spatial_idx.view(-1)
        indices = torch.empty(total_idx, device=input.device, dtype=torch.int64)
        # int32 index math needs the flat index below 2**24 and every address below 2**31; both
        # the gather address and the final index are bounded by input.numel(), so that is the test.
        use_wide = input.numel() >= 2**31
        n_rows = indices.numel() // out_hw  # == in_n * in_c * out_d
        block_c = triton.next_power_of_2(out_hw)
        # Rows per program: ~42 ns fixed and the row steps are unrolled, so this only trades
        # programs against code size (4/8/16 within 3%, 8 never the worst).
        block_r = 8
        for start, stop in _row_spans(n_rows, block_r):
            _merge_d2h_indices_kernel[(triton.cdiv(stop - start, block_r),)](
                d_arg_flat,
                spatial_flat[start * out_hw : stop * out_hw],
                indices[start * out_hw : stop * out_hw],
                hw_in,
                out_hw,
                win_d if in_d % out_d == 0 else 0,
                in_d,
                out_d,
                stop - start,
                start,
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                UNIFORM_D=(in_d % out_d == 0),
                WIDE=use_wide,
            )
        indices = indices.view(in_n, in_c, out_d, out_h, out_w)
        return output, indices
    else:
        pool_vals = _pool2d_triton(pool_input, out_h, out_w, return_indices=False)
        output = pool_vals.view(in_n, in_c, out_d, out_h, out_w)
        return output
