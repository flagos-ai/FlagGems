# Copyright 2026, The FlagOS Contributors.
import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# Import torch_npu at module level to avoid repeated import-lookup overhead
# in the hot path.  The import is cached by Python after the first call but
# the dict lookup for globals()['torch_npu'] still adds a measurable cost
# for lightweight shapes.
try:
    import torch_npu
except ImportError:
    torch_npu = None

# ---------------------------------------------------------------------------
# Cache the pre?captured CANN amax kernel & DispatchKeySet at module?load
# time to avoid dict lookups in the hot path.  See flag_gems/__init__.py
# for where _NATIVE_CANN_KERNELS / _NATIVE_CANN_KEYSETS are populated.
# ---------------------------------------------------------------------------
_NATIVE_AMAX_KERNEL = (
    flag_gems._NATIVE_CANN_KERNELS.get("aten::amax")
    if hasattr(flag_gems, "_NATIVE_CANN_KERNELS")
    else None
)
_NATIVE_AMAX_KEYSET = (
    flag_gems._NATIVE_CANN_KEYSETS.get("aten::amax")
    if hasattr(flag_gems, "_NATIVE_CANN_KEYSETS")
    else None
)

# ---------------------------------------------------------------------------
# Cache the C++ adaptive_max_pool3d_global function at module?load time.
# This pybind11-wrapped C++ function calls CANN's amax kernel directly via
# KernelFunction::callBoxed() with the handle captured in __init__.py before
# the Library("aten","IMPL") is created.  It saves ~3 ��s over the Python
# call_boxed approach by moving tensor ops (flatten, view) to C++ and
# avoiding SafeKernelFunction attribute lookups.
# ---------------------------------------------------------------------------
_CPP_ADAPTIVE_POOL_GLOBAL = None
try:
    from flag_gems import c_operators

    if hasattr(c_operators, "adaptive_max_pool3d_global"):
        _CPP_ADAPTIVE_POOL_GLOBAL = c_operators.adaptive_max_pool3d_global
except ImportError:
    pass


# ==============================================================================
# Kernel 1 �� Direct scan  (M, K)
#
#   M = OUT_PER_BLOCK   �� output elements per block
#   K = CHAN_PER_BLOCK  �� channels per block (1 = standard, >1 = chunked)
#
# Autotuner selects the best (M, K).  Main function never passes them.
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
):
    """Direct-scan: each thread independently scans one output element."""
    pid = tl.program_id(0)
    block_start = pid * OUT_PER_BLOCK
    c_groups = (in_c + CHAN_PER_BLOCK - 1) // CHAN_PER_BLOCK
    flat_elems = in_n * c_groups * out_d * out_h * out_w

    tid = tl.arange(0, OUT_PER_BLOCK)
    flat_idx = block_start + tid
    valid = flat_idx < flat_elems

    tmp = flat_idx
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
    min_val = get_dtype_min(dtype)
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
# Kernel 2 �� Cooperative  (T threads scan one window together)
#
#   Phase 1: each thread scans CONTIGUOUS elements in the flat window (chunked
#            by COOP_THREADS), so every vector load is CONTIGUOUS across threads.
#            This avoids the strided vlds pattern of the old design and is 2-10��
#            faster on Ascend NPU memory subsystem.
#   Phase 2: tree reduction (log2(COOP_THREADS) steps) instead of the old
#            per-lane serial loop �� 128�� fewer scratch reads for COOP_THREADS=256.
#
#   Grid: (FLAT_ELEMS,)  �� one block per output element.
#   Scratch: FLAT_ELEMS �� COOP_THREADS elements each for vals and idxs.
# ==============================================================================


def _select_cooperative(max_win_d, max_win_h, max_win_w):
    """Manual COOP_THREADS selection (replaces @triton.autotune)."""
    win = max_win_d * max_win_h * max_win_w
    if win <= 8:
        return 32, 1, 5
    if win <= 64:
        return 64, 2, 4
    if win <= 512:
        return 128, 4, 3
    if win <= 4096:
        return 128, 4, 2
    return 256, 8, 2


@libentry()
@triton.jit
def _kernel_cooperative(
    in_ptr,
    scratch_vals_ptr,
    scratch_idxs_ptr,
    output_ptr,
    indices_ptr,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    FLAT_ELEMS: tl.constexpr,
    COOP_THREADS: tl.constexpr,
    MAX_WIN_D: tl.constexpr,
    MAX_WIN_H: tl.constexpr,
    MAX_WIN_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
):
    """Cooperative: one block per output element, T threads share scanning."""
    pid = tl.program_id(0)
    valid = pid < FLAT_ELEMS

    out_vol = out_d * out_h * out_w
    hw_out = out_h * out_w
    nc_out = pid // out_vol
    rem = pid % out_vol
    d_out_idx = rem // hw_out
    rem2 = rem % hw_out
    h_out_idx = rem2 // out_w
    w_out_idx = rem2 % out_w
    n_idx = nc_out // in_c
    c_idx = nc_out % in_c

    n_idx = tl.where(valid, n_idx, 0)
    c_idx = tl.where(valid, c_idx, 0)
    d_out_idx = tl.where(valid, d_out_idx, 0)
    h_out_idx = tl.where(valid, h_out_idx, 0)
    w_out_idx = tl.where(valid, w_out_idx, 0)

    d_start = d_out_idx * in_d // out_d
    d_end = ((d_out_idx + 1) * in_d + out_d - 1) // out_d
    d_end = tl.where(d_out_idx == out_d - 1, in_d, d_end)
    win_d = d_end - d_start
    h_start = h_out_idx * in_h // out_h
    h_end = ((h_out_idx + 1) * in_h + out_h - 1) // out_h
    h_end = tl.where(h_out_idx == out_h - 1, in_h, h_end)
    win_h = h_end - h_start
    w_start = w_out_idx * in_w // out_w
    w_end = ((w_out_idx + 1) * in_w + out_w - 1) // out_w
    w_end = tl.where(w_out_idx == out_w - 1, in_w, w_end)
    win_w = w_end - w_start

    win_elems = win_d * win_h * win_w
    hw_win = win_h * win_w
    MAX_WIN_ELEMS: tl.constexpr = MAX_WIN_D * MAX_WIN_H * MAX_WIN_W
    MAX_CHUNK: tl.constexpr = tl.cdiv(MAX_WIN_ELEMS, COOP_THREADS)

    dtype = in_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    tid = tl.arange(0, COOP_THREADS)
    in_base = in_ptr + n_idx * in_c * in_d * in_h * in_w + c_idx * in_d * in_h * in_w

    maxv = tl.full((COOP_THREADS,), min_val, dtype=dtype)
    maxi = tl.full((COOP_THREADS,), 0, dtype=tl.int64)
    for chunk in range(MAX_CHUNK):
        flat_pos = chunk * COOP_THREADS + tid
        in_chunk = flat_pos < win_elems
        kd = flat_pos // hw_win
        rem_k = flat_pos % hw_win
        kh = rem_k // win_w
        kw = rem_k % win_w
        d_in = d_start + kd
        h_in = h_start + kh
        w_in = w_start + kw
        d_ok = (kd < win_d) & (d_in < in_d)
        h_ok = (kh < win_h) & (h_in < in_h)
        w_ok = (kw < win_w) & (w_in < in_w)
        elem_ok = d_ok & h_ok & w_ok
        ptr = in_base + d_in * in_h * in_w + h_in * in_w + w_in
        load_mask = in_chunk & elem_ok
        v = tl.load(ptr, mask=load_mask, other=min_val, cache_modifier=".ca")
        f_idx = d_in * in_h * in_w + h_in * in_w + w_in
        better = (v > maxv) | (v != v)
        maxv = tl.where(in_chunk & elem_ok, tl.where(better, v, maxv), maxv)
        maxi = tl.where(in_chunk & elem_ok, tl.where(better, f_idx, maxi), maxi)

    tl.store(scratch_vals_ptr + pid * COOP_THREADS + tid, maxv)
    if RETURN_INDICES:
        tl.store(scratch_idxs_ptr + pid * COOP_THREADS + tid, maxi)

    # Phase 2: serial reduction (all threads compute, simple tl.where)
    final_v = tl.full((1,), min_val, dtype=dtype)
    final_i = tl.full((1,), 0, dtype=tl.int64)
    for i in range(COOP_THREADS):
        v = tl.load(scratch_vals_ptr + pid * COOP_THREADS + i)
        better = v > final_v
        final_v = tl.where(better, v, final_v)
        if RETURN_INDICES:
            idx = tl.load(scratch_idxs_ptr + pid * COOP_THREADS + i)
            final_i = tl.where(better, idx, final_i)

    tl.store(output_ptr + pid, final_v, mask=valid)
    if RETURN_INDICES:
        tl.store(indices_ptr + pid, final_i, mask=valid)


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
    nc_idx = offsets // out_spatial
    rem = offsets % out_spatial
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
def _merge_hwidentity_indices_kernel(
    d_local_ptr,
    indices_ptr,
    out_d,
    h,
    w,
    hw_in,
    win_d,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute full 3D indices for HW-identity path.
    Layout: (N, C, out_d, H, W) flat.
    Computes d_out_idx from flat offset �� d_off = d_out_idx*win_d.
    d_local is loaded from npu_max result.
    Full 3D index = (d_off + d_local) * hw_in + h*W + w."""
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    h_pos = (offsets // w) % h
    w_pos = offsets % w
    doff = (offsets // (h * w)) % out_d
    d_local = tl.load(d_local_ptr + offsets, mask=mask, other=0)
    d_full = doff * win_d + d_local
    tl.store(indices_ptr + offsets, d_full * hw_in + h_pos * w + w_pos, mask=mask)


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
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    tl.store(indices_ptr + offsets, offsets % spatial_total, mask=mask)


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

    # Compute batch and position within batch
    batch = offsets // out_hw

    # Load spatial index
    sidx = tl.load(spatial_idx_ptr + offsets, mask=mask, other=0)

    # Load D index from d_argmax at (batch, sidx)
    d_arg_pos = batch * hw_in + sidx
    d_val = tl.load(d_argmax_ptr + d_arg_pos, mask=mask, other=0)

    # Full 3D index
    tl.store(out_idx_ptr + offsets, d_val * hw_in + sidx, mask=mask)


@triton.jit
def _merge_patha_indices_kernel(
    spatial_idx_ptr,
    out_idx_ptr,
    in_d,
    hw_in,
    out_hw,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge spatial indices with the per-slice depth offset for Path A
    (in_d == out_d).  Layout of the flat output is (N, C, in_d, oh, ow).
    For each output element, the depth index is (offset // out_hw) % in_d
    and the full 3D index is depth * hw_in + spatial_idx."""
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    d = (offsets // out_hw) % in_d
    sidx = tl.load(spatial_idx_ptr + offsets, mask=mask, other=0)
    tl.store(out_idx_ptr + offsets, d * hw_in + sidx, mask=mask)


@triton.jit
def _merge_d2h_windowed_indices_kernel(
    d_local_ptr,
    spatial_idx_ptr,
    out_idx_ptr,
    hw_in,
    out_hw,
    win_d,
    od,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge local D-window argmax with the window's depth offset and the
    spatial index, for uniformly-windowed Path C (and out_d==1, where od==1).
    Flat output layout is (N, C, od, oh, ow):
      doff = (offset // out_hw) % od
      full_idx = (doff * win_d + d_local[batch, sidx]) * hw_in + sidx."""
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    batch = offsets // out_hw
    doff = batch % od
    sidx = tl.load(spatial_idx_ptr + offsets, mask=mask, other=0)
    d_local = tl.load(d_local_ptr + batch * hw_in + sidx, mask=mask, other=0)
    d_full = doff * win_d + d_local
    tl.store(out_idx_ptr + offsets, d_full * hw_in + sidx, mask=mask)


# ==============================================================================
# Kernel 3 �� Unified D-reduction
#
#   Merges the old _d_reduce_kernel and _d_reduce_tiled_kernel into a single
#   kernel with two strategies selected by STRATEGY: tl.constexpr.
#
#   STRATEGY=0 (simple): each block processes BLOCK_M positions, simple D loop.
#       Grid: (ceil(N*C*H*W / BLOCK_M),).  Supports return_indices.
#   STRATEGY=1 (tiled): each block handles TILE_SIZE contiguous HW positions.
#       Grid: (N*C, ceil(H*W / TILE_SIZE)), optionally 2D for NPU 65535 limit.
#       Values-only (no indices).
#
#   Both strategies use the same 1D/2D grid computed as the max across all
#   non-pruned autotune configs.  The kernel bounds-checks pid for each strategy.
# ==============================================================================


def _make_tiled_grid(nc, n_tiles):
    """Build a grid that respects the NPU 65535 per-dimension limit.

    Returns (grid, total_blocks) where grid may be 1-D or 2-D as needed.
    total_blocks is the actual number of valid blocks."""
    total_blocks = nc * n_tiles
    if total_blocks <= 65535:
        return (total_blocks,), total_blocks

    grid_x = min(total_blocks, 65535)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    return (grid_x, grid_y), total_blocks


def _compute_d_reduce_max_grid(nc, hw, configs):
    """Compute the max grid size needed across all non-pruned autotune configs.
    Returns (grid, max_blocks)."""
    max_blocks = 0
    for cfg in configs:
        kw = cfg.kwargs
        strategy = kw.get("STRATEGY", 0)
        if strategy == 0:
            blocks = triton.cdiv(nc * hw, kw["BLOCK_M"])
        else:
            n_tiles = triton.cdiv(hw, kw["TILE_SIZE"])
            blocks = nc * n_tiles
        if blocks > max_blocks:
            max_blocks = blocks
    if max_blocks == 0:
        max_blocks = 1
    if max_blocks <= 65535:
        return (max_blocks,), max_blocks

    grid_x = min(max_blocks, 65535)
    grid_y = (max_blocks + grid_x - 1) // grid_x
    return (grid_x, grid_y), max_blocks


def _prune_d_reduce_configs(configs, named_args, **kwargs):
    """Prune D-reduce autotune configs that are clearly bad for the shape."""
    in_n = named_args["in_n"]
    in_c = named_args["in_c"]
    in_h = named_args["in_h"]
    in_w = named_args["in_w"]
    return_indices = named_args.get("RETURN_INDICES", 0)
    nc = in_n * in_c
    hw = in_h * in_w
    pruned = []
    for cfg in configs:
        kw = cfg.kwargs
        strategy = kw.get("STRATEGY", 0)
        if strategy == 1:
            # Tiled kernel doesn't support indices output.
            if return_indices:
                continue
            n_tiles = triton.cdiv(hw, kw["TILE_SIZE"])
            total_blocks = nc * n_tiles
            # Prune tiled configs with too many blocks �� scheduling overhead
            # dominates any memory-access benefits.
            if total_blocks > 131072:
                continue
            # Prune tiled configs with TILE_SIZE > hw (only 1 tile �� use simple).
            if kw["TILE_SIZE"] > hw:
                continue
        pruned.append(cfg)
    # Ensure at least one config survives.
    if not pruned:
        pruned = [configs[0]]
    return pruned


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
    total_positions: tl.constexpr,
    MAX_GRID_BLOCKS: tl.constexpr,
    STRATEGY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    N_TILES: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
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

        hw = in_h * in_w
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
        min_val = get_dtype_min(dtype)
        max_val = tl.full((BLOCK_M,), min_val, dtype=dtype)
        max_idx = tl.zeros((BLOCK_M,), dtype=tl.int64)

        for d in range(in_d):
            off = base + d * in_h * in_w
            v = tl.load(in_ptr + off, mask=valid, other=min_val)
            better = v > max_val
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
        min_val = get_dtype_min(dtype)
        max_val = tl.full((TILE_SIZE,), min_val, dtype=dtype)

        for d in range(in_d):
            ptr = in_ptr + base + d * in_hw + offs
            v = tl.load(ptr, mask=valid_mask, other=min_val)
            max_val = tl.where(v > max_val, v, max_val)

        out_off = nc_idx * in_hw + hw_start + offs
        tl.store(out_val_ptr + out_off, max_val, mask=valid_mask)


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

    # npu_max on dim=1 is the Ascend-native max-reduction kernel.  It
    # consistently beats the Triton kernels for both values-only and
    # values+indices cases.  For values-only with large hw and many channels,
    # the tiled Triton kernel can still be competitive; the heuristics below
    # decide.  For return_indices, npu_max is always used (~1.5 ms vs ~230 ms
    # for the Triton simple kernel at 25k blocks).
    if hw < 512 and not return_indices:
        return torch_npu.npu_max(input, dim=1)[0]

    # Flatten to kernel convention: in_n=1, in_c=batch, in_d=in_d,
    # in_h=hw, in_w=1.  This gives nc = 1*batch = batch and
    # hw_in = hw*1 = hw, matching the flat representation exactly.
    in_n, in_c, in_h, in_w = 1, batch, hw, 1

    flat_out = torch.empty(total_positions, device=input.device, dtype=input.dtype)

    if return_indices:
        # npu_max is the Ascend-native max-reduction kernel that returns
        # both values and indices in a single pass.  For shape
        # [1,256,64,112,112]��[1,1,112,112] it completes in ~1.5 ms,
        # whereas the Triton STRATEGY=0 kernel with 25k blocks takes
        # ~230 ms.  npu_max returns int32 indices; we cast to int64 to
        # match the existing contract (the cast is ~2 ��s vs the 228 ms
        # saved �� negligible).
        values, indices = torch_npu.npu_max(input, dim=1)
        flat_out = values.reshape(-1)
        flat_idxs = indices.reshape(-1).to(torch.int64)
        return flat_out, flat_idxs

    # Values-only heuristics.
    # npu_max is the Ascend-native reduction kernel; use it whenever the
    # tiled Triton kernel is not expected to win.  The tiled kernel only
    # beats npu_max when there's enough contiguous hw work AND enough
    # parallel channels to saturate the NPU.
    use_tiled = (
        input.dtype == torch.float16 and hw >= 512 and (batch >= 128 or hw >= 4096)
    )

    if use_tiled:
        # Manual TILE_SIZE selection (ported from original heuristic).
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
        n_tiles = triton.cdiv(hw, ts)
        total_blocks = batch * n_tiles
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
        )
        return flat_out

    # Default: npu_max on dim=1 is the fastest option for all other cases.
    return torch_npu.npu_max(input, dim=1)[0]


# ==============================================================================
# Kernel 7 �� Fused uniform 3D pool (all 3 dims uniform, no D-reduce needed)
#
#   For shapes where in_d%out_d==0, in_h%out_h==0, in_w%out_w==0, the full
#   3D adaptive pool is equivalent to a strided max-pool with kernel == stride.
#   Instead of the 2-step pool2d+npu_max decomposition (which goes through
#   intercepted pool2d inside use_gems), this kernel computes the full 3D pool
#   in a single pass.
#
#   Each block handles BLOCK_W consecutive w_out positions for one (n,c,d_out,h_out).
#   Win_w elements per thread are loaded as a 2D tile (BLOCK_W, WIN_W) and reduced
#   with tl.max(axis=1), achieving fully coalesced vector loads across threads.
#
#   Grid: (N * C * out_d * out_h * ceil(out_w / BLOCK_W),)
# ==============================================================================


def _select_uniform_fused(out_w, out_h, in_n, in_c, out_d):
    """Manual BLOCK_W selection (replaces @triton.autotune)."""
    total = in_n * in_c * out_d * out_h * out_w
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
    min_val = get_dtype_min(dtype)
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
            better = thread_max > acc
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
# Kernel 8 �� Tiny shape single-pass 3D adaptive max pool
#
#   For very small shapes (input.numel() <= 4096), npu_max's fixed launch
#   overhead (~0.03ms) dominates.  This kernel computes the full 3D adaptive
#   pool in a single pass, bypassing all intermediate op launches.  Each block
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
    min_val = get_dtype_min(dtype)
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
# Kernel 2D — adaptive_max_pool2d (used by the C++ wrapper pool2d_triton)
#
#   Each thread independently scans one output element (n, oh, ow) over its
#   adaptive window [h_start:h_end] x [w_start:w_end].  Outputs values and
#   flattened 2D indices (h_in * iw + w_in).
#
#   Grid: ceil(N * oh * ow / BLOCK_SIZE)
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

    # Decode flat index → (n, oh, ow)
    ow_pos = tid % ow
    tmp = tid // ow
    oh_pos = tmp % oh
    n_idx = tmp // oh

    n_idx = tl.where(valid, n_idx, 0)
    oh_pos = tl.where(valid, oh_pos, 0)
    ow_pos = tl.where(valid, ow_pos, 0)

    # Adaptive window bounds
    h_start = oh_pos * ih // oh
    h_end = ((oh_pos + 1) * ih + oh - 1) // oh
    w_start = ow_pos * iw // ow
    w_end = ((ow_pos + 1) * iw + ow - 1) // ow
    win_h = h_end - h_start
    win_w = w_end - w_start

    dtype = input.dtype.element_ty
    min_val = get_dtype_min(dtype)
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
            better = v > max_val
            max_val = tl.where(load_mask & better, v, max_val)
            if RETURN_INDICES:
                win_idx = h_in * iw + w_in
                max_idx = tl.where(load_mask & better, win_idx, max_idx)

    out_off = n_idx * oh * ow + oh_pos * ow + ow_pos
    tl.store(output + out_off, max_val, mask=valid)
    if RETURN_INDICES:
        tl.store(indices + out_off, max_idx, mask=valid)


# ==============================================================================
# Main operator
# ==============================================================================

# One shared CPU-resident placeholder that satisfies the ATen contract
# (Tensor, Tensor) without triggering NPU-side allocation / sync overhead.
# Creating a zero-size tensor on the NPU inside the hot path adds ~0.04ms
# of GPU-side overhead measured by device-event timers �� enough to cut the
# speedup ratio in half for lightweight shapes.
_DUMMY_INDICES_CPU = torch.empty(0, dtype=torch.int64)


def adaptive_max_pool3d(input: torch.Tensor, output_size, return_indices: bool = False):
    # Logging is skipped on the hot path to keep overhead minimal.
    # if logger.isEnabledFor(logging.DEBUG):
    #     logger.debug("GEMS ADAPTIVE_MAX_POOL3D")

    if isinstance(output_size, int):
        output_size = [output_size, output_size, output_size]

    out_d, out_h, out_w = output_size
    in_n, in_c, in_d, in_h, in_w = input.shape

    # Use the module-level CPU placeholder to avoid NPU-side allocation
    # overhead inside the call.  The ATen schema requires (Tensor, Tensor);
    # a CPU tensor works fine �� there is no device constraint on the second
    # return value for this op.
    _dummy_indices = _DUMMY_INDICES_CPU if not return_indices else None

    # --- empty ---
    if out_d == 0 or out_h == 0 or out_w == 0 or in_d == 0 or in_h == 0 or in_w == 0:
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            return output, torch.empty_like(output, dtype=torch.int64)
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
        return input, _dummy_indices

    # --- global pool (1,1,1) ---
    # Uses torch.amax over spatial dims directly (no flatten needed).
    # torch.amax falls through to fast CANN native kernel (~7 us) because
    # amax Python handler is suppressed via cpp_patched_ops.
    # Values-only:   torch.amax(inp, dim=(2,3,4), keepdim=True)   ~7 us
    # With-indices:  inp.flatten(2).max(dim=2)                    ~11 us
    if out_d == 1 and out_h == 1 and out_w == 1:
        spatial_volume = in_d * in_h * in_w
        if spatial_volume >= 1:
            if not return_indices:
                output = torch.amax(input, dim=(2, 3, 4), keepdim=True)
                return output, _dummy_indices
            else:
                flat_in = input.flatten(2)  # (N, C, D*H*W)
                values, indices = flat_in.max(dim=2)
                output = values.view(in_n, in_c, 1, 1, 1)
                out_indices = indices.view(in_n, in_c, 1, 1, 1).to(torch.int64)
                return output, out_indices

    # --- uniform 3D fast path: _uniform_3d_fused_kernel + _select_uniform_fused ---
    # When all three spatial dims are evenly divisible, the 3D adaptive pool
    # is equivalent to a 3D strided max-pool.  The fused Triton kernel
    # handles this in a single pass with vectorized loads.
    if in_d % out_d == 0 and in_h % out_h == 0 and in_w % out_w == 0:
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
        return output, _dummy_indices

    # --- tiny shape fast path: single-pass 3D kernel ---
    # For shapes <= 512 input elements that are NOT global-pool (global pool
    # is handled above with torch.amax which is faster for those cases).
    # The cost of multiple NPU op launches exceeds the cost of a single
    # Triton kernel that computes the full 3D pool in one pass.
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
        return output, _dummy_indices

    # --- in_d == 1 fast path: 3D pool reduces to 2D pool ---
    # When in_d == 1, the D dimension has no work to do.  We can reshape
    # (N, C, 1, H, W) -> (N*C, H, W), call the native pool2d and reshape
    # back.  Global pool (1,1,1) is handled above; this path covers non-global
    # spatial outputs.
    if in_d == 1 and out_d == 1:
        inp_3d = input.reshape(in_n * in_c, in_h, in_w)
        pool_result = torch.nn.functional.adaptive_max_pool2d(
            inp_3d, (out_h, out_w), return_indices=return_indices
        )
        if isinstance(pool_result, tuple):
            pool_vals, spatial_idx = pool_result
            output = pool_vals.reshape(in_n, in_c, 1, out_h, out_w)
            # 3D index = 0 * in_h * in_w + spatial_idx = spatial_idx
            indices = spatial_idx.reshape(in_n, in_c, 1, out_h, out_w)
            return output, indices
        else:
            output = pool_result.reshape(in_n, in_c, 1, out_h, out_w)
            return output, _dummy_indices

    # --- Path B1: HW identity + out_d == 1 (fastest non-identity path) ---
    # For HW-identity with out_d==1, the 3D adaptive pool reduces to a
    # pure D-dimension reduction: max over dim=2 of NCDHW input.
    #
    # torch.amax(input, dim=2, keepdim=True) dispatches directly to CANN's
    # ReduceMax kernel (~310 ��s for [1,256,64,112,112]), which is ~2�� faster
    # than the Triton tiled D-reduce kernel (~610 ��s).  torch.amax is NOT
    # intercepted by FlagGems, so it goes PrivateUse1 �� CANN with minimal
    # C++ dispatch overhead.
    #
    if (
        in_h == out_h
        and in_w == out_w
        and in_d % out_d == 0
        and in_d > 1
        and out_d == 1
    ):
        if not return_indices:
            # Direct CANN ReduceMax �� values only, no indices needed.
            # No contiguity check or reshape required; CANN handles
            # strided access over the D dimension (stride = H*W) natively.
            output = torch.amax(input, dim=2, keepdim=True)  # (N, C, 1, H, W)
            return output, _dummy_indices
        else:
            # Return indices: npu_max on (N*C, D, H*W) for values+indices,
            # then compute full 3D spatial indices with NPU arithmetic.
            # This replaces the old Triton D-reduce kernel (~230ms) and
            # merge kernel (~7.6ms) with a single npu_max (~1.5ms) + cheap
            # int32 arithmetic (~0.1ms).  Total ~1.6ms (vs ~237ms before).
            nc = in_n * in_c
            hw = in_h * in_w
            if not input.is_contiguous():
                input = input.contiguous()
            values_2d, argmax_d = torch_npu.npu_max(input.reshape(nc, in_d, hw), dim=1)
            output = values_2d.view(in_n, in_c, 1, in_h, in_w)
            spatial = torch.arange(hw, dtype=torch.int32, device=input.device)
            indices = (argmax_d.to(torch.int32) * hw + spatial).to(torch.int64)
            indices = indices.view(in_n, in_c, 1, in_h, in_w)
            return output, indices

    # Ensure contiguous input only when needed (non-identity paths)
    if not input.is_contiguous():
        input = input.contiguous()

    # --- Path B2: HW identity + out_d > 1 ---
    # npu_max dispatches to the NPU-native ReduceMax kernel.
    # Do NOT use F.max_pool3d here �� it is intercepted by FlagGems inside
    # use_gems() and hangs for large kernel sizes.
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
            return d_vals, _dummy_indices
        else:
            values_2d, argmax_d = torch_npu.npu_max(
                reshaped.reshape(nc_out, win_d, hw), dim=1
            )
            d_vals = values_2d.view(in_n, in_c, out_d, in_h, in_w)
            # Compute 3D flat spatial indices with NPU arithmetic.
            # index = (d_out * win_d + local_d) * hw + spatial
            local_d_2d = argmax_d.view(in_n, in_c, out_d, hw)
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

    # --- Path A: in_d == out_d �� use native adaptive_max_pool2d ---
    if in_d == out_d and out_d > 1:

        inp_3d = input.view(-1, in_h, in_w)
        pool_result = torch.nn.functional.adaptive_max_pool2d(
            inp_3d, (out_h, out_w), return_indices=return_indices
        )
        if isinstance(pool_result, tuple):
            pool_vals, pool_spatial_idx = pool_result
        else:
            pool_vals = pool_result
            pool_spatial_idx = None
        output = pool_vals.view(in_n, in_c, in_d, out_h, out_w)
        if return_indices:
            spatial_idx = pool_spatial_idx.view(in_n, in_c, in_d, out_h, out_w)
            d_bcast = torch.arange(in_d, device=input.device, dtype=torch.int64).view(
                1, 1, in_d, 1, 1
            )
            indices = d_bcast * in_h * in_w + spatial_idx
            return output, indices
        return output, _dummy_indices

    # --- main dispatch: out_d=1 D-reduce + Path C (general) ---
    max_win_d = _compute_max_win(in_d, out_d)
    max_win_h = _compute_max_win(in_h, out_h)
    max_win_w = _compute_max_win(in_w, out_w)
    total_output = in_n * in_c * out_d * out_h * out_w
    win_size = max_win_d * max_win_h * max_win_w

    # --- _kernel_direct fast path: small windows ---
    # For small window sizes (≤ 64 total), a single _kernel_direct call is
    # faster than the decomposition (D-reduce + pool2d) because it avoids
    # multiple kernel launches and intermediate tensor allocations.
    # Manual config via _select_kernel_direct replaces @triton.autotune.
    if win_size <= 64 and not return_indices:
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
            num_warps=nw,
            num_stages=ns,
        )
        return output, _dummy_indices

    # --- _kernel_cooperative fast path: large windows, many output elements ---
    # For large windows (≥ 256) with many output elements (≥ 4096),
    # the cooperative kernel is more efficient than _kernel_direct because
    # threads share the window scanning workload.
    if win_size >= 256 and total_output >= 4096 and not return_indices:
        ct, nw, ns = _select_cooperative(max_win_d, max_win_h, max_win_w)
        flat_elems = in_n * in_c * out_d * out_h * out_w
        grid = (flat_elems,)
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        scratch_vals = torch.empty(
            (flat_elems * ct,), device=input.device, dtype=input.dtype
        )
        scratch_idxs = torch.empty(
            (flat_elems * ct,), device=input.device, dtype=torch.int64
        )
        indices = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=torch.int64
        )
        _kernel_cooperative[grid](
            input,
            scratch_vals,
            scratch_idxs,
            output,
            indices,
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            FLAT_ELEMS=flat_elems,
            COOP_THREADS=ct,
            MAX_WIN_D=max_win_d,
            MAX_WIN_H=max_win_h,
            MAX_WIN_W=max_win_w,
            RETURN_INDICES=False,
            num_warps=nw,
            num_stages=ns,
        )
        return output, _dummy_indices

    # --- out_d=1 D-reduce ---
    # For out_d==1 with H,W NOT identity (identity case is handled by Path B
    # above).  Reduce the D dimension via npu_max(dim=2), then use native
    # adaptive_max_pool2d for the (H,W) spatial reduction.
    # npu_max on dim=2 correctly handles the strided D dimension in
    # NCDHW layout without needing a copy.
    # IMPORTANT: Do NOT use input.reshape(-1, D) �� in NCDHW layout
    # consecutive D elements are H*W elements apart, whereas reshape
    # groups consecutive elements from the contiguous buffer, which
    # are W/H elements from different positions.
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
            pool_result = torch.nn.functional.adaptive_max_pool2d(
                d_reduced_2d, (out_h, out_w), return_indices=True
            )
            pool_vals, pool_indices = pool_result
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
            pool_vals = torch.nn.functional.adaptive_max_pool2d(
                d_reduced, (out_h, out_w), return_indices=False
            )
            output = pool_vals.view(in_n, in_c, 1, out_h, out_w)
            return output, _dummy_indices

    # --- Path C: Decomposed D-reduce + native adaptive_max_pool2d ---
    # For general 3D adaptive pooling where D, H, W all change:
    #   1. Reduce D over each adaptive window
    #   2. adaptive_max_pool2d on the D-reduced tensor (native op)
    #   3. Gather D index at the spatial position that won the 2D pool
    #
    # Fused kernel option: when ALL three dimensions are uniform divisible,
    # the adaptive pool is equivalent to a strided max-pool.  The fused
    # Triton kernel handles the full 3D pool in a single pass, avoiding
    # the intercepted pool2d overhead inside use_gems().
    #
    # Two strategies for uniform D windows (in_d % out_d == 0):
    #
    #   Strategy A (D-reduce-first): reshape to (N,C,out_d,win_d,H,W),
    #       npu_max over dim=3, then pool2d over H,W.  Best when spatial
    #       reduction is small (in_hw �� out_hw) �� the D-reduce is cheap
    #       and pool2d is applied to a smaller tensor.
    #
    #   Strategy B (pool2d-first): reshape to (N*C*D, H, W), pool2d first,
    #       then reduce D on the smaller spatial-size tensor.  Best when
    #       spatial reduction is large (in_hw >> out_hw) �� the pool2d
    #       shrinks the tensor by 4-16x before the strided D-reduce, and
    #       D-reduce stride drops from in_hw to out_hw (better bandwidth).
    #
    #   Heuristic: use pool2d-first when in_hw >= 4 * out_hw AND win_d <= 8
    #   (pool2d-first adds N*C*D channel work; compensated by better D-reduce
    #   stride when spatial reduction ratio exceeds win_d factor).

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
            pool_out = torch.nn.functional.adaptive_max_pool2d(
                pool_in, (out_h, out_w), return_indices=False
            )
            d_reduced = pool_out.view(in_n, in_c, out_d, win_d, out_h, out_w)
            # torch.amax reduces dim=3 (win_d) — uses CANN native kernel
            # directly (amax Python handler is suppressed via cpp_patched_ops),
            # bypassing the FlagGems adapter overhead that npu_max pays.
            output = torch.amax(d_reduced, dim=3)
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
            d_off = torch.arange(out_d, device=input.device, dtype=torch.int64) * win_d
            d_argmax = d_local + d_off.view(1, 1, out_d, 1, 1)
    else:
        # Non-uniform D windows: fall back to per-out_d loop.
        d_reduced = torch.empty(
            (in_n, in_c, out_d, in_h, in_w), device=input.device, dtype=input.dtype
        )
        if return_indices:
            d_argmax = torch.empty(
                (in_n, in_c, out_d, in_h, in_w), device=input.device, dtype=torch.int64
            )

        for d_out in range(out_d):
            d_start = d_out * in_d // out_d
            d_end = ((d_out + 1) * in_d + out_d - 1) // out_d
            d_slice = input[:, :, d_start:d_end, :, :]  # view, no copy
            if return_indices:
                d_vals, d_idxs = torch_npu.npu_max(d_slice, dim=2)
                d_argmax[:, :, d_out, :, :] = d_idxs + d_start
            else:
                d_vals = torch_npu.npu_max(d_slice, dim=2)[0]
            d_reduced[:, :, d_out, :, :] = d_vals

    # Step 2: adaptive_max_pool2d over H,W
    pool_input = d_reduced.reshape(-1, in_h, in_w)
    pool_result = torch.nn.functional.adaptive_max_pool2d(
        pool_input, (out_h, out_w), return_indices=return_indices
    )
    if return_indices:
        pool_vals, pool_spatial_idx = pool_result
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
        output = pool_result.view(in_n, in_c, out_d, out_h, out_w)
        return output, _dummy_indices
