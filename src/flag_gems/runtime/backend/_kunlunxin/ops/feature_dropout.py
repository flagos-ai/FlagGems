import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# Fused feature-dropout kernels (kunlunxin / XPU override).
#
# The generic ops/feature_dropout.py uses TWO non-@libentry kernels (recompiled
# per shape -> IR explosion, source loc feature_dropout.py:17):
#   1. generate_feature_mask_kernel: materializes an (N, C) mask tensor.
#   2. apply_feature_mask_kernel: 1D grid over flat numel, per element computes
#      n = i // (C*spatial), c = (i % (C*spatial)) // spatial and GATHERS
#      mask[n*C + c] (integer div/mod, no HW divider on XPU + discrete gather).
#
# We split on the spatial size because the two regimes are fundamentally
# different work:
#   * spatial == 1 (2D input): feature dropout degenerates to ELEMENTWISE
#     dropout (each (n, c) is its own channel with its own random). Mirror the
#     proven kunlunxin dropout_forward pattern: wide 1D blocks, UNROLL=8, inline
#     philox, fused multiply, NO mask materialization (the generic path wastes a
#     full (N,C)-sized 2.6GB mask roundtrip here). @libentry -> compiles once.
#   * spatial > 1: feature dropout keeps/drops an ENTIRE channel, so the mask is
#     constant across the whole contiguous spatial run of a channel. The XPU
#     cannot do fast 2D runtime-addressed tiles (`ch[:,None]*spatial + s` defeats
#     OffsetAnalysis -> per-element discrete access, ~50x slower than flat 1D),
#     and tl.reshape/broadcast of the small per-channel m vector hits uni_sram
#     OutOfResources. So we stay on a WIDE FLAT 1D tile and pack the per-channel
#     keep/drop decisions of the (<= 32) channels spanned by the tile into a
#     SINGLE int32 bitmask: one philox draw over BLOCK_K lanes, `bits =
#     sum(keep << kk)`, then every element extracts its channel's bit with a
#     variable shift `bit = (bits >> local) & 1`. `local = channel(off) -
#     channel(tile_start)` is computed WITHOUT integer division:
#       - power-of-two spatial: exact right shifts (USE_SHIFT path);
#       - otherwise: exact magic-number multiplication ((n*M)>>K, verified
#         exactness for the covered range) on int64 for the tile base scalar and
#         int32 for the per-element delta.
#     A tile boundary mask (NEED_MASK) is a constexpr so aligned shapes keep the
#     fully unmasked fast load/store. Exotic shapes where no exact magic exists
#     (or the tile would span > 32 channels) fall back to the legacy 2D kernel.

UNROLL = 4


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_elementwise_bulk_kernel(
    X,
    Y,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Bulk kernel: every program's 4 sub-stores are FULLY in-bounds (the
    # launcher only grids over the aligned region n_full = floor(N/TILE)*TILE),
    # so there is NO store mask. This dodges the XPU masked-store legalization
    # bug where a program containing ANY fully-masked sub-store silently skips
    # ALL of its (even the valid) stores -> dropped tail.
    # 4 sub-stores of BLOCK (tile 4096 @ BLOCK=1024) is the measured sweet spot:
    # with 8 sub-vectors the fp32/bf16 paths run 2.2-2.6x slower than fp16
    # (register liveness / scheduling), tile 4096 puts all three dtypes on the
    # same philox-bound rate (~5G el/s).
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    cc = c0 + i4
    _O = cc * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, cc, c1, _O, _O)
    m0 = uint_to_uniform_float(r0) > p
    m1 = uint_to_uniform_float(r1) > p
    m2 = uint_to_uniform_float(r2) > p
    m3 = uint_to_uniform_float(r3) > p

    x0 = tl.load(X + i4)
    x1 = tl.load(X + i4 + BLOCK)
    x2 = tl.load(X + i4 + 2 * BLOCK)
    x3 = tl.load(X + i4 + 3 * BLOCK)

    tl.store(Y + i4, tl.where(m0, x0 * scale, 0.0).to(Y.dtype.element_ty))
    tl.store(Y + i4 + BLOCK, tl.where(m1, x1 * scale, 0.0).to(Y.dtype.element_ty))
    tl.store(
        Y + i4 + 2 * BLOCK, tl.where(m2, x2 * scale, 0.0).to(Y.dtype.element_ty)
    )
    tl.store(
        Y + i4 + 3 * BLOCK, tl.where(m3, x3 * scale, 0.0).to(Y.dtype.element_ty)
    )


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_elementwise_tail_kernel(
    X,
    Y,
    base,  # first flat index handled by the tail (n_full)
    N,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Tail kernel: handles the [n_full, N) remainder (< TILE elements) with a
    # SINGLE store per program. A single store with a partial (< full) mask is
    # correct on XPU (same pattern as dropout_backward / the channel kernel);
    # the bug only bites the multi-sub-store bulk path, hence the split.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0v = c0 + off.to(tl.uint32)
    _O = c0v * 0
    r0, _, _, _ = tl.philox(philox_seed, c0v, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    m = r0 > p

    mask = off < N
    x = tl.load(X + off, mask=mask, other=0.0)
    y = tl.where(m, x * scale, 0.0)
    tl.store(Y + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_channel_kernel(
    X,
    Y,
    NC,  # N * C (total number of channels)
    spatial,  # product of spatial dims (H*W*...)
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    ch = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # [BLOCK_C]
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    ch_valid = ch < NC
    s_valid = s < spatial

    # Per-channel philox random (deterministic in ch -> constant across spatial).
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = c0 + ch.to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand_vals = uint_to_uniform_float(r0)  # [BLOCK_C]
    m = tl.where(rand_vals > p, scale, 0.0)  # [BLOCK_C]

    # [BLOCK_C, BLOCK_S] contiguous tile: inner spatial axis is stride-1.
    offset = ch[:, None] * spatial + s[None, :]
    tile_mask = ch_valid[:, None] & s_valid[None, :]
    x = tl.load(X + offset, mask=tile_mask, other=0.0)
    y = x * m[:, None]
    tl.store(Y + offset, y, mask=tile_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "numel",
        "spatial",
        "magic_m",
        "magic_k",
        "dm_m",
        "dm_k",
        "p",
        "scale",
        "philox_seed",
        "philox_offset",
    ]
)
def _fd_channel_flat_kernel(
    X,
    Y,
    numel,
    spatial,
    magic_m,  # int64 magic: channel(start) = (start * magic_m) >> magic_k
    magic_k,
    dm_m,  # int32 magic: local = ((delta + r) * dm_m) >> dm_k
    dm_k,
    p,
    scale,
    philox_seed,
    philox_offset,
    TILE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG_BS: tl.constexpr,
    USE_SHIFT: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Flat 1D wide tile; per-channel keep/drop decisions bit-packed into one
    # int32 (BLOCK_K <= 32 channels spanned by the tile). Every element then
    # extracts its channel's bit with a variable shift -> zero integer
    # division, zero 2D addressing, fully coalesced load/store.
    pid = tl.program_id(0)
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    start = pid * TILE  # int32 (numel < 2^31, same assumption as the other kernels)
    off = start + tl.arange(0, TILE)

    if USE_SHIFT:
        # Power-of-two spatial: right shifts are exact floor division.
        ch0 = start >> LOG_BS
        local = tl.minimum((off >> LOG_BS) - ch0, 31)
        ch0 = ch0.to(tl.int64)
    else:
        ch0 = (start.to(tl.int64) * magic_m) >> magic_k
        r = (start - ch0 * spatial).to(tl.int32)  # start % spatial
        # channel(off) - channel(start) = floor((r + delta) / spatial), delta < TILE
        local = tl.minimum(((tl.arange(0, TILE) + r) * dm_m) >> dm_k, 31)

    kk = tl.arange(0, BLOCK_K)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = c0 + (ch0 + kk).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    keep = (uint_to_uniform_float(r0) > p).to(tl.int32)  # [BLOCK_K]
    bits = tl.sum(keep << kk, axis=0)  # one int32, bit kk = channel ch0+kk
    bit = (bits >> local) & 1
    m = bit.to(tl.float32) * scale

    if NEED_MASK:
        mask = off < numel
        x = tl.load(X + off, mask=mask, other=0.0)
        tl.store(Y + off, (x * m).to(Y.dtype.element_ty), mask=mask)
    else:
        x = tl.load(X + off)
        tl.store(Y + off, (x * m).to(Y.dtype.element_ty))


def _elementwise_launch_config(N):
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    else:
        return 1024, 16


def _tail_block(n_tail):
    b = triton.next_power_of_2(n_tail)
    if b < 64:
        b = 64
    if b > 1024:
        b = 1024
    return b


def _magic_div(spatial, max_n, k_hi, k_lo, m_limit):
    # Exact floor-division magic numbers: floor(n / spatial) == (n * M) >> K
    # for every 0 <= n <= max_n (classic round-up magic; exactness guaranteed by
    # M*spatial - 2^K >= 0 and max_n * (M*spatial - 2^K) < 2^K).
    for k in range(k_hi, k_lo - 1, -1):
        m = ((1 << k) // spatial) + 1
        if m * max_n >= m_limit:
            continue
        e = m * spatial - (1 << k)
        if e >= 0 and max_n * e < (1 << k):
            return m, k
    return None


def _channel_config(spatial, numel):
    # Flat-bits channel kernel config. The tile is halved until it spans at
    # most 31 channel boundaries, so the packed bitmask fits ONE int32
    # (BLOCK_K <= 32). Returns None when no exact magic exists for the spatial
    # size -> caller falls back to the legacy 2D kernel.
    tile = 8192
    while tile > 64 and triton.cdiv(tile, spatial) + 1 > 32:
        tile //= 2
    block_k = triton.next_power_of_2(triton.cdiv(tile, spatial) + 1)
    if block_k > 32:
        return None
    use_shift = (spatial & (spatial - 1)) == 0
    magic_m = magic_k = dm_m = dm_k = 0
    if not use_shift:
        m1 = _magic_div(spatial, numel + tile, 62, 32, 1 << 63)
        if m1 is None:
            return None
        m2 = _magic_div(spatial, tile + spatial, 31, 15, 1 << 31)
        if m2 is None:
            return None
        magic_m, magic_k = m1
        dm_m, dm_k = m2
    need_mask = (numel % tile) != 0
    num_warps = 8 if tile >= 4096 else 4
    return (tile, block_k, use_shift, need_mask, magic_m, magic_k, dm_m, dm_k, num_warps)


def _channel_config_2d(spatial):
    # Legacy 2D-tile config, kept as the fallback for exotic spatial sizes.
    # Size BLOCK_S to cover the whole spatial run in one block whenever feasible
    # (so the inline philox draw happens once per channel-block, not once per
    # spatial-tile). Target a ~8192-element tile with the inner (stride-1)
    # spatial axis wide for good block DMA.
    bs = triton.next_power_of_2(spatial)
    if bs > 2048:
        bs = 2048
    bc = max(1, 8192 // bs)
    return bc, bs, 16


def _feature_dropout_impl(input, out, p):
    device = input.device
    N = input.shape[0]
    C = input.shape[1]
    NC = N * C
    spatial = 1
    for i in range(2, input.ndim):
        spatial *= input.shape[i]
    scale = 1.0 / (1.0 - p)

    with torch_device_fn.device(device):
        if spatial == 1:
            # Elementwise regime (2D input): each (n, c) is its own channel.
            numel = NC
            block, num_warps = _elementwise_launch_config(numel)
            tile = block * UNROLL
            n_full = (numel // tile) * tile
            increment = triton.cdiv(numel, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            if n_full > 0:
                grid = (n_full // tile,)
                _fd_elementwise_bulk_kernel[grid](
                    input,
                    out,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    BLOCK=block,
                    num_warps=num_warps,
                )
            n_tail = numel - n_full
            if n_tail > 0:
                tblock = _tail_block(n_tail)
                tgrid = (triton.cdiv(n_tail, tblock),)
                _fd_elementwise_tail_kernel[tgrid](
                    input,
                    out,
                    n_full,
                    numel,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    BLOCK=tblock,
                    num_warps=4,
                )
        else:
            numel = NC * spatial
            # NC randoms consumed (one philox draw per channel).
            increment = triton.cdiv(NC, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            cfg = _channel_config(spatial, numel)
            if cfg is None:
                # No exact magic for this spatial size -> legacy 2D tile kernel.
                block_c, block_s, num_warps = _channel_config_2d(spatial)
                grid = (triton.cdiv(NC, block_c), triton.cdiv(spatial, block_s))
                _fd_channel_kernel[grid](
                    input,
                    out,
                    NC,
                    spatial,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    BLOCK_C=block_c,
                    BLOCK_S=block_s,
                    num_warps=num_warps,
                )
            else:
                (
                    tile,
                    block_k,
                    use_shift,
                    need_mask,
                    magic_m,
                    magic_k,
                    dm_m,
                    dm_k,
                    num_warps,
                ) = cfg
                grid = (triton.cdiv(numel, tile),)
                _fd_channel_flat_kernel[grid](
                    input,
                    out,
                    numel,
                    spatial,
                    magic_m,
                    magic_k,
                    dm_m,
                    dm_k,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    TILE=tile,
                    BLOCK_K=block_k,
                    LOG_BS=spatial.bit_length() - 1,
                    USE_SHIFT=use_shift,
                    NEED_MASK=need_mask,
                    num_warps=num_warps,
                )
    return out


def feature_dropout(input, p, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_DROPOUT")

    if not train or p == 0:
        return input.clone()
    if p == 1:
        return torch.zeros_like(input)
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    input = input.contiguous()
    out = torch.empty_like(input)
    return _feature_dropout_impl(input, out, p)


def feature_dropout_(input, p, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_DROPOUT_")

    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    # Each element is read and written at the same offset -> safe in-place; write
    # directly into `input` and skip the extra output buffer + copy.
    input = input.contiguous()
    _feature_dropout_impl(input, input, p)
    return input
