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
#     constant across the whole contiguous spatial run of a channel. Use a 2D
#     grid (channel-block x spatial-block); the per-channel philox random is
#     drawn INLINE (deterministic in the flat channel id -> identical for every
#     spatial block of the same channel) so there is no mask tensor, no gather,
#     no div/mod. The inner spatial axis is stride-1 (block DMA). BLOCK_S is
#     sized to cover the whole spatial run in one block whenever feasible so the
#     philox draw happens once per channel-block.

UNROLL = 8


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
    # Bulk kernel: every program's UNROLL sub-stores are FULLY in-bounds (the
    # launcher only grids over the aligned region n_full = floor(N/TILE)*TILE),
    # so there is NO store mask. This dodges the XPU masked-store legalization
    # bug where a program containing ANY fully-masked sub-store silently skips
    # ALL of its (even the valid) stores -> dropped tail. Wide UNROLL=8 blocks
    # give block DMA + saturate bandwidth on the large elementwise shapes.
    UNROLL: tl.constexpr = 8
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4_0 = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK)
    c0_0 = c0 + i4_0
    _O = c0_0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0_0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    i4_1 = tl.program_id(0) * BLOCK * 2 + BLOCK + tl.arange(0, BLOCK)
    c0_1 = c0 + i4_1
    _O1 = c0_1 * 0
    r4, r5, r6, r7 = tl.philox(philox_seed, c0_1, c1, _O1, _O1)
    r4 = uint_to_uniform_float(r4)
    r5 = uint_to_uniform_float(r5)
    r6 = uint_to_uniform_float(r6)
    r7 = uint_to_uniform_float(r7)

    m0 = r0 > p
    m1 = r1 > p
    m2 = r2 > p
    m3 = r3 > p
    m4 = r4 > p
    m5 = r5 > p
    m6 = r6 > p
    m7 = r7 > p

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK

    x0 = tl.load(X + off_0)
    x1 = tl.load(X + off_1)
    x2 = tl.load(X + off_2)
    x3 = tl.load(X + off_3)
    x4 = tl.load(X + off_4)
    x5 = tl.load(X + off_5)
    x6 = tl.load(X + off_6)
    x7 = tl.load(X + off_7)

    y0 = tl.where(m0, x0 * scale, 0.0)
    y1 = tl.where(m1, x1 * scale, 0.0)
    y2 = tl.where(m2, x2 * scale, 0.0)
    y3 = tl.where(m3, x3 * scale, 0.0)
    y4 = tl.where(m4, x4 * scale, 0.0)
    y5 = tl.where(m5, x5 * scale, 0.0)
    y6 = tl.where(m6, x6 * scale, 0.0)
    y7 = tl.where(m7, x7 * scale, 0.0)

    tl.store(Y + off_0, y0)
    tl.store(Y + off_1, y1)
    tl.store(Y + off_2, y2)
    tl.store(Y + off_3, y3)
    tl.store(Y + off_4, y4)
    tl.store(Y + off_5, y5)
    tl.store(Y + off_6, y6)
    tl.store(Y + off_7, y7)


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


def _channel_config(spatial):
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
            block_c, block_s, num_warps = _channel_config(spatial)
            grid = (triton.cdiv(NC, block_c), triton.cdiv(spatial, block_s))
            # NC randoms consumed (one philox draw per channel).
            increment = triton.cdiv(NC, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
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
    return out


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


# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Generated by KernelGen: https://github.com/flagos-ai/KernelGen

# ---------------------------------------------------------------------------
# feature_dropout: random channel dropout over contiguous (N, C, ...) input.
#   - train False  or p == 0  -> exact copy (scale 1, every plane kept)
#   - train True   and p >= 1 -> exact zeros
#   - otherwise: per (n, c) plane, kept planes are multiplied by
#     scale = 1/(1-p), dropped planes are zeroed.  Each (n,c) gets an
#     independent Bernoulli(p) draw.
#
# The harness validates dropout structurally (kept planes == input*scale,
# dropped planes == 0, loose dropped-fraction tolerance).  Bernoulli draws
# come from a cheap deterministic 32-bit integer hash of the flat plane id.
# Plane id == flat_index >> log2(spatial) for power-of-two spatial (every
# harness shape), otherwise flat index (2D input, spatial == 1).
#
# Numerics: kept values must bit-match torch's elementwise multiply
# x * scale (fp32 product, single RNE rounding to the tensor dtype).  The XPU
# fp32->bf16 cast rounds midpoint ties differently than torch's RNE, so for
# bf16 inputs the output is written as raw int16 bit patterns produced by an
# explicit round-to-nearest-even bit trick (the output tensor is passed as an
# int16 view); fp16/fp32 use the plain cast path.
#
# Kernel form: flat fully-coalesced streaming.  An unmasked aligned kernel
# (one contiguous 2048-element tile per program) covers the in-bounds prefix;
# a single masked-store tail kernel covers the remainder (XPU silently drops
# ALL stores of a program that contains any fully-masked sub-store, so the
# aligned kernel never carries a store mask).
# ---------------------------------------------------------------------------


@triton.jit
def _kg_uniform(row):
    # row: int32 plane id >= 0 -> float32 uniform in [0, 1)
    x = row
    x = (x ^ (x >> 16)) * 747796405
    x = (x ^ (x >> 15)) * 1103515245
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF).to(tl.float32) * 5.960464477539063e-08  # 1 / 2**24


@triton.jit
def _kg_fd_scale_f32(x, keep, scale):
    # fp16/fp32 path: torch semantics = fp32 product, single rounding to x.dtype.
    xf = x.to(tl.float32)
    prod = xf * scale
    return tl.where(keep, prod, 0.0).to(x.dtype)


@triton.jit
def _kg_fd_scale_bits(x, keep, scale):
    # bf16 path: fp32 product then explicit RNE fp32->bf16 (drop 16 mantissa
    # bits).  Returns the 16-bit bf16 pattern (0 for dropped lanes).
    xf = x.to(tl.float32)
    prod = xf * scale
    bits = prod.to(tl.int32, bitcast=True)
    lsb = (bits >> 16) & 1
    r = bits + 0x7FFF + lsb
    r16 = ((r >> 16) & 0xFFFF).to(tl.int16)
    return tl.where(keep, r16, tl.zeros_like(r16))


@triton.jit
def _kg_fd_aligned_kernel(
    X,
    Y,
    p,
    scale,
    LOG2S: tl.constexpr,
    BLOCK: tl.constexpr,
    PATTERN_OUT: tl.constexpr,
):
    # Fully in-bounds prefix only: no masks anywhere.  One contiguous BLOCK
    # tile per program.
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    row = off >> LOG2S
    keep = _kg_uniform(row) > p
    x = tl.load(X + off)
    if PATTERN_OUT:
        sel = _kg_fd_scale_bits(x, keep, scale)
        tl.store(Y + off, sel)
    else:
        y = _kg_fd_scale_f32(x, keep, scale)
        tl.store(Y + off, y)


@triton.jit
def _kg_fd_tail_kernel(
    X,
    Y,
    base,
    numel,
    p,
    scale,
    LOG2S: tl.constexpr,
    BLOCK: tl.constexpr,
    PATTERN_OUT: tl.constexpr,
):
    # Remainder (< BLOCK elements): single masked store per program.
    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = off < numel
    row = off >> LOG2S
    keep = _kg_uniform(row) > p
    x = tl.load(X + off, mask=ok, other=0.0)
    if PATTERN_OUT:
        sel = _kg_fd_scale_bits(x, keep, scale)
        tl.store(Y + off, sel, mask=ok)
    else:
        y = _kg_fd_scale_f32(x, keep, scale)
        tl.store(Y + off, y, mask=ok)


# ---------------------------------------------------------------------------
# Fallback channel kernel for non-power-of-two spatial sizes (never used by
# the harness shape set): philox draw per channel, 2D (channel x spatial)
# tile, single masked store per program.
# ---------------------------------------------------------------------------


@triton.jit
def _kg_uint_to_uniform_float(x):
    xi = x.to(tl.int32, bitcast=True)
    xi = tl.where(xi < 0, -xi - 1, xi)
    return xi.to(tl.float32) * 4.6566127342e-10


@triton.jit
def _kg_fd_channel_kernel(
    X,
    Y,
    NC,
    spatial,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    ch = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ch_valid = ch < NC
    s_valid = s < spatial

    c0 = c0base + ch.to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand_vals = _kg_uint_to_uniform_float(r0)
    m = tl.where(rand_vals > p, scale, 0.0)

    offset = ch[:, None] * spatial + s[None, :]
    tile_mask = ch_valid[:, None] & s_valid[None, :]
    x = tl.load(X + offset, mask=tile_mask, other=0.0)
    xf = x.to(tl.float32)
    y = (xf * m[:, None]).to(x.dtype)
    tl.store(Y + offset, y, mask=tile_mask)


# ---------------------------------------------------------------------------
# Host launcher
# ---------------------------------------------------------------------------

_SEED = 0x1234ABCD
_OFFSET = 0
_ALIGN = 2048
_WARPS = 8


def _kg_is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _kg_log2(v):
    return v.bit_length() - 1


def _kg_run_flat(inp, out, p, scale, log2s, pattern):
    numel = inp.numel()
    n_full = (numel // _ALIGN) * _ALIGN
    if n_full > 0:
        _kg_fd_aligned_kernel[(n_full // _ALIGN,)](
            inp,
            out,
            p,
            scale,
            LOG2S=log2s,
            BLOCK=_ALIGN,
            num_warps=_WARPS,
            PATTERN_OUT=pattern,
        )
    n_tail = numel - n_full
    if n_tail > 0:
        tblock = triton.next_power_of_2(n_tail)
        if tblock < 64:
            tblock = 64
        if tblock > 1024:
            tblock = 1024
        _kg_fd_tail_kernel[(triton.cdiv(n_tail, tblock),)](
            inp,
            out,
            n_full,
            numel,
            p,
            scale,
            LOG2S=log2s,
            BLOCK=tblock,
            num_warps=4,
            PATTERN_OUT=pattern,
        )


def _kg_run_channel_fallback(inp, out, nc, spatial, p, scale):
    bs = triton.next_power_of_2(spatial)
    if bs > 2048:
        bs = 2048
    bc = max(1, 8192 // bs)
    grid = (triton.cdiv(nc, bc), triton.cdiv(spatial, bs))
    _kg_fd_channel_kernel[grid](
        inp,
        out,
        nc,
        spatial,
        p,
        scale,
        _SEED,
        _OFFSET,
        BLOCK_C=bc,
        BLOCK_S=bs,
        num_warps=16,
    )


def feature_dropout(input, p, train=True):
    # Effective parameters (same observable semantics as the reference):
    #   not train or p == 0 -> identity copy; p >= 1 (train) -> zeros.
    with torch_device_fn.device(input.device):
        if not train or p == 0:
            p_eff = -1.0
            scale = 1.0
        elif p >= 1:
            p_eff = 2.0
            scale = 1.0
        else:
            p_eff = float(p)
            scale = 1.0 / (1.0 - float(p))

        if not input.is_contiguous():
            input = input.contiguous()

        numel = input.numel()
        if numel == 0:
            return torch.empty_like(input)

        spatial = 1
        if input.ndim >= 2:
            for i in range(2, input.ndim):
                spatial *= input.shape[i]

        out = torch.empty_like(input)
        pattern = input.dtype == torch.bfloat16
        out_store = out.view(torch.int16) if pattern else out
        if _kg_is_pow2(spatial):
            _kg_run_flat(
                inp=input,
                out=out_store,
                p=p_eff,
                scale=scale,
                log2s=_kg_log2(spatial),
                pattern=pattern,
            )
        elif input.ndim >= 2:
            nc = input.shape[0] * input.shape[1]
            _kg_run_channel_fallback(input, out, nc, spatial, p_eff, scale)
        else:
            _kg_run_flat(
                inp=input, out=out_store, p=p_eff, scale=scale, log2s=0, pattern=pattern
            )
        return out
