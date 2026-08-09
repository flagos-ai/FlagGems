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

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cdiv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        # Pre-offset the base pointers so the inner `ptr + n_offsets` access is a
        # scalar-base + stride-1 arange that OffsetAnalysis proves contiguous
        # (block DMA). The old inline `pid_m * N + n_offsets` addressing blocked
        # the analysis -> discrete scalar gather (~1-3 GB/s, e.g. [4096,4096] took
        # ~37ms). Pre-offsetting drops it to ~1.1ms (~35x).
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        tl.store(output_ptr + n_offsets, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            # it is possible that there are -inf's in the input
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        # Normalize pass. Iterate ASCENDING so each `input_ptr + n_offsets` load
        # and `output_ptr + n_offsets` store is a scalar-base + stride-1 arange
        # (block DMA). The old code walked the tiles DESCENDING
        # (`previous_multiple - start_n`) as a cache-locality trick, but on this
        # XPU the backward walk defeats OffsetAnalysis/prefetch -> discrete access
        # (~1-3 GB/s: [1024,65536] took ~154ms). Ascending drops it to ~4ms (~35x).
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)


# ------------------------  forward: XPU multirow 2D fast path (N <= 4096) --
#
# One program per [TILE_M, N] tile (up to 32768 lanes; block DMA, packing
# several rows per program removes the per-row launch overhead of the
# grid=(M,) fallback). Mirrors the validated softmax_backward_kernel_multirow
# design (measured on XPU: [1024,256] 0.87ms -> 0.02ms, [64,512,512]
# 19.5ms -> 0.83ms vs the per-row kernel).
#
# Correctness guards (XPU backend quirks, see HARNESS_SUMMARY):
#   * non-power-of-2 N silently miscompiles in the wide tile for a large
#     family of widths (odd N >= 17 and many N % 4 == 2; fp32/fp16 alike) ->
#     the multirow path is host-gated to pow2 N for ALL dtypes; non-pow2 falls
#     back to the per-row masked kernel (exact for any N).
#   * masked rows (M % TILE_M != 0) use the old per-row kernel so the
#     masked-load "other" semantics never appear in the multirow tile.
#   * TILE_M * N <= 8192 keeps tl.sum inside the XPU exact window; lanes
#     > 8192 (N=4096, TILE_M=8 -> 32768 lanes) need buffer_size_limit=2048
#     and were verified numerically (HARNESS_SUMMARY 2.5, safe_softmax C).

_SM_MR_MAX_N = 4096  # largest N handled by the 2D multirow tile
# TILE_M per N bucket; lanes = TILE_M * N (<= 32768 with buffer_size_limit).
# N=4096 bucket raised 2 -> 8 after A/B on XPU: [4096,4096] 1.21->0.57ms,
# [1024,4096] 0.31->0.15ms (fp16), numerics exact.
_SM_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 8), (2048, 4), (4096, 8)]


@triton.jit
def softmax_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    # Single-pass [TILE_M, N] tile (unmasked: M % TILE_M == 0 checked on host).
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    inp = tl.load(input_ptr + off).to(output_ptr.dtype.element_ty)
    m = tl.max(inp, 1)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = e / z[:, None]
    tl.store(output_ptr + off, out)


# -----------------------  forward: N > 32768 chunk split ---------------------
# The per-row two-pass kernel serializes each row in one program; wide rows
# (N > 32768) therefore reach only ~50-60 GB/s on XPU. Split every row into
# 8192-wide chunks computed by independent programs (flat pid*BN offsets ->
# contiguous block DMA), combine per-row (m, z) partials, then a final pass
# re-reads the chunks and writes exp(x - m) / z. Mirrors the validated
# log_softmax _fwd_chunk_split (08-19/20) structure:
#   * full 8192-wide chunks are unmasked (masked wide tiles miscompile);
#   * a non-8192 tail is decomposed into exact power-of-2 unmasked pieces
#     (<= 4096 lanes) plus a masked 64-lane remainder (masked reduces are
#     exact up to that width on this backend).
_SM_CHUNK_BN = 8192  # wide-chunk width for the split path
_SM_TAIL_PIECE = 4096  # largest unmasked pow2 tail piece


def _sm_pow2_tail_pieces(n, cap=_SM_TAIL_PIECE):
    """Split a row tail into (pieces, 64-lane remainder) - see log_softmax."""
    r = n % 64
    m = n - r
    pieces = []
    while m > 0:
        p = 1 << (m.bit_length() - 1)
        while p > cap:
            p >>= 1
        pieces.append(p)
        m -= p
    return pieces, r


@libentry()
@triton.jit
def softmax_kernel_chunk(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    # Flat (row * C_FULL + c) grid; offsets = pid * BN -> contiguous block DMA.
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


# The partial merge tile MUST be fully unmasked.  Probed on XPU 7 (2026-08-29,
# harness/results/functional/softmax_chunk_combine_xpu7_20260829, 264 configs
# against a CPU float64 reference): the previous form - a single
# `tl.arange(0, next_pow2(C))` tile read with
# `tl.load(..., mask=c < C, other=-inf/0.0)` - is silently wrong in 131/264
# configs (relative error up to 1.9e-1 on m+log(z) and 3.4e0 on z), which
# reaches production `softmax` / `softmax_out` as up to 6.8e-2 relative error on
# every element of a row (fp32 float64-reference row sum 1.068 instead of 1.0):
#   * C == C2 == 1024, i.e. the mask is identically true, fails 6/6 seeds;
#     dropping *only* the `other=` argument makes it exact, so `other=` pollutes
#     valid lanes at fp32 tile width 1024 (known XPU masked-memory defect);
#   * C < C2 (a real masked tail) is value-dependently wrong for every
#     C2 >= 128, with or without `other=`.
# Caller-side padding to the merge identity (m=-inf, z=0) plus fully unmasked
# groups of <= 1024 lanes had 0/264 failures (max 1.1e-7) for group widths
# 512/1024/2048 and for the degenerate single-group case.
_SM_COMBINE_GW = 1024  # hard cap on the merge tile width


def _sm_combine_geometry(C):
    """(group width, group count, padded per-row stride) for the partial merge.

    The width is clamped to [64, 1024]: tiles of <= 32 lanes miscompile on this
    XPU, a single tile spanning all partials is untrustworthy, and 64 keeps
    every padded row a whole number of 64-element store units.  The padded
    stride stays <= max(64, 2 * C), so the partial buffers grow at most 2x.
    """
    gw = min(max(64, triton.next_power_of_2(C)), _SM_COMBINE_GW)
    ng = triton.cdiv(C, gw)
    return gw, ng, ng * gw


@libentry()
@triton.jit
def softmax_combine_pad_init(
    partial_m_ptr,
    partial_z_ptr,
    CP: tl.constexpr,
):
    # grid = (M,): prime one padded partial row with the merge identity
    # (m = -inf, z = 0) so that the combine can read all CP lanes with no mask
    # at all.  CP is a multiple of 64 -> full unmasked contiguous store.
    pid = tl.program_id(0)
    off = pid * CP + tl.arange(0, CP)
    tl.store(partial_m_ptr + off, tl.full([CP], float("-inf"), tl.float32))
    tl.store(partial_z_ptr + off, tl.zeros([CP], tl.float32))


@libentry()
@triton.jit
def softmax_chunk_combine(
    m_ptr,
    z_ptr,
    partial_m_ptr,
    partial_z_ptr,
    NG: tl.constexpr,
    GW: tl.constexpr,
):
    # grid = (M,): per-row partials -> (row max, row sum-exp).  Fold NG fully
    # unmasked GW-lane groups over the padded row (see _sm_combine_geometry);
    # the padding lanes hold (-inf, 0) and contribute exactly nothing.  A row
    # that is entirely -inf still yields NaN, matching eager ATen.
    pid = tl.program_id(0)
    lane = tl.arange(0, GW)
    base = pid * NG * GW
    m = float("-inf")
    for g in tl.static_range(NG):  # tl.range breaks TritonXPUUnrollControl
        mc = tl.load(partial_m_ptr + base + g * GW + lane)
        m = tl.maximum(m, tl.max(mc, 0))
    z = 0.0
    for g in tl.static_range(NG):
        off = base + g * GW + lane
        mc = tl.load(partial_m_ptr + off)
        zc = tl.load(partial_z_ptr + off)
        z += tl.sum(zc * tl.exp(mc - m), 0)
    tl.store(m_ptr + pid, m)
    tl.store(z_ptr + pid, z)


@libentry()
@triton.jit
def softmax_chunk_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    # Flat grid (M * C_FULL): second read, out = exp(x - m) / z.
    pid = tl.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row).to(tl.float32)
    z = tl.load(z_ptr + row).to(tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_kernel_chunk_strided(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    # Strided variant (row*N + c*BLOCK_N) for rows with a tail (N % BN != 0),
    # where the flat pid*BN form drifts by the row tail.
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def softmax_chunk_pass_strided(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row).to(tl.float32)
    z = tl.load(z_ptr + row).to(tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    c = pid % C_FULL
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_tail_piece_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    M,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    # Unmasked exact-pow2 tail piece (fully inside the row).
    pid = tl.program_id(0)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def softmax_tail_piece_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.load(m_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_tail_masked_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    TAIL_LEN,
):
    # Masked 64-lane piece for the <64 column remainder of a row tail.
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def softmax_tail_masked_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    TAIL_BASE,
    TAIL_LEN,
):
    pid = tl.program_id(0)
    m = tl.load(m_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z, mask=within)


def _softmax_chunk_split(output, inp, M, N):
    """Chunked split forward for N > _SM_CHUNK_SPLIT_MIN (see dispatch)."""
    c_full = N // _SM_CHUNK_BN
    taillen = N - c_full * _SM_CHUNK_BN
    pieces, rrem = _sm_pow2_tail_pieces(taillen) if taillen else ([], 0)
    have_rem = rrem != 0
    C = c_full + len(pieces) + (1 if have_rem else 0)
    # Padded per-row stride: the merge reads CP lanes with no mask at all.
    GW, NG, CP = _sm_combine_geometry(C)
    pm = torch.empty((M * CP,), dtype=torch.float32, device=inp.device)
    pz = torch.empty((M * CP,), dtype=torch.float32, device=inp.device)
    m_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    z_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    if CP != C:
        # Only the padding lanes need priming: slots [0, C) are all written
        # below (c_full chunks + len(pieces) tail pieces + the remainder slot),
        # so an exactly-fitting stride skips this launch entirely.
        softmax_combine_pad_init[(M, 1, 1)](
            pm,
            pz,
            CP=CP,
            buffer_size_limit=2048,
            num_warps=8,
        )
    base = c_full * _SM_CHUNK_BN
    for slot, plen in enumerate(pieces):
        softmax_tail_piece_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            M,
            N,
            CP,
            c_full + slot,
            base,
            PLEN=plen,
            buffer_size_limit=2048,
            num_warps=8,
        )
        base += plen
    if have_rem:
        softmax_tail_masked_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            N,
            CP,
            c_full + len(pieces),
            base,
            rrem,
            buffer_size_limit=2048,
            num_warps=8,
        )
    if c_full:
        if pieces or have_rem:
            if M == 1:
                # Single row: row index is always 0, so the flat pid*BN form
                # stays perfectly contiguous (strided addressing with runtime
                # N would collapse to discrete gathers on XPU, 20-60x slower).
                softmax_kernel_chunk[(c_full, 1, 1)](
                    pm,
                    pz,
                    inp,
                    c_full,
                    CP,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
            else:
                softmax_kernel_chunk_strided[(M * c_full, 1, 1)](
                    pm,
                    pz,
                    inp,
                    N,
                    c_full,
                    CP,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
        else:
            softmax_kernel_chunk[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                c_full,
                CP,
                BLOCK_N=_SM_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    softmax_chunk_combine[(M, 1, 1)](
        m_out,
        z_out,
        pm,
        pz,
        NG=NG,
        GW=GW,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if c_full:
        if pieces or have_rem:
            if M == 1:
                softmax_chunk_pass[(c_full, 1, 1)](
                    output,
                    inp,
                    m_out,
                    z_out,
                    c_full,
                    C,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
            else:
                softmax_chunk_pass_strided[(M * c_full, 1, 1)](
                    output,
                    inp,
                    m_out,
                    z_out,
                    N,
                    c_full,
                    C,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
        else:
            softmax_chunk_pass[(M * c_full, 1, 1)](
                output,
                inp,
                m_out,
                z_out,
                c_full,
                C,
                BLOCK_N=_SM_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    base = c_full * _SM_CHUNK_BN
    for plen in pieces:
        softmax_tail_piece_pass[(M, 1, 1)](
            output,
            inp,
            m_out,
            z_out,
            N,
            base,
            PLEN=plen,
            buffer_size_limit=2048,
            num_warps=8,
        )
        base += plen
    if have_rem:
        softmax_tail_masked_pass[(M, 1, 1)](
            output,
            inp,
            m_out,
            z_out,
            N,
            base,
            rrem,
            buffer_size_limit=2048,
            num_warps=8,
        )


_SM_CHUNK_SPLIT_MAX_N = 8192 * 1024  # rows beyond this stay on per-row kernel


def _softmax_forward_launch(output, inp, M, N):
    """Inner launch on a contiguous [M, N] view (reduced dim innermost)."""
    # XPU correctness guard: 2D multirow tiles with non-pow2 N (a constexpr
    # `tl.arange(0, N)`) silently miscompile for a large family of widths
    # (odd N >= 17 and many N % 4 == 2; fp32/fp16 alike - probe: 224+224 of
    # ~340 scanned widths per dtype, maxdiff up to ~1e-1). Same XPU family as
    # log_softmax (08-20) / safe_softmax (08-21). Restrict the multirow fast
    # path to pow2 N for ALL dtypes (previously only bf16 was gated); non-pow2
    # falls back to the per-row masked kernel (exact for any N).
    use_multirow = N <= _SM_MR_MAX_N and ((N & (N - 1)) == 0)
    if use_multirow:
        tile_m = 1
        for n_hi, tm in _SM_N_TILE_M:
            if N <= n_hi:
                tile_m = tm
                break
        if M % tile_m == 0:
            grid = (M // tile_m,)
            if tile_m * N > 8192:
                softmax_kernel_multirow[grid](
                    output,
                    inp,
                    M,
                    N=N,
                    TILE_M=tile_m,
                    num_warps=4,
                    buffer_size_limit=2048,
                )
            else:
                softmax_kernel_multirow[grid](
                    output, inp, M, N=N, TILE_M=tile_m, num_warps=4
                )
            return
    if N > _SM_CHUNK_SPLIT_MAX_N:
        # Beyond the split window keep the per-row two-pass kernel
        # (grid=(M,), TILE_N and ONE_TILE_PER_CTA from the heuristics).
        grid = (M, 1, 1)
        softmax_kernel_inner[grid](
            output,
            inp,
            M,
            N,
            buffer_size_limit=2048,
            is_use_mask_zero=True,
        )
        return
    if N > _SM_MR_MAX_N:
        # Wide rows. The per-row two-pass kernel runs one program per row, a
        # serial chain of chunks: with few rows the whole row sits in one
        # program and the split (flat M x C grids, fully parallel over N) is
        # faster; with many rows the per-row kernel already has enough
        # programs and the split's extra launches/partial traffic only lose.
        # Threshold measured on XPU (2026-08-21): M * (N // 8192) < 1024.
        if M * (N // _SM_CHUNK_BN) < 1024:
            _softmax_chunk_split(output, inp, M, N)
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                output,
                inp,
                M,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
        return
    # Small/odd N: per-row masked single-tile kernel (exact for any N).
    grid = (M, 1, 1)
    softmax_kernel_inner[grid](
        output,
        inp,
        M,
        N,
        buffer_size_limit=2048,
        is_use_mask_zero=True,
    )


# ------------------------  backward -------------------------------


def softmax_backward_kernel_inner_heru_tile_n(args):
    N = args["N"]
    if N <= 32768:
        return triton.next_power_of_2(N)
    return 4096


def softmax_backward_kernel_inner_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@libentry()
@triton.heuristics(
    values={
        "TILE_N": softmax_backward_kernel_inner_heru_tile_n,
        "ONE_TILE_PER_CTA": softmax_backward_kernel_inner_heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # One program per row (grid=(M,)), mirroring the forward. Pre-offset the base
    # pointers so the inner `ptr + n_offsets` access is a scalar-base + stride-1
    # arange that OffsetAnalysis proves contiguous (block DMA). The old impl used a
    # fixed grid=(12,) with a [TILE_M, TILE_N] tile whose `m_offsets[:,None]*N +
    # n_offsets` addressing blocked the analysis -> discrete scalar gather
    # (~1-3 GB/s: [4096,4096] took ~38ms). It also computed in float64 (2x traffic,
    # unnecessary). float32 accumulation matches the forward and the generic backend.
    pid_m = ext.program_id(0)
    out_ptr += pid_m * N
    out_grad_ptr += pid_m * N
    in_grad_ptr += pid_m * N
    if ONE_TILE_PER_CTA:
        # KNOWN DEFECT (2026-08-29, unfixed): when TILE_N is exactly 1024 and
        # the dtype is fp32, `other=` pollutes *valid* lanes, so N =
        # 997/1000/1023/1024 on the K > 1 path come out 5e-3..1e-2 wrong
        # (relative) in fp32.  Isolated in
        # harness/results/functional/softmax_backward_sb_xpu7_20260829:
        # dropping `other=` makes exactly those cases exact, and widening the
        # tile to 2048 moves the failure onto N = 512 -> it is the 1024 width.
        # The clamp+tl.where rewrite that fixes it for TILE_N <= 4096 wedged
        # dev7 with a NOC timeout at TILE_N = 16384 (N = 8193) and was wrong for
        # TILE_N = 2, so it is NOT applied here; see the solution note.
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.sum(out_tile * out_grad_tile, 0)
        in_grad_tile = out_tile * (out_grad_tile - scale)
        tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)
    else:
        # Pass 1: accumulate scale = sum(out * out_grad) over the row. Iterate
        # ASCENDING so each load is a scalar-base + stride-1 arange (block DMA).
        scale = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale += out_tile * out_grad_tile
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale += out_tile * out_grad_tile
        scale = tl.sum(scale, 0)  # scalar

        # Pass 2: write in_grad = out * (out_grad - scale), ASCENDING.
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)


# ------------------------  backward: XPU-tuned fast paths (K==1) ----------
#
# Softmax backward over the innermost dim: in_grad[n] = out[n] * (out_grad[n] - s)
# with s = sum_n out[n] * out_grad[n] (per row).  Tuned 2026-08-16 on XPU after
# the special_log_softmax / log_softmax_backward_data experience:
#   * N <= 4096: single-pass 2D [TILE_M, N] tile (pow2-N constexpr; the row index
#     is left affine when M % TILE_M == 0 and clamped -- never masked -- only for
#     a partial last block; non-pow2 N uses a pow2 tile width with clamped
#     column indices -- see the kernels below for why masks are avoided and why
#     the clamp is confined to the partial-block case).
#   * N >  4096: per-row two-pass with wide unmasked tiles
#     (TILE 16384 for fp16/fp32, 8192 for bf16 - bf16 wider sums miscompile on XPU),
#     tail pieces kept <= 4096 masked lanes (masked reduces are exact only up to
#     4096 lanes on this backend).  Keep all loads/stores unwrapped; only the
#     flattened per-row data is contiguous [M, N].

_SB_MR_MAX_N = 4096  # largest N handled by the 2D multirow tile
# TILE_M per N bucket for the 2D tile: tile lanes (TILE_M * N <= 8192) stay
# within the XPU exact tl.sum window; N is a power of two in the target matrix.
_SB_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 8), (2048, 4), (4096, 2)]
_SB_WIDE = 8192  # wide tile for the per-row two-pass kernel (16384 spills uni_sram)


@triton.jit
def softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    # 2D single-pass tile [TILE_M, N]: scale per row = sum_n out*out_grad, then
    # in_grad = out * (out_grad - scale). N exact (no column padding) so each
    # [TILE_M, N] block is one contiguous region -> block DMA (mirrors
    # special_log_softmax's N<=4096 path).
    #
    # This is HEAD's `NEED_MASK=False` branch, unchanged, and it is now the ONLY
    # body: the caller only reaches it when N is a power of two AND
    # M % TILE_M == 0, so no lane can fall out of bounds and no mask (hence no
    # `other=`) is needed.  The row index MUST stay affine -- clamping it with
    # tl.minimum(row, M - 1) is memory-safe but destroys the stride-1/block-DMA
    # proof and cost 2.4x-168x in fixed Gems latency on every 2D benchmark cell
    # (measured 2026-08-30 on XPU 7: (4096,4096) fp16 0.311ms -> 52.14ms,
    # (1024,4096) fp32 0.087ms -> 12.53ms).  HEAD's `NEED_MASK=True` branch was
    # deleted instead: a masked 2D store writes the masked-out rows anyway on
    # this XPU, so it was a heap overwrite of (TILE_M - M % TILE_M) * N elements
    # past the end of in_grad (3840 lanes for the official (1, 256) case, 504
    # for (1, 8), 4096 for (1, 4096), measured with an allocator-level canary on
    # the production `_softmax_backward_data.out` path).  Those partial-block
    # shapes now go to softmax_backward_kernel_multirow_pad, which clamps.
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    o = tl.load(out_ptr + off).to(tl.float32)
    g = tl.load(out_grad_ptr + off).to(tl.float32)
    s = tl.sum(o * g, 1)  # [TILE_M]
    tl.store(in_grad_ptr + off, o * (g - s[:, None]))


@triton.jit
def softmax_backward_kernel_multirow_pad(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    W: tl.constexpr,
    TILE_M: tl.constexpr,
):
    # Same tile for a NON power-of-two N (N <= W = next_pow2(N) <= 4096) OR for a
    # partial last row block (M % TILE_M != 0), i.e. for every case the affine
    # unmasked kernel above cannot serve.  A non-pow2 constexpr tile dimension
    # mis-lowers on this XPU: with no mask at
    # all the [TILE_M, N] tile still produced wrong scales in every dtype
    # (N = 17/300/513/997/1023/1025/... , ds up to inf, bf16 -> NaN) and bf16
    # N = 2560 does not even compile (PassManager assertion).  So the tile width
    # is a power of two, both indices are clamped in-bounds (no mask, no
    # `other=`), the reduce is trimmed with tl.where, and the duplicated lanes
    # re-store column N-1 / row M-1 with the very same value.  For a pow2 N the
    # column clamp and the tl.where are no-ops and only the row clamp matters.
    # Clamped indices are a discrete gather (~2-100x slower than the affine
    # tile), which is why the caller keeps this kernel off the fast path.
    pid = tl.program_id(0)
    mo = tl.minimum(pid * TILE_M + tl.arange(0, TILE_M), M - 1)
    no = tl.arange(0, W)
    nc = tl.minimum(no, N - 1)
    off = mo[:, None] * N + nc[None, :]
    o = tl.load(out_ptr + off).to(tl.float32)
    g = tl.load(out_grad_ptr + off).to(tl.float32)
    s = tl.sum(tl.where(no[None, :] < N, o * g, 0.0), 1)  # [TILE_M]
    tl.store(in_grad_ptr + off, o * (g - s[:, None]))


@triton.jit
def softmax_backward_kernel_perrow_p2(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    W: tl.constexpr,
):
    # Per-row two-pass: pass1 accumulates scale = sum(out*out_grad) over full
    # tiles of width W; pass2 writes in_grad. Grid = (M,). Tail rows use the
    # split-row kernels below (fusing a masked tail into this kernel
    # miscompiles on XPU; the isolated 4096-lane masked kernel is exact).
    pid = tl.program_id(0)
    if pid < M:
        out_ptr += pid * N
        out_grad_ptr += pid * N
        in_grad_ptr += pid * N
        acc = tl.zeros([W], dtype=tl.float32)
        for start_n in range(0, N, W):
            n_offsets = start_n + tl.arange(0, W)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            acc += o * og
        scale = tl.sum(acc, 0)
        for start_n in range(0, N, W):
            n_offsets = start_n + tl.arange(0, W)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = o * (og - scale)
            tl.store(in_grad_ptr + n_offsets, ig)


@triton.jit
def softmax_backward_kernel_perrow_p2_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    p_tail_ptr,
    scale_ptr,
    N,
    PREV,
):
    # Full tiles only (runtime PREV, W=4096): scale = sum(acc) + p_tail[row];
    # also writes the per-row scale for the standalone tail pass.
    pid = tl.program_id(0)
    out_ptr += pid * N
    out_grad_ptr += pid * N
    in_grad_ptr += pid * N
    acc = tl.zeros([4096], dtype=tl.float32)
    for start_n in range(0, PREV, 4096):
        n_offsets = start_n + tl.arange(0, 4096)
        og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
        o = tl.load(out_ptr + n_offsets).to(tl.float32)
        acc += o * og
    scale = tl.sum(acc, 0) + tl.load(p_tail_ptr + pid)
    tl.store(scale_ptr + pid, scale)
    for start_n in range(0, PREV, 4096):
        n_offsets = start_n + tl.arange(0, 4096)
        og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
        o = tl.load(out_ptr + n_offsets).to(tl.float32)
        ig = o * (og - scale)
        tl.store(in_grad_ptr + n_offsets, ig)


@triton.jit
def softmax_backward_kernel_tail_partial(
    p_ptr,
    out_ptr,
    out_grad_ptr,
    N,
    PREV,
):
    # Masked 4096-lane tail partial: p = sum_tail(out * out_grad) per row.
    # Standalone kernel: masked 4096-lane reduces are exact on this XPU.
    pid = tl.program_id(0)
    tno = tl.arange(0, 4096)
    tmask = tno < (N - PREV)
    o = tl.load(out_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(
        tl.float32
    )
    tl.store(p_ptr + pid, tl.sum(o * g, 0))


@triton.jit
def softmax_backward_kernel_tail_pass(
    in_grad_ptr,
    scale_ptr,
    out_ptr,
    out_grad_ptr,
    N,
    PREV,
):
    # Standalone masked tail store: in_grad = o*(g - scale[row]).
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr + pid)
    tno = tl.arange(0, 4096)
    tmask = tno < (N - PREV)
    o = tl.load(out_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(
        tl.float32
    )
    tl.store(in_grad_ptr + pid * N + PREV + tno, o * (g - scale), mask=tmask)


# ---------------------------------------------------------------------------
# K > 1 (reduced dim is not innermost): the tensor is viewed as [M, N, K]
# (n = reduced dim, k = innermost) and transposed into [M*K, N] with
# aten._copy_from, then reduced by softmax_backward_kernel_inner (one program
# per row).  There is no [BN, K] partial/combine/pass trio in this file; the
# column-reduce design that this comment used to describe was never landed.


def _softmax_backward_launch_k1(output, grad_output, in_grad, M, N, input_dtype):
    # K == 1 (reduced dim is innermost): contiguous rows of length N.
    if N <= _SB_MR_MAX_N:
        TILE_M = 4
        for n_hi, tm in _SB_N_TILE_M:
            if N <= n_hi:
                TILE_M = tm
                break
        grid = (triton.cdiv(M, TILE_M),)
        if N == triton.next_power_of_2(N) and M % TILE_M == 0:
            # Fast path: pow2 constexpr tile width and a full last row block, so
            # the row index stays affine and nothing needs a mask or a clamp.
            # This is HEAD's codegen verbatim; keep it that way (a clamped row
            # index costs 2.4x-168x here, measured 2026-08-30 on XPU 7).
            softmax_backward_kernel_multirow[grid](
                output,
                grad_output,
                in_grad,
                M,
                N=N,
                TILE_M=TILE_M,
                num_warps=8,
            )
        else:
            # Non-pow2 N (the constexpr non-pow2 tile mis-lowers) or a partial
            # last row block (HEAD's row mask was a heap overwrite): pow2 tile
            # width + clamped indices.  TILE_M is still taken from the N bucket,
            # so TILE_M * W <= 8192 lanes.  No benchmark cell reaches here.
            softmax_backward_kernel_multirow_pad[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=triton.next_power_of_2(N),
                TILE_M=TILE_M,
                num_warps=8,
            )
    else:
        if N % _SB_WIDE == 0:
            grid = (M,)
            softmax_backward_kernel_perrow_p2[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=_SB_WIDE,
            )
        elif N % 4096 == 0:
            # A whole number of 4096-wide tiles: there is no masked tail, so the
            # split-row machinery below must not be used -- it unconditionally
            # reads `p_tail`, which is only written when `need_tail`.  HEAD came
            # here for every N = 4096 * odd (12288, 20480, 28672, ...) and added
            # an uninitialised `torch.empty((M,))` to the per-row scale, giving
            # arbitrarily wrong (sometimes inf/NaN) grads in all three dtypes.
            grid = (M,)
            softmax_backward_kernel_perrow_p2[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=4096,
            )
        else:
            # Tail rows: full tiles in the per-row kernel (W=4096); the tail
            # (1..4095 lanes, always non-empty here) is a standalone masked
            # kernel before and after it (fusing the masked tail into the
            # per-row kernel miscompiles on this XPU; the split-row pattern
            # mirrors special_log_softmax).
            prev = (N // 4096) * 4096
            p_tail = torch.empty((M,), dtype=torch.float32, device=in_grad.device)
            scale_buf = torch.empty((M,), dtype=torch.float32, device=in_grad.device)
            grid = (M,)
            softmax_backward_kernel_tail_partial[grid](
                p_tail, output, grad_output, N, prev
            )
            softmax_backward_kernel_perrow_p2_tail[grid](
                output,
                grad_output,
                in_grad,
                p_tail,
                scale_buf,
                N,
                prev,
            )
            softmax_backward_kernel_tail_pass[grid](
                in_grad, scale_buf, output, grad_output, N, prev
            )


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX")

    if self.ndim == 0:
        assert dim in (-1, 0), "Invalid dim"
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty_like(self, dtype=dtype)
        with torch_device_fn.device(self.device):
            softmax_kernel_inner[(1, 1, 1)](
                out,
                self,
                1,
                1,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
        return out

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    # special handling for dim = 0 and empty tensor
    if self.numel() == 0:
        out_shape = list(self.shape)
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty(out_shape, dtype=dtype, device=self.device)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]  # pre_dim
    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    K = self.numel() // M // N  # post_dim

    with torch_device_fn.device(self.device):
        if K > 1:
            # Rearrange [M, N, K] -> [M, K, N] so the reduced dim N is innermost
            # (the only fast axis on this XPU). Allocate the output tile directly
            # instead of `empty_like(self).view(...).transpose(...).contiguous()`,
            # which used to copy an uninitialized [M,K,N] buffer (a wasted
            # transpose-copy on top of the input transpose).
            inp_view = self.view(M, N, K).transpose(1, 2)
            inp_reshaped = torch.empty((M * K, N), dtype=self.dtype, device=self.device)
            # native strided copy (flag_gems never overrides _copy_from)
            torch.ops.aten._copy_from(inp_view, inp_reshaped, False)
            out_reshaped = torch.empty((M * K, N), dtype=dtype, device=self.device)

            _softmax_forward_launch(out_reshaped, inp_reshaped, M * K, N)

            # Restore the original rank and dimension order.
            out = out_reshaped.view(M, K, N).transpose(1, 2).reshape(self.shape)
        else:
            out = torch.empty_like(self, dtype=dtype)
            _softmax_forward_launch(out, self, M, N)
    return out


# N == 1 degenerate reduction: max == x and sum(exp(x - max)) == 1, so the
# result is exactly exp(x - x) / exp(x - x): 1.0 for finite inputs and NaN for
# +-inf / NaN, which is what eager ATen produces (verified on CPU). One flat
# contiguous pointwise pass replaces the whole reduction dispatch: no per-row
# programs and (for the K > 1 interior-dim case) no transpose copies at all.
# BLOCK probed on XPU 7 (probe_n1.py): 256/512/1024 all exact over 14 numels x
# 3 dtypes x (+-inf/NaN) with an out-of-range canary; 512 matches the
# risk-vetted log_softmax choice and is indistinguishable from 1024 at the
# benchmark numel (10000).
_SM_N1_BLOCK = 512


@libentry()
@triton.jit
def softmax_kernel_n1(
    output_ptr,
    input_ptr,
    n_elem,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    e = tl.exp(x - x)
    tl.store(output_ptr + offs, e / e, mask=mask)


def _softmax_n1_flat(out, inp):
    n_elem = inp.numel()
    softmax_kernel_n1[(triton.cdiv(n_elem, _SM_N1_BLOCK), 1, 1)](
        out,
        inp,
        n_elem,
        BLOCK=_SM_N1_BLOCK,
        buffer_size_limit=2048,
    )


def _native_contiguous(t):
    """Materialize `t` contiguously through the native strided copy.

    `Tensor.contiguous()` lowers to aten::contiguous -> aten::copy_, and
    `copy_` IS a gems-registered op, so inside `flag_gems.use_gems()` it turns
    into a gems strided pointwise copy (measured 57-1600x slower than the
    native XPU strided copy). `aten::_copy_from` is never overridden by gems.
    """
    dst = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    torch.ops.aten._copy_from(t, dst, False)
    return dst


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_OUT")

    if self.ndim == 0:
        assert dim in (-1, 0), "Invalid dim"
        dtype = torch.float32 if half_to_float else self.dtype
        if out.dtype != dtype:
            raise RuntimeError(
                f"_softmax.out: expected out dtype {dtype}, got {out.dtype}"
            )
        out.copy_(softmax(self, dim, half_to_float))
        return out

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")

    # Write the kernel result straight into `out` instead of
    # `out.copy_(softmax(...))`. `copy_` is a gems-registered op, so inside
    # `flag_gems.use_gems()` (which is where the benchmark times) the old
    # write-back became a gems pointwise copy running at ~1.25 GB/s: it alone
    # accounted for ~2070 ms of the 2113 ms baseline on (10000, 65536) fp16.
    # All layout movement below uses aten._copy_from (the native strided copy,
    # never overridden by gems) for the same reason.
    dim = dim % self.ndim
    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    inp = self if self.is_contiguous() else _native_contiguous(self)
    K = inp.numel() // M // N

    if N == 1 and out.is_contiguous():
        # Degenerate reduction axis: one flat pointwise pass over the whole
        # tensor (see softmax_kernel_n1). Layout-independent, so it also
        # covers K > 1 without any transpose copy.
        with torch_device_fn.device(inp.device):
            _softmax_n1_flat(out, inp)
        return out

    with torch_device_fn.device(inp.device):
        if K > 1:
            # Reduction over an interior dim: transpose so the reduced axis is
            # innermost, run the K == 1 launch family into a contiguous
            # scratch, then mirror it back through a transposed view of out.
            inp_t = torch.empty((M * K, N), dtype=inp.dtype, device=inp.device)
            torch.ops.aten._copy_from(
                inp.view(M, N, K).transpose(1, 2), inp_t.view(M, K, N), False
            )
            tmp = torch.empty((M * K, N), dtype=dtype, device=inp.device)
            _softmax_forward_launch(tmp, inp_t, M * K, N)
            src = tmp.view(M, K, N).transpose(1, 2)
            if out.is_contiguous():
                torch.ops.aten._copy_from(src, out.view(M, N, K), False)
            else:
                scratch = torch.empty((M, N, K), dtype=dtype, device=out.device)
                torch.ops.aten._copy_from(src, scratch, False)
                torch.ops.aten._copy_from(scratch.view(self.shape), out, False)
        elif not out.is_contiguous():
            # The launch kernels write flat [M, N] offsets; a strided out
            # (e.g. a slice view) would be corrupted. Compute into a
            # contiguous scratch and mirror it with the native strided copy.
            tmp = torch.empty(self.shape, dtype=dtype, device=self.device)
            _softmax_forward_launch(tmp, inp, M, N)
            torch.ops.aten._copy_from(tmp, out, False)
        else:
            _softmax_forward_launch(out, inp, M, N)
    return out


def softmax_backward(grad_output, output, dim, input_dtype, grad_input=None):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    K = output.numel() // M // N
    # The kernel computes in fp32 before storing, so an output buffer with the
    # requested dtype has the same values as the previous final `.to(...)`.
    # A contiguous out buffer can be written directly when no layout transform
    # is needed.
    if grad_input is not None and K == 1:
        in_grad = grad_input
    else:
        in_grad = torch.empty_like(output, dtype=input_dtype)

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            # Fallback (K > 8192 or unusual shapes): old transpose-based path
            # (correct but slow through gems copy_).
            # Transpose copies via aten._copy_from: flag_gems NEVER overrides
            # _copy_from, so these strided copies run at native speed (the
            # .contiguous() path dispatched to the gems copy_ override and was
            # ~300x slower; measured 308ms for [64,4096,64] fp16).
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2)
            out_view = output.view(M, N, K).transpose(1, 2)
            out_grad_reshaped = torch.empty(
                (M * K, N), dtype=grad_output.dtype, device=grad_output.device
            )
            out_reshaped = torch.empty(
                (M * K, N), dtype=output.dtype, device=output.device
            )
            torch.ops.aten._copy_from(out_grad_view, out_grad_reshaped, False)
            torch.ops.aten._copy_from(out_view, out_reshaped, False)
            in_grad_view = in_grad.view(M, N, K).transpose(1, 2)
            in_grad_reshaped = torch.empty(
                (M * K, N), dtype=in_grad.dtype, device=in_grad.device
            )
            torch.ops.aten._copy_from(in_grad_view, in_grad_reshaped, False)
            grid = lambda meta: (M * K, 1, 1)  # noqa: E731
            softmax_backward_kernel_inner[grid](
                out_reshaped,
                out_grad_reshaped,
                in_grad_reshaped,
                M * K,
                N,
                buffer_size_limit=2048,
            )
            origin_dim = output.ndim
            if output.ndim == 3:
                m, n, k = output.shape
            elif output.ndim == 2:
                m, n = output.shape
            if M == 1 and origin_dim == 2:
                in_grad = in_grad_reshaped.view(K, N).transpose(0, 1)
            elif M == 1 and origin_dim == 3:
                in_grad = in_grad_reshaped.transpose(0, 1).view(m, n, k)
            else:
                in_grad = in_grad_reshaped.view(m, k, n).transpose(1, 2)
        else:
            _softmax_backward_launch_k1(output, grad_output, in_grad, M, N, input_dtype)
    return in_grad


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP_OUT")
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected out dtype {input_dtype}, "
            f"got {grad_input.dtype}"
        )
    result = softmax_backward(
        grad_output,
        output,
        dim,
        input_dtype,
        grad_input=grad_input if grad_input.is_contiguous() else None,
    )
    if result is not grad_input:
        grad_input.copy_(result)
    return grad_input
