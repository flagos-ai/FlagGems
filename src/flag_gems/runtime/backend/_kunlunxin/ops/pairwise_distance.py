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

# Kunlunxin (XPU) override of pairwise_distance.
#
# Why a vendor override exists: the generic implementation
# (src/flag_gems/ops/pairwise_distance.py) is numerically unreliable on XPU:
#   * [BLOCK_M, BLOCK_D] 2D tile + axis-1 reduction miscompiles (rows >= 8 come
#     out wrong / noisy), which breaks the max/min (p=inf/-inf) kernels and
#     small-width bf16 rows;
#   * masked loads (mask=..., other=0.0) do not honour `other`: false lanes may
#     read out-of-bounds memory and the garbage participates in tl.sum. The
#     generic split-K tail block then pollutes (1, 10000000) sums (p in 0/1/2,
#     NaN for general p) and small-D bf16 rows;
#   * [1]-shaped accumulator tensors / [1]-block stores miscompile on XPU
#     (probe-verified) - only scalar (0-d) values are safe for stores;
#   * per-kernel live tiles must stay within the uni_sram budget
#     (~2048-4096 fp32 lanes total; 4096-lane mid loads + tail piece loads
#     together blow it up).
#
# This override only uses exact in-bounds UNMASKED loads, 1D reductions and
# scalar (0-d) accumulation:
#   * D <= 2048: one program per row; the row is covered by up to 4 binary
#     power-of-two "pieces" plus a short scalar loop for the remainder.
#   * D > 2048: a chunk kernel (grid (N, cdiv(D, 2048)), 2048-lane unmasked
#     tiles) writes fp32 per-chunk partials; a tail kernel (grid (N,)) reduces
#     the %-remainder with pieces into one extra partial; a mid-reduce kernel
#     combines partial groups of 2048 when needed; a final kernel reduces the
#     (zero/+-inf identity padded) partial buffer and applies the p-norm
#     finalization. Non-power-of-two remainders never touch masked memory
#     paths.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

# x ** p: vendor powf (see _pd_mode_reduce). exp2/log2 remain for the
# per-row finalization (single op, cost negligible).
exp2 = tl_extra_shim.exp2
log2 = tl_extra_shim.log2
pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

# Chunk width for the D axis. 2048 lanes keeps the fp32 tl.sum rounding well
# inside the fp32 rtol budget (1.3e-6) for the whole split-K chain and fits the
# XPU per-kernel live-tile budget.
_BLOCK_D = 2048
# Max power-of-two pieces covering a non-power-of-two remainder (D % _BLOCK_D
# or D <= _BLOCK_D). Remainder beyond the pieces is <= 15 lanes and is
# accumulated with a short scalar loop.
_MAX_PIECES = 4
# Max tl.sum lane count in mid/final reductions (uni_sram budget safe point).
_MID_BLOCK = 2048

# MODE: 0=p2, 1=p1, 2=p0, 3=inf, 4=-inf, 5=general


@triton.jit
def _pd_mode_reduce(diff, p_scalar, MODE: tl.constexpr):
    if MODE == 0:  # p == 2
        return tl.sum(diff * diff)
    elif MODE == 1:  # p == 1
        return tl.sum(diff)
    elif MODE == 2:  # p == 0: nonzero count
        return tl.sum((diff != 0).to(tl.float32))
    elif MODE == 3:  # inf
        return tl.max(diff)
    elif MODE == 4:  # -inf
        return tl.min(diff)
    else:  # general p: vendor powf (single libdevice call). The former
        # exp2(p * log2(x)) decomposition measured ~37x slower on XPU
        # (xpu log2f/exp2f are emulated; powf is a short sequence), and
        # matches exp2/log2 to <1e-6 relative on 2048/4096-lane chunks.
        # The exponent must be a constant tensor: `diff * 0.0 + p_scalar`
        # would poison the exponent to NaN on +/-inf diff (inf*0.0 = NaN),
        # while a constant preserves IEEE pow(x, p): inf^p -> inf (p>0) / 0
        # (p<0), matching the old exp2(p*log2(x)) on non-finite inputs.
        p_const = tl.full(diff.shape, p_scalar, diff.dtype)
        return tl.sum(pow(diff, p_const))


@triton.jit
def _pd_combine(acc, part, MODE: tl.constexpr):
    if MODE == 3:
        return tl.maximum(acc, part)
    elif MODE == 4:
        return tl.minimum(acc, part)
    else:
        return acc + part


@triton.jit
def _pd_finalize(acc, p_scalar, MODE: tl.constexpr):
    if MODE == 0:
        return tl.sqrt(acc)
    elif MODE == 5:
        return exp2((1.0 / p_scalar) * log2(acc))
    else:
        return acc


@triton.jit
def _pd_piece_sum(
    x1_ptr,
    x2_ptr,
    base,  # row start + optional tail start offset
    eps,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,  # uniform piece width (power of two)
    NP: tl.constexpr,  # number of uniform pieces (offset i * S)
    NSCALAR: tl.constexpr,  # scalar-loop remainder lanes
):
    # Pure scalar (0-d) accumulation: [1]-vectors miscompile on XPU. Piece
    # loads are all the SAME width (mixed-width tiles blow uni_sram).
    acc = 0.0
    if NP >= 1:
        a = tl.load(x1_ptr + base + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + tl.arange(0, S)).to(tl.float32)
        acc = _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE)
    if NP >= 2:
        a = tl.load(x1_ptr + base + S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 3:
        a = tl.load(x1_ptr + base + 2 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 2 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 4:
        a = tl.load(x1_ptr + base + 3 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 3 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 5:
        a = tl.load(x1_ptr + base + 4 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 4 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 6:
        a = tl.load(x1_ptr + base + 5 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 5 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 7:
        a = tl.load(x1_ptr + base + 6 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 6 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 8:
        a = tl.load(x1_ptr + base + 7 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 7 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NSCALAR > 0:
        # Plain `range` (not static_range): a non-unrolled scf.for keeps the
        # live set small. A static_range loop with a long remainder (e.g. the
        # D % 4096 lane tail of a 10M-D row, NSCALAR=128) pushes the ELF stack
        # over the hardware budget ("Failed to tune buffer size.").
        for j in range(NSCALAR):
            av = tl.load(x1_ptr + base + NP * S + j).to(tl.float32)
            bv = tl.load(x2_ptr + base + NP * S + j).to(tl.float32)
            diff = tl.abs(av - bv + eps)
            if MODE == 0:
                part = diff * diff
            elif MODE == 2:
                part = (diff != 0).to(tl.float32)
            elif MODE == 5:
                part = exp2(p_scalar * log2(diff))
            else:
                part = diff
            acc = _pd_combine(acc, part, MODE)
    return acc


@libentry()
@triton.jit
def _pd_small_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * D
    acc = _pd_piece_sum(
        x1_ptr,
        x2_ptr,
        base,
        eps,
        p_scalar,
        MODE,
        S,
        NP,
        NSCALAR,
    )
    tl.store(out_ptr + pid, _pd_finalize(acc, p_scalar, MODE))


# Multi-row variant of _pd_small_kernel: ROWS independent rows per program to
# cut the program count (and therefore the launch overhead) when N is large
# but D is small (launch-bound regime, e.g. (10000, 1) / (10000, 256)). Each
# row is processed exactly as in _pd_small_kernel; rows are independent so the
# loads pipeline across the ROWS iterations.
_ROWS = 8
# Route to _pd_small_multi_kernel only in the launch-bound regime: many
# independent small-D rows (e.g. (10000, 1) / (10000, 256)).
_MULTI_MIN_N = 1024
# Flat elementwise block for the D == 1 path (_pd_d1_kernel).
_D1_BLOCK = 1024


@libentry()
@triton.jit
def _pd_small_multi_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    for r in tl.static_range(ROWS):
        row = pid * ROWS + r
        if row < N:
            base = row * D
            acc = _pd_piece_sum(
                x1_ptr, x2_ptr, base, eps, p_scalar, MODE, S, NP, NSCALAR,
            )
            tl.store(out_ptr + row, _pd_finalize(acc, p_scalar, MODE))


# D == 1: every row is a single element, so all p-norms collapse to
# |x1 - x2 + eps| (p == 0 counts non-zeros: 1.0 for every lane since
# eps > 0 makes |x1 - x2 + eps| > 0 except a measure-zero set handled by the
# comparison). A single flat elementwise pass over the N rows replaces the
# per-row reduction kernel, which measured ~90x slower (row-serial latency).
# Loads use a clamped index so no masked (possibly OOB) load is ever emitted;
# the store mask only discards valid load results.
@libentry()
@triton.jit
def _pd_d1_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    eps,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    safe = tl.minimum(off, N - 1)
    a = tl.load(x1_ptr + safe).to(tl.float32)
    b = tl.load(x2_ptr + safe).to(tl.float32)
    d = tl.abs(a - b + eps)
    if MODE == 2:  # p == 0: nonzero count of a 1-element row.  Count the
        # reference condition |a - b + eps| != 0 exactly (a - b != -eps, which
        # also covers the a == b tie case: |eps| != 0).  d is computed in fp32
        # after an exact fp16/bf16 -> fp32 conversion, so the comparison is
        # exact for those dtypes.
        d = (d != 0).to(tl.float32)
    tl.store(out_ptr + off, d, mask=off < N)


@libentry()
@triton.jit
def _pd_chunk_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    D,
    eps,
    p_scalar,
    MID_STRIDE,
    MID_SIZE,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_n * D + pid_c * BLOCK
    a = tl.load(x1_ptr + base + tl.arange(0, BLOCK)).to(tl.float32)
    b = tl.load(x2_ptr + base + tl.arange(0, BLOCK)).to(tl.float32)
    diff = tl.abs(a - b + eps)
    m = _pd_mode_reduce(diff, p_scalar, MODE)
    tl.store(mid_ptr + pid_n * MID_STRIDE + pid_c, m)


@libentry()
@triton.jit
def _pd_tail_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    N,
    D,
    T,
    eps,
    p_scalar,
    MID_SIZE,
    MID_STRIDE,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * D + D - T
    acc = _pd_piece_sum(
        x1_ptr,
        x2_ptr,
        base,
        eps,
        p_scalar,
        MODE,
        S,
        NP,
        NSCALAR,
    )
    tl.store(mid_ptr + pid * MID_STRIDE + MID_SIZE, acc)


@libentry()
@triton.jit
def _pd_mid_reduce_kernel(
    mid_ptr,
    out_ptr,
    MID,
    STRIDE_IN,
    STRIDE_OUT,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    off = pid_c * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(mid_ptr + pid_n * STRIDE_IN + off).to(tl.float32)
    if MODE == 3:
        acc = tl.max(m)
    elif MODE == 4:
        acc = tl.min(m)
    else:
        acc = tl.sum(m)
    tl.store(out_ptr + pid_n * STRIDE_OUT + pid_c, acc)


@libentry()
@triton.jit
def _pd_final_kernel(
    mid_ptr,
    out_ptr,
    p_scalar,
    MID_STRIDE,
    MODE: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK_MID)
    m = tl.load(mid_ptr + pid * MID_STRIDE + off).to(tl.float32)
    if MODE == 3:
        acc = tl.max(m)
    elif MODE == 4:
        acc = tl.min(m)
    else:
        acc = tl.sum(m)
    tl.store(out_ptr + pid, _pd_finalize(acc, p_scalar, MODE))


def _mode_of(p):
    if p == 0.0:
        return 2
    if p == 1.0:
        return 1
    if p == 2.0:
        return 0
    if math.isinf(p):
        return 3 if p > 0 else 4
    return 5


def _piece_args(t):
    """Uniform-width piece decomposition of t.

    Returns (S, NP, NSCALAR): NP tiles of uniform width S (power of two) plus
    NSCALAR trailing lanes covered by a scalar loop. Widths are kept uniform
    because mixed-width live tiles blow the XPU uni_sram budget; S maximises
    the covered width (min(NP, 8) * S) over powers of two <= 512.
    """
    if t <= 0:
        return 0, 0, 0
    best = (0, 0)  # (coverage, S)
    S = 512
    while S > 0:
        n = t // S
        if n > 0:
            n_used = min(n, 8)
            cov = n_used * S
            if cov > best[0]:
                best = (cov, S)
        S //= 2
    _, S = best
    np = min(t // S, 8)
    return S, np, t - np * S


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN PAIRWISE_DISTANCE")
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]

    # Empty feature dim: torch returns 0 for finite p; inf/-inf have no identity
    # element over an empty reduction and torch raises.
    if D == 0:
        if p == float("inf") or p == float("-inf"):
            raise RuntimeError(
                "pairwise_distance cannot compute the inf/-inf norm on an empty "
                "reduction dimension (no identity element)"
            )
        out = torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        if keepdim:
            out = out.unsqueeze(-1)
        return out

    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)

    mode = _mode_of(p)
    p_scalar = float(p) if mode == 5 else 1.0

    with torch_device_fn.device(x1.device):
        if D == 1:
            # Single-lane rows: one flat elementwise pass (grid N // _D1_BLOCK)
            # instead of N per-row reduction programs (row-serial latency,
            # measured ~90x slower for this shape class).
            _pd_d1_kernel[(triton.cdiv(N, _D1_BLOCK),)](
                x1,
                x2,
                out,
                N,
                eps,
                MODE=mode,
                BLOCK=_D1_BLOCK,
            )
        elif D <= _BLOCK_D:
            PS, PNP, PNSC = _piece_args(D)
            if N >= _MULTI_MIN_N:
                # Launch-bound regime (many small-D rows): _ROWS independent
                # rows per program cut the program count by _ROWS (see
                # _pd_small_multi_kernel).
                _pd_small_multi_kernel[(triton.cdiv(N, _ROWS),)](
                    x1,
                    x2,
                    out,
                    N,
                    D,
                    eps,
                    p_scalar,
                    MODE=mode,
                    S=PS,
                    NP=PNP,
                    NSCALAR=PNSC,
                    ROWS=_ROWS,
                )
            else:
                _pd_small_kernel[(N,)](
                    x1,
                    x2,
                    out,
                    N,
                    D,
                    eps,
                    p_scalar,
                    MODE=mode,
                    S=PS,
                    NP=PNP,
                    NSCALAR=PNSC,
                )
        else:
            # All modes use 4096-lane chunks when D >= 4096 (halves the chunk
            # program count vs 2048 and is numerically safe: tl.sum is complete
            # for BLOCK <= 8192). D in (2048, 4096) keeps 2048-lane chunks so
            # the remainder stays short; the small path handles D <= 2048.
            chunk_block = 4096 if (mode in (3, 4) or D >= 2 * _BLOCK_D) else _BLOCK_D
            MID = D // chunk_block
            T = D - MID * chunk_block
            P = MID + (1 if T > 0 else 0)
            # Padded lanes must hold the MODE identity so unmasked partial
            # reductions stay correct: 0.0 for sums/counts, -inf for max,
            # +inf for min.
            if mode == 3:
                pad = -float("inf")
            elif mode == 4:
                pad = float("inf")
            else:
                pad = 0.0
            stride = triton.next_power_of_2(P)
            if stride < 2:
                stride = 2
            if stride > _MID_BLOCK:
                stride = triton.cdiv(P, _MID_BLOCK) * _MID_BLOCK
            mid = torch.full((N * stride,), pad, device=x1.device, dtype=torch.float32)
            # NOTE: chunk_block must be the one used to derive MID/T/P above (a
            # later re-assignment to a different width used to silently drop the
            # last MID*(4096-BLOCK) lanes of every row, halving sum-mode results
            # for D >= 4096).
            _pd_chunk_kernel[(N, MID)](
                x1,
                x2,
                mid,
                D,
                eps,
                p_scalar,
                stride,
                MID,
                MODE=mode,
                BLOCK=chunk_block,
            )
            if T > 0:
                PS, PNP, NCSC = _piece_args(T)
                _pd_tail_kernel[(N,)](
                    x1,
                    x2,
                    mid,
                    N,
                    D,
                    T,
                    eps,
                    p_scalar,
                    MID,
                    stride,
                    MODE=mode,
                    S=PS,
                    NP=PNP,
                    NSCALAR=NCSC,
                )
            cur_mid, cur_stride, cur_n = mid, stride, P
            while triton.next_power_of_2(cur_n) > _MID_BLOCK:
                g = triton.cdiv(cur_n, _MID_BLOCK)
                nstride = triton.next_power_of_2(g)
                cur_out = torch.full(
                    (N * nstride,), pad, device=x1.device, dtype=torch.float32
                )
                _pd_mid_reduce_kernel[(N, g)](
                    cur_mid,
                    cur_out,
                    MID=cur_n,
                    STRIDE_IN=cur_stride,
                    STRIDE_OUT=nstride,
                    MODE=mode,
                    BLOCK=_MID_BLOCK,
                )
                cur_mid, cur_stride, cur_n = cur_out, nstride, g
            _pd_final_kernel[(N,)](
                cur_mid,
                out,
                p_scalar,
                cur_stride,
                MODE=mode,
                BLOCK_MID=triton.next_power_of_2(cur_n),
            )

    return out
