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

# Kunlunxin (XPU) override of dist.
#
# Why a vendor override exists: the generic implementation
# (src/flag_gems/ops/dist.py) is numerically unreliable on XPU:
#   * masked loads (mask=..., other=0.0) do not honour `other`: false lanes may
#     read out-of-bounds memory and the garbage participates in tl.sum. The
#     generic two-stage path (N > 16384) uses one 4096-lane masked block per
#     chunk, so every N that is not a multiple of 4096 (e.g. 65537, 1000000)
#     produces a wrong tail partial (measured: 7983.1 instead of 1.02 for the
#     single-lane tail of N=65537; rel. error ~3e-2..1e-1 for (65537,)/(1,1e6)).
#     N that are exact multiples of 4096 (e.g. 32768, 524288) are unaffected,
#     which is why only the non-divisible shapes fail.
#   * Per-kernel live tiles must stay within the uni_sram budget
#     (~2048-4096 fp32 lanes total; 4096-lane tiles plus a second live tile
#     can blow the budget, cf. cdist_backward).
#
# This override only uses exact in-bounds UNMASKED loads, 1D reductions and
# scalar (0-d) accumulation:
#   * N <= 2048: one program; the range is decomposed into up to 8 uniform
#     power-of-two "pieces" (S = 512/256/.../1) plus a short scalar loop for
#     the remainder (never touching a masked-memory path).
#   * N > 2048: a chunk kernel (grid (N // 2048,), one 2048-lane unmasked tile
#     per program) writes fp32 per-chunk partials; a tail kernel (grid (1,))
#     reduces the %-remainder with the piece decomposition into one extra
#     partial; the partial buffer is pre-filled with the MODE identity (0.0 /
#     -inf / +inf) so the unmasked mid/final reductions stay correct; a
#     mid-reduce kernel re-reduces groups of 2048 while the partial count
#     exceeds 2048; a final kernel reduces the (identity-padded) partial buffer
#     and applies the p-norm finalization.
#   * All accumulation is fp32 (XPU has no fp64; fp16/bf16 inputs are exactly
#     upcast, so p == 0 counts and all comparisons are exact).

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

# x ** p: vendor powf (see _dist_mode_reduce); exp2/log2 remain for the
# per-piece finalization (single op, cost negligible).
exp2 = tl_extra_shim.exp2
log2 = tl_extra_shim.log2
pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

# Width of one chunk (uniform unmasked tile) in the two-stage path. 2048 lanes
# keeps the tl.sum rounding well inside the fp32 rtol budget and fits the XPU
# per-kernel live-tile budget (cf. _MID_BLOCK).
_BLOCK = 2048
# Max tl.sum lane count in mid/final reductions (uni_sram budget safe point).
_MID_BLOCK = 2048
# Max uniform-width pieces covering a non-power-of-two remainder.
_MAX_PIECES = 8

# MODE: 0=p2, 1=p1, 2=p0, 3=inf, 4=-inf, 5=general


@triton.jit
def _dist_mode_reduce(diff, p_scalar, MODE: tl.constexpr):
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
    else:  # general p: vendor powf (single libdevice call; see pairwise_distance).
        # The exponent must be a constant tensor: `diff * 0.0 + p_scalar`
        # would poison the exponent to NaN on +/-inf diff (inf*0.0 = NaN),
        # while a constant preserves IEEE pow(x, p).
        p_const = tl.full(diff.shape, p_scalar, diff.dtype)
        return tl.sum(pow(diff, p_const))


@triton.jit
def _dist_combine(acc, part, MODE: tl.constexpr):
    if MODE == 3:
        return tl.maximum(acc, part)
    elif MODE == 4:
        return tl.minimum(acc, part)
    else:
        return acc + part


@triton.jit
def _dist_finalize(acc, p_scalar, MODE: tl.constexpr):
    if MODE == 0:
        return tl.sqrt(acc)
    elif MODE == 5:
        # Direct libdevice pow: exp2((1/p)*log2(acc)) costs ~5e-7 relative
        # (log2+exp2 double round), which is amplified by 1/p (e.g. 2x for
        # p=0.5) and can exceed the fp32 rtol budget; powf is ~1 ulp.
        return pow(acc, 1.0 / p_scalar)
    else:
        return acc


@triton.jit
def _dist_piece_sum(
    x_ptr,
    y_ptr,
    base,  # segment start (flattened element offset)
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,  # uniform piece width (power of two)
    NP: tl.constexpr,  # number of uniform pieces (offset i * S)
    NSCALAR: tl.constexpr,  # scalar-loop remainder lanes
):
    # Pure scalar (0-d) accumulation: [1]-vectors miscompile on XPU. Piece
    # loads are all the SAME width (mixed-width tiles blow uni_sram). The
    # caller guarantees base + NP*S + NSCALAR <= N so every load is exactly
    # in-bounds (no masked-memory path).
    acc = tl.zeros((), dtype=tl.float32)
    if NP >= 1:
        a = tl.load(x_ptr + base + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + tl.arange(0, S)).to(tl.float32)
        acc = _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE)
    if NP >= 2:
        a = tl.load(x_ptr + base + S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NP >= 3:
        a = tl.load(x_ptr + base + 2 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + 2 * S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NP >= 4:
        a = tl.load(x_ptr + base + 3 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + 3 * S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NP >= 5:
        a = tl.load(x_ptr + base + 4 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + 4 * S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NP >= 6:
        a = tl.load(x_ptr + base + 5 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + 5 * S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NP >= 7:
        a = tl.load(x_ptr + base + 6 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + 6 * S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NP >= 8:
        a = tl.load(x_ptr + base + 7 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(y_ptr + base + 7 * S + tl.arange(0, S)).to(tl.float32)
        acc = _dist_combine(
            acc, _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE), MODE
        )
    if NSCALAR > 0:
        # Plain `range` (not static_range): a non-unrolled scf.for keeps the
        # live set small (a long remainder would otherwise push the ELF stack
        # over the hardware budget, "Failed to tune buffer size.").
        for j in range(NSCALAR):
            av = tl.load(x_ptr + base + NP * S + j).to(tl.float32)
            bv = tl.load(y_ptr + base + NP * S + j).to(tl.float32)
            diff = tl.abs(av - bv)
            if MODE == 0:
                part = diff * diff
            elif MODE == 2:
                part = (diff != 0).to(tl.float32)
            elif MODE == 5:
                part = exp2(p_scalar * log2(diff))
            else:
                part = diff
            acc = _dist_combine(acc, part, MODE)
    return acc


@libentry()
@triton.jit
def _dist_piece_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    # Single-launch path for 0 < N <= _BLOCK: one program covers the whole
    # flattened tensor and writes the final scalar directly.
    pid = tl.program_id(0)
    acc = _dist_piece_sum(
        x_ptr,
        y_ptr,
        pid * (NP * S + NSCALAR),
        p_scalar,
        MODE,
        S,
        NP,
        NSCALAR,
    )
    tl.store(out_ptr, _dist_finalize(acc, p_scalar, MODE))


@libentry()
@triton.jit
def _dist_chunk_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    p_scalar,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * BLOCK
    a = tl.load(x_ptr + base + tl.arange(0, BLOCK)).to(tl.float32)
    b = tl.load(y_ptr + base + tl.arange(0, BLOCK)).to(tl.float32)
    m = _dist_mode_reduce(tl.abs(a - b), p_scalar, MODE)
    tl.store(mid_ptr + pid, m)


@libentry()
@triton.jit
def _dist_tail_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    base,
    mid_idx,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    acc = _dist_piece_sum(
        x_ptr,
        y_ptr,
        base,
        p_scalar,
        MODE,
        S,
        NP,
        NSCALAR,
    )
    tl.store(mid_ptr + mid_idx, acc)


@libentry()
@triton.jit
def _dist_mid_reduce_kernel(
    mid_ptr,
    out_ptr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(mid_ptr + off).to(tl.float32)
    if MODE == 3:
        acc = tl.max(m)
    elif MODE == 4:
        acc = tl.min(m)
    else:
        acc = tl.sum(m)
    tl.store(out_ptr + pid, acc)


@libentry()
@triton.jit
def _dist_final_kernel(
    mid_ptr,
    out_ptr,
    p_scalar,
    MODE: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    off = tl.arange(0, BLOCK_MID)
    m = tl.load(mid_ptr + off).to(tl.float32)
    if MODE == 3:
        acc = tl.max(m)
    elif MODE == 4:
        acc = tl.min(m)
    else:
        acc = tl.sum(m)
    tl.store(out_ptr, _dist_finalize(acc, p_scalar, MODE))


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
            n_used = min(n, _MAX_PIECES)
            cov = n_used * S
            if cov > best[0]:
                best = (cov, S)
        S //= 2
    _, S = best
    np = min(t // S, _MAX_PIECES)
    return S, np, t - np * S


def dist(input, other, p=2):
    logger.debug("GEMS_KUNLUNXIN DIST")
    if input.shape != other.shape:
        input, other = torch.broadcast_tensors(input, other)
    if not input.is_contiguous():
        input = input.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()

    n = input.numel()

    # torch returns 0 for finite non-negative p on an empty reduction; for
    # inf / -inf / negative p there is no identity element and torch raises.
    if n == 0:
        if p == float("inf") or p == float("-inf") or p < 0:
            raise RuntimeError(
                f"dist cannot compute the {p} norm on an empty tensor "
                "(no identity element over an empty reduction)"
            )
        return torch.zeros([], dtype=input.dtype, device=input.device)

    out = torch.empty([], dtype=input.dtype, device=input.device)
    mode = _mode_of(p)
    p_scalar = float(p) if mode == 5 else 1.0

    with torch_device_fn.device(input.device):
        if n <= _BLOCK:
            s, np, ns = _piece_args(n)
            _dist_piece_kernel[(1,)](
                input,
                other,
                out,
                p_scalar,
                MODE=mode,
                S=s,
                NP=np,
                NSCALAR=ns,
            )
            return out

        p_full = n // _BLOCK
        tail = n - p_full * _BLOCK
        p_cnt = p_full + (1 if tail else 0)
        # Padded lanes must hold the MODE identity so unmasked partial
        # reductions stay correct: 0.0 for sums/counts, -inf for max,
        # +inf for min.
        if mode == 3:
            pad = -float("inf")
        elif mode == 4:
            pad = float("inf")
        else:
            pad = 0.0
        # The partial buffer is padded to a power of two (>= p_cnt) so the
        # unmasked mid/final reductions never read out of bounds.
        size = triton.next_power_of_2(max(p_cnt, 2))
        mid = torch.full((size,), pad, device=input.device, dtype=torch.float32)
        _dist_chunk_kernel[(p_full,)](
            input, other, mid, p_scalar, MODE=mode, BLOCK=_BLOCK
        )
        if tail:
            s, np, ns = _piece_args(tail)
            _dist_tail_kernel[(1,)](
                input,
                other,
                mid,
                p_full * _BLOCK,
                p_full,
                p_scalar,
                MODE=mode,
                S=s,
                NP=np,
                NSCALAR=ns,
            )
        cur = mid
        cur_n = p_cnt
        while triton.next_power_of_2(cur_n) > _MID_BLOCK:
            g = triton.cdiv(cur_n, _MID_BLOCK)
            nxt = torch.full(
                (triton.next_power_of_2(g),),
                pad,
                device=input.device,
                dtype=torch.float32,
            )
            _dist_mid_reduce_kernel[(g,)](
                cur, nxt, MODE=mode, BLOCK=_MID_BLOCK
            )
            cur = nxt
            cur_n = g
        _dist_final_kernel[(1,)](
            cur,
            out,
            p_scalar,
            MODE=mode,
            BLOCK_MID=triton.next_power_of_2(cur_n),
        )

    return out