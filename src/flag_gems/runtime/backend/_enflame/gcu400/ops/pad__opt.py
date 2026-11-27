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

"""pad v6: constant-mode 1D/2D/3D aligned-load phase kernel + tiered launch tuning.

On top of the v5.7 structure (2026-11-18 second-round probe calibration, kernel body untouched,
host dispatch params only):
- rank3 num_warps tiering: large shapes (n_out>=2^23 and row groups>=3, copy dominated) use nw1
  (1024^3 both dtypes +10~15%, (64,512,512) +6~12%); fill-dominated small-group shapes keep nw4.
- rank3 U self-adaptation: U takes the largest power-of-two factor of IN_D1 that is <= u_cap, so
  nfull1>=1 always holds and the whole row no longer drops into the per-row slow path when
  IN_D1<U ((1024,64,4096) f16 measured 4.1x).
- rank3 np1: fixed at 8 when row groups<=2 (span/bad-plane fill parallelism); fp32 small-batch np0
  raised to 16.
- rank2 nw tiering: row groups>=48 use nw1 (4096^2 +2~6%), otherwise nw4.

GCU400 load alignment law (2026-11-18 probe calibration):
- load-side misalignment (e.g. by -PB_last elements) triggers vectorization degradation and the
  bandwidth collapses 5x (fp16 2^30: 595GB/s vs 2947GB/s aligned); store-side misalignment costs
  only ~10%. Therefore the copy backbone tiles by input-space alignment and stores may be misaligned.
- In-kernel algebraically equivalent round-trip expressions such as `V0 + g*U + rows - V0` are not
  folded by the front end, which breaks load-address alignment analysis (81us vs 36us for the
  direct form). Source/destination row vectors must each be affine expressions of the loop
  induction variables directly.
- Masked 2-D tile stores risk leakage/out-of-bounds when some lanes are masked (P15 family), so
  the large-tile backbone must be completely unmasked; fills go through flat spans (1-D masked)
  or small 2-D fill blocks (FILLB<=64, the P15 safe zone).
- Per-row strided small ops cost ~0.6us/row, so per-row fill is unusable; fills must fuse the
  whole group ([U, FILLB] single op, measured 5.9us/16M elements) or use a flat span.

Structure (v5.7 branch-free phases): all phases are unconditional grid-stride loops (zero cost
when trip=0):
  1) valid row groups [V0, V0+NFULL*U): pure 2-D tile load/store + 2-D column fill
  2) right-edge residual rows (< U rows): per-row 1-D masked
  3/4) top/bottom pad regions: flat span fill
rank3 does the same two phases at plane level (valid plane copy / bad plane tiled fill), with row
groups aligned to W2 to eliminate the left crossing. Small shapes (n_out < 2^18) and negative pad
go to the v4 flat kernel (verified); non-constant, non-contiguous, rank>3 fall back to production.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# --------------------------- v4 flat kernels (small shapes) ---------------------------


@triton.jit
def _pad_rank1_kernel(
    X,
    Out,
    N_OUT: tl.constexpr,
    VALUE: tl.constexpr,
    LEFT: tl.constexpr,
    N_IN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src_coord = out_offset - LEFT
    valid = (out_offset < N_OUT) & (src_coord >= 0) & (src_coord < N_IN)
    value_or_input = tl.load(X + src_coord, mask=valid, other=VALUE)
    tl.store(Out + out_offset, value_or_input, mask=out_offset < N_OUT)


@triton.jit
def _pad_rank2_kernel(
    X,
    Out,
    N_OUT: tl.constexpr,
    VALUE: tl.constexpr,
    OUT_S0: tl.constexpr,
    IN_S0: tl.constexpr,
    PB0: tl.constexpr,
    PB1: tl.constexpr,
    IN_D0: tl.constexpr,
    IN_D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    coord0 = out_offset // OUT_S0
    coord1 = out_offset - coord0 * OUT_S0
    src0 = coord0 - PB0
    src1 = coord1 - PB1
    valid = (
        (out_offset < N_OUT)
        & (src0 >= 0)
        & (src0 < IN_D0)
        & (src1 >= 0)
        & (src1 < IN_D1)
    )
    src_offset = src0 * IN_S0 + src1
    # Masked loads on the GCU backend still require pointers to lie in a valid range, so negative offsets must be
    # clamped first
    safe_src_offset = tl.where(valid, src_offset, 0)
    value_or_input = tl.load(X + safe_src_offset, mask=valid, other=VALUE)
    tl.store(Out + out_offset, value_or_input, mask=out_offset < N_OUT)


@triton.jit
def _pad_rank3_kernel(
    X,
    Out,
    N_OUT: tl.constexpr,
    VALUE: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    IN_S0: tl.constexpr,
    IN_S1: tl.constexpr,
    PB0: tl.constexpr,
    PB1: tl.constexpr,
    PB2: tl.constexpr,
    IN_D0: tl.constexpr,
    IN_D1: tl.constexpr,
    IN_D2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    coord0 = out_offset // OUT_S0
    rest0 = out_offset - coord0 * OUT_S0
    coord1 = rest0 // OUT_S1
    coord2 = rest0 - coord1 * OUT_S1
    src0 = coord0 - PB0
    src1 = coord1 - PB1
    src2 = coord2 - PB2
    valid = (
        (out_offset < N_OUT)
        & (src0 >= 0)
        & (src0 < IN_D0)
        & (src1 >= 0)
        & (src1 < IN_D1)
        & (src2 >= 0)
        & (src2 < IN_D2)
    )
    src_offset = src0 * IN_S0 + src1 * IN_S1 + src2
    safe_src_offset = tl.where(valid, src_offset, 0)
    value_or_input = tl.load(X + safe_src_offset, mask=valid, other=VALUE)
    tl.store(Out + out_offset, value_or_input, mask=out_offset < N_OUT)


def _pick_block(n_out):
    if n_out < 65536:
        return 1024, 2
    if n_out < (1 << 24):
        return 16384, 2
    return 32768, 4


def _launch(A, pad_before, pad_after, value):
    in_shape = tuple(A.shape)
    rank = len(in_shape)
    out_shape = tuple(in_shape[d] + pad_before[d] + pad_after[d] for d in range(rank))
    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    n_out = out.numel()
    BLOCK, num_warps = _pick_block(n_out)
    grid = (triton.cdiv(n_out, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n_out, BLOCK),)
    if rank == 1:
        _pad_rank1_kernel[grid](
            A,
            out,
            n_out,
            value,
            pad_before[0],
            in_shape[0],
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    elif rank == 2:
        _pad_rank2_kernel[grid](
            A,
            out,
            n_out,
            value,
            out_shape[1],
            in_shape[1],
            pad_before[0],
            pad_before[1],
            in_shape[0],
            in_shape[1],
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    else:
        _pad_rank3_kernel[grid](
            A,
            out,
            n_out,
            value,
            out_shape[1] * out_shape[2],
            out_shape[2],
            in_shape[1] * in_shape[2],
            in_shape[2],
            pad_before[0],
            pad_before[1],
            pad_before[2],
            in_shape[0],
            in_shape[1],
            in_shape[2],
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out


# --------------------------- v5 aligned phase kernels ---------------------------


@triton.jit
def _pad1_bulk_kernel(
    X,
    Out,
    VALUE,
    N_OUT: tl.constexpr,
    LEFT: tl.constexpr,
    N_IN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Input-aligned bulk: program p purely copies input block [p*B, (p+1)*B) (load/store fully
    # aligned or with +LEFT misalignment on the store side, both harmless); the last program handles the left/right pad
    # flat fill. Requires BLOCK | N_IN (guaranteed by host).
    pid = tl.program_id(0)
    NBLK = N_IN // BLOCK
    if pid < NBLK:
        off = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(Out + off + LEFT, tl.load(X + off))
    else:
        cols = tl.arange(0, BLOCK)
        for c0 in tl.range(0, LEFT, BLOCK):
            cc = c0 + cols
            tl.store(Out + cc, VALUE, mask=cc < LEFT)
        for c0 in tl.range(0, N_OUT - LEFT - N_IN, BLOCK):
            cc = c0 + cols
            tl.store(Out + LEFT + N_IN + cc, VALUE, mask=cc < N_OUT - LEFT - N_IN)


@triton.jit
def _pad2_phase_kernel(
    X,
    Out,
    VALUE,
    IN_D0: tl.constexpr,
    IN_D1: tl.constexpr,
    PB0: tl.constexpr,
    PA0: tl.constexpr,
    PB1: tl.constexpr,
    PA1: tl.constexpr,
    OUT_D0: tl.constexpr,
    OUT_D1: tl.constexpr,
    NPROG: tl.constexpr,
    NFULL: tl.constexpr,
    NPART: tl.constexpr,
    U: tl.constexpr,
    BC: tl.constexpr,
    NCHUNK: tl.constexpr,
    FILLB: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BC)
    V0 = PB0
    V1 = PB0 + IN_D0
    rows_v = tl.arange(0, U)
    # phase 1: valid row groups [V0, V0 + NFULL*U) — pure tile copy + 2D column fill
    for j in tl.range(pid, NFULL, NPROG):
        src_rows = j * U + rows_v
        rows = V0 + j * U + rows_v
        for k in tl.static_range(NCHUNK):
            cc = k * BC + cols
            v = tl.load(X + src_rows[:, None] * IN_D1 + cc[None, :])
            tl.store(Out + rows[:, None] * OUT_D1 + PB1 + cc[None, :], v)
        if FILLB > 0:
            fcols = tl.arange(0, FILLB)
            if PB1 > 0:
                tl.store(
                    Out + rows[:, None] * OUT_D1 + fcols[None, :],
                    VALUE,
                    mask=(fcols < PB1)[None, :],
                )
            if PA1 > 0:
                dstc = PB1 + IN_D1 + fcols
                tl.store(
                    Out + rows[:, None] * OUT_D1 + dstc[None, :],
                    VALUE,
                    mask=(dstc < OUT_D1)[None, :],
                )
    # phase 2: right-edge remaining rows [V0 + NFULL*U, V1)
    for pr in tl.range(pid, NPART, NPROG):
        src_row = NFULL * U + pr
        r = V0 + NFULL * U + pr
        for k in tl.static_range(NCHUNK):
            cc = k * BC + cols
            v = tl.load(
                X + src_row * IN_D1 + tl.minimum(cc, IN_D1 - 1), mask=cc < IN_D1
            )
            tl.store(
                Out + r * OUT_D1 + PB1 + cc,
                v,
                mask=(cc < IN_D1) & (PB1 + cc < OUT_D1),
            )
        if FILLB > 0:
            fcols = tl.arange(0, FILLB)
            if PB1 > 0:
                tl.store(Out + r * OUT_D1 + fcols, VALUE, mask=fcols < PB1)
            if PA1 > 0:
                dstc = PB1 + IN_D1 + fcols
                tl.store(Out + r * OUT_D1 + dstc, VALUE, mask=dstc < OUT_D1)
    # phase 3: top pad flat span [0, V0 * OUT_D1)
    span_top = V0 * OUT_D1
    for c0 in tl.range(pid * BC, span_top, NPROG * BC):
        cc = c0 + cols
        tl.store(Out + cc, VALUE, mask=cc < span_top)
    # phase 4: bottom pad flat span [V1 * OUT_D1, OUT_D0 * OUT_D1)
    span_bot = (OUT_D0 - V1) * OUT_D1
    for c0 in tl.range(pid * BC, span_bot, NPROG * BC):
        cc = c0 + cols
        tl.store(Out + V1 * OUT_D1 + cc, VALUE, mask=cc < span_bot)


@triton.jit
def _pad3_phase_kernel(
    X,
    Out,
    VALUE,
    IN_D0: tl.constexpr,
    IN_D1: tl.constexpr,
    IN_D2: tl.constexpr,
    PB0: tl.constexpr,
    PA0: tl.constexpr,
    PB1: tl.constexpr,
    PA1: tl.constexpr,
    PB2: tl.constexpr,
    PA2: tl.constexpr,
    OUT_D0: tl.constexpr,
    OUT_D1: tl.constexpr,
    OUT_D2: tl.constexpr,
    NP0: tl.constexpr,
    NP1: tl.constexpr,
    NFULL1: tl.constexpr,
    NPART1: tl.constexpr,
    NBAD0: tl.constexpr,
    U: tl.constexpr,
    BC: tl.constexpr,
    NCHUNK: tl.constexpr,
    FILLB: tl.constexpr,
    SPANB: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    cols = tl.arange(0, BC)
    scols = tl.arange(0, SPANB)
    W0 = PB0
    W1 = PB0 + IN_D0
    W2 = PB1
    W3 = PB1 + IN_D1
    rows_v = tl.arange(0, U)
    # phase A: valid planes [W0, W1) (plane-atomic, no cross-boundary)
    for pj in tl.range(pid0, IN_D0, NP0):
        row_in = pj * IN_D1
        row_out = (W0 + pj) * OUT_D1
        # clean d1 row groups [W2, W2 + NFULL1*U)
        for g in tl.range(pid1, NFULL1, NP1):
            src_rows = row_in + g * U + rows_v
            dst_rows = row_out + W2 + g * U + rows_v
            for k in tl.static_range(NCHUNK):
                cc = k * BC + cols
                v = tl.load(X + src_rows[:, None] * IN_D2 + cc[None, :])
                tl.store(Out + dst_rows[:, None] * OUT_D2 + PB2 + cc[None, :], v)
            if FILLB > 0:
                fcols = tl.arange(0, FILLB)
                if PB2 > 0:
                    tl.store(
                        Out + dst_rows[:, None] * OUT_D2 + fcols[None, :],
                        VALUE,
                        mask=(fcols < PB2)[None, :],
                    )
                if PA2 > 0:
                    dstc = PB2 + IN_D2 + fcols
                    tl.store(
                        Out + dst_rows[:, None] * OUT_D2 + dstc[None, :],
                        VALUE,
                        mask=(dstc < OUT_D2)[None, :],
                    )
        # d1 right-edge remaining rows
        for pr in tl.range(pid1, NPART1, NP1):
            src_row = row_in + NFULL1 * U + pr
            d1 = W2 + NFULL1 * U + pr
            for k in tl.static_range(NCHUNK):
                cc = k * BC + cols
                v = tl.load(
                    X + src_row * IN_D2 + tl.minimum(cc, IN_D2 - 1),
                    mask=cc < IN_D2,
                )
                tl.store(
                    Out + (row_out + d1) * OUT_D2 + PB2 + cc,
                    v,
                    mask=(cc < IN_D2) & (PB2 + cc < OUT_D2),
                )
            if FILLB > 0:
                fcols = tl.arange(0, FILLB)
                if PB2 > 0:
                    tl.store(
                        Out + (row_out + d1) * OUT_D2 + fcols,
                        VALUE,
                        mask=fcols < PB2,
                    )
                if PA2 > 0:
                    dstc = PB2 + IN_D2 + fcols
                    tl.store(
                        Out + (row_out + d1) * OUT_D2 + dstc,
                        VALUE,
                        mask=dstc < OUT_D2,
                    )
        # d1 top span (flat within a plane; large SPANB blocks amortize per-op overhead)
        span_t = W2 * OUT_D2
        for c0 in tl.range(pid1 * SPANB, span_t, NP1 * SPANB):
            cc = c0 + scols
            tl.store(Out + row_out * OUT_D2 + cc, VALUE, mask=cc < span_t)
        # d1 bottom span
        span_b = (OUT_D1 - W3) * OUT_D2
        for c0 in tl.range(pid1 * SPANB, span_b, NP1 * SPANB):
            cc = c0 + scols
            tl.store(Out + (row_out + W3) * OUT_D2 + cc, VALUE, mask=cc < span_b)
    # phase B: bad planes [0, W0) ∪ [W1, OUT_D0): flat fill of whole planes
    PSPAN = OUT_D1 * OUT_D2
    for bp in tl.range(pid0, NBAD0, NP0):
        p = tl.where(bp < W0, bp, W1 + bp - W0)
        base = p * PSPAN
        for c0 in tl.range(pid1 * SPANB, PSPAN, NP1 * SPANB):
            cc = c0 + scols
            tl.store(Out + base + cc, VALUE, mask=cc < PSPAN)


# --------------------------- dispatch ---------------------------

_SMALL_OUT = 1 << 18
_INT32_LIMIT = 1 << 31
_MAX_FILL_COLS = 64


def _pow2_divisor(n, cap, floor):
    """Largest power-of-two factor of n not exceeding cap; returns 0 when it is below floor."""
    b = cap
    while b >= floor and n % b != 0:
        b //= 2
    return b if b >= floor else 0


def _launch_v5(A, pad_before, pad_after, value):
    """Aligned phase path; returns None when the conditions do not hold (fall back to the generic one)."""
    in_shape = tuple(A.shape)
    rank = len(in_shape)
    if any(p < 0 for p in pad_before) or any(p < 0 for p in pad_after):
        return None
    out_shape = tuple(in_shape[d] + pad_before[d] + pad_after[d] for d in range(rank))
    n_out = 1
    for d in out_shape:
        n_out *= d
    if n_out == 0:
        return None
    if n_out > _INT32_LIMIT or A.numel() > _INT32_LIMIT:
        return None
    itemsize = A.element_size()
    if itemsize not in (2, 4):
        return None
    out = None
    if rank == 1:
        n_in = in_shape[0]
        if n_in < _SMALL_OUT:
            return None
        block = _pow2_divisor(n_in, min(1 << 17, (1 << 18) // itemsize), 1 << 14)
        if block == 0 or n_in // block > 65534:
            return None
        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        _pad1_bulk_kernel[(n_in // block + 1,)](
            A,
            out,
            value,
            n_out,
            pad_before[0],
            n_in,
            BLOCK=block,
            num_warps=2,
        )
        return out
    in_last = in_shape[-1]
    pb_last = pad_before[-1]
    pa_last = pad_after[-1]
    if pb_last > _MAX_FILL_COLS or pa_last > _MAX_FILL_COLS:
        return None
    fillb = 1
    while fillb < max(pb_last, pa_last, 1):
        fillb *= 2
    if rank == 2:
        if n_out < _SMALL_OUT:
            return None
        in_d1 = in_last
        bc = _pow2_divisor(in_d1, min(4096, (1 << 18) // (32 * itemsize)), 128)
        if bc == 0:
            return None
        nchunk = in_d1 // bc
        nfull = in_shape[0] // 32
        npart = in_shape[0] % 32
        nprog = min(48, max(nfull + npart, 1))
        # Single warp is faster when the number of row groups fills all 48 programs (4096^2 f16 +2% / f32 +6%);
        # small shapes with too few groups instead need nw4 ((1024,4096) f16 nw1 -6%)
        nw = 1 if nfull + npart >= 48 else 4
        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        _pad2_phase_kernel[(nprog,)](
            A,
            out,
            value,
            IN_D0=in_shape[0],
            IN_D1=in_d1,
            PB0=pad_before[0],
            PA0=pad_after[0],
            PB1=pb_last,
            PA1=pa_last,
            OUT_D0=out_shape[0],
            OUT_D1=out_shape[1],
            NPROG=nprog,
            NFULL=nfull,
            NPART=npart,
            U=32,
            BC=bc,
            NCHUNK=nchunk,
            FILLB=fillb,
            num_warps=nw,
        )
        return out
    # rank == 3
    if n_out < _SMALL_OUT:
        return None
    in_d2 = in_last
    u_cap = 128 if itemsize == 2 else 64
    # Adaptive U: choose U as the largest power-of-2 factor of IN_D1 that is <=u_cap (lower bound 8), ensuring
    # nfull1>=1,
    # to avoid all rows falling into the slow per-row path when IN_D1 < U ((1024,64,4096) f16 measured 4.1x)
    u = _pow2_divisor(in_shape[1], u_cap, 1)
    if u < 8:
        u = u_cap if in_shape[1] < 8 else 8
    bc = _pow2_divisor(in_d2, min(1024, (1 << 18) // (u * itemsize)), 128)
    if bc == 0:
        return None
    if u * bc * itemsize > (1 << 18):
        u //= 2
        bc = _pow2_divisor(in_d2, min(1024, (1 << 18) // (u * itemsize)), 128)
        if bc == 0:
            return None
    nchunk = in_d2 // bc
    nfull1 = in_shape[1] // u
    npart1 = in_shape[1] % u
    nbad0 = pad_before[0] + pad_after[0]
    # When there are too few row groups, spread span/bad-plane filling over 8 pid1 (span is a serial bottleneck for
    # small shapes)
    if nfull1 + npart1 <= 2:
        np1 = 8
    else:
        np1 = min(8, nfull1 + npart1)
    if itemsize == 4:
        # f32: increase pid0 parallelism for small batches (IN_D0<=128) ((64,512,512) f32 104->93us)
        if in_shape[0] <= 128:
            np0 = min(16, max(4, in_shape[0] // 4))
        else:
            np0 = min(12, max(4, in_shape[0] // 8))
    else:
        np0 = min(24, max(4, in_shape[0] // 8))
    np0 = max(np0, 1)
    # nw1 is enabled only for large shapes with >=3 row groups (copy-dominated): 1024^3 f16/f32 +10~15%,
    # (64,512,512) both dtypes +6~12%; fill-dominated shapes with few groups degrade instead (keep nw4)
    nw = 1 if (n_out >= (1 << 23) and nfull1 + npart1 >= 3) else 4
    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    _pad3_phase_kernel[(np0, np1)](
        A,
        out,
        value,
        IN_D0=in_shape[0],
        IN_D1=in_shape[1],
        IN_D2=in_d2,
        PB0=pad_before[0],
        PA0=pad_after[0],
        PB1=pad_before[1],
        PA1=pad_after[1],
        PB2=pb_last,
        PA2=pa_last,
        OUT_D0=out_shape[0],
        OUT_D1=out_shape[1],
        OUT_D2=out_shape[2],
        NP0=np0,
        NP1=np1,
        NFULL1=nfull1,
        NPART1=npart1,
        NBAD0=nbad0,
        U=u,
        BC=bc,
        NCHUNK=nchunk,
        FILLB=fillb,
        SPANB=4096,
        num_warps=nw,
    )
    return out


def pad(self, pad, mode="constant", value=None):
    """Same signature as the production entry: self is the input tensor and pad is a list of length 2*ndim."""
    logger.debug("GEMS_ENFLAME PAD")
    if value is None:
        value = 0.0
    rank = self.ndim
    if (
        mode == "constant"
        and self.is_contiguous()
        and rank <= 3
        and len(pad) == 2 * rank
    ):
        # F.pad convention: padding is given in (left, right) pairs starting from the innermost dimension
        pad_before = [0] * rank
        pad_after = [0] * rank
        for dim_from_last in range(rank):
            dim = rank - dim_from_last - 1
            pad_before[dim] = int(pad[2 * dim_from_last])
            pad_after[dim] = int(pad[2 * dim_from_last + 1])
        out = _launch_v5(self, pad_before, pad_after, float(value))
        if out is not None:
            return out
        n_out = 1
        for d in range(rank):
            n_out *= self.shape[d] + pad_before[d] + pad_after[d]
        if n_out < (1 << 18) and n_out > 0:
            return _launch(self, pad_before, pad_after, float(value))
    from flag_gems.runtime.backend._enflame.gcu400.ops.pad import pad as _public

    return _public(self, pad, mode, value)
