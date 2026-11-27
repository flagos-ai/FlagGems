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

"""Triton implementation of aten::glu_jvp (forward-mode JVP of aten::glu).

Semantics
---------
With ``d`` the (non-negative) narrowed dim, ``N = glu.size(d)`` and
``a = x[..., 0:N, ...]``, ``b = x[..., N:2N, ...]`` (narrowed along ``d``), the
primal is ``glu = a * sigmoid(b)`` and the pushforward is

    out = da * sigmoid(b) + glu * (1 - sigmoid(b)) * db

so the first half of ``x`` is never read.  The output shape is the broadcast of
``glu`` and the three narrowed operands over every axis except ``d``, with the
``d`` axis fixed to ``N`` (native ``narrow(dim, m, m)`` semantics).

Kernel shape
------------
The output is viewed as ``[M, K]`` (``M = prod(out_shape[:-1])``, ``K =
out_shape[-1]``).  Two device kernels serve the two layout families:

* **row-tile streaming kernel** (the dense case, every benchmark shape): the
  output is contiguous and every operand follows a uniform row-step layout
  that the wrapper proves in Python (the same ``_is_uniform`` contract the
  old 2-D FAST mode used), so each tile covers ``ROWS`` whole output rows by
  ``BLOCK >= K`` lanes — 32K lanes per tile for 2-byte inputs, 16K for 4-byte
  — with pure 2-D affine addressing.  The grid is a *fixed* 48 CTAs that
  stride over the row blocks, which keeps GCU400's 65535-program x limit
  (P19) satisfied for any input while giving the DMA engine long contiguous
  streams.  The affine
  form is required, not a preference: the flat ``idx // K`` per-element
  decomposition silently miscompiles on this stack (P13) while the 2-D
  affine form is the documented bypass.  ``EVEN`` drops the bounds masks
  when ``M`` divides ``ROWS`` and ``K`` fills ``BLOCK``, and ``IS32``
  keeps the address arithmetic in int32 whenever every operand's exact
  offset bound fits — int64 addressing costs ~+58% on GCU400
  (glu_backward_jvp probe, 2026-11-17).
* **general kernel**: the legacy rank-5 digit-decomposed addressing for
  broadcast and non-uniform strides.  Its launch folds the
  ``(row_block, col_block)`` tile pair into a flat program id with a
  grid-stride loop, so oversized M can no longer overflow grid.x either.

Broadcast operands never reach the flat kernel: a size-1 outer axis makes the
row step zero and a size-1 last axis makes the column step zero, and both are
explicitly routed to the general kernel.  Besides being unnecessary there, the
``rows * 0`` tile-load form is a device-side SIGKILL on this stack
(compiler_issues/flaggems_glu_jvp_broadcast_stride0_load_fault.md) while the
digit-addition form used by the general kernel is verified safe on the same
data.  The same routing removes the old FAST mode's strided-``out`` mis-address
(the hardcoded ``o_r = K`` row step), because a non-contiguous output now
always takes the general path.

The wrapper performs metadata work only (dim wrap, extension checks, zero-copy
``narrow`` views, broadcast-stride alignment, output allocation) and a single
device kernel evaluates the expression above with fp32 intermediates for
half/float inputs and fp64 for double inputs, rounding exactly once on store.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# The op is float-only and, like the framework reference, supports exactly the
# four real floating point widths (integers, bool, fp8 and complex are
# rejected).  torch reports fp8 dtypes as floating point, so an explicit
# membership test is what rejects them.
_FLOAT_DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float64,
)

# Row-index slots in the general kernel.  Up to ``_SLOTS`` leading output axes
# are addressed directly; every workload in the frozen contract has rank <= 6.
_SLOTS = 5
_ROWS_RANK_MAX = _SLOTS

# GCU400 grid hard limits (P19): the x axis holds at most 65535 programs and
# the y/z axes only 255 — CUDA's 2**31 x-axis assumption does not hold here.
_GRID_X_HARD_LIMIT = 65535
_GRID_Y_HARD_LIMIT = 255

# Flat streaming kernel geometry.  A fixed 48-CTA grid striding over big row
# tiles won the same-job parameter sweep on the core shapes and matches the
# Fixed-Grid recipe the other wide-stream ops in this backend converged on.
# The lane count is dtype-banded so the per-buffer tile footprint stays
# constant: 2-byte inputs take 32K lanes, 4-byte inputs 16K (interleaved
# 2-round sweep, 2026-11-18: 32K lanes win -8%..-16% on every large 16-bit
# core case — 16Mx64 4440->4081us, 4096^2 63.6->56.7us, 1024^3 3341->2816us
# — while the same 32K lanes on fp32 press the DSM budget and win nothing).
_FLAT_NCTA = 48
_FLAT_TILE_2B = 32768
_FLAT_TILE_4B = 16384
_FLAT_WARPS = 1

# General-path launch cap: one CTA per (row, col) tile up to this many, then a
# grid-stride loop takes over (probe sweet spot ~1024 from glu_backward_jvp).
_GENERAL_NCTA_CAP = 1024


@triton.jit
def _glu_jvp_rowtile_kernel(
    g_ptr,
    xb_ptr,
    da_ptr,
    db_ptr,
    o_ptr,
    M,
    g_r,
    x_r,
    a_r,
    b_r,
    g_c,
    x_c,
    a_c,
    b_c,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    NRBLK: tl.constexpr,
    NCTA: tl.constexpr,
    EVEN: tl.constexpr,
    IS64: tl.constexpr,
    F64: tl.constexpr,
):
    """Row-tile streaming kernel for the dense layout.

    Every program strides over ``ceil(M / ROWS)`` row blocks; one tile covers
    ``ROWS`` whole output rows by ``BLOCK >= K`` lanes, so every address is the
    2-D affine ``row[:, None] * row_step + col[None, :] * col_step`` form with
    ``row``/``col`` taken straight from ``arange`` — no per-element index
    division anywhere.  That is deliberate: the flat ``idx // K`` /
    ``idx - row * K`` decomposition this host used first is a P13
    index-derived-addressing miscompile on this stack (row 0 correct, every
    later row silently reading at the wrong stride, see
    compiler_problems/P13_index_derived_addressing_miscompile), while the
    2-D affine form is the documented P13 bypass and matches the addressing
    the previous verified kernel used.

    ``EVEN`` drops the bounds masks when ``M`` divides ``ROWS`` and ``K``
    fills ``BLOCK`` exactly; ``IS64`` widens the address chain only when some
    operand's exact offset bound exceeds int32.
    """
    pid = tl.program_id(0)
    if IS64:
        pid = pid.to(tl.int64)
    for rb in range(pid, NRBLK, NCTA):
        rows = rb * ROWS + tl.arange(0, ROWS)
        if IS64:
            rows = rows.to(tl.int64)
        cols = tl.arange(0, BLOCK)
        g_off = rows[:, None] * g_r + cols[None, :] * g_c
        x_off = rows[:, None] * x_r + cols[None, :] * x_c
        a_off = rows[:, None] * a_r + cols[None, :] * a_c
        b_off = rows[:, None] * b_r + cols[None, :] * b_c
        o_off = rows[:, None] * K + cols[None, :]
        if EVEN:
            g = tl.load(g_ptr + g_off)
            xb = tl.load(xb_ptr + x_off)
            da = tl.load(da_ptr + a_off)
            db = tl.load(db_ptr + b_off)
        else:
            m = (rows < M)[:, None] & (cols < K)[None, :]
            g = tl.load(g_ptr + g_off, mask=m, other=0.0)
            xb = tl.load(xb_ptr + x_off, mask=m, other=0.0)
            da = tl.load(da_ptr + a_off, mask=m, other=0.0)
            db = tl.load(db_ptr + b_off, mask=m, other=0.0)

        if F64:
            xf = xb.to(tl.float64)
            gf = g.to(tl.float64)
            af = da.to(tl.float64)
            bf = db.to(tl.float64)
        else:
            xf = xb.to(tl.float32)
            gf = g.to(tl.float32)
            af = da.to(tl.float32)
            bf = db.to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-xf))
        res = af * sig + gf * ((1.0 - sig) * bf)

        if EVEN:
            tl.store(o_ptr + o_off, res)
        else:
            tl.store(o_ptr + o_off, res, mask=m)


@triton.jit
def _glu_jvp_kernel(
    g_ptr,
    xb_ptr,
    da_ptr,
    db_ptr,
    o_ptr,
    M,
    K,
    g_b0,
    g_b1,
    g_b2,
    g_b3,
    g_b4,
    g_s,
    x_b0,
    x_b1,
    x_b2,
    x_b3,
    x_b4,
    x_s,
    a_b0,
    a_b1,
    a_b2,
    a_b3,
    a_b4,
    a_s,
    b_b0,
    b_b1,
    b_b2,
    b_b3,
    b_b4,
    b_s,
    o_b0,
    o_b1,
    o_b2,
    o_b3,
    o_b4,
    o_s,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    T4: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    NKB: tl.constexpr,
    NTILE: tl.constexpr,
    NCTA: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    F64: tl.constexpr,
    BC_G: tl.constexpr,
    BC_X: tl.constexpr,
    BC_A: tl.constexpr,
    BC_B: tl.constexpr,
):
    """General kernel: rank-5 digit-decomposed addressing for broadcast and
    non-uniform layouts.

    The (row-block, col-block) tile pair is folded into one flat program id
    with a grid-stride loop, so an M of any size stays inside GCU400's
    grid limits.  Broadcast strides ride the digit additions (``i_k * 0``),
    the form verified not to fault (unlike the ``rows * 0`` multiply form the
    old FAST mode used).

    ``BC_*`` mark operands whose *last* axis is broadcast (column stride 0):
    the column term would be the ``cols * 0`` multiply-zero form — the same
    device-fault family as ``rows * 0`` (verified 2026-11-18: the general
    kernel SIGKILLs on ``glu[:, :, :1]`` without these flags) — so those
    operands load once per row through the digit add-chain and broadcast
    across the tile instead.
    """
    pid = tl.program_id(0) + tl.program_id(1) * GRID_STRIDE
    for t in range(pid, NTILE, NCTA):
        rb = t // NKB
        cb = t - rb * NKB
        rows = (rb * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        cols = cb * BLOCK + tl.arange(0, BLOCK)

        r = rows
        i0 = r // T1
        r = r % T1
        i1 = r // T2
        r = r % T2
        i2 = r // T3
        r = r % T3
        i3 = r // T4
        i4 = r % T4

        g_b = i0 * g_b0 + i1 * g_b1 + i2 * g_b2 + i3 * g_b3 + i4 * g_b4
        x_b = i0 * x_b0 + i1 * x_b1 + i2 * x_b2 + i3 * x_b3 + i4 * x_b4
        a_b = i0 * a_b0 + i1 * a_b1 + i2 * a_b2 + i3 * a_b3 + i4 * a_b4
        b_b = i0 * b_b0 + i1 * b_b1 + i2 * b_b2 + i3 * b_b3 + i4 * b_b4
        o_b = i0 * o_b0 + i1 * o_b1 + i2 * o_b2 + i3 * o_b3 + i4 * o_b4

        rm = rows < M
        m = rm[:, None] & (cols < K)[None, :]
        c = cols[None, :]

        if BC_G:
            g = tl.load(g_ptr + g_b, mask=rm, other=0.0)[:, None]
        else:
            g = tl.load(g_ptr + g_b[:, None] + c * g_s, mask=m, other=0.0)
        if BC_X:
            xb = tl.load(xb_ptr + x_b, mask=rm, other=0.0)[:, None]
        else:
            xb = tl.load(xb_ptr + x_b[:, None] + c * x_s, mask=m, other=0.0)
        if BC_A:
            da = tl.load(da_ptr + a_b, mask=rm, other=0.0)[:, None]
        else:
            da = tl.load(da_ptr + a_b[:, None] + c * a_s, mask=m, other=0.0)
        if BC_B:
            db = tl.load(db_ptr + b_b, mask=rm, other=0.0)[:, None]
        else:
            db = tl.load(db_ptr + b_b[:, None] + c * b_s, mask=m, other=0.0)

        if F64:
            xf = xb.to(tl.float64)
            gf = g.to(tl.float64)
            af = da.to(tl.float64)
            bf = db.to(tl.float64)
        else:
            xf = xb.to(tl.float32)
            gf = g.to(tl.float32)
            af = da.to(tl.float32)
            bf = db.to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-xf))
        res = af * sig + gf * ((1.0 - sig) * bf)

        tl.store(o_ptr + o_b[:, None] + c * o_s, res, mask=m)


@triton.jit
def _glu_jvp_row_kernel(
    g_ptr,
    xb_ptr,
    da_ptr,
    db_ptr,
    o_ptr,
    K,
    g_s,
    x_s,
    a_s,
    b_s,
    o_s,
    F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single output row (``M == 1``), evaluated with a 1-D tile over ``K``.

    The 2-D tile kernel faults the device when the row axis carries a
    single valid row (the following synchronisation SIGABRTs the process), so
    this variant drops the row axis entirely: every operand base is the row-0
    offset, which the launcher already folded into the tensor pointer.
    """
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0).to(tl.int64)
    for start in tl.range(pid * BLOCK, K, num_programs * BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < K

        g = tl.load(g_ptr + cols * g_s, mask=m, other=0.0)
        xb = tl.load(xb_ptr + cols * x_s, mask=m, other=0.0)
        da = tl.load(da_ptr + cols * a_s, mask=m, other=0.0)
        db = tl.load(db_ptr + cols * b_s, mask=m, other=0.0)

        if F64:
            xf = xb.to(tl.float64)
            gf = g.to(tl.float64)
            af = da.to(tl.float64)
            bf = db.to(tl.float64)
        else:
            xf = xb.to(tl.float32)
            gf = g.to(tl.float32)
            af = da.to(tl.float32)
            bf = db.to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-xf))
        res = af * sig + gf * ((1.0 - sig) * bf)

        tl.store(o_ptr + cols * o_s, res, mask=m)


def _strides(t):
    """Strides with the broadcast (size-1) axes zeroed so they read one element."""
    return [0 if s == 1 else st for s, st in zip(t.shape, t.stride())]


def _is_uniform(lead, outer, row_step):
    """True when ``outer`` is exactly the dense layout of ``[*lead, K]`` scaled
    by ``row_step``, i.e. the flat row index maps linearly onto the operand."""
    if len(outer) != len(lead):
        return False
    tail = 1
    for k in range(len(lead) - 1, -1, -1):
        if outer[k] != row_step * tail:
            return False
        tail *= lead[k]
    return True


def _offset_bound(t):
    """Exact max element offset ``sum((size_i - 1) * stride_i) + 1`` reachable
    from the tensor's data pointer (glu_backward_jvp precedent, 2026-11-17).

    The coarse ``numel * max(stride)`` form flags every dense big tensor as
    needing 64-bit arithmetic even when the footprint fits int32; negative
    strides (never produced by the aten path) fall back to the conservative
    bound.
    """
    if t.numel() == 0:
        return 0
    strides = t.stride()
    if any(s < 0 for s in strides):
        return t.numel() * max(abs(s) for s in strides)
    return 1 + sum(
        (t.shape[i] - 1) * strides[i] for i in range(t.dim()) if t.shape[i] > 1
    )


def glu_jvp(glu, x, dx, dim, out=None):
    logger.debug("GEMS_ENFLAME GLU_JVP")
    # The op is float-only, and only the four real float types are supported:
    # torch reports fp8 dtypes as floating point, so membership in this explicit
    # set (rather than ``is_floating_point``) is what rejects them along with
    # integers, bool and complex.  The pytest expects (RuntimeError, TypeError)
    # for those rejection workloads.
    if not all(t.dtype in _FLOAT_DTYPES for t in (glu, x, dx)):
        raise TypeError(
            "glu_jvp: only float16/float32/bfloat16/float64 tensors are supported"
        )

    nd = x.dim()
    d = dim + nd if dim < 0 else dim
    if d < 0 or d >= nd:
        raise IndexError(
            f"glu_jvp: dimension out of range (expected to be in range of "
            f"[{-nd}, {nd - 1}], but got {dim})"
        )
    for t in (glu, dx):
        if d >= t.dim():
            raise IndexError(
                f"glu_jvp: dimension out of range (expected to be in range of "
                f"[{-t.dim()}, {t.dim() - 1}], but got {dim})"
            )

    N = glu.size(d)
    if 2 * N > x.size(d) or 2 * N > dx.size(d):
        raise RuntimeError(
            "glu_jvp: the narrowed extent does not fit within the operand"
        )

    ops = (glu, x.narrow(d, N, N), dx.narrow(d, 0, N), dx.narrow(d, N, N))
    rank = max(t.dim() for t in ops)
    if rank - 1 > _ROWS_RANK_MAX:
        raise RuntimeError(
            f"glu_jvp: output rank above {_ROWS_RANK_MAX + 1} is not supported"
        )

    # Output shape: broadcast over the axes that are not narrowed, with N on d.
    out_shape = []
    for a in range(rank):
        size = 1
        for t in ops:
            k = a - (rank - t.dim())
            if k < 0:
                continue
            s = t.size(k)
            if s != 1:
                if size != 1 and s != size:
                    raise RuntimeError(
                        "glu_jvp: operands are not broadcastable to a common shape"
                    )
                size = s
        out_shape.append(size)

    if out is None:
        res = torch.empty(out_shape, dtype=glu.dtype, device=glu.device)
    else:
        if tuple(out.shape) != tuple(out_shape):
            raise RuntimeError(
                "glu_jvp: the out buffer does not match the expected output shape"
            )
        if out.dtype != glu.dtype:
            raise RuntimeError("glu_jvp: the out buffer dtype must match glu")
        res = out

    M = 1
    for s in out_shape[:-1]:
        M *= s
    K = out_shape[-1]
    if M == 0 or K == 0:
        return res

    # Per-operand broadcast strides (0 on size-1 axes), right-aligned to the
    # output rank and left-padded to ``_SLOTS`` slots so the fastest row axis
    # lands in the last slot.
    rows_rank = rank - 1
    padn = _SLOTS - rows_rank
    lead = out_shape[:-1]
    stride_vecs = []
    last_stride = []
    outer = []
    for t in ops:
        pad = rank - t.dim()
        stride = [0] * pad + _strides(t)
        stride_vecs.append([0] * padn + stride[:-1])
        last_stride.append(stride[-1])
        outer.append(stride[:-1])

    o_stride = _strides(res)
    o_vec = [0] * padn + o_stride[:-1]

    if M == 1:
        # One output row: use the 1-D row kernel (the 2-D tile kernel faults
        # the device when the row axis holds a single valid row).
        row_block = 256
        row_grid = (min(triton.cdiv(K, row_block), 48),)
        _glu_jvp_row_kernel[row_grid](
            glu,
            ops[1],
            ops[2],
            ops[3],
            res,
            K,
            last_stride[0],
            last_stride[1],
            last_stride[2],
            last_stride[3],
            o_stride[-1],
            F64=(glu.dtype == torch.float64),
            BLOCK=row_block,
        )
        return res

    # ---- row-tile streaming path ---------------------------------------------
    # Eligibility: a contiguous output plus, for every operand, a uniform outer
    # layout with a non-zero row step and a non-zero column step.  A zero row
    # step (outer broadcast) or zero column step (last-axis broadcast) routes
    # to the general kernel: besides being slower, the ``rows * 0`` multiply
    # form is a device SIGKILL
    # (compiler_issues/flaggems_glu_jvp_broadcast_stride0_load_fault.md).
    # K must fit one tile column (``next_pow2(K) <= tile``); wider K falls
    # through to the general kernel.
    tile = (
        _FLAT_TILE_2B if glu.dtype in (torch.float16, torch.bfloat16) else _FLAT_TILE_4B
    )
    flat = res.is_contiguous() and triton.next_power_of_2(K) <= tile
    if flat:
        for s, cs in zip(outer, last_stride):
            if cs == 0 or s[-1] == 0 or not _is_uniform(lead, s, s[-1]):
                flat = False
                break
    if flat:
        block = triton.next_power_of_2(K)
        # Keep the banded lane count per tile by growing the row count as the
        # column count shrinks, but never past M itself.
        rows_per = min(tile // block, triton.next_power_of_2(max(M, 1)))
        nrblk = triton.cdiv(M, rows_per)
        is64 = max(_offset_bound(t) for t in ops + (res,)) >= 2**31
        row_steps = [s[-1] for s in outer]
        _glu_jvp_rowtile_kernel[(_FLAT_NCTA,)](
            glu,
            ops[1],
            ops[2],
            ops[3],
            res,
            M,
            row_steps[0],
            row_steps[1],
            row_steps[2],
            row_steps[3],
            last_stride[0],
            last_stride[1],
            last_stride[2],
            last_stride[3],
            K=K,
            ROWS=rows_per,
            BLOCK=block,
            NRBLK=nrblk,
            NCTA=_FLAT_NCTA,
            EVEN=(M % rows_per == 0 and block == K),
            IS64=is64,
            F64=(glu.dtype == torch.float64),
            num_warps=_FLAT_WARPS,
        )
        return res

    # ---- general path -------------------------------------------------------
    slots = [1] * padn + lead
    T1 = slots[1] * slots[2] * slots[3] * slots[4]
    T2 = slots[2] * slots[3] * slots[4]
    T3 = slots[3] * slots[4]
    T4 = slots[4]

    # Tile geometry: a fixed 16 rows per program with the column count following
    # K (clamped to 16..64).  Same-job sweeps over the core cases put the
    # tested geometries within 2% of each other, so this keeps the geometry
    # that measured best.
    BLOCK = triton.next_power_of_2(K)
    BLOCK = 64 if BLOCK > 64 else (16 if BLOCK < 16 else BLOCK)
    ROWS = 16

    nkb = triton.cdiv(K, BLOCK)
    nrb = triton.cdiv(M, ROWS)
    ntile = nrb * nkb
    # Fold the (row-block, col-block) pair into a flat program id; the CTA
    # count is capped and the kernel strides over the surplus tiles, which is
    # what keeps oversized M inside the GCU400 grid limits (P19).
    ncta = min(ntile, _GENERAL_NCTA_CAP)
    grid_stride = min(
        ncta,
        min(
            _GRID_X_HARD_LIMIT,
            max(_GENERAL_NCTA_CAP, triton.cdiv(ncta, _GRID_Y_HARD_LIMIT)),
        ),
    )
    grid = (grid_stride, triton.cdiv(ncta, grid_stride))

    g4, x4, a4, b4 = stride_vecs
    _glu_jvp_kernel[grid](
        glu,
        ops[1],
        ops[2],
        ops[3],
        res,
        M,
        K,
        g4[0],
        g4[1],
        g4[2],
        g4[3],
        g4[4],
        last_stride[0],
        x4[0],
        x4[1],
        x4[2],
        x4[3],
        x4[4],
        last_stride[1],
        a4[0],
        a4[1],
        a4[2],
        a4[3],
        a4[4],
        last_stride[2],
        b4[0],
        b4[1],
        b4[2],
        b4[3],
        b4[4],
        last_stride[3],
        o_vec[0],
        o_vec[1],
        o_vec[2],
        o_vec[3],
        o_vec[4],
        o_stride[-1],
        T1=T1,
        T2=T2,
        T3=T3,
        T4=T4,
        ROWS=ROWS,
        BLOCK=BLOCK,
        NKB=nkb,
        NTILE=ntile,
        NCTA=ncta,
        GRID_STRIDE=grid_stride,
        F64=(glu.dtype == torch.float64),
        BC_G=(last_stride[0] == 0),
        BC_X=(last_stride[1] == 0),
        BC_A=(last_stride[2] == 0),
        BC_B=(last_stride[3] == 0),
    )
    return res


# ---------------------------------------------------------------------------
# Namespace sync.  ``SpecOpRegistrar`` only rebinds the top-level ``flag_gems``
# namespace, so the ``flag_gems.ops.glu_jvp`` submodule attribute — the object
# benchmark/test_glu_jvp.py binds as ``gems_op`` — keeps pointing at the
# generic implementation.  The generic launch keeps CUDA's grid.x <= 2**31
# assumption and overflows GCU400's 65535-program hard limit (P19) on the
# ((16777216, 64), dim=-1) core case: grid = (1048576, 1).  Rebind the package
# attribute to this specialized host so every reference path (top level, ATen
# registration, ops submodule) runs the GCU400 kernel.  ``flag_gems.ops`` is
# fully imported before the registrar loads this module, so the lookup below
# is a plain sys.modules hit.
try:
    import flag_gems.ops as _generic_ops_pkg
except ImportError:  # pragma: no cover - flag_gems.ops always importable
    _generic_ops_pkg = None

if (
    _generic_ops_pkg is not None
    and getattr(_generic_ops_pkg, "glu_jvp", None) is not glu_jvp
):
    _generic_ops_pkg.glu_jvp = glu_jvp
