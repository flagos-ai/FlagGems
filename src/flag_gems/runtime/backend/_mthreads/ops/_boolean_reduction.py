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

"""Shared fused boolean reduction core for any/all (single codegen).

``all(x, dims) == !any(x == 0, dims)``: both operators share the same
execution architecture and differ only in combine polarity. This module
owns that architecture so ``ops.any`` and ``ops.all`` stay thin, symmetric
wrappers:

- host-side dim coalescing (output dims fastest-first in output-flat
  order, reduce dims fastest-first by input stride) so a generic
  rank-specialized kernel addresses any input layout with its real
  strides (no permute/contiguous materialization — strided views are
  reduced in place);
- rank-specialized codegen kernels, cached per (output-rank, reduce-rank,
  mode), with an ``IS_ALL`` constexpr selecting AND/zero-found polarity
  at compile time (single source, two pruned binaries);
- a 1D grid (this backend's 2D-grid program_id mapping is broken),
  decomposed inside the kernel into (output-tile, split);
- split-K along the reduction space with "partial" (workspace + merge
  kernel), "atomic" (int32 flags) or "none" (force single-pass) merge
  strategies;
- a chunk-unrolled k-loop (``UNROLL`` blocks per chunk): each chunk
  accumulates a 2D predicate tile elementwise and does a single
  row-reduce per chunk, and the per-row early-exit mask is
  loop-invariant inside a chunk, so the unrolled loads keep their
  memory-level parallelism. ``UNROLL=1`` reproduces the classic
  per-block structure; the polarity-specific tile/split tables live in
  ``_boolean_dims_config``;
- per-row early exit is expressed as load masking (``other`` neutral
  element), never as loop control flow, so the software pipeline stays
  intact;
- int32 in-kernel addressing with a headroom gate (``IDX64`` constexpr):
  offsets are computed in int32 whenever the host proves every derived
  index — including masked-lane slack — fits in int31;
- three host-side routes layered on top of the fused kernel (all
  disablable via env knobs for A/B testing):
  1. small dense full reductions go to the ``boolean_whole_impl``
     two-kernel cascade (launch-bound shapes where the cascade's
     coalesced linear scan beats the fused kernel's split machinery);
  2. contiguous inputs whose reduce dims form a memory-order band that
     excludes the fastest dim go to a transposed segment-column
     streaming kernel (``_boolean_segcol_kernel``): the input is viewed
     as [SEG, ROWS, M] and each CTA streams row-major (BLOCK_M-wide)
     contiguous tiles down its row band, replacing the 32-byte gather
     transactions the fused kernel suffers on m-contiguous layouts;
  3. contiguous narrow-M (32..256 columns) fp32-class deep-K bands
     (>= 256 MiB) go to a column-strip streaming kernel
     (``_mthreads_boolean_colstrip_*``): each CTA owns a contiguous row
     band cut so every (INNER, 32) tile load is fully unmasked with the
     warp's 32 lanes on 32 consecutive columns (128-byte requests), and
     the columns decompose into exact 32-wide chunks plus an exact
     tail — measured 1.14x Torch (1.43x the fused kernel) on
     [100, 65536, 100] fp32 where the fused kernel's 32-byte gather
     caps it at 0.78x;
  4. ``all`` over a dense tensor with a scalar output and >= 256 MiB
     footprint upgrades the split merge from partials to atomic int32
     flags (wave-granularity early exit; measured never-behind at
     >= 312 MiB, behind at <= 156 MiB — and strictly excluded for
     ``any``, where the per-CTA mask early exit already wins).
"""

import importlib.util
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

_NUM_STAGES = 3

# Test/A-B knob: force the int64 addressing variant to exercise the IDX64
# codegen branch on small tensors (never set in production).
_FORCE_IDX64 = os.environ.get("FLAGGEMS_BOOL_FORCE_IDX64", "0") not in (
    "0",
    "",
    "false",
    "False",
)

# Headroom for the int32 fast path: masked k lanes legitimately reach
# K + BLOCK_N * UNROLL and m lanes reach M + BLOCK_M; int32 wraparound
# there would turn `kk < k_rem` / `m < M` true for negative offsets and
# dereference them (proven silent-wrong-answer at K = 2**31).
_IDX32_HEADROOM = 1 << 16
_IDX32_LIMIT = 2**31 - _IDX32_HEADROOM


def _env_flag(name):
    return os.environ.get(name, "1") not in ("0", "", "false", "False")


# Host-side routing knobs (ablation / A-B testing; all default on).
_SEGCOL_ENABLED = _env_flag("FLAGGEMS_BOOL_SEGCOL")
_CASCADE_ROUTE_ENABLED = _env_flag("FLAGGEMS_BOOL_CASCADE_ROUTE")
_ATOMIC_ALL_ROUTE_ENABLED = _env_flag("FLAGGEMS_BOOL_ATOMIC_ALL")
_COLSTRIP_ENABLED = _env_flag("FLAGGEMS_BOOL_COLSTRIP")

# Narrow-M deep-K column-strip route: routed footprints (>= 256 MiB) and
# column counts (32 <= M <= 256, fp32-class 4-byte elements). Below 256 MiB
# the fused kernel's split machinery wins (measured at 96/156 MiB: 0.89x and
# 1.03x the fused kernel; at >= 312 MiB the strip wins everywhere measured).
_COLSTRIP_MIN_BYTES = 256 * 1024 * 1024
# Column-tile width. 32 keeps one warp instruction's lanes on 32 consecutive
# columns (a 128-byte contiguous request on 4-byte elements) — the layout the
# Triton-MUSA coalescer picks for (INNER, 32) tiles; wider (INNER, 64) tiles
# fall back to an 8x4 lane mapping (16-byte rows) and measured far slower.
_COLSTRIP_CHUNK_W = 32
# Rows per loop iteration (INNER): measured optimum 16 on
# [100, 65536, 100] fp32 (2.05 ms vs 2.07/2.39/2.87 for 8/32/64-shaped bands).
_COLSTRIP_INNER = 16
# Cap on routed column counts: each CTA holds ceil(M/32) column accumulators
# plus one (INNER, 32) tile in flight per chunk.
_COLSTRIP_MAX_M = 256

# Dense full reductions in this byte window route to the
# boolean_whole_impl cascade ([10000, 256]-class shapes: measured
# 1.2-2.6x over the fused path). The upper cap is polarity- and
# element-size-dependent: above it the fused kernel's finer-grained
# early exit wins (any on [4096, 4096] fp16 at 32 MiB measured 2.4x
# ahead of the cascade, all on the bool variant at 16 MiB 1.5x, any on
# [10000, 512] bool at 4.9 MiB 1.25x; every measured cell inside its
# window wins or ties). The cap is exclusive: the pow2 shapes sitting
# at exactly cap bytes give the fused kernel an ideal config and
# measured 1.03-1.09x ahead of the cascade. Boundary validated at
# +/- one row/column. The
# 1 MiB floor keeps tiny tensors on the single-launch fused path (64x64
# fp16 measured 1.2-1.4x behind on the cascade).
_CASCADE_MIN_BYTES = 1 * 1024 * 1024
_CASCADE_MAX_BYTES_ANY = 5 * 1024 * 1024
_CASCADE_MAX_BYTES_ANY_1B = 3 * 1024 * 1024
_CASCADE_MAX_BYTES_ALL = 10 * 1024 * 1024

# all-only scalar-output atomic upgrade threshold: >= 312 MiB measured
# never behind, 156 MiB behind by 2-3%, 39 MiB by 31-45%.
_ATOMIC_ALL_MIN_BYTES = 256 * 1024 * 1024


def _np2(x):
    return triton.next_power_of_2(max(int(x), 1))


# ---------------------------------------------------------------------------
# whole-tensor reduction (dim=None): two-kernel cascade over a dense input
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _boolean_whole_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    IS_ALL: tl.constexpr,
    IDX64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    # Widen before the multiply: pid * BLOCK_SIZE overflows int32 once the
    # linear offset passes 2^31 (inputs >= 2^31 elements).
    if IDX64:
        offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    if IS_ALL:
        inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
        val = tl.min(inp_val != 0, axis=0)
    else:
        inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
        val = tl.max(inp_val != 0, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, val)


@libentry()
@triton.jit
def _boolean_whole_kernel_2(
    mid, out, MID_SIZE, IS_ALL: tl.constexpr, BLOCK_MID: tl.constexpr
):
    if IS_ALL:
        acc = tl.full((BLOCK_MID,), 1, tl.int1)
    else:
        acc = tl.zeros((BLOCK_MID,), dtype=tl.int1)
    for off in range(0, MID_SIZE, BLOCK_MID):
        offset = off + tl.arange(0, BLOCK_MID)
        mid_ptrs = mid + offset
        mask = offset < MID_SIZE
        if IS_ALL:
            # AND-merge: out-of-range blocks must read as "all true".
            mid_val = tl.load(mid_ptrs, mask=mask, other=1).to(tl.int1)
            acc = acc & tl.min(mid_val, axis=0).to(tl.int1)
        else:
            mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
            acc = acc | tl.max(mid_val, axis=0).to(tl.int1)
    if IS_ALL:
        tl.store(out, tl.min(acc, axis=0).to(out.dtype.element_ty))
    else:
        tl.store(out, tl.max(acc, axis=0).to(out.dtype.element_ty))


def boolean_whole_impl(inp, is_all):
    """Two-kernel linear cascade for a dense whole-tensor reduction.

    The caller must have proven the layout dense (linear offsets are
    exactly ``{0 .. n_elements-1}``); strided inputs use
    ``boolean_dims_impl`` with every dim reduced instead.
    """
    n_elements = inp.numel()
    # ATen legacy: any/all keep uint8 dtype for uint8 inputs, bool otherwise.
    out_dtype = torch.uint8 if inp.dtype == torch.uint8 else torch.bool

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    block_size = min(block_size * 2, 4096, triton.next_power_of_2(n_elements))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = min(triton.next_power_of_2(mid_size), 4096)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=out_dtype, device=inp.device)

    num_warps_block = min(8, max(1, block_size // 128))
    num_warps_mid = min(8, max(1, block_mid // 128))

    with torch_device_fn.device(inp.device):
        _boolean_whole_kernel_1[(mid_size, 1)](
            inp,
            mid,
            n_elements,
            mid_size,
            IS_ALL=is_all,
            IDX64=n_elements >= 2**31,
            BLOCK_SIZE=block_size,
            num_warps=num_warps_block,
            num_stages=2,
        )
        _boolean_whole_kernel_2[(1, 1)](
            mid,
            out,
            mid_size,
            IS_ALL=is_all,
            BLOCK_MID=block_mid,
            num_warps=num_warps_mid,
            num_stages=2,
        )
    return out


# ---------------------------------------------------------------------------
# transposed segment-column streaming kernel ("segcol")
#
# For a contiguous input viewed as [SEG, ROWS, M] where the reduce dims
# form the ROWS band (a memory-order band that excludes the fastest
# dim), each CTA owns a BLOCK_M-wide column chunk and streams consecutive
# rows of its band: every load is a BLOCK_M-element contiguous tile
# (0.5-2 KiB), replacing the fused kernel's BLOCK_M-wide (16-32 B)
# gather at K*es stride that saturates L1 transaction issue rate before
# DRAM on m-contiguous layouts. Cross-CTA combine reuses the split-K
# partials machinery (SEG == 1) or one direct CTA per (segment, column
# chunk) (SEG > 1).
#
# Correctness invariants carried over from the fused kernel:
# - early exit is a tile-level local while-exit (all column lanes
#   resolved); dead columns keep loading but their acc update is
#   idempotent, so the load mask stays row-only — constant along the
#   contiguous axis, or the vectorizer degrades to per-element loads;
# - no cross-CTA state is ever polled (the wave-granularity atomic
#   strategy owns that job);
# - int32 addressing is gated by an IDX64 constexpr with headroom for
#   the masked-lane row overshoot (r_base + rr can overshoot ROWS by up
#   to CTA_ROWS + INNER rows; wraparound there is masked-off, but the
#   headroom keeps even those lanes' offsets in int31 range).
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _boolean_segcol_kernel(
    inp,
    out,
    M,
    ROWS,
    CTA_ROWS,
    IS_ALL: tl.constexpr,
    IDX64: tl.constexpr,
    M_TAIL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    mchunks = tl.cdiv(M, BLOCK_M)
    outer = pid // mchunks
    mc = pid % mchunks
    cols = mc * BLOCK_M + tl.arange(0, BLOCK_M)
    colp = cols[None, :]
    r_base = outer * CTA_ROWS
    row_live = r_base < ROWS
    if IS_ALL:
        acc = tl.full((1, BLOCK_M), 1, tl.int1)
    else:
        acc = tl.zeros((1, BLOCK_M), dtype=tl.int1)
    # Fully-out-of-range CTAs (split rounding) just store the identity.
    r = tl.where(row_live, 0, CTA_ROWS)
    while r < CTA_ROWS:
        for i in range(0, INNER):
            rr = r + i + tl.arange(0, 1)
            # rr < CTA_ROWS keeps the (rounded-up) last INNER band inside
            # this CTA's own row range. In direct mode (CTA_ROWS == rows of
            # one segment while ROWS is the global row count) the overshoot
            # lanes would otherwise read the *next* segment's rows and
            # corrupt that segment's directly-stored output. In split mode
            # CTA_ROWS is a multiple of INNER, so r + i never reaches
            # CTA_ROWS and the term is vacuously true.
            live = row_live & (rr < CTA_ROWS) & ((r_base + rr) < ROWS)
            if IDX64:
                row = (r_base + rr).to(tl.int64)
            else:
                row = r_base + rr
            ptrs = inp + row[:, None] * M + colp
            if M_TAIL:
                # Column tail (M not a multiple of BLOCK_M): the mask now
                # varies along the contiguous axis, which costs some
                # vectorization — only the last column chunk needs it.
                if IS_ALL:
                    v = tl.load(ptrs, mask=live[:, None] & (cols < M)[None, :], other=1)
                else:
                    v = tl.load(ptrs, mask=live[:, None] & (cols < M)[None, :], other=0)
            else:
                if IS_ALL:
                    v = tl.load(ptrs, mask=live[:, None], other=1)
                else:
                    v = tl.load(ptrs, mask=live[:, None], other=0)
            if IS_ALL:
                acc = (acc & (v != 0)).to(tl.int1)
            else:
                acc = (acc | (v != 0)).to(tl.int1)
        r += INNER
        # Tile-level early exit (local state only): stop paying loop ALU
        # once every column lane is resolved.
        if IS_ALL:
            if tl.max(acc.to(tl.int32)) == 0:
                r = CTA_ROWS
        else:
            if tl.min(acc.to(tl.int32)) == 1:
                r = CTA_ROWS
    if IDX64:
        optr = out + (outer.to(tl.int64) * M + cols)[None, :]
    else:
        optr = out + (outer * M + cols)[None, :]
    if M_TAIL:
        tl.store(optr, acc.to(out.dtype.element_ty), mask=(cols < M)[None, :])
    else:
        tl.store(optr, acc.to(out.dtype.element_ty))


def _segcol_view(shape, reduce_dims):
    """[SEG, ROWS, M] view of a contiguous shape, or None.

    Valid when the reduce dims are a contiguous dim band that does not
    include the fastest dim (M = product of the dims after the band,
    ROWS = product of the band, SEG = product of the dims before it).
    """
    red = sorted(set(reduce_dims))
    if not red or red != list(range(red[0], red[0] + len(red))):
        return None
    seg = math.prod(shape[: red[0]])
    rows = math.prod(shape[d] for d in red)
    m_cols = math.prod(shape[red[-1] + 1 :])
    return seg, rows, m_cols


def _segcol_config(seg, rows, M, dtype, element_size):
    """Tile plan for the segcol kernel, or None when the shape cannot
    fill the GPU (the fused kernel keeps those).

    Tuned on S5000 (prototype sweeps + merge A/B, do_bench protocol):
    - SEG == 1 (deep-K band): split along rows; base band count
      1024 (fp16 / 1-byte) or 2048 (bf16 / 4-byte — measured optima on
      1024^3 dim=[0,1]), scaled down with the footprint (a 2048-band
      plan on the 64 MiB [64, 512, 512] regressed 1.5-2.3x; 512 bands
      is the measured optimum class there).
    - SEG > 1 (mid-dim band): direct mode, one CTA per (segment,
      column chunk); needs >= 1024 CTAs or the direct grid starves.
      fp16 full scans want INNER=16 (INNER=8 measured 0.81-0.82x).
    """
    if seg == 1:
        if dtype == torch.float16:
            block_m, base_outer, num_warps = 1024, 1024, 4
        elif dtype == torch.bfloat16:
            block_m, base_outer, num_warps = 512, 2048, 2
        elif element_size == 4:
            block_m, base_outer, num_warps = 512, 2048, 2
        elif element_size == 1:
            block_m, base_outer, num_warps = 1024, 1024, 4
        else:
            block_m, base_outer, num_warps = 512, 2048, 2
        block_m = min(_np2(M), block_m)
        inner = 8
        target_outer = min(
            base_outer, max(512, seg * rows * M * element_size // (1 << 20))
        )
        outer = max(1, min(target_outer, triton.cdiv(rows, inner)))
        cta_rows = max(inner, triton.cdiv(rows, outer))
        cta_rows = triton.cdiv(cta_rows, inner) * inner
        outer = triton.cdiv(rows, cta_rows)
        grid = outer * triton.cdiv(M, block_m)
        if grid < 256:
            return None
        return {
            "direct": False,
            "BLOCK_M": block_m,
            "INNER": inner,
            "num_warps": num_warps,
            "cta_rows": cta_rows,
            "outer": outer,
            "grid": grid,
        }
    block_m = min(_np2(M), 1024)
    inner = 16 if dtype == torch.float16 else 8
    grid = seg * triton.cdiv(M, block_m)
    if grid < 1024:
        return None
    return {
        "direct": True,
        "BLOCK_M": block_m,
        "INNER": inner,
        "num_warps": 2,
        "cta_rows": rows,
        "outer": seg,
        "grid": grid,
    }


_SEGCOL_PLAN_CACHE = {}


def _plan_segcol(inp, reduce_dims):
    """Segcol applicability + plan, cached per (shape, dims, dtype).

    The segcol kernel addresses the input storage as a dense row-major
    ``[SEG * ROWS, M]`` array, so this route is only valid for
    contiguous inputs. The (shape, dims, dtype) cache key is sound only
    under that restriction: contiguous inputs of one shape share a
    single addressing, while a strided twin (transpose/permute view
    with the same shape/dims/dtype) is addressed differently and must
    never reuse a plan cached for its contiguous twin — doing so reads
    a permuted element set and silently produces wrong results
    (round7 micro P0). Non-contiguous inputs therefore return before
    touching the cache.

    Exclusions (all host-side, from the prototype report + merge A/B):
    - non-contiguous input, or reduce dims not a memory-order band
      excluding the fastest dim;
    - M < 512 or footprint < 512 MiB: narrow or small column bands
      measured behind the fused kernel in the merge A/B ([100, 256, 100]
      / [100, 65536, 100] dim=[0,1]: 100-wide tail-masked tiles 1.2-12.8x
      behind; [64, 512, 512] / [4096, 4096] dim=[0]: 32-64 MiB bands
      1.1-2.5x behind — the fused path's finer-grained early exit wins
      there). Every measured >= 512 MiB M>=512 cell wins for both
      polarities;
    - SEG == 1 with K far below M (1024^3 dim=[0]: measured 0.74x the
      fused kernel) unless SEG >= 64;
    - grid too small to fill the GPU (starved direct grids measured
      0.04-0.06x).
    """
    if not inp.is_contiguous():
        return None
    key = (tuple(inp.shape), tuple(reduce_dims), inp.dtype)
    if key in _SEGCOL_PLAN_CACHE:
        return _SEGCOL_PLAN_CACHE[key]
    plan = None
    view = _segcol_view(inp.shape, reduce_dims)
    if view is not None:
        seg, rows, m_cols = view
        if (
            m_cols >= 512
            and inp.numel() * inp.element_size() >= 512 * 1024 * 1024
            and (4 * rows >= m_cols or seg >= 64)
        ):
            cfg = _segcol_config(seg, rows, m_cols, inp.dtype, inp.element_size())
            if cfg is not None:
                plan = (seg, rows, m_cols, cfg)
    if len(_SEGCOL_PLAN_CACHE) < 4096:
        _SEGCOL_PLAN_CACHE[key] = plan
    return plan


def _segcol_impl(inp, plan, is_all, seg_plan):
    seg, rows, M, cfg = seg_plan
    # ATen legacy: any/all keep uint8 dtype for uint8 inputs, bool otherwise.
    out_dtype = torch.uint8 if inp.dtype == torch.uint8 else torch.bool
    block_m = cfg["BLOCK_M"]
    inner = cfg["INNER"]
    # int32 gate with masked-lane headroom: live lanes stay below numel,
    # the row overshoot of the last (rounded) band adds up to
    # (CTA_ROWS + INNER) rows and the column chunk adds M + BLOCK_M.
    idx64 = (
        _FORCE_IDX64
        or inp.numel() + (cfg["cta_rows"] + inner + 2) * M + block_m >= _IDX32_LIMIT
    )
    m_tail = M % block_m != 0
    with torch_device_fn.device(inp.device):
        if cfg["direct"]:
            out_flat = torch.empty(seg * M, dtype=torch.bool, device=inp.device)
            # Direct mode: segments tile the global row space exactly.
            _boolean_segcol_kernel[(cfg["grid"],)](
                inp,
                out_flat,
                M,
                seg * rows,
                cfg["cta_rows"],
                IS_ALL=is_all,
                IDX64=idx64,
                M_TAIL=m_tail,
                BLOCK_M=block_m,
                INNER=inner,
                num_warps=cfg["num_warps"],
                num_stages=_NUM_STAGES,
            )
        else:
            partial = torch.empty(
                (cfg["outer"], M), dtype=torch.int8, device=inp.device
            )
            _boolean_segcol_kernel[(cfg["grid"],)](
                inp,
                partial,
                M,
                rows,
                cfg["cta_rows"],
                IS_ALL=is_all,
                IDX64=idx64,
                M_TAIL=m_tail,
                BLOCK_M=block_m,
                INNER=inner,
                num_warps=cfg["num_warps"],
                num_stages=_NUM_STAGES,
            )
            out_flat = torch.empty(seg * M, dtype=torch.bool, device=inp.device)
            # Narrow output tiles: see the fused merge site note (2-CTA
            # starvation on multi-row outputs, 18x measured; M=1-class rows
            # prefer the wide tile).
            if seg * M >= 16:
                merge_bm = min(8, _np2(seg * M))
                merge_bs = min(64, _np2(cfg["outer"]))
                merge_warps = 2
            else:
                merge_bm = min(64, _np2(seg * M))
                merge_bs = min(128, _np2(cfg["outer"]))
                merge_warps = 4
            _boolean_merge_partial_kernel[(triton.cdiv(seg * M, merge_bm),)](
                partial,
                out_flat,
                seg * M,
                cfg["outer"],
                IS_ALL=is_all,
                BLOCK_M=merge_bm,
                BLOCK_S=merge_bs,
                num_warps=merge_warps,
                num_stages=2,
            )
    out = out_flat.view(plan["view_shape"])
    if out_dtype != torch.bool:
        out = out.to(out_dtype)
    return out


# ---------------------------------------------------------------------------
# narrow-M column-strip kernel ("colstrip")
#
# For a contiguous input viewed as [ROWS, M] (SEG == 1 segcol view: the reduce
# dims form a memory-order band excluding the fastest dim) with a *narrow*
# column count, each CTA streams a contiguous row band and reduces
# column-striped tiles: the row band is cut so that every load is a fully
# unmasked (INNER, 32) tile whose 32 column lanes are contiguous addresses
# (128-byte warp requests), and M is decomposed into 32-wide column chunks
# plus one exact-width tail chunk so no load ever needs a column mask.
#
# This replaces the fused kernel's 32-byte gather on the m-contiguous
# deep-K class (measured 0.783x Torch on [100, 65536, 100] fp32): the strip
# reaches ~1.29 TB/s vs the fused kernel's ~0.88 TB/s and Torch's ~1.12 TB/s
# (S5000, interleaved do_bench, official shape/dtype/distribution).
#
# Correctness invariants:
# - the band plan guarantees bands * RPC == ROWS and RPC % INNER == 0, so
#   unmasked loads never leave the tensor (no masked-lane row overshoot and
#   no cross-band reads: AND/OR over rows is idempotent, but band-local
#   reads keep partials independent of the merge order anyway);
# - the tail chunk is loaded unmasked only when its width covers exactly the
#   remaining columns (tail_w == M - nfull * 32); otherwise it is masked
#   (mask varies along the contiguous axis but only on the one small tile);
# - int32 addressing is gated by an IDX64 constexpr (row * M products are
#   widened before the multiply when numel crosses the int31 headroom).
# ---------------------------------------------------------------------------
def _generate_colstrip_kernel_source(nfull, tail_w, tail_exact):
    kernel_name = (
        f"_mthreads_boolean_colstrip_f{nfull}"
        + (f"_t{tail_w}" if tail_w else "")
        + ("" if tail_exact or not tail_w else "_m")
    )
    code = IndentedBuffer()
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("inp,")
        code.writeline("partial,")
        code.writeline("M,")
        code.writeline("BANDS,")
        code.writeline("RPC,")
        code.writeline("INNER: tl.constexpr,")
        code.writeline("IS_ALL: tl.constexpr,")
        code.writeline("IDX64: tl.constexpr,")
    code.writeline("):")
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("r_base = pid * RPC")
        # per-chunk column accumulators (kept in registers)
        for c in range(nfull):
            if c == 0:
                code.writeline("if IS_ALL:")
                with code.indent():
                    code.writeline(f"acc{c} = tl.full((32,), 1, tl.int1)")
                code.writeline("else:")
                with code.indent():
                    code.writeline(f"acc{c} = tl.zeros((32,), dtype=tl.int1)")
            else:
                code.writeline("if IS_ALL:")
                with code.indent():
                    code.writeline(f"acc{c} = tl.full((32,), 1, tl.int1)")
                code.writeline("else:")
                with code.indent():
                    code.writeline(f"acc{c} = tl.zeros((32,), dtype=tl.int1)")
        if tail_w:
            code.writeline("if IS_ALL:")
            with code.indent():
                code.writeline(f"acct = tl.full(({tail_w},), 1, tl.int1)")
            code.writeline("else:")
            with code.indent():
                code.writeline(f"acct = tl.zeros(({tail_w},), dtype=tl.int1)")
        code.writeline("for r in range(0, RPC, INNER):")
        with code.indent():
            code.writeline("if IDX64:")
            with code.indent():
                code.writeline("rows = (r_base + r + tl.arange(0, INNER)).to(tl.int64)")
            code.writeline("else:")
            with code.indent():
                code.writeline("rows = r_base + r + tl.arange(0, INNER)")
            code.writeline("p = inp + rows[:, None] * M")
            for c in range(nfull):
                off = c * 32
                code.writeline(f"v{c} = tl.load(p + {off} + tl.arange(0, 32)[None, :])")
                if c == 0:
                    code.writeline("if IS_ALL:")
                    with code.indent():
                        code.writeline(
                            f"acc{c} = acc{c} & (tl.min(v{c} != 0, axis=0) == 1)"
                        )
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(
                            f"acc{c} = acc{c} | (tl.max(v{c} != 0, axis=0) == 1)"
                        )
                else:
                    code.writeline("if IS_ALL:")
                    with code.indent():
                        code.writeline(
                            f"acc{c} = acc{c} & (tl.min(v{c} != 0, axis=0) == 1)"
                        )
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(
                            f"acc{c} = acc{c} | (tl.max(v{c} != 0, axis=0) == 1)"
                        )
            if tail_w:
                off = nfull * 32
                if tail_exact:
                    code.writeline(
                        f"vt = tl.load(p + {off} + tl.arange(0, {tail_w})[None, :])"
                    )
                else:
                    code.writeline(f"ct = {off} + tl.arange(0, {tail_w})")
                    code.writeline(
                        "vt = tl.load("
                        "p + ct[None, :], mask=(ct < M)[None, :], "
                        "other=1.0 if IS_ALL else 0.0)"
                    )
                code.writeline("if IS_ALL:")
                with code.indent():
                    code.writeline("acct = acct & (tl.min(vt != 0, axis=0) == 1)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("acct = acct | (tl.max(vt != 0, axis=0) == 1)")
        # column-major partial stores: partial[col, band]
        for c in range(nfull):
            code.writeline(
                f"tl.store(partial + ({c * 32} + tl.arange(0, 32)) * BANDS"
                f" + pid, acc{c})"
            )
        if tail_w:
            if tail_exact:
                code.writeline(
                    f"tl.store(partial + ({nfull * 32} + tl.arange(0, {tail_w}))"
                    f" * BANDS + pid, acct)"
                )
            else:
                code.writeline(f"ct = {nfull * 32} + tl.arange(0, {tail_w})")
                code.writeline(
                    "tl.store(partial + ct * BANDS + pid, acct, mask=ct < M)"
                )
    return kernel_name, code.getvalue()


def _colstrip_kernel_for(nfull, tail_w, tail_exact):
    key = (nfull, tail_w, tail_exact)
    cache = getattr(_colstrip_kernel_for, "overloads", None)
    if cache is None:
        cache = {}
        _colstrip_kernel_for.overloads = cache
    if key in cache:
        return cache[key]

    kernel_name, source = _generate_colstrip_kernel_source(*key)
    file_name = f"{kernel_name}.py"
    file_path = code_cache_dir() / file_name
    write_atomic(file_path, source)

    spec = importlib.util.spec_from_file_location(
        f"_mthreads_boolean_colstrip_gen_{kernel_name}", file_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = libentry()(getattr(module, kernel_name))
    cache[key] = kernel
    return kernel


@libentry()
@triton.jit
def _boolean_colstrip_merge_kernel(
    partial,
    out,
    M,
    BANDS,
    IS_ALL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # partial is column-major [M, BANDS]: each m row is BANDS contiguous
    # bytes, so the merge reads fully coalesced 1D runs.
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M
    if IS_ALL:
        acc = tl.full((BLOCK_M,), 1, tl.int1)
    else:
        acc = tl.zeros((BLOCK_M,), dtype=tl.int1)
    for s0 in range(0, BANDS, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        s_mask = s < BANDS
        if IS_ALL:
            v = tl.load(
                partial + m[:, None] * BANDS + s[None, :],
                mask=m_mask[:, None] & s_mask[None, :],
                other=1,
            )
            acc = acc & tl.min(v != 0, axis=1).to(tl.int1)
        else:
            v = tl.load(
                partial + m[:, None] * BANDS + s[None, :],
                mask=m_mask[:, None] & s_mask[None, :],
                other=0,
            )
            acc = acc | tl.max(v != 0, axis=1).to(tl.int1)
    tl.store(out + m, acc, mask=m_mask)


def _colstrip_band_plan(rows):
    """(RPC, bands) with bands * RPC == rows and RPC % INNER == 0, or None.

    Prefers a power-of-two band count near 4096 CTAs (measured optimum class
    on [100, 65536, 100] fp32: 4096/8192 within 0.5%, 2048 -10%, 1024 -11%),
    then any multiple-of-16 RPC whose band count still fills the GPU.
    """
    for bands in (4096, 8192, 2048, 1024):
        rpc = rows // bands
        if rows % bands == 0 and rpc % _COLSTRIP_INNER == 0 and rpc > 0:
            return rpc, bands
    base = triton.cdiv(triton.cdiv(rows, 4096), _COLSTRIP_INNER) * _COLSTRIP_INNER
    for k in range(64):
        rpc = base + k * _COLSTRIP_INNER
        if rpc > 0 and rows % rpc == 0 and rows // rpc >= 1024:
            return rpc, rows // rpc
    return None


_COLSTRIP_PLAN_CACHE = {}


def _plan_colstrip(inp, reduce_dims, is_all):
    """Colstrip applicability + plan, cached per (shape, dims, dtype).

    Route conditions (all host-side, from the prototype measurements):
    - contiguous input, all-only polarity, 4-byte elements;
    - segcol view with SEG == 1 (a deep reduce band over the fastest dim);
    - 32 <= M <= 256 and >= 256 MiB footprint (below that the fused kernel's
      split machinery measured ahead: 0.89x at 96 MiB / M=128, 1.34x vs
      1.36x at 78 MiB / M=100);
    - an exact band plan exists (bands * RPC == rows, RPC % 16 == 0) —
      awkward row counts simply stay on the fused path.
    The cache key is sound only under the contiguity restriction (same
    argument as the segcol plan cache).
    """
    if not (_COLSTRIP_ENABLED and is_all and inp.element_size() == 4):
        return None
    if not inp.is_contiguous():
        return None
    key = (tuple(inp.shape), tuple(reduce_dims), inp.dtype)
    if key in _COLSTRIP_PLAN_CACHE:
        return _COLSTRIP_PLAN_CACHE[key]
    plan = None
    view = _segcol_view(inp.shape, reduce_dims)
    if view is not None:
        seg, rows, m_cols = view
        nbytes = inp.numel() * inp.element_size()
        if (
            seg == 1
            and _COLSTRIP_CHUNK_W <= m_cols <= _COLSTRIP_MAX_M
            and nbytes >= _COLSTRIP_MIN_BYTES
        ):
            band = _colstrip_band_plan(rows)
            if band is not None:
                rpc, bands = band
                nfull = m_cols // _COLSTRIP_CHUNK_W
                tail = m_cols - nfull * _COLSTRIP_CHUNK_W
                if tail == 0:
                    tail_w, tail_exact = 0, True
                else:
                    tail_w = triton.next_power_of_2(tail)
                    tail_exact = tail_w == tail
                plan = {
                    "rows": rows,
                    "M": m_cols,
                    "RPC": rpc,
                    "bands": bands,
                    "nfull": nfull,
                    "tail_w": tail_w,
                    "tail_exact": tail_exact,
                    "idx64": (
                        _FORCE_IDX64
                        or inp.numel() >= _IDX32_LIMIT
                        or rows * m_cols >= _IDX32_LIMIT
                    ),
                }
    if len(_COLSTRIP_PLAN_CACHE) < 4096:
        _COLSTRIP_PLAN_CACHE[key] = plan
    return plan


def _colstrip_impl(inp, plan, is_all, cs_plan):
    # ATen legacy: any/all keep uint8 dtype for uint8 inputs, bool otherwise.
    out_dtype = torch.uint8 if inp.dtype == torch.uint8 else torch.bool
    M = cs_plan["M"]
    bands = cs_plan["bands"]
    kernel = _colstrip_kernel_for(
        cs_plan["nfull"], cs_plan["tail_w"], cs_plan["tail_exact"]
    )
    with torch_device_fn.device(inp.device):
        partial = torch.empty((M, bands), dtype=torch.int8, device=inp.device)
        kernel[(bands,)](
            inp,
            partial,
            M,
            bands,
            cs_plan["RPC"],
            INNER=_COLSTRIP_INNER,
            IS_ALL=is_all,
            IDX64=cs_plan["idx64"],
            num_warps=1,
            num_stages=2,
        )
        out_flat = torch.empty(M, dtype=torch.bool, device=inp.device)
        merge_bm = min(8, _np2(M))
        merge_bs = min(512, _np2(bands))
        _boolean_colstrip_merge_kernel[(triton.cdiv(M, merge_bm),)](
            partial,
            out_flat,
            M,
            bands,
            IS_ALL=is_all,
            BLOCK_M=merge_bm,
            BLOCK_S=merge_bs,
            num_warps=2,
            num_stages=2,
        )
    out = out_flat.view(plan["view_shape"])
    if out_dtype != torch.bool:
        out = out.to(out_dtype)
    return out


# ---------------------------------------------------------------------------
# split-reduction merge kernel (strategy "partial")
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _boolean_merge_partial_kernel(
    partial,
    out,
    M,
    SPLITS,
    IS_ALL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    m = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    m_mask = m < M
    if IS_ALL:
        acc = tl.full((BLOCK_M,), 1, tl.int1)
    else:
        acc = tl.zeros((BLOCK_M,), dtype=tl.int1)
    for s0 in range(0, SPLITS, BLOCK_S):
        s = (s0 + tl.arange(0, BLOCK_S)).to(tl.int64)
        s_mask = s < SPLITS
        if IS_ALL:
            # AND-merge: out-of-range splits must read as "all true".
            v = tl.load(
                partial + s[:, None] * M + m[None, :],
                mask=s_mask[:, None] & m_mask[None, :],
                other=1,
            )
            acc = acc & tl.min(v != 0, axis=0).to(tl.int1)
        else:
            v = tl.load(
                partial + s[:, None] * M + m[None, :],
                mask=s_mask[:, None] & m_mask[None, :],
                other=0,
            )
            acc = acc | tl.max(v != 0, axis=0).to(tl.int1)
    tl.store(out + m, acc, mask=m_mask)


# ---------------------------------------------------------------------------
# rank-specialized fused boolean_dims kernel (codegen, cached per rank/mode)
#
# One launch reduces every requested dim. The grid is 1D with size
# ceil(M / BLOCK_M) * SPLITS: decomposed in-kernel into (output-tile, split).
# Output dims and reduce dims are coalesced on the host and passed as
# shape/stride lists; the kernel expands them with an unrolled div/mod chain
# (output dims first = fastest-varying, giving coalesced base addresses; the
# last dim of each chain needs no modulo). No do_not_specialize: the
# stride/shape specialization hints (== 1, % 16) are what lets Triton
# vectorize the loads. Offsets are computed in int32 whenever the host proves
# every derived index fits (IDX64 widens before every multiply: sid * KCHUNK,
# pid * BLOCK_M and (c * UNROLL + u) * BLOCK_N all overflow int32 as soon as
# the product reaches 2^31, which silently produces negative offsets and
# out-of-bounds reads on >4 GiB tensors).
# ---------------------------------------------------------------------------
def _generate_boolean_dims_kernel_source(
    num_output_dims, num_reduce_dims, mode, chunked
):
    kernel_name = f"_mthreads_boolean_dims_{mode}_o{num_output_dims}_r{num_reduce_dims}"
    if not chunked:
        # UNROLL == 1: the classic flat per-block k-loop. A degenerate
        # chunked loop (one block per chunk) breaks Triton's software
        # pipelining across the outer loop and costs 20-70% on
        # early-exit-heavy shapes, so the flat structure is emitted
        # separately and cached under its own name.
        kernel_name += "_flat"
    meta_args = (
        ["M", "K"]
        + (["KCHUNK"] if mode != "direct" else [])
        + [f"output_shape{i}" for i in range(num_output_dims)]
        + [f"input_output_stride{i}" for i in range(num_output_dims)]
        + [f"reduce_shape{j}" for j in range(num_reduce_dims)]
        + [f"input_reduce_stride{j}" for j in range(num_reduce_dims)]
    )
    code = IndentedBuffer()
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("inp,")
        code.writeline("out,")
        for arg_name in meta_args:
            code.writeline(f"{arg_name},")
        code.writeline("IS_ALL: tl.constexpr,")
        code.writeline("IDX64: tl.constexpr,")
        code.writeline("BLOCK_M: tl.constexpr,")
        code.writeline("BLOCK_N: tl.constexpr,")
        code.writeline("UNROLL: tl.constexpr,")
    code.writeline("):")
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        if mode != "direct":
            # NOTE: this backend's 2D-grid program_id mapping is unreliable;
            # always launch a 1D grid and decompose it inside the kernel.
            code.writeline("tiles = tl.cdiv(M, BLOCK_M)")
            code.writeline("sid = pid // tiles")
            code.writeline("pid = pid % tiles")
        code.writeline("if IDX64:")
        with code.indent():
            code.writeline("m = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)")
        code.writeline("else:")
        with code.indent():
            code.writeline("m = pid * BLOCK_M + tl.arange(0, BLOCK_M)")
        code.writeline("m_mask = m < M")
        code.writeline("if IDX64:")
        with code.indent():
            code.writeline("base = tl.zeros((BLOCK_M,), dtype=tl.int64)")
        code.writeline("else:")
        with code.indent():
            code.writeline("base = tl.zeros((BLOCK_M,), dtype=tl.int32)")
        code.writeline("lin = m")
        for i in range(num_output_dims):
            # The last peeled dim needs no modulo: lin is already < its shape
            # for every unmasked lane (m < M).
            if i < num_output_dims - 1:
                code.writeline(f"idx = lin % output_shape{i}")
                code.writeline(f"lin = lin // output_shape{i}")
            else:
                code.writeline("idx = lin")
            code.writeline(f"base += idx * input_output_stride{i}")
        if mode != "direct":
            code.writeline("if IDX64:")
            with code.indent():
                code.writeline("k_start = sid.to(tl.int64) * KCHUNK")
            code.writeline("else:")
            with code.indent():
                code.writeline("k_start = sid * KCHUNK")
            code.writeline("k_rem = tl.minimum(KCHUNK, tl.maximum(K - k_start, 0))")
        else:
            code.writeline("k_start = 0")
            code.writeline("k_rem = K")
        if mode == "atomic":
            code.writeline("if IS_ALL:")
            with code.indent():
                # int32 flag 0 means "no zero found yet": alive rows start as
                # True and die permanently once any split raises the flag.
                code.writeline(
                    "acc = (tl.load(out + m, mask=m_mask, other=0) == 0).to(tl.int1)"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline(
                    "acc = (tl.load(out + m, mask=m_mask, other=0) != 0).to(tl.int1)"
                )
        else:
            code.writeline("if IS_ALL:")
            with code.indent():
                code.writeline("acc = tl.full((BLOCK_M,), 1, tl.int1)")
            code.writeline("else:")
            with code.indent():
                code.writeline("acc = tl.zeros((BLOCK_M,), dtype=tl.int1)")
        if not chunked:
            # Flat per-block k-loop: one row-reduce per block; the early-exit
            # mask updates every block (finest granularity).
            code.writeline("for i in range(0, tl.cdiv(k_rem, BLOCK_N)):")
            with code.indent():
                code.writeline("if IDX64:")
                with code.indent():
                    code.writeline(
                        "k = k_start + i.to(tl.int64) * BLOCK_N"
                        " + tl.arange(0, BLOCK_N)"
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline("k = k_start + i * BLOCK_N + tl.arange(0, BLOCK_N)")
                code.writeline("k_mask = k < K")
                code.writeline("if IDX64:")
                with code.indent():
                    code.writeline("roff = tl.zeros((BLOCK_N,), dtype=tl.int64)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("roff = tl.zeros((BLOCK_N,), dtype=tl.int32)")
                code.writeline("klin = k")
                for j in range(num_reduce_dims):
                    if j < num_reduce_dims - 1:
                        code.writeline(f"r_idx = klin % reduce_shape{j}")
                        code.writeline(f"klin = klin // reduce_shape{j}")
                    else:
                        code.writeline("r_idx = klin")
                    code.writeline(f"roff += r_idx * input_reduce_stride{j}")
                # Early exit is a per-row load mask, not loop control flow.
                code.writeline("if IS_ALL:")
                with code.indent():
                    code.writeline(
                        "mask = m_mask[:, None] & k_mask[None, :] & acc[:, None]"
                    )
                    code.writeline(
                        "vals = tl.load("
                        "inp + base[:, None] + roff[None, :], mask=mask, other=1.0)"
                    )
                    code.writeline("acc = acc & tl.min(vals != 0, axis=1).to(tl.int1)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("act = acc == 0")
                    code.writeline(
                        "mask = m_mask[:, None] & k_mask[None, :] & act[:, None]"
                    )
                    code.writeline(
                        "vals = tl.load("
                        "inp + base[:, None] + roff[None, :], mask=mask, other=0.0)"
                    )
                    code.writeline("acc = acc | tl.max(vals != 0, axis=1).to(tl.int1)")
        else:
            # Chunked k-loop: UNROLL blocks are accumulated into a 2D predicate
            # tile elementwise and reduced once per chunk. The early-exit mask
            # (acc) is loop-invariant inside a chunk, so the unrolled loads stay
            # independent and pipeline; reducing every block instead serializes
            # each load on the previous block's result.
            code.writeline("for c in range(0, tl.cdiv(k_rem, BLOCK_N * UNROLL)):")
            with code.indent():
                code.writeline("if IS_ALL:")
                with code.indent():
                    code.writeline("acc2d = tl.full((BLOCK_M, BLOCK_N), 1, tl.int1)")
                code.writeline("else:")
                with code.indent():
                    code.writeline(
                        "acc2d = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int1)"
                    )
                code.writeline("for u in range(UNROLL):")
                with code.indent():
                    code.writeline("if IDX64:")
                    with code.indent():
                        code.writeline(
                            "kk = ((c * UNROLL + u) * BLOCK_N).to(tl.int64)"
                            " + tl.arange(0, BLOCK_N)"
                        )
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(
                            "kk = (c * UNROLL + u) * BLOCK_N + tl.arange(0, BLOCK_N)"
                        )
                    code.writeline("k = k_start + kk")
                    # kk is chunk-local: bound it against this split's remainder
                    # (never read into the next split's range).
                    code.writeline("k_mask = kk < k_rem")
                    code.writeline("if IDX64:")
                    with code.indent():
                        code.writeline("roff = tl.zeros((BLOCK_N,), dtype=tl.int64)")
                    code.writeline("else:")
                    with code.indent():
                        code.writeline("roff = tl.zeros((BLOCK_N,), dtype=tl.int32)")
                    # The reduce index must be derived from the GLOBAL k, not the
                    # chunk-local kk, or every split would scan the same prefix.
                    code.writeline("klin = k")
                    for j in range(num_reduce_dims):
                        if j < num_reduce_dims - 1:
                            code.writeline(f"r_idx = klin % reduce_shape{j}")
                            code.writeline(f"klin = klin // reduce_shape{j}")
                        else:
                            code.writeline("r_idx = klin")
                        code.writeline(f"roff += r_idx * input_reduce_stride{j}")
                    # Early exit is a per-row load mask, not loop control flow.
                    code.writeline("if IS_ALL:")
                    with code.indent():
                        code.writeline(
                            "mask = m_mask[:, None] & k_mask[None, :] & acc[:, None]"
                        )
                        code.writeline(
                            "vals = tl.load("
                            "inp + base[:, None] + roff[None, :], mask=mask, other=1.0)"
                        )
                        code.writeline("acc2d = acc2d & (vals != 0)")
                    code.writeline("else:")
                    with code.indent():
                        code.writeline(
                            "mask = m_mask[:, None] & k_mask[None, :] & (acc == 0)[:, None]"
                        )
                        code.writeline(
                            "vals = tl.load("
                            "inp + base[:, None] + roff[None, :], mask=mask, other=0.0)"
                        )
                        code.writeline("acc2d = acc2d | (vals != 0)")
                code.writeline("if IS_ALL:")
                with code.indent():
                    code.writeline("acc = acc & tl.min(acc2d, axis=1).to(tl.int1)")
                code.writeline("else:")
                with code.indent():
                    code.writeline("acc = acc | tl.max(acc2d, axis=1).to(tl.int1)")
        if mode == "direct":
            code.writeline("tl.store(out + m, acc, mask=m_mask)")
        elif mode == "atomic":
            code.writeline("if IS_ALL:")
            with code.indent():
                code.writeline("found = (acc == 0).to(tl.int32)")
                code.writeline(
                    "tl.atomic_max(out + m, found, mask=m_mask & (found != 0))"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline("local = acc.to(tl.int32)")
                code.writeline(
                    "tl.atomic_max(out + m, local, mask=m_mask & (local != 0))"
                )
        else:
            code.writeline("if IDX64:")
            with code.indent():
                code.writeline(
                    "tl.store(out + sid.to(tl.int64) * M + m, acc.to(tl.int8), mask=m_mask)"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline(
                    "tl.store(out + sid * M + m, acc.to(tl.int8), mask=m_mask)"
                )
    return kernel_name, code.getvalue()


def _boolean_dims_kernel_for_rank(num_output_dims, num_reduce_dims, mode, chunked):
    key = (num_output_dims, num_reduce_dims, mode, chunked)
    cache = getattr(_boolean_dims_kernel_for_rank, "overloads", None)
    if cache is None:
        cache = {}
        _boolean_dims_kernel_for_rank.overloads = cache
    if key in cache:
        return cache[key]

    kernel_name, source = _generate_boolean_dims_kernel_source(
        num_output_dims, num_reduce_dims, mode, chunked
    )
    file_name = f"{kernel_name}.py"
    file_path = code_cache_dir() / file_name
    write_atomic(file_path, source)

    spec = importlib.util.spec_from_file_location(
        f"_mthreads_boolean_dims_gen_{kernel_name}", file_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # libentry caches launch metadata per specialization: raw JITFunction
    # dispatch costs ~20us per launch, which dominates small reductions.
    kernel = libentry()(getattr(module, kernel_name))
    cache[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# host-side layout handling
# ---------------------------------------------------------------------------
def _coalesce_layout(items):
    """Coalesce (shape, stride) pairs ordered fastest-first.

    Size-1 dims are dropped (they never move the base offset). Adjacent dims
    are merged when they form a contiguous run, keeping the first (smallest)
    stride. This shortens the in-kernel div/mod chains without changing the
    offset sum.
    """
    coalesced = []
    for shape, stride in items:
        if shape == 1:
            continue
        if coalesced and coalesced[-1][0] * coalesced[-1][1] == stride:
            coalesced[-1][0] *= shape
        else:
            coalesced.append([shape, stride])
    return [(shape, stride) for shape, stride in coalesced]


def _boolean_dims_config(
    M, K, element_size, m_contiguous, k_contiguous, is_all, k_span=1
):
    """Tile/split plan for the fused kernel, per polarity.

    Both tables are tuned on S5000 over the official all_dims/any_dims
    workloads (dim=[0,1]). Layout classes:
    - k-contiguous (reduce lanes contiguous, the dominant trailing-dim
      reduction): wide BLOCK_N tiles along the contiguous reduce axis,
      chunk-unrolled to keep loads in flight.
    - m-contiguous (output lanes contiguous, reduce lanes strided): small
      (BLOCK_M, BLOCK_N) tiles with many split CTAs; the strided k axis
      cannot coalesce, so parallelism comes from the split count.
    - strided (both axes non-contiguous, e.g. [::2] slice or transpose
      views with partial dims): short BM=8 tiles + split-K fill; these
      scans are latency-bound and the unsplit grid starves the GPU.
    ``k_span`` is the byte distance between consecutive k lanes on the
    fastest reduce dim (1 when truly contiguous).
    Starvation rule (both polarities): whenever the unsplit grid
    (output_tiles) is below the CTA target, split K up to one BLOCK_N per
    split — the ~10us merge launch is noise next to a 5-30x kernel-time
    win on starved scans (16x32x64x128 / slice partial-dims clusters).
    """
    if K <= 2:
        # Trivial reduce domain (folded broadcast dims or a size-<=2 dim):
        # the kernel is an elementwise (inp != 0) gather. Give each CTA a
        # wide run along the fastest axis and no k machinery.
        BLOCK_M = min(_np2(M), 256 if m_contiguous else 32)
        BLOCK_N = _np2(K)
        return BLOCK_M, BLOCK_N, 1, 4, 1

    strided = not m_contiguous and not k_contiguous
    if is_all:
        nbytes = M * K * element_size
        if m_contiguous:
            if M <= 128:
                # Few output rows: the strided k axis cannot coalesce, so
                # parallelism comes from narrow tiles + many split CTAs.
                BLOCK_M = min(_np2(M), 8)
                BLOCK_N = 128
            elif K <= 64:
                # Tiny reduce domain (e.g. dims=[0] on a 4D tensor):
                # parallelism comes from narrow m tiles (more output
                # tiles), not deep k scanning.
                BLOCK_M = min(_np2(M), 8)
                BLOCK_N = min(256, _np2(K))
            elif 1024 <= K < 4096:
                # Mid-depth k streams: narrow m tiles already give the grid
                # enough CTAs; splitting adds a merge launch and over-splits
                # these small scans (measured on 16x32x64x128 dims=[1,2] /
                # [0,2] contiguous). Shallower (K < 1024) and deeper streams
                # keep the classic wide-m tile. 1-byte elements keep the
                # narrow 64-lane tile (measured 0.118 -> 0.155 with BN=256
                # on 16x32x64x128 dims=[1,2] bool).
                BLOCK_M = min(_np2(M), 8)
                BLOCK_N = min(256 if element_size > 1 else 64, _np2(K))
                if M > 128 and nbytes > (32 << 20):
                    # Large tensors do not stay L2-resident: bound each k
                    # row's gather span so one load does not fan out over
                    # thousands of pages (k stride can reach the whole
                    # trailing-dim product).
                    BLOCK_N = min(BLOCK_N, max(64, 8192 // max(k_span, 1)))
            else:
                BLOCK_M = min(_np2(M), 16 if element_size <= 2 else 32)
                BLOCK_N = 64
            UNROLL = 8
        elif strided:
            # Keep one k row's gather span bounded (~8 KiB): a wide BLOCK_N
            # on a large k stride (e.g. a permuted view, stride 128) makes
            # every load touch dispersed lines and the tile waits on the
            # slowest of them. The m direction cannot coalesce here (unlike
            # the m-contiguous class), so span control matters at every
            # tensor size. Two measured regimes (S5000):
            # - dense-ish k (k_span <= 16B, e.g. [::2] slice views):
            #   BM=8 with a 128-lane tile ([1024^3] slice dims=[1,2] fp16:
            #   0.84ms vs 0.93-1.25 for the alternatives);
            # - sparse k (transpose/permute views): BM=16 with a 32-lane
            #   tile (16x32x64x128 dims=[1,2] permute: 0.15ms vs 0.19+).
            if k_span <= 16:
                BLOCK_M = min(_np2(M), 8)
                BLOCK_N = min(128, _np2(K))
            else:
                BLOCK_M = min(_np2(M), 16)
                BLOCK_N = min(32, _np2(K))
            UNROLL = 4
        else:
            BLOCK_M = min(_np2(M), min(64, max(16, 128 // element_size)))
            if nbytes >= (1 << 30):
                BLOCK_N = min(4096, _np2(K))
                UNROLL = 16
            else:
                BLOCK_N = min(2048, _np2(K))
                UNROLL = 8
            while BLOCK_M * BLOCK_N > 8192 and BLOCK_N > 32:
                BLOCK_N //= 2
            if BLOCK_M >= 16:
                # Wide output tiles with a strided reduce axis: keep the
                # in-flight footprint modest.
                UNROLL = min(UNROLL, 4)
        # Never reserve k lanes or unrolled loads beyond the reduce domain.
        BLOCK_N = min(BLOCK_N, _np2(K))
        UNROLL = min(UNROLL, max(1, triton.cdiv(K, BLOCK_N)))
        tile = BLOCK_M * BLOCK_N
        num_warps = (
            8 if (BLOCK_N >= 1024 or tile >= 4096) else (4 if tile >= 512 else 2)
        )
        output_tiles = triton.cdiv(M, BLOCK_M)
        # Scale the target grid with the reduction's byte footprint; scans
        # whose k lanes gather (m-contiguous, strided) want a higher floor:
        # their per-CTA throughput is latency-bound. Deep m-contiguous
        # streams (>= 16384) keep the 2048 floor measured on the official
        # dim=[0,1] shapes; shallower ones fill at most ~512 CTAs.
        if strided:
            target_ctas = min(8192, max(4096, nbytes // 524288))
        elif m_contiguous:
            target_ctas = min(8192, max(2048 if K >= 16384 else 512, nbytes // 524288))
        else:
            target_ctas = min(8192, max(512, nbytes // 524288))
        blocks = triton.cdiv(K, BLOCK_N)
        chunks = triton.cdiv(K, BLOCK_N * UNROLL)
        if output_tiles == 0 or output_tiles >= target_ctas:
            # output_tiles == 0: empty output (M == 0) — nothing to launch;
            # the caller returns early, but the plan is still derived.
            splits = 1
        else:
            # Deep m-contiguous splits keep the one-chunk-per-CTA floor
            # measured on [100,256,100]-class shapes; starved shallower
            # scans may split as fine as one block per CTA.
            cap = chunks if (m_contiguous and K >= 16384) else blocks
            splits = max(1, min(cap, triton.cdiv(target_ctas, output_tiles)))
            if m_contiguous and K >= 65536 and output_tiles <= 16:
                # Deep m-contiguous all-scans with few output tiles: the
                # byte-scaled target underfills the grid. Measured on
                # [100, 65536, 100] fp32 (S5000): 385 splits from the byte
                # rule vs 512 measured 14% faster (1024 within 2% of 512);
                # shapes whose target already exceeds the floor are
                # unaffected.
                splits = max(1, min(cap, max(splits, 512)))
        return BLOCK_M, BLOCK_N, UNROLL, num_warps, splits

    # any: per-block early-exit granularity (UNROLL=1, flat loop) is the
    # default; the per-chunk reduce chain only dominates for long per-CTA
    # streams (see the unroll gates below). All thresholds were measured
    # with the official protocol (triton.testing.do_bench, cold L2) on
    # S5000.
    if m_contiguous:
        if 8 <= K <= 256 and M >= 64:
            # Shallow m-contiguous any-scans (e.g. [100, 1, 100] dim=[0,1]):
            # the work is one or two tiny loads per output, so per-CTA fixed
            # cost dominates. One CTA per 1-2 outputs (BM<=2, BN=64) hides
            # the prologue latency across many more CTAs; measured 1.24x
            # Torch vs 0.79x for the 16-wide tile on S5000 (interleaved
            # do_bench, cold L2).
            BLOCK_M = min(_np2(M), 2)
            BLOCK_N = 64
            num_warps = 2
        else:
            BLOCK_M = min(_np2(M), 16)
            BLOCK_N = 128
            num_warps = 4
    elif strided:
        BLOCK_M = min(_np2(M), 8)
        BLOCK_N = min(256, _np2(K), max(32, 8192 // max(k_span, 1)))
        tile = BLOCK_M * BLOCK_N
        num_warps = 4 if tile >= 512 else 2
    else:
        BLOCK_M = min(_np2(M), min(64, max(16, 128 // element_size)))
        BLOCK_N = min(2048, _np2(K))
        while BLOCK_M * BLOCK_N > 8192 and BLOCK_N > 32:
            BLOCK_N //= 2
        tile = BLOCK_M * BLOCK_N
        num_warps = 8 if tile >= 4096 else (4 if tile >= 512 else 2)
    BLOCK_N = min(BLOCK_N, _np2(K))
    UNROLL = 1
    blocks = triton.cdiv(K, BLOCK_N)
    output_tiles = triton.cdiv(M, BLOCK_M)
    if strided:
        target_ctas = min(16384, max(4096, (M * K * element_size) // 262144))
    else:
        target_ctas = min(16384, max(512, (M * K * element_size) // 262144))
    if output_tiles == 0 or output_tiles >= target_ctas or K <= 256:
        # output_tiles == 0: empty output (M == 0) — nothing to launch.
        # K <= 256: splitting a tiny reduce domain only adds a merge launch
        # that dwarfs the scan itself (both polarities, measured on the
        # [100, 1, 100] dim=[0,1] class).
        splits = 1
    elif m_contiguous and K >= 16384 and output_tiles < 128:
        # Deep m-contiguous reductions with a small output tiling (strided
        # per-CTA streams): the byte-scaled target both under- and over-
        # shoots here. Measured optima are captured by three bounds:
        # ~8-16 serial blocks per CTA rounded up to a power of two
        # ([100,256,100] wants 32 not 74, [64,512,512] wants 16-32 not 128),
        # a 128-split cap ([100,65536,100] wants 128 not 715-1429), and a
        # ~4096-CTA cap ([1024^3] wants 64 not 256). Large output tilings
        # (>=128 tiles, e.g. [65537,32768] dim=[0]) keep the byte-scaled
        # rule: forcing deep splits there is -65%.
        splits = min(
            128,
            max(1, _np2(blocks // 8)),
            max(1, triton.cdiv(4096, output_tiles)),
        )
        if triton.cdiv(blocks, splits) >= 48:
            # Long per-CTA strided stream: the per-chunk reduce chain
            # dominates and chunk unrolling wins ([1024^3] -14%,
            # [100,65536,100] -32%). Below ~48 blocks per CTA the
            # per-block early-exit granularity measured better.
            UNROLL = 8
    else:
        splits = max(
            1,
            min(blocks, triton.cdiv(target_ctas, output_tiles)),
        )
    if not m_contiguous and k_contiguous and splits > 1:
        # k-contig (reduce lanes truly contiguous, stride 1): once a CTA
        # scans >= 24 serial blocks the per-chunk reduce chain dominates
        # and chunk unrolling wins (measured on S5000, official do_bench
        # protocol: [10000,65536] fp32 U2 -14%, fp16 U2 -31%, bf16 U2
        # -19%; [4096,4096] at 16 blocks wants the flat loop, and fp16
        # wants U2 where bf16 tolerates U8). 1-byte elements only unroll
        # with a single m row (BM=1 full reductions keep wide vectorized
        # 1-byte loads): wider m tiles in the chunked codegen degrade to
        # ~186 GB/s narrow strips (measured bool [10000,65536] /
        # [1024^3] rowwise/deep partial dims: 0.14-0.17x the flat loop).
        # Strided reduce lanes (e.g. a [::2] slice view, reduce stride 2)
        # keep the flat loop: measured +32% with U8 on [1024^3] slice
        # dims=(0,2) bool.
        if triton.cdiv(triton.cdiv(K, splits), BLOCK_N) >= 24:
            if element_size == 1:
                UNROLL = 8 if BLOCK_M == 1 else 1
            else:
                UNROLL = 2
    return BLOCK_M, BLOCK_N, UNROLL, num_warps, splits


# Geometry/config plans are pure functions of (shape, strides, dims, keepdim,
# element size, polarity): cache them so hot small-tensor calls pay one dict
# lookup instead of re-deriving the layout and tile config in Python every
# call.
_PLAN_CACHE = {}


def _plan_boolean_dims(shape, strides, reduce_dims, keepdim, element_size, is_all):
    key = (shape, strides, reduce_dims, keepdim, element_size, is_all)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan

    ndim = len(shape)
    reduce_set = set(reduce_dims)
    out_dims = [d for d in range(ndim) if d not in reduce_set]

    # Output dims are enumerated fastest-first in OUTPUT-FLAT order (reverse
    # natural dim order) so the kernel's linear output index matches
    # out_flat.view(shape) exactly. Coalescing keeps the first (fastest)
    # input stride and is only applied when the input-side strides stay
    # consistent, so the base-offset sum is preserved.
    out_items = _coalesce_layout((shape[d], strides[d]) for d in reversed(out_dims))
    # Reduce dims are enumerated fastest-first by INPUT stride so that
    # consecutive k lanes touch consecutive addresses when possible.
    # Broadcast (stride-0) reduce dims fold to a single sample: AND/OR over
    # s > 1 copies of one element is that element, so such a dim contributes
    # a factor of 1 to K instead of its shape (a shape-0 dim stays 0: the
    # reduction domain is empty and handled by the K == 0 early return).
    # Without the fold, a broadcast reduce dim (e.g. expand().all(dim=0))
    # leaves the kernel scanning K lanes of which only one address is live.
    red_items = _coalesce_layout(
        (
            1 if (strides[d] == 0 and shape[d] > 1) else shape[d],
            strides[d],
        )
        for d in sorted(reduce_dims, key=lambda d: strides[d])
    )
    M = math.prod(s for s, _ in out_items)
    K = math.prod(s for s, _ in red_items)
    # The m-lane direction is memory-contiguous when the fastest output dim
    # has input stride 1 (then BLOCK_M spans adjacent elements).
    m_contiguous = bool(out_items) and out_items[0][1] == 1
    # The k-lane direction is truly contiguous when the fastest reduce dim
    # has input stride 1 (then BLOCK_N spans adjacent elements).
    k_contiguous = bool(red_items) and red_items[0][1] == 1
    max_elem_offset = sum((s - 1) * st for s, st in zip(shape, strides) if s > 1)
    view_shape = (
        [1 if d in reduce_set else shape[d] for d in range(ndim)]
        if keepdim
        else [shape[d] for d in out_dims]
    )
    BLOCK_M, BLOCK_N, UNROLL, num_warps, splits = _boolean_dims_config(
        M,
        K,
        element_size,
        m_contiguous,
        k_contiguous,
        is_all,
        red_items[0][1] * element_size if red_items else 1,
    )
    plan = {
        "out_items": out_items,
        "red_items": red_items,
        "M": M,
        "K": K,
        "max_elem_offset": max_elem_offset,
        "view_shape": view_shape,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "UNROLL": UNROLL,
        "num_warps": num_warps,
        "splits": splits,
    }
    if len(_PLAN_CACHE) < 4096:
        _PLAN_CACHE[key] = plan
    return plan


# ---------------------------------------------------------------------------
# shared host implementation
# ---------------------------------------------------------------------------
def boolean_dims_impl(inp, reduce_dims, keepdim, is_all, merge_strategy="partial"):
    """Single-launch fused multi-dim any/all over `reduce_dims`.

    ``reduce_dims`` must be normalized (positive, sorted, unique). Polarity:
    ``is_all=False`` computes OR over (x != 0); ``is_all=True`` computes AND
    over (x != 0) with early exit on the first zero.
    """
    plan = _plan_boolean_dims(
        inp.shape,
        inp.stride(),
        tuple(reduce_dims),
        keepdim,
        inp.element_size(),
        is_all,
    )
    M = plan["M"]
    K = plan["K"]
    # ATen legacy: any/all keep uint8 dtype for uint8 inputs, bool otherwise.
    out_dtype = torch.uint8 if inp.dtype == torch.uint8 else torch.bool

    if M == 0:
        return torch.empty(plan["view_shape"], dtype=out_dtype, device=inp.device)
    if K == 0:
        # Empty reduction domain: any -> False, all -> vacuously True.
        return torch.full(
            plan["view_shape"], 1 if is_all else 0, dtype=out_dtype, device=inp.device
        )

    nbytes = inp.numel() * inp.element_size()

    # Route: small dense full reductions -> two-kernel linear cascade.
    # The [10000, 256]-class shapes are launch/latency-bound and the
    # cascade's coalesced linear scan measures 1.2-2.1x the fused path
    # there; above the polarity-dependent byte cap the fused kernel's
    # finer-grained early exit wins again (validated at the +/- one-row
    # boundary).
    cascade_cap = (
        _CASCADE_MAX_BYTES_ALL
        if is_all
        else (
            _CASCADE_MAX_BYTES_ANY_1B
            if inp.element_size() == 1
            else _CASCADE_MAX_BYTES_ANY
        )
    )
    if (
        _CASCADE_ROUTE_ENABLED
        and len(reduce_dims) == inp.ndim
        and _CASCADE_MIN_BYTES <= nbytes < cascade_cap
        and torch.ops.aten.is_non_overlapping_and_dense(inp)
    ):
        out = boolean_whole_impl(inp, is_all)
        if keepdim:
            out = out.view(plan["view_shape"])
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out

    # Route: transposed segment-column streaming for contiguous
    # m-contiguous / non-k-contiguous band layouts (see
    # _boolean_segcol_kernel).
    if _SEGCOL_ENABLED:
        seg_plan = _plan_segcol(inp, tuple(reduce_dims))
        if seg_plan:
            return _segcol_impl(inp, plan, is_all, seg_plan)

    # Route: narrow-M deep-K column-strip streaming (see
    # _generate_colstrip_kernel_source) — contiguous fp32-class inputs whose
    # reduce band spans every dim but the fastest and whose column count is
    # too narrow for the segcol kernel's 512-wide tiles.
    if _COLSTRIP_ENABLED and merge_strategy == "partial":
        cs_plan = _plan_colstrip(inp, tuple(reduce_dims), is_all)
        if cs_plan:
            return _colstrip_impl(inp, plan, is_all, cs_plan)

    # Route: all-only scalar-output atomic upgrade. any must stay on
    # partials (atomic measured 0.64-0.71x: it pays memset + convert
    # for an early exit any's per-CTA mask already provides), and the
    # 256 MiB floor keeps the launch-overhead flip (156 MiB -2~3%,
    # 39 MiB -31~45%) out of range.
    if (
        _ATOMIC_ALL_ROUTE_ENABLED
        and merge_strategy == "partial"
        and is_all
        and M == 1
        and nbytes >= _ATOMIC_ALL_MIN_BYTES
        and torch.ops.aten.is_non_overlapping_and_dense(inp)
    ):
        merge_strategy = "atomic"

    BLOCK_M = plan["BLOCK_M"]
    BLOCK_N = plan["BLOCK_N"]
    UNROLL = plan["UNROLL"]
    num_warps = plan["num_warps"]
    splits = plan["splits"]
    if merge_strategy == "none":
        splits = 1

    # int32 in-kernel addressing is ~1.4x faster on strided layouts, but only
    # safe when every derived index fits in int31: the max element offset,
    # the padded last chunk of the k loop (K + BLOCK_N * UNROLL), m lanes
    # (M + BLOCK_M) and the partials row index (splits * M). The headroom
    # keeps masked-lane wraparound from turning the bound checks true for
    # negatives.
    idx64 = (
        _FORCE_IDX64
        or plan["max_elem_offset"] >= _IDX32_LIMIT
        or K + BLOCK_N * UNROLL >= _IDX32_LIMIT
        or M + BLOCK_M >= _IDX32_LIMIT
        or (splits > 1 and splits * M >= _IDX32_LIMIT)
    )

    kernel_args = [M, K]
    if splits > 1:
        kernel_args.append(triton.cdiv(K, splits))
    kernel_args += [s for s, _ in plan["out_items"]]
    kernel_args += [t for _, t in plan["out_items"]]
    kernel_args += [s for s, _ in plan["red_items"]]
    kernel_args += [t for _, t in plan["red_items"]]

    output_tiles = triton.cdiv(M, BLOCK_M)
    with torch_device_fn.device(inp.device):
        if splits == 1:
            out_flat = torch.empty(M, dtype=torch.bool, device=inp.device)
            kernel = _boolean_dims_kernel_for_rank(
                len(plan["out_items"]),
                len(plan["red_items"]),
                "direct",
                UNROLL > 1,
            )
            kernel[(output_tiles,)](
                inp,
                out_flat,
                *kernel_args,
                IS_ALL=is_all,
                IDX64=idx64,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                num_warps=num_warps,
                num_stages=_NUM_STAGES,
            )
        elif merge_strategy == "atomic":
            out_i32 = torch.zeros(M, dtype=torch.int32, device=inp.device)
            kernel = _boolean_dims_kernel_for_rank(
                len(plan["out_items"]),
                len(plan["red_items"]),
                "atomic",
                UNROLL > 1,
            )
            kernel[(output_tiles * splits,)](
                inp,
                out_i32,
                *kernel_args,
                IS_ALL=is_all,
                IDX64=idx64,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                num_warps=num_warps,
                num_stages=_NUM_STAGES,
            )
            if is_all:
                # flag == 0 <=> no split found a zero <=> all true
                out_flat = out_i32 == 0
            else:
                out_flat = out_i32.to(torch.bool)
        else:
            # Every (split, row) slot is written unconditionally by its CTA,
            # so the workspace needs no initialization.
            partial = torch.empty((splits, M), dtype=torch.int8, device=inp.device)
            kernel = _boolean_dims_kernel_for_rank(
                len(plan["out_items"]),
                len(plan["red_items"]),
                "partial",
                UNROLL > 1,
            )
            kernel[(output_tiles * splits,)](
                inp,
                partial,
                *kernel_args,
                IS_ALL=is_all,
                IDX64=idx64,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                num_warps=num_warps,
                num_stages=_NUM_STAGES,
            )
            out_flat = torch.empty(M, dtype=torch.bool, device=inp.device)
            # Narrow output tiles keep the merge grid from starving on
            # multi-row outputs: with a min(64, ...) block the M=100 /
            # splits=512 merge ran on 2 CTAs and was latency-bound
            # (measured 18x slower than an 8-wide tile on S5000). M=1
            # merges read one contiguous partials row and prefer the wide
            # 128-lane tile (measured 5-10% end-to-end regression with the
            # narrow tile there).
            if M >= 16:
                merge_bm = min(8, _np2(M))
                merge_bs = min(64, _np2(splits))
                merge_warps = 2
            else:
                merge_bm = min(64, _np2(M))
                merge_bs = min(128, _np2(splits))
                merge_warps = 4
            _boolean_merge_partial_kernel[(triton.cdiv(M, merge_bm),)](
                partial,
                out_flat,
                M,
                splits,
                IS_ALL=is_all,
                BLOCK_M=merge_bm,
                BLOCK_S=merge_bs,
                num_warps=merge_warps,
                num_stages=2,
            )

    out = out_flat.view(plan["view_shape"])
    if out_dtype != torch.bool:
        out = out.to(out_dtype)
    return out
