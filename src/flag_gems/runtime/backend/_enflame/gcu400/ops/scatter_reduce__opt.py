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
"""GCU400 scatter_reduce: contention-safe output-centric implementation.

The generic implementation reduces with ``tl.atomic_add(..., sem="relaxed")``
and a CAS loop for prod. On GCU these lose / replay updates when many source
elements target the same output element (e.g. 256 sources into one index):
``sum`` returned 24.491182 instead of 22.770594 and the runtime aborted for
sum/mean/amax. This backend keeps the tuned atomic-free scan for the aligned
fast-path domain, routes every other aten-valid scatter of bounded scan work
through a single deterministic general kernel, and only delegates cases
beyond the work budget (or with invalid contracts) to the generic code.
"""

import importlib
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime  # noqa: F401

# ``flag_gems.ops`` re-exports ``scatter_reduce`` as a function, so importing
# the attribute from the package yields that function instead of the module.
# Resolve the submodule explicitly for the generic delegate.
_generic_sr = importlib.import_module("flag_gems.ops.scatter_reduce")

logger = logging.getLogger(__name__)

# Keep the scan affordable: at most 2**20 (output, scan) pairs per call.
_MAX_SCAN_WORK = 1 << 20
_MAX_SCAN_AXIS = 256
# General deterministic-scan budget: with L2-resident index/src the scan is
# latency-bounded by (out_numel * idx_axis) pair loads; once the data spills
# to DRAM only the much smaller DRAM pair budget keeps the launch sane.
_MAX_SCAN_PAIRS_L2 = 1 << 36
_MAX_SCAN_PAIRS_DRAM = 1 << 28
_MAX_SCAN_DATA_BYTES = 48 << 20
_MAX_SCAN_NUMEL = (1 << 31) - 1
_REDUCE_IDS = {"sum": 0, "prod": 1, "mean": 2, "amax": 3, "amin": 4}

# Lanes the generic fallback may cover in a single launch: 65535 programs times
# the smallest BLOCK * LOOP tile it uses. Larger inputs are split into slabs.
_FALLBACK_LANES_PER_LAUNCH = 65535 * 128

# Threshold for the inner-lane kernel: only when the inner dimension is long enough (≥128) does the coalescing benefit
# of consecutive lanes
# outweigh the parallelism waste of "one program computing only a few elements". Measured ~7.5x at idx_inner=4096,
# while it is 9x slower at idx_inner=1 (dim is the last dimension), so a narrow inner dimension must fall back to the
# old kernel.
_MIN_INNER_FOR_LANE_KERNEL = 128
# Threshold for the 2D tile kernel: the tile gives 1.7~2.0x for inner∈[16,128); for inner<16 it is slower than the old
# kernel
# (measured 8x slower at inner=1: the 2D tile degrades into 32 single-lane scalar accesses).
_MIN_INNER_FOR_TILE_KERNEL = 16
_DEFAULT_GEN_BLOCK_I = 2048
_DEFAULT_SCAN_BLOCK_I = 256

# Large 5D extrema scatters are canonicalized to an equivalent 3D problem. The
# gate and the canonicalization live in the generic module, so the cases above
# this element threshold delegate there instead of decoding 5D coordinates in
# every lane here.
_CANONICALIZE_5D_MIN_ELEMENTS = 1 << 23


@triton.jit(
    do_not_specialize=[
        "OUT_AXIS",
        "IDX_AXIS",
        "OUT_INNER",
        "OUT_NUMEL",
        "REDUCE_ID",
        "INCLUDE_SELF",
    ]
)
def _gcu_scatter_reduce_scan_kernel(
    InPtr,
    IdxPtr,
    SrcPtr,
    OutPtr,
    OUT_AXIS,
    IDX_AXIS,
    OUT_INNER,
    OUT_NUMEL,
    REDUCE_ID,
    INCLUDE_SELF,
    BLOCK_SCAN: tl.constexpr,
):
    """One program per output element, scalar loop over the index axis.

    GCU drops lanes for vector loads with a runtime predicate and for 256-lane
    vector reductions, so this kernel deliberately uses scalar loads in a loop
    instead. Small / high-contention cases only; see _scan_candidate.
    """
    pid = tl.program_id(axis=0).to(tl.int32)
    i = pid % OUT_INNER
    t = (pid // OUT_INNER) % OUT_AXIS
    o = pid // (OUT_INNER * OUT_AXIS)
    out_off = (o * OUT_AXIS + t) * OUT_INNER + i
    seen = 0

    if INCLUDE_SELF:
        acc = tl.load(InPtr + out_off).to(tl.float32)
        denom = 1.0
    else:
        if REDUCE_ID == 1:
            acc = 1.0
        elif REDUCE_ID == 3:
            acc = float("-inf")
        elif REDUCE_ID == 4:
            acc = float("inf")
        else:
            acc = 0.0
        denom = 0.0

    for k in range(IDX_AXIS):
        off = o * IDX_AXIS * OUT_INNER + k * OUT_INNER + i
        iv = tl.load(IdxPtr + off).to(tl.int32)
        sv = tl.load(SrcPtr + off).to(tl.float32)
        hit = iv == t
        seen = seen + tl.where(hit, 1, 0)
        if REDUCE_ID == 0:
            acc += tl.where(hit, sv, 0.0)
        elif REDUCE_ID == 1:
            acc *= tl.where(hit, sv, 1.0)
        elif REDUCE_ID == 2:
            acc += tl.where(hit, sv, 0.0)
            denom += tl.where(hit, 1.0, 0.0)
        elif REDUCE_ID == 3:
            acc = tl.maximum(acc, tl.where(hit, sv, float("-inf")))
        else:
            acc = tl.minimum(acc, tl.where(hit, sv, float("inf")))

    if REDUCE_ID == 2:
        acc = acc / tl.maximum(denom, 1.0)
    if not INCLUDE_SELF:
        acc = tl.where(seen > 0, acc, tl.load(InPtr + out_off).to(tl.float32))
    tl.store(OutPtr + out_off, acc)


@triton.jit(
    do_not_specialize=[
        "OUT_AXIS",
        "IDX_AXIS",
        "OUT_INNER",
        "OUT_NUMEL",
        "REDUCE_ID",
        "INCLUDE_SELF",
    ]
)
def _gcu_scatter_reduce_vec_kernel(
    InPtr,
    IdxPtr,
    SrcPtr,
    OutPtr,
    OUT_AXIS,
    IDX_AXIS,
    OUT_INNER,
    OUT_NUMEL,
    REDUCE_ID,
    INCLUDE_SELF,
    IDX_BLOCK: tl.constexpr,
):
    """Blocked vector scan over the index axis.

    t9 batch Step 4: the original kernel walked IDX_AXIS with a scalar loop (up to 256 iterations),
    measured at 8-11ms on GCU400 / Triton 3.6. Switching to one chunked tl.arange load plus
    tl.sum / tl.max / tl.min reduction brought it down to 1.1-1.4ms, bit-identical to the scalar
    result. prod is not in this kernel (this Triton version has no tl.prod), the wrapper falls back.
    """
    pid = tl.program_id(axis=0).to(tl.int32)
    i = pid % OUT_INNER
    t = (pid // OUT_INNER) % OUT_AXIS
    o = pid // (OUT_INNER * OUT_AXIS)
    out_off = (o * OUT_AXIS + t) * OUT_INNER + i
    orig = tl.load(InPtr + out_off).to(tl.float32)
    if INCLUDE_SELF:
        acc = orig
        denom = 1.0
    else:
        if REDUCE_ID == 3:
            acc = float("-inf")
        elif REDUCE_ID == 4:
            acc = float("inf")
        else:
            acc = 0.0
        denom = 0.0
    seen = 0
    for kb in range(tl.cdiv(IDX_AXIS, IDX_BLOCK)):
        kk = kb * IDX_BLOCK + tl.arange(0, IDX_BLOCK)
        off = o * IDX_AXIS * OUT_INNER + kk * OUT_INNER + i
        m = kk < IDX_AXIS
        iv = tl.load(IdxPtr + off, mask=m, other=-1).to(tl.int32)
        sv = tl.load(SrcPtr + off, mask=m, other=0.0).to(tl.float32)
        hit = (iv == t) & m
        seen = seen + tl.sum(tl.where(hit, 1, 0), 0)
        if REDUCE_ID == 0:
            acc += tl.sum(tl.where(hit, sv, 0.0), 0)
        elif REDUCE_ID == 2:
            acc += tl.sum(tl.where(hit, sv, 0.0), 0)
            denom += tl.sum(tl.where(hit, 1.0, 0.0), 0)
        elif REDUCE_ID == 3:
            acc = tl.maximum(acc, tl.max(tl.where(hit, sv, float("-inf")), 0))
        else:
            acc = tl.minimum(acc, tl.min(tl.where(hit, sv, float("inf")), 0))
    if REDUCE_ID == 2:
        acc = acc / tl.maximum(denom, 1.0)
    if not INCLUDE_SELF:
        acc = tl.where(seen > 0, acc, orig)
    tl.store(OutPtr + out_off, acc)


# ---------------------------------------------------------------------------
# Eager relaxed-atomic SUM path (R01, 2026-11-18)
#
# The historical note in ops/scatter_reduce.py reported that relaxed atomic_add
# lost updates on GCU. Probe 15 re-checked that claim on the current torch_gcu
# build: a minimal relaxed-atomic scatter-add matched the CPU golden bit for bit
# on (256, 256) / (1024, 1024) / (10000, 256) fp16, and it beat the CPU round
# trip by 1.33-1.36x once the element count passed the launch-overhead knee.
# This kernel is that minimal form: no mask plane, no count plane, int32
# offsets, and a 1D-or-2D grid because grid.x is capped at 65535.
# ---------------------------------------------------------------------------
_MAX_ATOMIC_GRID_X = 65535
# Element-count floor for the single-pass atomic path. FLAGGEMS_SR_ATOMIC_MIN
# overrides it so the scan/atomic crossover can be re-probed on new hardware.
_MIN_ATOMIC_NUMEL = int(os.environ.get("FLAGGEMS_SR_ATOMIC_MIN", str(1 << 10)))
# Probe 18 BLOCK sweep on (1024,1024) and (4096,4096): 1024 gives 6.647ms /
# 106.7ms against 7.083ms / 113.0ms at 512 and 169.3ms at 256.
_ATOMIC_BLOCK = 1024


@triton.jit(do_not_specialize=["N", "NCOLS"])
def _gcu_atomic_reduce_lastdim_kernel(
    IdxPtr,
    SrcPtr,
    AccPtr,
    AuxPtr,
    N,
    NCOLS,
    REDUCE_ID: tl.constexpr,
    MEAN: tl.constexpr,
    MARKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One pass of relaxed atomics over the source elements (sum/mean/amax/amin).

    Every program consumes BLOCK consecutive (index, src) pairs; ``row * NCOLS``
    recovers the output base, which is only valid for dim == ndim - 1 with
    matching outer shapes (the caller enforces that).

    ``MARKED`` additionally marks every written position, which the finalize
    kernel needs to reproduce the aten ``include_self=False`` contract (a
    position that receives no scatter keeps its original self value). The mark
    is a plain (racy) store of a constant: every writer stores the same value,
    so no atomic unit is needed and the mark costs a store instead of an
    integer atomic, which measured roughly 2x slower than the sum atomic alone.

    ``MEAN`` needs the exact number of contributions per output position, so it
    pays a second float atomic (``AuxPtr`` is the float32 count plane). Probe on
    GCU400: sum 3.5ms, sum+count 7.0ms, atomic_max 3.7ms for 1M elements.
    """
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    row = offs // NCOLS
    dst = row * NCOLS
    iv = tl.load(IdxPtr + offs, mask=mask, other=0).to(tl.int32)
    sv = tl.load(SrcPtr + offs, mask=mask, other=0.0).to(tl.float32)
    if REDUCE_ID == 3:
        tl.atomic_max(AccPtr + dst + iv, sv, mask=mask, sem="relaxed")
    elif REDUCE_ID == 4:
        tl.atomic_min(AccPtr + dst + iv, sv, mask=mask, sem="relaxed")
    else:
        tl.atomic_add(AccPtr + dst + iv, sv, mask=mask, sem="relaxed")
    if MEAN:
        tl.atomic_add(
            AuxPtr + dst + iv,
            tl.full((BLOCK,), 1.0, tl.float32),
            mask=mask,
            sem="relaxed",
        )
    elif MARKED:
        tl.store(AuxPtr + dst + iv, tl.full((BLOCK,), 1, tl.int32), mask=mask)


@triton.jit(do_not_specialize=["N"])
def _gcu_atomic_reduce_finalize_kernel(
    AccPtr,
    AuxPtr,
    SelfPtr,
    OutPtr,
    N,
    REDUCE_ID: tl.constexpr,
    MEAN: tl.constexpr,
    MARKED: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold the self value into the accumulator and cast back to the input dtype.

    Merging self here (instead of pre-copying it into the accumulator) removes a
    whole read+write pass; the self plane is read exactly once either way.

    * mean: ``(acc + self) / (count + 1)``, or ``acc / count`` when
      include_self=False, where untouched positions keep self.
    * amax/amin/sum with include_self: ``max/min/add(acc, self)``.
    * anything with include_self=False: the mark plane selects acc or self.
    """
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.load(AccPtr + offs, mask=mask, other=0.0)
    self_v = tl.load(SelfPtr + offs, mask=mask, other=0.0).to(tl.float32)
    if MEAN:
        cnt = tl.load(AuxPtr + offs, mask=mask, other=0.0)
        if INCLUDE_SELF:
            # cnt >= 0, so the divisor is at least 1; untouched positions reduce to self.
            val = (acc + self_v) / (cnt + 1.0)
        else:
            val = tl.where(cnt > 0.0, acc / tl.maximum(cnt, 1.0), self_v)
    elif MARKED:
        hit = tl.load(AuxPtr + offs, mask=mask, other=0)
        val = tl.where(hit > 0, acc, self_v)
    elif REDUCE_ID == 3:
        val = tl.maximum(acc, self_v)
    elif REDUCE_ID == 4:
        val = tl.minimum(acc, self_v)
    else:
        val = acc + self_v
    tl.store(OutPtr + offs, val.to(OutPtr.dtype.element_ty), mask=mask)


@triton.jit(
    do_not_specialize=[
        "OUT_AXIS",
        "IDX_AXIS",
        "OUT_INNER",
        "REDUCE_ID",
        "INCLUDE_SELF",
    ]
)
def _gcu_scatter_reduce_inner_kernel(
    InPtr,
    IdxPtr,
    SrcPtr,
    OutPtr,
    OUT_AXIS,
    IDX_AXIS,
    OUT_INNER,
    REDUCE_ID,
    INCLUDE_SELF,
    BLOCK_I: tl.constexpr,
):
    """Lane-vectorised variant: each program handles BLOCK_I consecutive output elements on the inner axis.

    The only difference from ``_gcu_scatter_reduce_vec_kernel`` is the memory access direction: the
    vec version computes one output element per program, so while scanning the index axis the lane
    stride is OUT_INNER and each of the 256 lanes hits an independent cache line (only 1/32 of each
    load is used); this kernel spreads lanes over the **inner axis**, so while scanning the index
    axis the BLOCK_I lanes of ``IdxPtr/SrcPtr`` are fully contiguous and one 256x4B load lands in 8
    cache lines, improving memory efficiency about 8x. The reduction count
    (out_numel x idx_axis) is unchanged, so this is a pure memory-side optimization and the result
    is bit-identical to the vec version.
    prod does not use this kernel (this Triton version has no tl.prod).
    """
    pid = tl.program_id(axis=0).to(tl.int32)
    n_iblk = tl.cdiv(OUT_INNER, BLOCK_I)
    i_blk = pid % n_iblk
    t = (pid // n_iblk) % OUT_AXIS
    o = pid // (n_iblk * OUT_AXIS)
    i = i_blk * BLOCK_I + tl.arange(0, BLOCK_I)
    m_i = i < OUT_INNER
    out_off = (o * OUT_AXIS + t) * OUT_INNER + i
    orig = tl.load(InPtr + out_off, mask=m_i, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    if REDUCE_ID == 3:
        acc = tl.full([BLOCK_I], float("-inf"), tl.float32)
    elif REDUCE_ID == 4:
        acc = tl.full([BLOCK_I], float("inf"), tl.float32)
    denom = tl.zeros([BLOCK_I], dtype=tl.float32)
    if INCLUDE_SELF:
        # The initial values of amax/amin must be "replaced" by input values, not accumulated:
        # -inf + x is still -inf, which makes amax/amin with include_self completely wrong.
        if REDUCE_ID == 3 or REDUCE_ID == 4:
            acc = orig
        else:
            acc += orig
        denom += 1.0
    seen = tl.zeros([BLOCK_I], dtype=tl.int32)

    for k in range(IDX_AXIS):
        off = o * IDX_AXIS * OUT_INNER + k * OUT_INNER + i
        iv = tl.load(IdxPtr + off, mask=m_i, other=-1).to(tl.int32)
        sv = tl.load(SrcPtr + off, mask=m_i, other=0.0).to(tl.float32)
        hit = (iv == t) & m_i
        seen += tl.where(hit, 1, 0)
        if REDUCE_ID == 0:
            acc += tl.where(hit, sv, 0.0)
        elif REDUCE_ID == 2:
            acc += tl.where(hit, sv, 0.0)
            denom += tl.where(hit, 1.0, 0.0)
        elif REDUCE_ID == 3:
            acc = tl.maximum(acc, tl.where(hit, sv, float("-inf")))
        else:
            acc = tl.minimum(acc, tl.where(hit, sv, float("inf")))

    if REDUCE_ID == 2:
        acc = acc / tl.maximum(denom, 1.0)
    if not INCLUDE_SELF:
        acc = tl.where(seen > 0, acc, orig)
    tl.store(OutPtr + out_off, acc, mask=m_i)


def _prod(shape):
    n = 1
    for v in shape:
        n *= int(v)
    return n


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit(
    do_not_specialize=[
        "OUT_NUMEL",
        "OUT_AXIS",
        "OUT_INNER",
        "IDX_AXIS",
        "IDX_INNER",
        "REDUCE_ID",
        "INCLUDE_SELF",
    ]
)
def _gcu_scatter_reduce_general_kernel(
    InPtr,
    IdxPtr,
    SrcPtr,
    OutPtr,
    OUT_NUMEL,
    OUT_AXIS,
    OUT_INNER,
    IDX_AXIS,
    IDX_INNER,
    REDUCE_ID,
    INCLUDE_SELF,
    IDX_BLOCK: tl.constexpr,
):
    """Deterministic output-centric scan for every case the fast paths reject.

    The host narrows inp/src/index to the index-defined active prefix first, so
    output and index share the (outer, axis, inner) decomposition and every
    (o, t, i) output element scans the matching index row in IDX_BLOCK vector
    blocks, reducing the hits in fp32 without atomics. Programs stride over
    the outputs (grid capped at 65535), so out_numel is unbounded; the ``t9 batch generalization``
    adds grid-stride, the prod branch (tl.reduce multiply) and independent
    OUT/IDX strides relative to the aligned-only fast-path kernels.
    """
    pid = tl.program_id(axis=0).to(tl.int32)
    nprog = tl.num_programs(axis=0)
    for out_id in range(pid, OUT_NUMEL, nprog):
        o = out_id // (OUT_AXIS * OUT_INNER)
        rem = out_id % (OUT_AXIS * OUT_INNER)
        t = rem // OUT_INNER
        i = rem % OUT_INNER
        orig = tl.load(InPtr + out_id).to(tl.float32)
        if INCLUDE_SELF:
            acc = orig
            denom = 1.0
        else:
            if REDUCE_ID == 1:
                acc = 1.0
            elif REDUCE_ID == 3:
                acc = float("-inf")
            elif REDUCE_ID == 4:
                acc = float("inf")
            else:
                acc = 0.0
            denom = 0.0
        seen = 0
        for kb in range(tl.cdiv(IDX_AXIS, IDX_BLOCK)):
            kk = kb * IDX_BLOCK + tl.arange(0, IDX_BLOCK)
            m = kk < IDX_AXIS
            off = o * IDX_AXIS * IDX_INNER + kk * IDX_INNER + i
            iv = tl.load(IdxPtr + off, mask=m, other=-1).to(tl.int32)
            sv = tl.load(SrcPtr + off, mask=m, other=0.0).to(tl.float32)
            hit = (iv == t) & m
            seen = seen + tl.sum(tl.where(hit, 1, 0), 0)
            if REDUCE_ID == 0:
                acc += tl.sum(tl.where(hit, sv, 0.0), 0)
            elif REDUCE_ID == 1:
                acc *= tl.reduce(tl.where(hit, sv, 1.0), axis=0, combine_fn=_multiply)
            elif REDUCE_ID == 2:
                acc += tl.sum(tl.where(hit, sv, 0.0), 0)
                denom += tl.sum(tl.where(hit, 1.0, 0.0), 0)
            elif REDUCE_ID == 3:
                acc = tl.maximum(acc, tl.max(tl.where(hit, sv, float("-inf")), 0))
            else:
                acc = tl.minimum(acc, tl.min(tl.where(hit, sv, float("inf")), 0))
        if REDUCE_ID == 2:
            acc = acc / tl.maximum(denom, 1.0)
        if not INCLUDE_SELF:
            acc = tl.where(seen > 0, acc, orig)
        tl.store(OutPtr + out_id, acc)


@triton.jit(
    do_not_specialize=[
        "OUTER",
        "OUT_AXIS",
        "OUT_INNER",
        "IDX_AXIS",
        "IDX_INNER",
        "REDUCE_ID",
        "INCLUDE_SELF",
    ]
)
def _gcu_scatter_reduce_gen_inner_kernel(
    InPtr,
    IdxPtr,
    SrcPtr,
    OutPtr,
    OUTER,
    OUT_AXIS,
    OUT_INNER,
    IDX_AXIS,
    IDX_INNER,
    REDUCE_ID,
    INCLUDE_SELF,
    BLOCK_I: tl.constexpr,
):
    """General deterministic scan kernel (inner-axis lane vectorization + grid-stride).

    Mathematically equivalent to ``_gcu_scatter_reduce_general_kernel`` (same output-centric scan,
    no atomics, deterministic), changing only the memory access direction and the parallel split:
    * lanes span the **inner axis** (``BLOCK_I`` consecutive elements), so while scanning the index
      axis one load of ``IdxPtr/SrcPtr`` is a contiguous BLOCK_I x elem_size; the old kernel spread
      lanes over the index axis (stride IDX_INNER) and each lane hit its own cache line.
    * grid-stride in units of blocks: the total block count
      ``outer x out_axis x cdiv(inner, BLOCK_I)`` can far exceed the grid cap.
    * the index axis becomes a scalar loop, so prod degrades to per-element multiplication (no tl.prod).
    """
    pid = tl.program_id(axis=0).to(tl.int32)
    nprog = tl.num_programs(axis=0).to(tl.int32)
    n_iblk = tl.cdiv(OUT_INNER, BLOCK_I)
    total_blocks = OUTER * OUT_AXIS * n_iblk

    for bid in range(pid, total_blocks, nprog):
        i_blk = bid % n_iblk
        t = (bid // n_iblk) % OUT_AXIS
        o = bid // (n_iblk * OUT_AXIS)
        i = i_blk * BLOCK_I + tl.arange(0, BLOCK_I)
        m_i = i < OUT_INNER
        out_off = (o * OUT_AXIS + t) * OUT_INNER + i
        orig = tl.load(InPtr + out_off, mask=m_i, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_I], dtype=tl.float32)
        if REDUCE_ID == 1:
            acc = tl.full([BLOCK_I], 1.0, tl.float32)
        elif REDUCE_ID == 3:
            acc = tl.full([BLOCK_I], float("-inf"), tl.float32)
        elif REDUCE_ID == 4:
            acc = tl.full([BLOCK_I], float("inf"), tl.float32)
        denom = tl.zeros([BLOCK_I], dtype=tl.float32)
        if INCLUDE_SELF:
            # Same as the inner kernel: amax/amin replace the initial ±inf with input values; prod initializes by
            # multiplication.
            if REDUCE_ID == 1:
                acc *= orig
            elif REDUCE_ID == 3 or REDUCE_ID == 4:
                acc = orig
            else:
                acc += orig
            denom += 1.0
        seen = tl.zeros([BLOCK_I], dtype=tl.int32)

        for k in range(IDX_AXIS):
            off = o * IDX_AXIS * IDX_INNER + k * IDX_INNER + i
            iv = tl.load(IdxPtr + off, mask=m_i, other=-1).to(tl.int32)
            sv = tl.load(SrcPtr + off, mask=m_i, other=0.0).to(tl.float32)
            hit = (iv == t) & m_i
            seen += tl.where(hit, 1, 0)
            if REDUCE_ID == 0:
                acc += tl.where(hit, sv, 0.0)
            elif REDUCE_ID == 1:
                acc *= tl.where(hit, sv, 1.0)
            elif REDUCE_ID == 2:
                acc += tl.where(hit, sv, 0.0)
                denom += tl.where(hit, 1.0, 0.0)
            elif REDUCE_ID == 3:
                acc = tl.maximum(acc, tl.where(hit, sv, float("-inf")))
            else:
                acc = tl.minimum(acc, tl.where(hit, sv, float("inf")))

        if REDUCE_ID == 2:
            acc = acc / tl.maximum(denom, 1.0)
        if not INCLUDE_SELF:
            acc = tl.where(seen > 0, acc, orig)
        tl.store(OutPtr + out_off, acc, mask=m_i)


@triton.jit(
    do_not_specialize=[
        "OUTER",
        "OUT_AXIS",
        "OUT_INNER",
        "IDX_AXIS",
        "IDX_INNER",
        "REDUCE_ID",
        "INCLUDE_SELF",
    ]
)
def _gcu_scatter_reduce_tile_kernel(
    InPtr,
    IdxPtr,
    SrcPtr,
    OutPtr,
    OUTER,
    OUT_AXIS,
    OUT_INNER,
    IDX_AXIS,
    IDX_INNER,
    REDUCE_ID,
    INCLUDE_SELF,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Narrow-inner (``out_inner < 128``) special case: a 2-D tile covers (index axis x inner axis).

    With a narrow inner axis the inner axis alone cannot fill a vector and
    ``_gcu_scatter_reduce_gen_inner_kernel`` would waste lanes; this kernel combines the inner axis
    (BLOCK_I, rounded up to a power of two and aligned to IDX_INNER) with the index axis (BLOCK_K)
    into one 2-D tile so a single load covers ``BLOCK_K x BLOCK_I`` **physically contiguous**
    elements (when IDX_INNER is small the whole tile is contiguous), then reduces along the index
    axis. prod has no usable reduction primitive and still uses the old kernel.
    """
    pid = tl.program_id(axis=0).to(tl.int32)
    nprog = tl.num_programs(axis=0).to(tl.int32)
    n_iblk = tl.cdiv(OUT_INNER, BLOCK_I)
    total_blocks = OUTER * OUT_AXIS * n_iblk
    ii = tl.arange(0, BLOCK_I)
    kk = tl.arange(0, BLOCK_K)

    for bid in range(pid, total_blocks, nprog):
        i_blk = bid % n_iblk
        t = (bid // n_iblk) % OUT_AXIS
        o = bid // (n_iblk * OUT_AXIS)
        i = i_blk * BLOCK_I + ii
        m_i = i < OUT_INNER
        out_off = (o * OUT_AXIS + t) * OUT_INNER + i
        orig = tl.load(InPtr + out_off, mask=m_i, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_I], dtype=tl.float32)
        if REDUCE_ID == 3:
            acc = tl.full([BLOCK_I], float("-inf"), tl.float32)
        elif REDUCE_ID == 4:
            acc = tl.full([BLOCK_I], float("inf"), tl.float32)
        denom = tl.zeros([BLOCK_I], dtype=tl.float32)
        if INCLUDE_SELF:
            if REDUCE_ID == 3 or REDUCE_ID == 4:
                acc = orig
            else:
                acc += orig
            denom += 1.0
        seen = tl.zeros([BLOCK_I], dtype=tl.int32)

        for kb in range(tl.cdiv(IDX_AXIS, BLOCK_K)):
            kv = kb * BLOCK_K + kk
            off = o * IDX_AXIS * IDX_INNER + kv[:, None] * IDX_INNER + i[None, :]
            m = (kv < IDX_AXIS)[:, None] & m_i[None, :]
            iv = tl.load(IdxPtr + off, mask=m, other=-1).to(tl.int32)
            sv = tl.load(SrcPtr + off, mask=m, other=0.0).to(tl.float32)
            hit = (iv == t) & m
            seen += tl.sum(tl.where(hit, 1, 0), axis=0)
            if REDUCE_ID == 0 or REDUCE_ID == 2:
                acc += tl.sum(tl.where(hit, sv, 0.0), axis=0)
                denom += tl.sum(tl.where(hit, 1.0, 0.0), axis=0)
            elif REDUCE_ID == 3:
                acc = tl.maximum(acc, tl.max(tl.where(hit, sv, float("-inf")), axis=0))
            else:
                acc = tl.minimum(acc, tl.min(tl.where(hit, sv, float("inf")), axis=0))

        if REDUCE_ID == 2:
            acc = acc / tl.maximum(denom, 1.0)
        if not INCLUDE_SELF:
            acc = tl.where(seen > 0, acc, orig)
        tl.store(OutPtr + out_off, acc, mask=m_i)


def _general_candidate(inp, dim, index, src, reduce):
    """Accept every aten-valid scatter whose deterministic scan fits the budget.

    The remaining cases (invalid reduce/dim, empty tensors, aten contract
    violations, or scan work beyond the budget) still fall back to the generic
    implementation so error contracts and the tuned large-shape path stay put.
    """
    if reduce not in _REDUCE_IDS:
        return None
    if inp.ndim != index.ndim or index.ndim != src.ndim or inp.ndim == 0:
        return None
    if dim < -inp.ndim or dim >= inp.ndim:
        return None
    dim %= inp.ndim
    if index.numel() == 0 or inp.numel() == 0 or inp.numel() > _MAX_SCAN_NUMEL:
        return None
    for k in range(inp.ndim):
        if int(index.shape[k]) > int(src.shape[k]):
            return None
        if k != dim and int(index.shape[k]) > int(inp.shape[k]):
            return None
    idx_axis = int(index.shape[dim])
    pairs = inp.numel() * idx_axis
    data_bytes = index.numel() * index.element_size() + src.numel() * src.element_size()
    if pairs > _MAX_SCAN_PAIRS_L2:
        return None
    if data_bytes > _MAX_SCAN_DATA_BYTES and pairs > _MAX_SCAN_PAIRS_DRAM:
        return None
    return dim


def _general_scatter_reduce(inp, dim, index, src, reduce, include_self):
    dim %= inp.ndim
    index_shape = [int(size) for size in index.shape]
    inp_shape = [int(size) for size in inp.shape]
    active_inp = inp
    active_src = src
    for axis, size in enumerate(index_shape):
        if int(active_src.shape[axis]) > size:
            active_src = active_src.narrow(axis, 0, size)
        if axis != dim and inp_shape[axis] > size:
            active_inp = active_inp.narrow(axis, 0, size)
    outer = _prod(index_shape[:dim])
    idx_axis = index_shape[dim]
    idx_inner = _prod(index_shape[dim + 1 :])
    out_axis = inp_shape[dim]
    out_numel = active_inp.numel()
    assert out_numel == outer * out_axis * idx_inner

    inp_c = active_inp.contiguous()
    src_c = active_src.contiguous()
    idx_c = index.to(torch.int32).contiguous()
    out = torch.empty_like(inp_c)
    from flag_gems.runtime import torch_device_fn

    # Kernel selection:
    #  * wide inner (≥128): inner-lane kernel, with BLOCK_I set to _DEFAULT_GEN_BLOCK_I;
    #  * narrow inner: 2D tile kernel (except prod, which has no usable reduction primitive);
    #  * FLAGGEMS_SR_GEN_BLOCK=0: old kernel, for A/B and regression comparison.
    gen_block = int(os.environ.get("FLAGGEMS_SR_GEN_BLOCK", str(_DEFAULT_GEN_BLOCK_I)))
    gen_warps = int(os.environ.get("FLAGGEMS_SR_WARPS", "4"))
    with torch_device_fn.device(inp.device):
        use_tile = (
            gen_block > 0
            and reduce != "prod"
            and _MIN_INNER_FOR_TILE_KERNEL <= idx_inner < _MIN_INNER_FOR_LANE_KERNEL
        )
        if use_tile:
            tile_i = 1 << max(0, min(idx_inner, 128) - 1).bit_length()
            tile_k = max(1, 256 // tile_i)
            n_iblk = (idx_inner + tile_i - 1) // tile_i
            total_blocks = outer * out_axis * n_iblk
            _gcu_scatter_reduce_tile_kernel[(min(total_blocks, 65535),)](
                inp_c,
                idx_c,
                src_c,
                out,
                outer,
                out_axis,
                idx_inner,
                idx_axis,
                idx_inner,
                _REDUCE_IDS[reduce],
                include_self,
                BLOCK_I=tile_i,
                BLOCK_K=tile_k,
                num_warps=gen_warps,
            )
        elif gen_block > 0 and idx_inner >= _MIN_INNER_FOR_TILE_KERNEL:
            # BLOCK_I must not exceed the nearest power of two of the inner axis, otherwise idle lanes are wasted
            # proportionally
            # (at inner=128 + BLOCK_I=2048, 15/16 of the lanes idle).
            if idx_inner < gen_block:
                gen_block = 1 << (idx_inner - 1).bit_length()
            n_iblk = (idx_inner + gen_block - 1) // gen_block
            total_blocks = outer * out_axis * n_iblk
            _gcu_scatter_reduce_gen_inner_kernel[(min(total_blocks, 65535),)](
                inp_c,
                idx_c,
                src_c,
                out,
                outer,
                out_axis,
                idx_inner,
                idx_axis,
                idx_inner,
                _REDUCE_IDS[reduce],
                include_self,
                BLOCK_I=gen_block,
                num_warps=gen_warps,
            )
        else:
            # Inner dimension too narrow (<16) or prod: both inner-lane and 2D tile degrade, so use the old kernel.
            _gcu_scatter_reduce_general_kernel[(min(out_numel, 65535),)](
                inp_c,
                idx_c,
                src_c,
                out,
                out_numel,
                out_axis,
                idx_inner,
                idx_axis,
                idx_inner,
                _REDUCE_IDS[reduce],
                include_self,
                IDX_BLOCK=256,
                num_warps=1,
            )

    active_is_full = all(
        axis == dim or index_shape[axis] == inp_shape[axis] for axis in range(inp.ndim)
    )
    if active_is_full:
        return out
    # Elements outside the index-defined active prefix keep their input value,
    # mirroring the generic high-rank path: clone the untouched input and copy
    # the reduced active box back into it (both are pure data movement).
    full = inp.contiguous().clone()
    dst = full
    for axis, size in enumerate(index_shape):
        if axis != dim and inp_shape[axis] > size:
            dst = dst.narrow(axis, 0, size)
    dst.copy_(out)
    return full


def _scan_candidate(inp, dim, index, src, reduce, include_self):
    """Only take small, shape-aligned cases; everything else uses the tuned path."""
    if reduce not in _REDUCE_IDS:
        return None
    if inp.ndim != index.ndim or inp.ndim != src.ndim or inp.ndim == 0:
        return None
    if index.shape != src.shape or index.numel() == 0 or inp.numel() == 0:
        return None
    if dim < -inp.ndim or dim >= inp.ndim:
        return None
    dim %= inp.ndim
    for k in range(inp.ndim):
        if k != dim and int(index.shape[k]) != int(inp.shape[k]):
            return None
    idx_axis = int(index.shape[dim])
    out_numel = inp.numel()
    if idx_axis > _MAX_SCAN_AXIS or out_numel > 65535:
        return None
    if out_numel * idx_axis > _MAX_SCAN_WORK:
        return None
    if inp.dtype not in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
        return None
    return dim, idx_axis, out_numel


def _should_canonicalize_5d(inp, dim, index, src, reduce):
    """Whether a 5D extrema scatter should go through the canonicalizing path."""
    if (
        inp.ndim != 5
        or index.ndim != inp.ndim
        or src.ndim != inp.ndim
        or reduce not in ("amax", "amin")
        or inp.numel() < _CANONICALIZE_5D_MIN_ELEMENTS
    ):
        return False
    dim %= inp.ndim
    return all(
        axis == dim or index.size(axis) == inp.size(axis) for axis in range(inp.ndim)
    )


def _fallback_scatter_reduce(inp, dim, index, src, reduce, **kwargs):
    """Generic implementation, split so it never exceeds the device grid.x limit.

    The generic 2D kernels take one program per ``BLOCK * LOOP`` lanes and the
    launch grid is capped at 65535 programs, so a single launch covers at most
    ``65535 * BLOCK * LOOP`` lanes (8.4M at the smallest block this backend
    uses). Axes other than the reduced one never interact in a scatter, so an
    oversized case is split along the largest remaining axis and each slab is
    scattered independently.
    """
    lanes = int(index.numel())
    if lanes <= _FALLBACK_LANES_PER_LAUNCH:
        return _generic_sr.scatter_reduce(inp, dim, index, src, reduce, **kwargs)

    split_axes = [k for k in range(inp.ndim) if k != dim % inp.ndim]
    if not split_axes:
        return _generic_sr.scatter_reduce(inp, dim, index, src, reduce, **kwargs)
    axis = max(split_axes, key=lambda k: int(inp.shape[k]))
    axis_size = int(inp.shape[axis])
    slab_count = -(-lanes // _FALLBACK_LANES_PER_LAUNCH)
    slab_size = max(1, -(-axis_size // slab_count))

    result = None
    for start in range(0, axis_size, slab_size):
        length = min(slab_size, axis_size - start)
        slab = _generic_sr.scatter_reduce(
            inp.narrow(axis, start, length).contiguous(),
            dim,
            index.narrow(axis, start, length).contiguous(),
            src.narrow(axis, start, length).contiguous(),
            reduce,
            **kwargs,
        )
        if result is None:
            result = torch.empty_like(inp)
        result.narrow(axis, start, length).copy_(slab)
    return result


def scatter_reduce(
    inp,
    dim,
    index,
    src,
    reduce,
    *,
    include_self=True,
    _deduplicate_programs=False,
    _use_product_lock=False,
    _product_grid_limit=None,
):
    logger.debug("GEMS_ENFLAME SCATTER_REDUCE")
    if _should_canonicalize_5d(inp, dim, index, src, reduce):
        return _fallback_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            _deduplicate_programs=_deduplicate_programs,
            _use_product_lock=_use_product_lock,
            _product_grid_limit=_product_grid_limit,
        )
    # Last-dim scatter with index/src matching inp runs 1-2 orders of magnitude
    # faster as a single relaxed-atomic pass than through the scan or general
    # kernels (sum: 142ms -> 3.4ms, mean: 148ms -> 8ms, amax: 162ms -> 5ms at
    # 1024x1024), so it short-circuits first.
    atomic = _atomic_candidate(inp, dim, src, index, reduce, include_self)
    if atomic is not None:
        idx_offsets, src_values, acc, aux, ncols, aux_kind = atomic
        _launch_atomic_reduce(
            idx_offsets, src_values, acc, aux, ncols, reduce, aux_kind
        )
        # The accumulator only ever holds the scattered contributions; the self
        # value is folded in by the finalize pass.
        return _finalize_atomic_reduce(acc, aux, inp, reduce, aux_kind, include_self)
    candidate = _scan_candidate(inp, dim, index, src, reduce, include_self)
    if candidate is None:
        general = _general_candidate(inp, dim, index, src, reduce)
        if general is not None:
            return _general_scatter_reduce(
                inp, general, index, src, reduce, include_self
            )
        return _fallback_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            _deduplicate_programs=_deduplicate_programs,
            _use_product_lock=_use_product_lock,
            _product_grid_limit=_product_grid_limit,
        )

    dim, idx_axis, out_numel = candidate
    out_inner = _prod(inp.shape[dim + 1 :])
    out_axis = int(inp.shape[dim])

    # t9 batch 5/5: fp32 + all-contiguous inputs skip 4 dtype conversions / copies
    # (under ENFLAME_PT_OP_DEBUG_CONFIG=fallback_cpu=all_ops the fixed cost of those
    #  .to()/.contiguous() dominates), and use num_warps=1.
    # bf16/fp16/fp64 keep the 1/5 vectorized implementation unchanged to avoid low-precision regressions.
    if (
        inp.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and src.dtype == inp.dtype
        and _REDUCE_IDS[reduce] != 1
    ):
        from flag_gems.runtime import torch_device_fn as _tdf

        _inp = inp.contiguous()
        _src = src.contiguous()
        _out = torch.empty_like(_inp)
        # Inner-axis lane-vectorized kernel: when BLOCK_I=0, fall back to the original "one program, one output
        # element" vec version, for A/B and regression comparison.
        block_i = int(
            os.environ.get("FLAGGEMS_SR_INNER_BLOCK", str(_DEFAULT_SCAN_BLOCK_I))
        )
        if out_inner < _MIN_INNER_FOR_LANE_KERNEL:
            block_i = 0
        with _tdf.device(inp.device):
            if block_i > 0:
                grid = ((out_inner + block_i - 1) // block_i) * out_axis
                grid *= max(1, out_numel // (out_axis * out_inner))
                _gcu_scatter_reduce_inner_kernel[(grid,)](
                    _inp,
                    index.contiguous(),
                    _src,
                    _out,
                    out_axis,
                    idx_axis,
                    out_inner,
                    _REDUCE_IDS[reduce],
                    include_self,
                    BLOCK_I=block_i,
                    num_warps=4,
                )
            else:
                _gcu_scatter_reduce_vec_kernel[(out_numel,)](
                    _inp,
                    index.contiguous(),
                    _src,
                    _out,
                    out_axis,
                    idx_axis,
                    out_inner,
                    out_numel,
                    _REDUCE_IDS[reduce],
                    include_self,
                    IDX_BLOCK=256,
                    num_warps=1,
                )
        return _out

    inp_f32 = inp.to(torch.float32).contiguous()
    src_f32 = src.to(torch.float32).contiguous()
    idx_i64 = index.to(torch.int32).contiguous()
    out = torch.empty_like(inp_f32)
    from flag_gems.runtime import torch_device_fn

    with torch_device_fn.device(inp.device):
        if _REDUCE_IDS[reduce] == 1:
            # prod: that Triton version lacks tl.prod, keep the original scalar scan kernel.
            _gcu_scatter_reduce_scan_kernel[(out_numel,)](
                inp_f32,
                idx_i64,
                src_f32,
                out,
                out_axis,
                idx_axis,
                out_inner,
                out_numel,
                _REDUCE_IDS[reduce],
                include_self,
                256,
                num_warps=4,
            )
        else:
            # t9 batch Step 4 B03: IDX_BLOCK=256 / num_warps=4,
            # sum/mean/amax/amin measured 7.9~11.4x, bit-identical.
            _gcu_scatter_reduce_vec_kernel[(out_numel,)](
                inp_f32,
                idx_i64,
                src_f32,
                out,
                out_axis,
                idx_axis,
                out_inner,
                out_numel,
                _REDUCE_IDS[reduce],
                include_self,
                IDX_BLOCK=512,
                num_warps=8,
            )
    return out.to(inp.dtype)


# Reductions that the single-pass atomic kernel can carry. prod is excluded
# because there is no atomic multiply primitive and ``0 * x`` would poison every
# position that is hit only after another writer already stored a zero.
_ATOMIC_REDUCES = ("sum", "mean", "amax", "amin")


def _atomic_candidate(inp, dim, src, index, reduce, include_self):
    """Return (idx_offsets, src_values, acc, aux, ncols, aux_kind) or None.

    The kernel recovers the output base as ``row * ncols``, which only holds for
    last-dimension scatter with identical index/src/inp shapes, so every other
    layout keeps the scan / general kernels.

    ``aux_kind`` tells the launcher what the auxiliary plane holds: ``"none"``
    when only ``acc`` is needed, ``"count"`` for the float32 contribution count
    of mean, and ``"mark"`` for the racy hit marker that reproduces the
    ``include_self=False`` contract of sum/amax/amin.
    """
    if reduce not in _ATOMIC_REDUCES:
        return None
    if inp.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return None
    if src.dtype != inp.dtype or index.dtype != torch.int64:
        return None
    if inp.ndim == 0 or inp.ndim > 5:
        return None
    dim = dim % inp.ndim
    if dim != inp.ndim - 1:
        return None
    if index.shape != src.shape or index.shape != inp.shape:
        return None
    if index.numel() < _MIN_ATOMIC_NUMEL:
        return None
    if inp.numel() > _MAX_SCAN_NUMEL:
        return None

    ncols = int(inp.shape[-1])
    if ncols <= 0:
        return None
    # fp16/bf16 src and the int64 index are upcast inside the kernel instead of on
    # the host: that skips a full cast pass (about 96MB at 4096^2 fp16, and half
    # the index traffic at 1e5 columns).
    src_c = src.contiguous()
    idx_c = index.contiguous()
    shape = inp.shape
    dev = inp.device
    # The self value is never copied into the accumulator: the finalize kernel
    # folds it in, which keeps the accumulator at its neutral element and saves a
    # full read+write pass.
    neutral = {3: float("-inf"), 4: float("inf")}.get(_REDUCE_IDS[reduce])
    if reduce == "mean":
        # mean needs the exact divisor, so the plane is a real counter and the
        # float32 atomic (probe: about 2x cheaper than an int32 one) accumulates
        # it. Both planes are zero so one allocation and one fill serve them.
        planes = torch.zeros((2,) + tuple(shape), dtype=torch.float32, device=dev)
        return idx_c, src_c, planes[0], planes[1], ncols, "count"
    if neutral is None:
        acc = torch.zeros(shape, dtype=torch.float32, device=dev)
    else:
        # include_self=False amax/amin start from the neutral element; the mark
        # plane selects the untouched positions back to self.
        acc = torch.full(shape, neutral, dtype=torch.float32, device=dev)
    if include_self:
        return idx_c, src_c, acc, None, ncols, "none"
    # A plain atomic accumulator cannot tell a position that summed to exactly
    # zero from one that was never written, so a separate plane marks every
    # write and the finalize kernel selects acc or self per element.
    return (
        idx_c,
        src_c,
        acc,
        torch.zeros(shape, dtype=torch.int32, device=dev),
        ncols,
        "mark",
    )


def _atomic_grid(total, block):
    programs = triton.cdiv(total, block)
    return (
        min(programs, _MAX_ATOMIC_GRID_X),
        triton.cdiv(programs, _MAX_ATOMIC_GRID_X),
    )


def _launch_atomic_reduce(
    idx_offsets,
    src_values,
    acc,
    aux,
    ncols,
    reduce,
    aux_kind,
    block=_ATOMIC_BLOCK,
    warps=4,
):
    from flag_gems.runtime import torch_device_fn

    total = idx_offsets.numel()
    # The kernel never dereferences the auxiliary pointer when MEAN/MARKED are
    # both False, so the accumulator stands in for it and no extra allocation is
    # needed.
    with torch_device_fn.device(acc.device):
        _gcu_atomic_reduce_lastdim_kernel[_atomic_grid(total, block)](
            idx_offsets,
            src_values,
            acc,
            acc if aux is None else aux,
            total,
            ncols,
            REDUCE_ID=_REDUCE_IDS[reduce],
            MEAN=aux_kind == "count",
            MARKED=aux_kind == "mark",
            BLOCK=block,
            num_warps=warps,
        )
    return acc


def _finalize_atomic_reduce(
    acc,
    aux,
    self_tensor,
    reduce,
    aux_kind,
    include_self,
    block=_ATOMIC_BLOCK,
    warps=4,
):
    """Materialize the reduced value and cast back to the input dtype."""
    from flag_gems.runtime import torch_device_fn

    self_c = self_tensor.contiguous()
    out = torch.empty(self_c.shape, dtype=self_c.dtype, device=self_c.device)
    total = out.numel()
    with torch_device_fn.device(out.device):
        _gcu_atomic_reduce_finalize_kernel[_atomic_grid(total, block)](
            acc,
            acc if aux is None else aux,
            self_c,
            out,
            total,
            REDUCE_ID=_REDUCE_IDS[reduce],
            MEAN=aux_kind == "count",
            MARKED=aux_kind == "mark",
            INCLUDE_SELF=include_self,
            BLOCK=block,
            num_warps=warps,
        )
    return out
