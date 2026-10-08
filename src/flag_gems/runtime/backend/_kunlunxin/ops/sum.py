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
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress

try:
    import triton.experimental.tle.language as tle
    from triton.runtime import driver
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)

# `tritonxpu-tle-pipeline` (KTC-119) rotates a loop's GM->LM fill into a prefetch,
# and it only runs when this is set at pass-construction time, which is the first
# compile -- long after import, so setting it here is early enough. It is a no-op
# for any loop that does not also carry `tt.num_stages >= 2`, which is why turning it
# on here does not change any other operator: the attribute comes from
# `tl.range(..., num_stages=)`, and the only loop in this file that asks for it is the
# fold below. Without it the fold is a
# synchronous loop -- correct, just with no DMA/compute overlap (1G f32 along dim=1:
# 2.83ms against 2.42ms pipelined).
os.environ.setdefault("TRITONXPU_TLE_PIPELINE", "1")

# =============================================================================
# XPU correctness constraints (see harness/HARNESS_SUMMARY.md §2.5 / §3.6):
#   - `tl.sum` is only exact for tiles <= 8192 lanes (no buffer limit) or
#     == 32768 lanes WITH buffer_size_limit=2048; 16384/65536 miscompile.
#   - Masked loads (mask + other=0) are NOT reliable inside reductions for
#     tiles >= 32768 lanes, and unreliable INSIDE runtime loops; single-shot
#     masked tiles <= 8192 lanes ARE reliable (validated 2026-08-22).
#   - `tl.where(mask, x, 0)` inside a reduction miscompiles; avoid.
# All kernels below are built exclusively from: unmasked exact-size tiles,
# single-shot masked tiles <= 8192 lanes, static merges <= 8192 lanes.
# =============================================================================

# =============================================================================
# tle row-reduce -- the only kernel either path needs when the dtype fits
# =============================================================================
# The pointer kernels below read the reduction axis with `tl.load`, and the tail
# kernel additionally walks its rows with a static loop. Measured against aten on
# the benchmark shapes that is 0.01-0.45x (1024x1024x1024 f32 along dim=1: 172ms
# against 2.39ms). `tle.gpu` moves the same tile with the cluster DMA instead --
# one descriptor per [XBLOCK, YBLOCK] tile, GM -> LM -> registers -- and lands at
# 0.8-1.05x, so it is the path taken whenever the layout and dtype fit.
#
# It also needs only one kernel for both operators, where the pointer path needed
# seven: it cannot mask inside a reduction (see the constraints at the top of this
# file), so every tail -- a short chunk, a short row, a partial buffer that is not a
# power of two -- had to be split off into its own launch or staged into a zero-padded
# buffer. On the tle side a short tile is not masked but zero-filled before the copy,
# and the copy clamps itself to the descriptor, so an arbitrary [M, N] needs one
# launch and the pointer path is down to the two kernels f64 and bool still need.
_TLE_CORE_NUM = 64
# LM per core. 4 KB is what test/tle/test_tle_copy_1d.py and utils/tle_copy.py both
# settle on; 8 KB fails to allocate here ("TLE kernel stack is over the local-memory
# budget"), since the tile buffer is only part of the footprint -- the accumulator
# and the address vectors live in the same budget.
_TLE_LM_BYTES_PER_CORE = 4096
# Rows per program. The reduce is core-local at XBLOCK >= core_num, and wider rows
# amortise the launch: on 1024x1024x1024 f32, XBLOCK 64/128/256/512 measured
# 4.46/3.45/2.86/2.50ms against aten's 2.57ms. 1024 regresses (2.64ms).
_TLE_XBLOCK = 512
# Clusters on the chip; the grid wants about one program each.
_TLE_CLUSTERS = 8
# A row this wide has to buy a long YBLOCK, even at the cost of a 64-row XBLOCK and its
# expensive result write.
_TLE_WIDE_ROW_BYTES = 32768
# Below this the reduce is all launch and result write, so take the widest XBLOCK and
# the smallest grid.
_TLE_NARROW_ROW_BYTES = 2048
# Flat reduce: reshape to [n // COLS, COLS] and row-reduce. COLS stays small so the
# grid stays wide -- on a 1G-element f32 reduce, [4096, 262144] measured 3.78ms and
# [262144, 4096] 2.46ms against aten's 2.36ms, i.e. the same bytes at 0.64x and 0.96x.
_TLE_FLAT_COLS = 4096
# What one program's loop finishes on its own. A [1, n] reduce leaves 63 of the 64
# cores idle, so it is only worth it when staging would cost more than it saves: below
# this, a stage plus its partial buffer plus a second launch is the slower half of a
# reduce that is already launch-bound.
_TLE_FLAT_ONESHOT = 8192

# Only dtypes verified to survive an LM round trip on KL3.
#
# float64 is absent because it is *broken* on this path, not merely imprecise: an
# all-ones (64, 64) f64 row-reduce with an f64 accumulator returns 1.0 / 1.75 alternating
# instead of 64.0, which is what two 32-bit lanes reinterpreted as one f64 looks like.
# The identical kernel in f32 is exact, and a tile as small as 64x64 (512 B/core) still
# fails, so it is neither the LM budget nor the accumulator. (The earlier "f64 is off by
# 1.9e-6" note was measured while `_tle_row_plan` hardcoded ACC_DTYPE=tl.float32, which
# kept f64 arithmetic out of the kernel entirely and hid this.) f64 therefore keeps the
# pointer fallback; everything else here reduces exactly, the integers as long as the
# accumulator is int64, which is also aten's rule.
_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
}
# Accumulator per input dtype, matching `_resolve_acc_dtype`. Summing integers in f32
# would start losing counts at 2^24.
_TLE_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.float32: tl.float32,
    torch.bfloat16: tl.float32,
    torch.int8: tl.int64,
    torch.int16: tl.int64,
    torch.int32: tl.int64,
    torch.int64: tl.int64,
    torch.uint8: tl.int64,
}


def _npo2(x):
    """`triton.next_power_of_2` costs 3.2us a call, which a 64x64 reduce cannot
    afford: two of them were a third of that launch's host time. This is 0.08us."""
    return 1 << (x - 1).bit_length() if x > 1 else 1


# (M, N, in dtype, out dtype) -> everything the launch needs that does not depend on
# the pointers. Sizing a tile is pure arithmetic, but at ~10us of host time per launch
# and a 2us kernel, arithmetic is the cost. Bounded by the distinct shapes an
# application reduces.
_TLE_ROW_GEOM = {}
_TLE_ROW_PLANS = {}
_TLE_FOLD_PLANS = {}
_TLE_PLAN_MISS = object()
_FLAT_LAUNCHERS = _TLE_PLAN_MISS


def _flat_launchers():
    """`driver.active.flat_launchers`, resolved once. `driver.active` is a lazy proxy,
    so the attribute walk is not free at ~12us a launch."""
    global _FLAT_LAUNCHERS
    if _FLAT_LAUNCHERS is _TLE_PLAN_MISS:
        _FLAT_LAUNCHERS = getattr(driver.active, "flat_launchers", None)
    return _FLAT_LAUNCHERS


def _tle_available():
    """`tle.gpu` exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


# Evaluated once: two `os.environ.get` calls per reduce are not free when the whole
# launch is ~12us, and neither variable can change under a running process.
_TLE_AVAILABLE = _tle_available()


@triton.jit(
    # Both callers hand this kernel a freshly allocated buffer, and the XPU allocator
    # returns pointers from varying divisibility classes. With alignment on the
    # specialization key every new class is a fresh compile: the 1G-element flat
    # reduce measured 767ms per call that way. Nothing here reads the alignment, and
    # `N` only bounds the loop.
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_sum_row_kernel(
    a_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
):
    """Sum a [XBLOCK, YBLOCK]-tiled slice of a [M, N] input along axis=1.

    tritonxpu-tle-core-tiling hands each of the 64 cores XBLOCK/core_num whole rows
    of the tile, so the reduce stays core-local and needs no barrier. The reduction
    axis is streamed in YBLOCK columns; the running sum is kept as [XBLOCK] rather
    than [XBLOCK, YBLOCK] (the layernorm pattern) because a 2D accumulator is what
    forces the tile down: with it XBLOCK=512 does not fit the LM budget at all, and
    the largest tile that does measured 3.08ms on 1024x1024x1024 f32 against 2.50ms
    here.

    Partial tiles need no mask, on either axis. `tle.gpu.copy` clamps the transfer to
    the descriptor extents, so a tile that hangs off the edge is simply not fully
    written -- the bad values are the *stale* LM bytes it left behind, not the copy.
    Zeroing the buffer first therefore makes the pad exact, and NEED_ZERO does that on
    the one iteration that can be partial. Masking after the load is what does not
    work: `tl.where(coff + col_ids < N, aval, 0.0)` returns wrong numbers in bf16 and
    blows the LM budget in f16 at every tile size, and `tl.load(..., mask=)` has no
    lowering for a local pointer at all. A row block past M needs nothing either way,
    since the output copy clamps the same way on the way out.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    acc = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        # Only the last step can be short, and only then is the buffer worth clearing:
        # every other step overwrites it whole.
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        acc = acc + tl.sum(tl.load(a_ptrs).to(ACC_DTYPE), axis=1)
    tl.store(c_ptrs, (acc * SCALE).to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_row_geom(M, N, itemsize):
    """Cached tiling for a [M, N] row-reduce: `(xblock, yblock, row_blocks)`.

    XBLOCK is rows per program and YBLOCK is the LM budget divided by it, so the two
    trade off, and the trade is not subtle: on 24 shapes x 2 dtypes the wrong choice
    costs between 1.3x and 6.4x. Three effects set it.

    First, `tle.gpu.copy(c_lmem, c_desc, [XBLOCK], ...)` -- the LM -> GM write of the
    result -- costs ~9us at XBLOCK == 64 and ~0.7us at XBLOCK == 512, measured with a
    kernel that does nothing else. At 64 rows a core owns a single element of the
    result, and the write degenerates into 64 four-byte transfers. That alone is more
    than aten spends on a whole (64, 64) reduce, so XBLOCK == 64 is only ever worth it
    when the reduction axis is long enough to pay for it.

    Second, the grid is `ceil(M / XBLOCK)`, and the machine wants roughly one program
    per cluster. Two programs leave it idle -- (600, 4096) f32 measured 47.0us at
    XBLOCK 512 (grid 2) against 22.8us at 128 (grid 5) -- and dozens of programs pay
    the per-program cost too many times: (4096, 40999) f32 measured 489us at 64
    (grid 64) against 426us at 128 (grid 32).

    Third, a long row wants a long YBLOCK, because that is what makes the transfers
    big and the loop short. (600, 40999) f32: 79.5us at XBLOCK 64 (YBLOCK 1024)
    against 511us at 512 (YBLOCK 128), a 6.4x swing on the same bytes.

    So: aim for one program per cluster, then let the row length pull XBLOCK down when
    it is long enough to need the YBLOCK, or up when the whole reduce is small enough
    that the launch and the result write are all there is. XBLOCK may exceed M -- the
    padding rows are clamped by both copies, in and out.

    None is never returned: there is no fallback left to decline to.
    """
    geom_key = (M, N, itemsize)
    geom = _TLE_ROW_GEOM.get(geom_key, _TLE_PLAN_MISS)
    if geom is not _TLE_PLAN_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        xblock = max(_TLE_CORE_NUM, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 256
    tile_elems = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM // itemsize
    yblock = max(1, min(_npo2(N), tile_elems // xblock))
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, -(-M // xblock))
    _TLE_ROW_GEOM[geom_key] = geom
    return geom


def _tle_row_plan(M, N, row_stride, in_dtype, out_dtype, geom, scale=1.0):
    """Cached launch plan for reducing `[M, N]` columns at row pitch `row_stride`.

    `geom` is the caller's `_tle_row_geom` result, and `row_stride` is the parent row
    pitch, which differs from `N` exactly when the caller passes a column slice.
    It is a launch operand and part of the key: the flat launcher replays the operand
    list verbatim, so a plan built for a contiguous view would otherwise be reused for
    a sliced one and stride the DMA wrong -- silently reducing the wrong columns.

    `scale` multiplies the result before the store (`mean = sum * (1/N)`) and is a
    **constexpr of the kernel**, so it must be part of the key: replaying a plan built
    for one scale with another would silently return the wrong values.
    """
    plan_key = (M, N, row_stride, in_dtype, out_dtype, scale)
    plan = _TLE_ROW_PLANS.get(plan_key)
    if plan is not None:
        return plan
    xblock, yblock, row_blocks = geom
    consts = (
        xblock,
        yblock,
        _TLE_TL_DTYPE[in_dtype],
        _TLE_ACC_DTYPE[in_dtype],
        _TLE_TL_DTYPE[out_dtype],
        N % yblock != 0,
        scale,
    )
    key = (M, N, row_stride, row_blocks, in_dtype, out_dtype) + consts[:2] + consts[5:]
    # The last element is the launch's operand list minus the two pointers. Building it
    # here rather than per call keeps `a.shape` and `a.stride(0)` -- two dispatcher round
    # trips -- off a path whose whole budget is ~12us.
    plan = ((row_blocks,), consts, key, (M, N, row_stride, 1), (M, 1, N))
    _TLE_ROW_PLANS[plan_key] = plan
    return plan


def _tle_row_reduce(a, c, plan):
    """`c[m] = sum(a[m, :])` for a 2-D `a` whose last stride is 1, and 1-D `c`.

    `plan` comes from `_tle_row_plan`. Both sides are addressed through tensor
    descriptors, so the launch is bound once and replayed flat afterwards: a small
    row-reduce is host-bound, and going through `JITFunction.run` every call costs
    ~9us of the ~12us a 64x64 reduce takes. The launcher key carries everything the
    compiled kernel depends on -- dtypes, both block sizes, the extents, and the
    grid, which participates in compilation on XPU. Pointers are deliberately absent,
    which is only sound because the kernel is compiled without alignment
    specialization.
    """
    grid, consts, key, a_meta, c_meta = plan

    launchers = _flat_launchers()
    if launchers is None:  # triton without the launcher cache: correct, just slower
        _tle_sum_row_kernel[grid](
            TensorDescriptor.from_tensor(a, block_shape=[consts[0], consts[1]]),
            TensorDescriptor.from_tensor(c, block_shape=[consts[0]]),
            a_meta[1],
            *consts,
        )
        return

    launch, stream = launchers.acquire(_tle_sum_row_kernel, key)
    if launch is None:
        kernel = _tle_sum_row_kernel[grid](
            TensorDescriptor.from_tensor(a, block_shape=[consts[0], consts[1]]),
            TensorDescriptor.from_tensor(c, block_shape=[consts[0]]),
            a_meta[1],
            *consts,
        )
        launchers.bind(_tle_sum_row_kernel, key, kernel, grid)
        return
    # Descriptor ABI: base pointer, then `.shape` (i32) and `.strides` (i64).
    launch(stream, a.data_ptr(), *a_meta, c.data_ptr(), *c_meta)


def _tle_sum_dim(inp, out, M, N, scale=1.0):
    """Row-reduce `inp` into `out` with tle; False if tle cannot express it."""
    if not _TLE_AVAILABLE:
        return False
    # N == 1 makes `inp.view(M, 1)` a descriptor whose last stride is M, not 1 (torch
    # is free to pick any stride for an extent-1 dimension), which
    # `TensorDescriptor.__post_init__` rejects outright. Reducing a single element is a
    # copy anyway, so it belongs on the pointer path.
    if N < 2:
        return False
    if inp.dtype not in _TLE_TL_DTYPE or out.dtype not in _TLE_TL_DTYPE:
        return False
    if not inp.is_contiguous() or not out.is_contiguous():
        return False
    geom = _tle_row_geom(M, N, inp.element_size())
    plan = _tle_row_plan(M, N, N, inp.dtype, out.dtype, geom, scale)
    a = inp if inp.ndim == 2 and inp.shape[0] == M else inp.view(M, N)
    c = out if out.ndim == 1 else out.view(M)
    with torch_device_fn.device(inp.device):
        _tle_row_reduce(a, c, plan)
    return True


def _tle_sum_flat(inp, out, acc_dtype):
    """Full reduction as repeated row-reduces of a [n // COLS, COLS] view.

    Feeding the output back in is the whole algorithm: 1G elements go 1G -> 262144 in
    one launch, and what is left is three orders of magnitude smaller. COLS shrinks once
    the tensor does, so the later stages still have rows for every core, and the last
    one -- everything that fits a single program's loop -- is a [1, n] reduce, which is
    exact but leaves 63 cores idle, so it is only ever handed a small residue. A
    `n % COLS` residue is not reduced: it is carried into the next stage's buffer as-is,
    which costs a copy of under COLS elements and saves a launch.

    Returns False without touching `out` only when the dtype or the layout is out of
    reach; any size is in reach.
    """
    if not _TLE_AVAILABLE:
        return False
    # `acc_dtype` is `_resolve_acc_dtype(inp.dtype)`, which for every dtype in
    # `_TLE_TL_DTYPE` is itself in `_TLE_TL_DTYPE`, so accepting `inp.dtype` implies it.
    if inp.dtype not in _TLE_TL_DTYPE:
        return False
    cur = inp.reshape(-1) if inp.is_contiguous() else inp.contiguous().view(-1)
    with torch_device_fn.device(inp.device):
        while cur.numel() > _TLE_FLAT_ONESHOT:
            n = cur.numel()
            # Wide tiles while there is a lot left; once the tensor is within a couple of
            # tiles, narrow them so the grid still covers the cores.
            cols = (
                _TLE_FLAT_COLS
                if n >= 2 * _TLE_FLAT_COLS
                else max(2, _npo2(n) // _TLE_CORE_NUM)
            )
            rows = n // cols
            if rows < 2:
                break
            geom = _tle_row_geom(rows, cols, cur.element_size())
            plan = _tle_row_plan(rows, cols, cols, cur.dtype, acc_dtype, geom)
            main = rows * cols
            tail = n - main
            nxt = torch.empty((rows + tail,), dtype=acc_dtype, device=inp.device)
            _tle_row_reduce(cur[:main].view(rows, cols), nxt[:rows], plan)
            if tail:
                nxt[rows:].copy_(cur[main:])
            cur = nxt
        n = cur.numel()
        logger.debug("GEMS_KUNLUNXIN SUM_FLAT tle numel=%d -> %d", inp.numel(), n)
        if n < 2:
            # A single element (or none) is a copy; `out` is 0-d or (1,).
            out.view(-1).copy_(cur.view(-1))
            return True
        geom = _tle_row_geom(1, n, cur.element_size())
        plan = _tle_row_plan(1, n, n, cur.dtype, out.dtype, geom)
        _tle_row_reduce(cur.view(1, n), out.view(1), plan)
    return True


def _resolve_acc_dtype(inp_dtype):
    """ATen reduction accumulate type: fp32 for 16-bit fp, fp64 for fp64,
    int64 for all integers/bool. 32-bit fp accumulates in fp32."""
    if inp_dtype is torch.float64:
        return torch.float64
    if inp_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    return torch.int64


def _resolve_out_dtype(inp_dtype, dtype):
    """torch.sum dtype=None semantics: int/bool -> int64, float -> input dtype."""
    if dtype is not None:
        return dtype
    if inp_dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32):
        return torch.int64
    return inp_dtype


# Middle-axis reduce. FlagGems otherwise calls `dim_compress`, which permutes the
# reduced axis to the end and materialises the result -- a transposing copy of the
# whole tensor, so 4 GB in and 4 GB out before the reduce has read anything (1G f32
# along dim=1: 5.69ms of copy plus 2.50ms of reduce, against 2.39ms for the whole
# aten call). XDNN never does that: `reduce_calc_common.cpp` normalises to (m, t, n)
# and dispatches `reduce_mtn`, which keeps an n-wide accumulator and folds t across
# loop iterations, so the DMA only ever reads contiguous n-runs.
#
# This is that, with the tile sized to keep the transfer large. One row per step --
# XDNN's !load_multi_n branch -- is DMA-issue bound here (34.9ms on 1G f32), so a
# step reads NBLOCK whole rows and the accumulator is [NBLOCK, KBLOCK]: still pure
# elementwise adds, no axis=0 reduce, which matters because core-tiling cannot
# express one (`triton_xpu.reduce` infers a per-core slice of the result). What comes
# out is the same reduction with N shrunk to NBLOCK, small enough that the ordinary
# dim_compress + row-reduce path finishes it for a few percent of the total.
#
# Tile: [32, 1024] f32. KBLOCK wants to be the whole row -- at kb=1024 the fold
# measured 2.83ms on 1G f32 against 3.48ms at kb=512 and 3.58ms at kb=256 -- and
# 32*1024 is the largest that fits, since the f32 accumulator, the f32 output buffer
# and the input tile share the 8 KB per-core budget.
_TLE_PIPELINE = os.environ.get("TRITONXPU_TLE_PIPELINE") == "1"
# Depth 2 (num_stages=3) is the measured optimum, and it pays for the halved tile
# several times over. On 1G f32 along dim=1, (tile, num_stages) measured:
# (32768, 1) 2.83ms, (32768, 2) does not fit the budget, (16384, 2) 3.01ms,
# (16384, 3) 2.42ms -- against aten's 2.39ms. f16/bf16 go 2.31/2.89ms -> 1.46/1.51ms.
# The pass doubles the rotated buffer, so the tile has to come down to pay for it:
# `fitBudget` counts only `local_alloc` and runs even with the pass off.
_TLE_FOLD_TILE = 16384 if _TLE_PIPELINE else 32768
_TLE_FOLD_STAGES = 3 if _TLE_PIPELINE else 1
# The pipeline pass needs at least one full 64-byte vector per core in the buffer it
# rotates; below that, lowering fails outright ("invalid element type in
# packLLElements. Expected '!llvm.ptr<2>' but got '!llvm.ptr'"). The boundary is exactly
# 64 B/core, swept over NBLOCK x KBLOCK x dtype: every tile at or above it lowers, every
# tile below it crashes, and the same tiles all lower fine at num_stages=1. So this only
# turns the overlap off -- it never costs a shape its tle path.
_TLE_FOLD_MIN_STAGE_BYTES = 64 * _TLE_CORE_NUM


@triton.jit(
    do_not_specialize=["N", "K", "kblocks"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_sum_fold_kernel(
    a_desc,
    c_desc,
    N,
    K,
    kblocks,
    NBLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
):
    """`c[j, b, k] = sum_i a[b, i * NBLOCK + j, k]` over a (B, N, K) input.

    A program owns one batch and one K block. The input is addressed as (B * N, K),
    so a transfer is a run of whole contiguous rows at row `b * N + noff`.

    `N % NBLOCK == 0` is the caller's contract; there is no mask. K needs none -- the
    last block is shifted back to end at K (see below). The elementwise `tl.where` that
    would cover a tail was measured wrong on both axes -- (3, 33, 512) f32 (N tail) came
    back off by 398, (3, 32, 1000) f32 (K tail) by 13.6, and (3, 33, 1000) returned
    nan/2e38 -- so masking a tile loaded through LM is not something to rely on here,
    elementwise or not. `_tle_sum_mid` picks a divisible NBLOCK or runs one pass at
    NBLOCK == 1. The
    partials come out with the surviving reduction index OUTERMOST -- (NBLOCK, B * K)
    -- which is what makes the second stage another fold rather than a transpose: its
    reduction axis is the outer one and its contiguous axis is all of B * K, so it
    reads 128 KB runs where a (B, NBLOCK, K) partial would have needed dim_compress
    (measured: that layout cost ~700us on the 1G f32 case, this one ~90us).
    """
    pid = tl.program_id(0)
    kb = pid % kblocks
    b = pid // kblocks
    # The last K block is shifted back to end at K rather than hanging off the row.
    # K is the contiguous axis, so an overhanging block would read the NEXT row -- live
    # data, which no clamp stops and no zero-fill hides (measured: (3, 32, 100) f32 off
    # by 6.6). Shifting instead re-reads columns the previous block already covered, and
    # that is harmless here in a way it would not be on the reduction axis: each program
    # sums all of N for its own columns, so the overlap writes the same result twice
    # rather than double-counting it. It costs at most one block of re-read per row.
    k_off = tl.minimum(kb * KBLOCK, K - KBLOCK)

    a_lmem = tle.gpu.alloc(
        [NBLOCK, KBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc(
        [NBLOCK, KBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    n_ids = tl.broadcast_to(tl.arange(0, NBLOCK)[:, None], (NBLOCK, KBLOCK))
    k_ids = tl.broadcast_to(tl.arange(0, KBLOCK)[None, :], (NBLOCK, KBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (n_ids, k_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (n_ids, k_ids))

    acc = tl.zeros([NBLOCK, KBLOCK], ACC_DTYPE)
    for noff in tl.range(0, N, NBLOCK, num_stages=NUM_STAGES):
        tle.gpu.copy(a_desc, a_lmem, [NBLOCK, KBLOCK], [b * N + noff, k_off])
        acc = acc + tl.load(a_ptrs).to(ACC_DTYPE)
    tl.store(c_ptrs, (acc * SCALE).to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [NBLOCK, KBLOCK], [0, b * K + k_off])


def _tle_fold(inp, part, B, N, K, nblock, kblock, scale=1.0):
    """Launch the fold, binding the launcher once as `_tle_row_reduce` does."""
    a = inp.view(B * N, K)
    c = part.view(nblock, B * K)
    plan_key = (B, N, K, nblock, kblock, a.dtype, c.dtype, scale)
    plan = _TLE_FOLD_PLANS.get(plan_key)
    if plan is not None:
        grid, args, key = plan
        launchers = _flat_launchers()
        if launchers is not None:
            launch, stream = launchers.acquire(_tle_sum_fold_kernel, key)
            if launch is not None:
                launch(
                    stream,
                    a.data_ptr(),
                    B * N,
                    K,
                    K,
                    1,
                    c.data_ptr(),
                    nblock,
                    B * K,
                    B * K,
                    1,
                    N,
                    K,
                    args[2],
                )
                return
    kblocks = -(-K // kblock)
    grid = (B * kblocks,)
    consts = (
        nblock,
        kblock,
        _TLE_TL_DTYPE[a.dtype],
        _TLE_ACC_DTYPE[a.dtype],
        _TLE_TL_DTYPE[c.dtype],
        # A one-row tile cannot be rotated at all, and neither can one whose per-core
        # slice is under a vector wide (see _TLE_FOLD_MIN_STAGE_BYTES) -- both fail in
        # the same lowering. Those callers are the NBLOCK == 1 ones (the second stage,
        # and the first stage when no power-of-two block divides N); they are a small
        # share of the work, so they run synchronous.
        (
            _TLE_FOLD_STAGES
            if nblock > 1
            and nblock * kblock * a.element_size() >= _TLE_FOLD_MIN_STAGE_BYTES
            else 1
        ),
        scale,
    )
    args = (N, K, kblocks) + consts

    launchers = _flat_launchers()
    if launchers is None:
        _tle_sum_fold_kernel[grid](
            TensorDescriptor.from_tensor(a, block_shape=[nblock, kblock]),
            TensorDescriptor.from_tensor(c, block_shape=[nblock, kblock]),
            *args,
        )
        return

    key = (B, N, K, grid[0], a.dtype, c.dtype) + consts[:2] + consts[5:8]
    _TLE_FOLD_PLANS[plan_key] = (grid, args, key)
    launch, stream = launchers.acquire(_tle_sum_fold_kernel, key)
    if launch is None:
        kernel = _tle_sum_fold_kernel[grid](
            TensorDescriptor.from_tensor(a, block_shape=[nblock, kblock]),
            TensorDescriptor.from_tensor(c, block_shape=[nblock, kblock]),
            *args,
        )
        launchers.bind(_tle_sum_fold_kernel, key, kernel, grid)
        return
    launch(
        stream,
        a.data_ptr(),
        B * N,
        K,
        K,
        1,
        c.data_ptr(),
        nblock,
        B * K,
        B * K,
        1,
        N,
        K,
        kblocks,
    )


def _tle_sum_mid(inp, out, dims, N, scale=1.0):
    """Reduce a single non-last axis without dim_compress; False if it does not fit.

    Two folds. The first shrinks the reduced axis from N to NBLOCK and writes the
    partials reduction-index-outermost; the second folds those NBLOCK planes into one
    with NBLOCK == 1, reading the whole B * K plane as contiguous runs. Neither stage
    transposes anything, and the second touches N/NBLOCK times fewer bytes than the
    first.
    """
    if not _TLE_AVAILABLE:
        return False
    d = dims[0]
    if inp.dtype not in _TLE_TL_DTYPE or out.dtype not in _TLE_TL_DTYPE:
        return False
    if not inp.is_contiguous() or not out.is_contiguous():
        return False
    B = 1
    for s in inp.shape[:d]:
        B *= s
    K = 1
    for s in inp.shape[d + 1 :]:
        K *= s
    # A K narrower than the core count leaves cores idle inside the tile.
    if K < _TLE_CORE_NUM:
        return False
    # K needs no divisibility (the kernel shifts its last block back into the row); it
    # only has to fit one, so that `TensorDescriptor` accepts the block shape.
    kblock = min(_npo2(K), _TLE_FOLD_TILE)
    while kblock > K:
        kblock >>= 1
    nblock = _TLE_FOLD_TILE // kblock
    while nblock > 1 and (N % nblock or nblock >= N):
        nblock >>= 1
    # NBLOCK == 1 when no power-of-two block divides N (N prime, N = 7): the reduction
    # axis cannot take the shift trick -- overlapping rows would be counted twice. One
    # row per transfer makes that stage DMA-issue bound, and it used to skip the second
    # fold by writing `out` directly with SCALE applied. That store is exactly the
    # `(acc * SCALE).to(<16-bit>)` shape that corrupts the last lanes of a 256-element
    # block (mean_dim bf16 flake, 2026-09-18), so NBLOCK == 1 now runs the same two
    # folds as every other shape: SCALE only ever lands on the fp32 partials.
    acc_dtype = _resolve_acc_dtype(inp.dtype)
    part = torch.empty((nblock, B, K), dtype=acc_dtype, device=inp.device)
    plane = B * K
    # Stage two reduces the NBLOCK planes with NBLOCK == 1, over a contiguous axis of
    # B * K. Same contract: the block must divide the plane. It wants the LARGEST such
    # block, which is the opposite of what stage one wants: NBLOCK == 1 leaves the
    # pipeline pass nothing to rotate, so the loop is synchronous and its cost is DMA
    # issue rather than bandwidth. On (64, 512, 512) f16, stage two measured
    # 77.7/44.4/30.4/19.3/19.9/24.1us at kblock2 512/1024/2048/4096/8192/16384 -- so
    # insisting on a wide grid more than doubled it. Four programs is where the curve
    # flattens.
    kblock2 = min(_npo2(plane), _TLE_FOLD_TILE)
    while kblock2 > 1 and (kblock2 > plane or plane // kblock2 < 4):
        kblock2 >>= 1
    with torch_device_fn.device(inp.device):
        _tle_fold(inp, part, B, N, K, nblock, kblock, scale)
        _tle_fold(part, out, 1, nblock, plane, 1, kblock2)
    logger.debug(
        "GEMS_KUNLUNXIN SUM_DIM tle fold B=%d N=%d K=%d tile=%dx%d then %dx%d",
        B,
        N,
        K,
        nblock,
        kblock,
        1,
        kblock2,
    )
    return True


def _launch_sum_flat(inp, out, acc_dtype):
    """Exact flat reduction writing into `out` (0-d or (1,))."""
    if not _tle_sum_flat(inp, out, acc_dtype):
        raise NotImplementedError(f"kunlunxin sum: no tle path for dtype {inp.dtype}")


def _launch_sum_dim(inp, out, M, N, scale=1.0):
    if scale != 1.0 and (M == 1 or N == 1):
        # The two degenerate branches below bypass the TLE kernels (flat / copy)
        # and carry no scale, so refuse rather than return an unscaled result.
        raise NotImplementedError(
            "kunlunxin sum: scale != 1 is only supported on the tle row/fold paths"
        )
    if M == 1:
        # Degenerate: whole tensor reduces to one element -> route to the flat
        # machinery, which parallelises over N. The row-reduce must not see this
        # shape: one row is one program, so it would walk all of N in a single
        # core-tiled loop (1G f32 measured 765ms against 2.5ms for the flat path).
        _launch_sum_flat(inp.view(-1), out, _resolve_acc_dtype(inp.dtype))
        return
    if N == 1:
        # Summing one element is a copy, and a descriptor cannot express it anyway:
        # torch is free to give an extent-1 dimension any stride, and
        # `TensorDescriptor.__post_init__` requires the last one to be 1. `copy_` also
        # applies the dtype promotion the caller's `out` asks for.
        out.view(M).copy_(inp.view(M))
        return
    if inp.dtype is torch.bool:
        # There is no LM dtype for bool, but its values are 0 and 1, so int8 with an
        # int64 accumulator reduces it exactly.
        inp = inp.to(torch.int8)
    if not inp.is_contiguous():
        # A descriptor needs the last stride to be 1.
        inp = inp.contiguous()
    if not _tle_sum_dim(inp, out, M, N, scale):
        raise NotImplementedError(
            f"kunlunxin sum: no tle path for dtype {inp.dtype} -> {out.dtype}"
        )


def _reduce_view(inp, dims):
    """`inp` reshaped so the reduced dims are the trailing, contiguous ones.

    `dim_compress` is `inp.permute(order).contiguous()`, which for dims that already
    sit at the end of a contiguous tensor is a no-op that still costs a list
    comprehension, a sort, a permute and a dispatcher round trip through
    `contiguous()`. On a reduce whose whole cost is ~15us of host time that is not
    noise, and the trailing case is the common one -- it is what `torch.sum(x, -1)`
    and every already-compressed caller hand us.
    """
    if inp.is_contiguous() and sorted(dims) == list(
        range(inp.ndim - len(dims), inp.ndim)
    ):
        return inp
    return dim_compress(inp, dims)


def _prep_flat(inp, dtype):
    if dtype is None and inp.dtype is torch.bool:
        inp = inp.to(torch.int64)
    out_dtype = _resolve_out_dtype(inp.dtype, dtype)
    acc_dtype = _resolve_acc_dtype(inp.dtype)
    return inp, out_dtype, acc_dtype


def sum(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SUM")
    inp, out_dtype, acc_dtype = _prep_flat(inp, dtype)
    out = torch.empty([], dtype=out_dtype, device=inp.device)
    _launch_sum_flat(inp, out, acc_dtype)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN SUM_OUT")
    inp, _, acc_dtype = _prep_flat(inp, dtype)
    _launch_sum_flat(inp, out, acc_dtype)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None, scale=1.0):
    logger.debug("GEMS_KUNLUNXIN SUM_DIM")
    out_dtype = _resolve_out_dtype(inp.dtype, dtype)

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is None or dim == []:
            out_shape = [1] * len(out_shape) if keepdim else []
        else:
            dims = dim if isinstance(dim, (list, tuple)) else [dim]
            if keepdim:
                for d in dims:
                    out_shape[d % inp.ndim] = 1
            else:
                for d in sorted(dims, key=lambda x: x % inp.ndim, reverse=True):
                    out_shape.pop(d % inp.ndim)
        out = torch.empty(out_shape, dtype=out_dtype, device=inp.device)
        zero_(out)
        return out

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=out_dtype, device=inp.device)

    if not (
        len(dim) == 1
        and dim[0] != inp.ndim - 1
        and _tle_sum_mid(inp, out, dim, N, scale)
    ):
        _launch_sum_dim(_reduce_view(inp, dim), out, M, N, scale)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN SUM_DIM_OUT")

    if inp.numel() == 0:
        dims = (
            dim
            if isinstance(dim, (list, tuple))
            else ([dim] if dim is not None else [])
        )
        if keepdim:
            for d in dims:
                pass  # out shape already correct from caller
        zero_(out)
        return out

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if list(out.shape) != shape:
        out.resize_(shape)
    if not (
        len(dim) == 1 and dim[0] != inp.ndim - 1 and _tle_sum_mid(inp, out, dim, N)
    ):
        _launch_sum_dim(_reduce_view(inp, dim), out, M, N)
    if not keepdim:
        # Compute squeezed shape and resize in-place
        out_shape = [s for i, s in enumerate(shape) if i not in dim]
        out.resize_(out_shape)
    return out
