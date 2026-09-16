import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

try:
    import triton.experimental.tle.language as tle
    from triton.runtime import driver
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger("flag_gems.ops.native_group_norm")
rsqrt = tl_extra_shim.rsqrt

# =============================================================================
# tle.gpu path
# =============================================================================
# group norm is a row-reduce plus a row-affine over the SAME contiguous data:
#
#   stats   (N * group,  group_size * HxW) -> mean, rstd     one row per group
#   affine  (N * C,      HxW)              -> y = x * scale + shift
#                                                     one row per (batch, channel)
#
# Both rows are contiguous in a contiguous NCHW input, so both fit the
# `[XBLOCK, YBLOCK]` cluster-DMA tile that sum.py's `_tle_sum_row_kernel` uses:
# GM -> LM -> registers, no per-element GM pointer arithmetic. The fused
# GM-pointer kernel below walks the same bytes with `tl.load(X + base + idx)`,
# which is why it measured 0.06x of aten.
#
# TWO passes over the data, not one, and the reason is a hardware gap rather than
# taste. The hand-written XDNN kernel does group norm in ONE kernel
# (api/src/kernel/kunlun3cpp/kunlun3cpp_aten/group_norm_fwd.xpu): it gives each
# cluster a slice of the N * group groups, splits one group's `n` elements across
# SEVERAL CORES, and reduces across those cores through shared memory and a
# cluster barrier -- `sum_sm[transposed_cid] = sum; mfence_sm(); sync_all();
# groupsum_sm(...)`, lines 968-973 -- after which every core has the group's mean
# and can apply the affine in the same kernel.
#
# tle.gpu has the shared-memory scope and the SM fence but NOT that barrier: the
# XPU op set is copy / normcopy / raw / dma_wait, and there is no `core_id()`
# either, because `tritonxpu-tle-core-tiling` is what assigns tile ROWS to cores.
# So a single tle kernel has to keep a whole group in one program, and that
# program's tile is only `group_size` rows -- 2 on the benchmark shapes, leaving
# 62 of 64 cores idle. That is exactly the corner the fused GM-pointer kernel
# below is in, and why it measured 0.06x of aten.
#
# Splitting at the barrier instead lets each pass pick the tile it wants.
_TLE_CORE_NUM = 64
# LM per core. Same budget sum.py settles on -- 8 KB fails to allocate, since the
# accumulators and the address vectors share it with the tile.
_TLE_LM_BYTES_PER_CORE = 4096
_TLE_XBLOCK = 512
_TLE_CLUSTERS = 8
_TLE_WIDE_ROW_BYTES = 32768
_TLE_NARROW_ROW_BYTES = 2048
# Below this the extra tle launches cost more than the cluster DMA saves: the
# fused kernel is one launch, and on the accuracy shapes (2, 4, 8) the whole
# operator is launch overhead.
_TLE_MIN_NUMEL = 1 << 15
# The coefficient kernel is a single-shot masked tile, so sum.py's reliable
# ceiling of 8192 lanes applies -- but the binding constraint here is uni_sram,
# not that: at 1024 lanes the kernel stopped COMPILING once the narrow mean/rstd
# stores were added ("out of resource: uni_sram", fp16 with no weight/bias). It
# only ever touches N * C elements, so a narrow tile costs nothing but programs.
_TLE_COEFF_BLOCK = 256
# Two-stage stats reduce. The single-stage stats kernel gives each core ONE
# group-row and reduces it core-local, so the stats grid is `ceil(m_stat /
# xblock)` programs -- and on a wide row xblock is pinned to the 64-core count,
# so a shape with few groups but a long row (e.g. (16, 8, 128, 128): m_stat 64,
# lrow 32768) collapses to grid=1: ONE cluster moves the whole 8 MB while the
# other seven idle, and the reduce measured 0.25x of aten. Splitting the row
# into column chunks -- partial (sum, sumsq) per chunk, then a cheap GM-pointer
# finalize over the chunks -- lets stage 1 run `row_blocks * n_chunks` programs
# and fill the clusters. Only worth it when the row is long enough that the
# extra launch + partial buffer pays for itself and the single-stage grid was
# actually starved.
_TLE_STATS2_ENABLED = True
_TLE_STATS2_MIN_LROW = 8192
_TLE_STATS2_GEOM = {}
# Fused single-kernel path (layernorm pattern): one program owns whole
# group-rows, reduces each row core-locally and normalizes it in place -- one GM
# read, one GM write, no barrier, the way tle_layernorm does it. It needs the
# WHOLE group-row resident in LM, so it only applies when the padded row width
# fits the per-core stack. test_tle_layernorm caps that at epc = XBLOCK*W/64 <=
# 256 (~20 B/element against an 8 KB stack); with XBLOCK == 1 that is a hard
# width ceiling of 256 * 64 = 16384 columns. Wider groups -- (16, 8, 128, 128)
# has group_size*HxW = 32768 -- fall back to the split stats/affine path.
# The SM computed-index gather this kernel needs is CORRECT as of the
# tritonxpu-tle-core-tiling fixes (anchor scan skipping [M,1] row-vector
# candidates, the cyclic 1D make_range, the bf16 SM load) -- see
# sm_gather_task/HANDOFF.md for the chain. What remains is a ROUTING decision:
# the SM weight/bias gather is a scalar per-register chain, so fusion only pays
# on an unpadded narrow row with few groups (measured: (4,16,64,4) 2.0-2.3x,
# (16,16,64) 0.9x, ties elsewhere; 0.2-0.6x on wide or padded rows). The two
# route constants below select that corner; everything else takes the split
# path.
_TLE_FUSE_ENABLED = True
_TLE_FUSE_EPC = 256
_TLE_FUSE_WMAX = _TLE_FUSE_EPC * 64
_TLE_FUSE_ROUTE_WMAX = 512
_TLE_FUSE_ROUTE_MGRP = 64

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


def _npo2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _coeff_block(n):
    """BLOCK for the two GM-pointer helpers, never wider than the buffer.

    A masked load still ISSUES the addresses it masks off -- the reduce skill's
    envelope notes record a power-of-two tile faulting past the last row with an
    illegal access -- so a fixed 256-lane tile over the 4-element stats buffer of
    `(1, 8, 4, 4)` reaches ~1 KB past a 16-byte allocation and HUNG THE CARD
    (`kl3_dev7: kl3_wait_for_noc_idle() timeout`). Clamp to the extent instead;
    these kernels only ever cover a few thousand elements, so a narrow tile costs
    nothing but a few more programs.
    """
    return min(_TLE_COEFF_BLOCK, _npo2(n))


def _tle_available():
    """`tle.gpu` exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()

# Launch overhead is the whole budget here. Measured on card 7 with
# XPU_EVENT_KL3_ENABLE=1, the three-launch tle path sat at 48-61us on EVERY shape
# from 16K to 1M elements -- 16K and 1M cost the same, so that floor is not data
# movement, it is `JITFunction.run` plus four `TensorDescriptor.from_tensor`
# constructions per call. sum.py measured the same thing (~9us of a ~12us launch)
# and the fix is the driver's launcher cache: bind the compiled kernel once, then
# replay the operand list flat. Descriptor ABI is base pointer, then `.shape`
# (i32), then `.strides` (i64); constexprs are compiled in and absent.
#
# Pointers are deliberately NOT in the key, which is only sound because every
# kernel here is compiled `do_not_specialize_on_alignment` on its descriptors and
# pointers -- otherwise each fresh allocation's divisibility class would be a new
# compile.
_TLE_STATS_GEOM = {}
_TLE_AFFINE_GEOM = {}
_TLE_PLAN_MISS = object()
_FLAT_LAUNCHERS = _TLE_PLAN_MISS


def _flat_launchers():
    """`driver.active.flat_launchers`, resolved once -- `driver.active` is a lazy
    proxy and the attribute walk is not free at ~12us a launch."""
    global _FLAT_LAUNCHERS
    if _FLAT_LAUNCHERS is _TLE_PLAN_MISS:
        _FLAT_LAUNCHERS = getattr(driver.active, "flat_launchers", None)
    return _FLAT_LAUNCHERS


def _tle_row_geom(M, N, itemsize, extra_bytes_per_row, cache):
    """`(xblock, yblock, row_blocks)` for a `[M, N]` row tile.

    This is sum.py's `_tle_row_geom` rule verbatim -- aim for one program per
    cluster, let a long row buy YBLOCK and a short one buy the wide result
    write -- with the budget reduced by the per-row vectors the kernel allocates
    beside the tile (mean/rstd on the stats side, scale/shift on the affine
    side). Those are `[XBLOCK]`, small next to `[XBLOCK, YBLOCK]`, but
    `fitBudget` counts every `local_alloc`, so they have to come out of the
    budget or the tile fails to allocate.

    XBLOCK never falls below the 64-row core count. Sub-64 tiles were tried to
    raise the grid on the small-M wide shapes and returned wrong numbers on BOTH
    passes: the stats reduce because `tl.sum(axis=1)` is core-local only at
    XBLOCK >= core_num (below it a row's reduction is split across cores with no
    combine), and the affine pass too (measured: (16, 8, 128, 128) off by 17
    either way). Raising the grid on those shapes needs the two-stage reduce, not
    a smaller tile.

    Cached: sizing a tile is pure arithmetic, but at a ~15us launch budget
    arithmetic is the cost, and the distinct shapes an application normalises are
    bounded.
    """
    key = (M, N, itemsize)
    geom = cache.get(key, _TLE_PLAN_MISS)
    if geom is not _TLE_PLAN_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        xblock = max(_TLE_CORE_NUM, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 256
    # The narrow-row branch above assumes the whole pass is launch and
    # result-write bound, which stops being true once there are many rows -- and
    # the threshold is in BYTES, so 1024 fp16 elements a row lands in it. On
    # (32, 32, 32, 32) fp16 that gave the affine pass xblock 512 over 1024 rows,
    # i.e. grid 2 on a 1M-element tensor, where fp32 got grid 8. Give the grid
    # back as long as halving actually buys a program and xblock stays off 64,
    # where the result write degenerates into per-element transfers.
    while (
        xblock > 128
        and -(-M // xblock) < _TLE_CLUSTERS
        and -(-M // (xblock >> 1)) > -(-M // xblock)
    ):
        xblock >>= 1
    budget = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM
    yblock = max(1, min(_npo2(N), budget // itemsize // xblock))
    while yblock > 1 and xblock * (yblock * itemsize + extra_bytes_per_row) > budget:
        yblock >>= 1
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, -(-M // xblock))
    cache[key] = geom
    return geom


@triton.jit(
    # `L` only bounds the loop and divides the accumulators; `eps` is a value.
    # The descriptors carry freshly allocated pointers, and with alignment on the
    # specialization key every new allocation class is a fresh compile.
    do_not_specialize=["L", "eps"],
    do_not_specialize_on_alignment=["x_desc", "mean_desc", "rstd_desc"],
)
def _tle_group_stats_kernel(
    x_desc,
    mean_desc,
    rstd_desc,
    L,
    eps,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """mean/rstd of each row of a `[N * group, L]` view, L = group_size * HxW.

    Two `[XBLOCK]` fp32 accumulators -- sum and sum of squares -- so one pass
    over the data produces both moments. Keeping them 1-D rather than
    `[XBLOCK, YBLOCK]` is what lets XBLOCK reach 512: a 2-D accumulator forces
    the tile down (sum.py measured 3.08ms against 2.50ms on 1024^3 f32).

    The short last column step is ZERO-FILLED, not masked. `tle.gpu.copy` clamps
    itself to the descriptor, so a partial tile leaves STALE LM bytes behind
    rather than a bad transfer, and squaring stale bytes is what turns them into
    inf. `tl.where` inside the reduction is not an option -- it returns wrong
    numbers in bf16 and blows the LM budget in f16 -- and `tl.load(local_ptr,
    mask=)` has no lowering at all. L is `group_size * HxW`, which for a 3-D
    input like (2, 4, 8) is 16 against a 256-wide tile, so this branch is the
    common case here, not an edge one.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    x_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    mean_lmem = tle.gpu.alloc(
        [XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem
    )
    rstd_lmem = tle.gpu.alloc(
        [XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem
    )

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    x_ptrs = tle.gpu.local_ptr(x_lmem, (row_ids, col_ids))
    stat_ids = tl.arange(0, XBLOCK)
    mean_ptrs = tle.gpu.local_ptr(mean_lmem, (stat_ids,))
    rstd_ptrs = tle.gpu.local_ptr(rstd_lmem, (stat_ids,))

    acc = tl.zeros([XBLOCK], tl.float32)
    acc_sq = tl.zeros([XBLOCK], tl.float32)
    for coff in tl.range(0, L, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > L:  # only the last step can be short
                tl.store(x_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(x_desc, x_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        xv = tl.load(x_ptrs).to(tl.float32)
        acc += tl.sum(xv, axis=1)
        acc_sq += tl.sum(xv * xv, axis=1)

    mean = acc / L
    # E[x^2] - E[x]^2 can land a hair below zero on a near-constant row, and
    # rsqrt of that is nan. aten runs Welford and cannot produce a negative
    # variance at all, so clamp rather than propagate.
    var = tl.maximum(acc_sq / L - mean * mean, 0.0)
    tl.store(mean_ptrs, mean)
    tl.store(rstd_ptrs, rsqrt(var + eps))
    tle.gpu.copy(mean_lmem, mean_desc, [XBLOCK], [row_off])
    tle.gpu.copy(rstd_lmem, rstd_desc, [XBLOCK], [row_off])
    # The input-dtype copy of mean/rstd that aten also returns is NOT written
    # here. Two more `[XBLOCK]` LM buffers beside the tile and the two fp32
    # accumulators overran the budget outright -- "TLE kernel stack is over the
    # local-memory budget after pressure relief and vrf_budget escalation" -- so
    # the cast rides along in the coefficient kernel instead, which reads these
    # same fp32 values anyway and costs nothing extra to have it.


@triton.jit(
    do_not_specialize=["L", "CW", "M"],
    do_not_specialize_on_alignment=["x_desc", "psum_desc", "psumsq_desc"],
)
def _tle_stats_partial_kernel(
    x_desc,
    psum_desc,
    psumsq_desc,
    L,
    CW,
    M,
    ROW_BLOCKS: tl.constexpr,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """Stage 1 of the two-stage stats reduce: partial (sum, sumsq) per column
    chunk of a `[M, L]` row view.

    The program grid is `ROW_BLOCKS * n_chunks`; each program owns one XBLOCK
    band of rows and one CW-wide column chunk, reduces that rectangle
    core-local, and writes the two `[XBLOCK]` partials into row `chunk` of the
    `[n_chunks, M]` partial buffers. Laying the partials out chunk-major keeps
    each write contiguous (`chunk * M + row_off`), and stage 2 then reduces the
    small chunk axis with ordinary GM pointers.

    Same zero-fill rule as the single-stage kernel: CW is a multiple of YBLOCK
    so only the very last chunk's last step can be short, and a stale LM lane
    squared would poison the accumulator.
    """
    pid = tl.program_id(0)
    rblk = pid % ROW_BLOCKS
    cblk = pid // ROW_BLOCKS
    row_off = rblk * XBLOCK
    col_start = cblk * CW
    col_end = tl.minimum(col_start + CW, L)

    x_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    psum_lmem = tle.gpu.alloc(
        [XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem
    )
    psumsq_lmem = tle.gpu.alloc(
        [XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem
    )

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    x_ptrs = tle.gpu.local_ptr(x_lmem, (row_ids, col_ids))
    stat_ids = tl.arange(0, XBLOCK)
    psum_ptrs = tle.gpu.local_ptr(psum_lmem, (stat_ids,))
    psumsq_ptrs = tle.gpu.local_ptr(psumsq_lmem, (stat_ids,))

    acc = tl.zeros([XBLOCK], tl.float32)
    acc_sq = tl.zeros([XBLOCK], tl.float32)
    for coff in tl.range(col_start, col_end, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > L:  # only the last chunk's last step can be short
                tl.store(x_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(x_desc, x_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        xv = tl.load(x_ptrs).to(tl.float32)
        acc += tl.sum(xv, axis=1)
        acc_sq += tl.sum(xv * xv, axis=1)

    tl.store(psum_ptrs, acc)
    tl.store(psumsq_ptrs, acc_sq)
    out_off = cblk * M + row_off
    tle.gpu.copy(psum_lmem, psum_desc, [XBLOCK], [out_off])
    tle.gpu.copy(psumsq_lmem, psumsq_desc, [XBLOCK], [out_off])


@triton.jit(
    do_not_specialize=["M", "L", "eps"],
    do_not_specialize_on_alignment=["Psum", "Psumsq", "Mean", "Rstd"],
)
def _tle_stats_finalize_kernel(
    Psum,
    Psumsq,
    Mean,
    Rstd,
    M,
    L,
    eps,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """Stage 2: fold the `[NCHUNK, M]` partials into mean/rstd.

    NCHUNK is small (<= a dozen) and constexpr, so the chunk loop unrolls; the
    index space is M (a few thousand at most), the same reason the coefficient
    and cast kernels stay on the GM-pointer path rather than a cluster tile.
    """
    idx = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < M
    s = tl.zeros([BLOCK], tl.float32)
    sq = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(NCHUNK):
        s += tl.load(Psum + k * M + idx, mask=m, other=0.0)
        sq += tl.load(Psumsq + k * M + idx, mask=m, other=0.0)
    mean = s / L
    var = tl.maximum(sq / L - mean * mean, 0.0)
    tl.store(Mean + idx, mean, mask=m)
    tl.store(Rstd + idx, rsqrt(var + eps), mask=m)


@triton.jit(
    do_not_specialize=["MC", "C", "group_size", "num_groups"],
    do_not_specialize_on_alignment=["Mean", "Rstd", "W", "B", "Scale", "Shift"],
)
def _group_affine_coeff_kernel(
    Mean,
    Rstd,
    W,
    B,
    Scale,
    Shift,
    MC,
    C,
    group_size,
    num_groups,
    BLOCK: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    """Fold mean/rstd/weight/bias into one per-channel scale and shift.

    `(x - mean) * rstd * w + b` is `x * (rstd * w) + (b - mean * rstd * w)`, so
    the affine pass only needs two numbers per (batch, channel) row and never has
    to gather a stat or a weight itself.

    That indirection is not bookkeeping, it is what keeps the tle kernel alive.
    The obvious shortcut -- drop this kernel and read mean/rstd/weight/bias with
    ordinary GM pointers inside `_tle_group_affine_kernel`, indexing them by the
    row block -- was tried and HUNG THE CARD: 60 of 84 sweep cases failed and
    dmesg showed `kl3_wait_for_noc_idle() timeout` on the device. A cluster tle
    kernel reaching into GM outside its descriptors is not something to retry.
    Here the coefficients arrive as one more contiguous `[XBLOCK]` descriptor
    copy, which is an access the core-tiling pass already understands.

    This kernel itself stays on the GM pointer path on purpose: it touches N * C
    elements, thousands at most, where a tile kernel would be pure launch
    overhead. The masked tile is single-shot and BLOCK-wide, the spelling sum.py's
    constraint list marks reliable.

    `HAS_W` / `HAS_B` are constexpr rather than `W is None` so that the operand
    list has a FIXED LENGTH: a None argument is dropped from the launch signature,
    and the flat launcher replays operands positionally. The caller passes a
    harmless stand-in pointer when there is no weight or bias.
    """
    pid = ext.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < MC
    c = idx % C
    # Row (n, g) of the stats matrix owns channels [g * group_size, ...), so the
    # stat row for channel c of batch n is n * num_groups + c // group_size.
    srow = (idx // C) * num_groups + c // group_size
    mean = tl.load(Mean + srow, mask=m, other=0.0)
    rstd = tl.load(Rstd + srow, mask=m, other=0.0)
    if HAS_W:
        scale = rstd * tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    else:
        scale = rstd
    if HAS_B:
        shift = tl.load(B + c, mask=m, other=0.0).to(tl.float32) - mean * scale
    else:
        shift = -mean * scale
    tl.store(Scale + idx, scale, mask=m)
    tl.store(Shift + idx, shift, mask=m)


@triton.jit(
    do_not_specialize=["M"],
    do_not_specialize_on_alignment=["Mean32", "Rstd32", "MeanOut", "RstdOut"],
)
def _stats_cast_kernel(
    Mean32,
    Rstd32,
    MeanOut,
    RstdOut,
    M,
    BLOCK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Narrow the fp32 stats pair to the input dtype aten also returns.

    An fp16/bf16 group norm has two different mean/rstd results to produce: fp32
    for the coefficient fold, and the input dtype for the caller. This is the
    third place that cast has lived, and the only cheap one:

    * in the stats kernel, two more `[XBLOCK]` LM buffers beside the tile and the
      fp32 accumulators overran the budget -- "TLE kernel stack is over the
      local-memory budget after pressure relief and vrf_budget escalation";
    * in the coefficient kernel, whose index space is N * C rather than N * group,
      it had to be a SCATTER masked to one lane per stat row, and that took the
      kernel from 11us to 118us;
    * here it is two contiguous loads and two contiguous stores over N * group
      elements -- one extra launch, and the only shape it touches is a few
      thousand elements wide.
    """
    idx = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < M
    tl.store(
        MeanOut + idx, tl.load(Mean32 + idx, mask=m, other=0.0).to(OUT_DTYPE), mask=m
    )
    tl.store(
        RstdOut + idx, tl.load(Rstd32 + idx, mask=m, other=0.0).to(OUT_DTYPE), mask=m
    )


@triton.jit(
    do_not_specialize=["HW"],
    do_not_specialize_on_alignment=["x_desc", "y_desc", "scale_desc", "shift_desc"],
)
def _tle_group_affine_kernel(
    x_desc,
    y_desc,
    scale_desc,
    shift_desc,
    HW,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
):
    """`y[r, :] = x[r, :] * scale[r] + shift[r]` over a `[N * C, HxW]` view.

    ONE LM buffer serves both directions: the tile comes in, the affine is
    applied in registers, the result is stored back into the same buffer and
    copied out. That is the shape `_tle_tile_copy_kernel` in utils/tle_copy.py
    already uses for a plain copy, and halving the LM footprint is what buys the
    long YBLOCK a 16K-wide row wants.

    The coefficients are hoisted out of the column loop: they depend on the row
    block only, and a `[XBLOCK]` copy is the expensive kind (at XBLOCK 64 a core
    owns a single element of it), so it is worth paying once per program rather
    than once per tile.

    No zero-fill and no mask here: nothing is reduced, so a short tile only
    computes garbage in lanes the clamped copy-out never writes. The stats kernel
    needs the fill because it squares those lanes into its accumulator.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    buf = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    scale_lmem = tle.gpu.alloc(
        [XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem
    )
    shift_lmem = tle.gpu.alloc(
        [XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem
    )

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    buf_ptrs = tle.gpu.local_ptr(buf, (row_ids, col_ids))
    coeff_ids = tl.arange(0, XBLOCK)
    scale_ptrs = tle.gpu.local_ptr(scale_lmem, (coeff_ids,))
    shift_ptrs = tle.gpu.local_ptr(shift_lmem, (coeff_ids,))

    tle.gpu.copy(scale_desc, scale_lmem, [XBLOCK], [row_off])
    tle.gpu.copy(shift_desc, shift_lmem, [XBLOCK], [row_off])
    scale = tl.load(scale_ptrs)[:, None]
    shift = tl.load(shift_ptrs)[:, None]

    for coff in tl.range(0, HW, YBLOCK):
        tle.gpu.copy(x_desc, buf, [XBLOCK, YBLOCK], [row_off, coff])
        yv = tl.load(buf_ptrs).to(tl.float32) * scale + shift
        tl.store(buf_ptrs, yv.to(IN_DTYPE))
        tle.gpu.copy(buf, y_desc, [XBLOCK, YBLOCK], [row_off, coff])


@triton.jit(
    do_not_specialize=["L", "HW", "eps", "C", "group_size", "num_groups"],
    do_not_specialize_on_alignment=[
        "x_desc",
        "y_desc",
        "W",
        "B",
        "mean_desc",
        "rstd_desc",
    ],
)
def _tle_group_norm_fused_kernel(
    x_desc,
    y_desc,
    W,
    B,
    mean_desc,
    rstd_desc,
    L,
    HW,
    eps,
    C,
    group_size,
    num_groups,
    XBLOCK: tl.constexpr,
    WT: tl.constexpr,
    CBLK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    TAIL: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    """Whole group norm for one tile of group-rows, the tle_layernorm way.

    A program owns XBLOCK contiguous groups; core tiling hands each core whole
    rows, so the reduce over the L = group_size * HxW columns is core-local and
    needs no barrier. mean/rstd are computed and the row is normalized IN PLACE,
    one GM read and one GM write -- half the traffic of the split stats+affine
    path, which reads the input twice.

    The per-channel weight/bias is the one thing layernorm does not have: within
    a group-row, column j belongs to channel `group*group_size + j // HxW`, and
    the group differs per row. Weight/bias are staged into cluster SHARED memory
    (smem, not per-core lmem) and indexed by that computed [XBLOCK, WT] channel
    map with `local_ptr`. The scope matters: an LM `local_ptr` DROPS its index and
    hands back the core's own contiguous slice (that is how the `a` tile is
    addressed), so an LM weight buffer would ignore `ch` and multiply by the
    wrong lane -- measured as y off by 4.8 while mean/rstd stayed exact. Only the
    SM `local_ptr` honours an arbitrary index. Either way it stays off the
    GM-gather path that hangs the cluster.
    """
    pid = tl.program_id(0)
    row0 = pid * XBLOCK
    rid = tl.arange(0, XBLOCK)
    rows = tl.broadcast_to(rid[:, None], (XBLOCK, WT))
    cols = tl.broadcast_to(tl.arange(0, WT)[None, :], (XBLOCK, WT))

    a_lmem = tle.gpu.alloc(
        [XBLOCK, WT], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    a_ptr = tle.gpu.local_ptr(a_lmem, (rows, cols))
    tle.gpu.copy(x_desc, a_lmem, [XBLOCK, WT], [row0, 0])
    x = tl.load(a_ptr).to(tl.float32)
    if TAIL:  # padding columns must not pollute the reduce
        x = tl.where(cols < L, x, 0.0)
    mean = tl.sum(x, 1) / L
    var = tl.maximum(tl.sum(x * x, 1) / L - mean * mean, 0.0)
    rstd = rsqrt(var + eps)
    y = (x - mean[:, None]) * rstd[:, None]
    if HAS_W or HAS_B:
        grp = (row0 + rid) % num_groups
        ch = grp[:, None] * group_size + (cols // HW)
        if TAIL:
            # WT is the npo2 padding of L, and the gather has no mask: a
            # padding column j >= L computes ch = grp*group_size + j//HW one
            # PAST the [CBLK] smem buffer (e.g. (16,16,8,48): CBLK 16, padding
            # j//48 = 2 -> ch 16) and the out-of-bounds SM read faults the
            # card. Valid columns always have ch <= C-1 <= CBLK-1, so clamping
            # to CBLK-1 only redirects the discarded padding lanes.
            ch = tl.minimum(ch, CBLK - 1)
        if HAS_W:
            w_smem = tle.gpu.alloc(
                [CBLK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.smem
            )
            tle.gpu.copy(W, w_smem, [C], [0])
            y = y * tl.load(tle.gpu.local_ptr(w_smem, (ch,))).to(tl.float32)
        if HAS_B:
            b_smem = tle.gpu.alloc(
                [CBLK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.smem
            )
            tle.gpu.copy(B, b_smem, [C], [0])
            y = y + tl.load(tle.gpu.local_ptr(b_smem, (ch,))).to(tl.float32)
    tl.store(a_ptr, y.to(IN_DTYPE))
    tle.gpu.copy(a_lmem, y_desc, [XBLOCK, WT], [row0, 0])

    m_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)
    r_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)
    tl.store(tle.gpu.local_ptr(m_lmem, (rid,)), mean.to(OUT_DTYPE))
    tl.store(tle.gpu.local_ptr(r_lmem, (rid,)), rstd.to(OUT_DTYPE))
    tle.gpu.copy(m_lmem, mean_desc, [XBLOCK], [row0])
    tle.gpu.copy(r_lmem, rstd_desc, [XBLOCK], [row0])


def _tle_group_norm_fused(
    input,
    y,
    weight,
    bias,
    mean,
    rstd,
    m_grp,
    L,
    HxW,
    C,
    group_size,
    group,
    eps,
    tl_dtype,
):
    """One-kernel group norm; returns False if the group-row is too wide to fit.

    WT is the padded (power-of-two) row width `tl.arange` needs; the whole row
    must live in LM, so `epc = XBLOCK * WT / 64 <= 256` -- with XBLOCK 1 that caps
    WT at 16384. Wider groups have no fused config and take the split path.
    """
    WT = _npo2(L)
    if WT > _TLE_FUSE_WMAX:
        return False
    xblock = 1
    while xblock * 2 <= m_grp and (xblock * 2) * WT // _TLE_CORE_NUM <= _TLE_FUSE_EPC:
        xblock *= 2
    # Give the grid back toward one program per cluster when rows are plentiful.
    while (
        xblock > 1
        and -(-m_grp // xblock) < _TLE_CLUSTERS
        and -(-m_grp // (xblock >> 1)) > -(-m_grp // xblock)
    ):
        xblock >>= 1
    # The fused kernel needs `tritonxpu-tle-core-tiling` to ENGAGE on it: the
    # SM local_ptr weight/bias gather is only correct when every tensor's
    # ClusterLayout is retiled consistently. With xblock == 1 the only 2D tile
    # is [1, WT] -- a broadcast-source row vector the anchor scan rejects by
    # design -- so the pass no-ops, the kernel stays on the DEFAULT layout,
    # and the gather/store slot counts disagree with the copy drain plan
    # (the sm_gather repro measured that as garbage and an illegal access at
    # L=8196). Wide rows that force xblock to 1 keep the split path.
    if xblock < 2:
        return False
    # Fusion is a per-shape win, not a blanket one (A/B, min over rounds): the
    # SM weight/bias gather is a scalar per-register chain, so on wide rows it
    # costs more than the second input pass it saves. Measured on card 7:
    # fused WINS (4,16,64,4) 2.0-2.3x, (16,16,64) 0.9x, ties (1,8,4,4) and
    # (16,16,128); LOSES (16,16,8,48) 0.5x, (16,16,8,88) 0.4x, (16,16,1024)
    # 0.4-0.6x and (32,32,32,32) 0.2x. The winning corner is an unpadded
    # narrow row with few groups; route everything else to the split path.
    if WT != L or WT > _TLE_FUSE_ROUTE_WMAX or m_grp > _TLE_FUSE_ROUTE_MGRP:
        return False
    grid = (-(-m_grp // xblock),)

    has_w = weight is not None
    has_b = bias is not None
    dummy = (
        None
        if has_w and has_b
        else torch.empty(C, dtype=input.dtype, device=input.device)
    )
    w_t = weight if has_w else dummy
    b_t = bias if has_b else dummy

    _tle_group_norm_fused_kernel[grid](
        TensorDescriptor.from_tensor(input.view(m_grp, L), [xblock, WT]),
        TensorDescriptor.from_tensor(y.view(m_grp, L), [xblock, WT]),
        TensorDescriptor.from_tensor(w_t, [C]),
        TensorDescriptor.from_tensor(b_t, [C]),
        TensorDescriptor.from_tensor(mean.view(m_grp), [xblock]),
        TensorDescriptor.from_tensor(rstd.view(m_grp), [xblock]),
        L,
        HxW,
        eps,
        C,
        group_size,
        group,
        xblock,
        WT,
        _npo2(C),
        tl_dtype,
        tl_dtype,
        WT != L,
        has_w,
        has_b,
        isCloseCoreTiling=False,
    )
    logger.debug(
        "GEMS_KUNLUNXIN NATIVE_GROUP_NORM tle FUSED m_grp=%d L=%d tile=%dx%d grid=%d",
        m_grp,
        L,
        xblock,
        WT,
        grid[0],
    )
    return True


def _tle_stats2_geom(m_stat, lrow, itemsize):
    """`(xblock, yblock, row_blocks, chunk_width, n_chunks)` for the two-stage
    stats reduce, or None when splitting the row would not fill idle clusters.

    Reuse the single-stage tile sizing (`_tle_row_geom` on the full row) so the
    partial kernel moves the same [XBLOCK, YBLOCK] tiles, then split the columns
    into `n_chunks` so that `row_blocks * n_chunks` reaches the cluster count.
    The chunk width is rounded UP to a YBLOCK multiple so every chunk boundary
    lands on a tile edge and only the last chunk can be short.
    """
    key = (m_stat, lrow, itemsize)
    geom = _TLE_STATS2_GEOM.get(key, _TLE_PLAN_MISS)
    if geom is not _TLE_PLAN_MISS:
        return geom
    xblock, yblock, row_blocks = _tle_row_geom(
        m_stat, lrow, itemsize, 8, _TLE_STATS_GEOM
    )
    # Split only when the single-stage reduce collapses to ONE cluster. With
    # row_blocks >= 2 the rows already spread across clusters, and then the
    # partial buffer write plus the finalize launch cost more than the split
    # saves -- measured on (16, 16, 4098) fp32 (row_blocks 2): 0.95x, a
    # regression -- while its fp16 sibling (row_blocks 1) gains 1.06x.
    if row_blocks > 1:
        geom = None
        _TLE_STATS2_GEOM[key] = geom
        return geom
    n_chunks = _TLE_CLUSTERS
    if n_chunks <= 1:  # single-stage already spreads across the clusters
        geom = None
        _TLE_STATS2_GEOM[key] = geom
        return geom
    # Chunk width on a YBLOCK grid; recompute n_chunks from it so the last chunk
    # is the only short one and the grid matches the buffer layout exactly.
    chunk_target = -(-lrow // n_chunks)
    cw = max(yblock, -(-chunk_target // yblock) * yblock)
    n_chunks = -(-lrow // cw)
    if n_chunks <= 1:
        geom = None
    else:
        geom = (xblock, yblock, row_blocks, cw, n_chunks)
    _TLE_STATS2_GEOM[key] = geom
    return geom


def _launch_stats_2stage(x, mean_f32, rstd_f32, m_stat, lrow, eps, geom, tl_dtype):
    """Two-stage stats: partial (sum, sumsq) per column chunk, then finalize.

    Not put behind the flat-launcher cache: it allocates two `[n_chunks *
    m_stat]` scratch buffers, so the operand pointers change every call and the
    partial buffers must be sized per shape anyway. The launch it replaces was
    grid-starved (one cluster), so its own launch overhead is the smaller cost.
    """
    xblock, yblock, row_blocks, cw, n_chunks = geom
    psum = torch.empty(n_chunks * m_stat, dtype=torch.float32, device=x.device)
    psumsq = torch.empty(n_chunks * m_stat, dtype=torch.float32, device=x.device)
    grid1 = (row_blocks * n_chunks,)
    _tle_stats_partial_kernel[grid1](
        TensorDescriptor.from_tensor(x.view(m_stat, lrow), [xblock, yblock]),
        TensorDescriptor.from_tensor(psum, [xblock]),
        TensorDescriptor.from_tensor(psumsq, [xblock]),
        lrow,
        cw,
        m_stat,
        row_blocks,
        xblock,
        yblock,
        tl_dtype,
        lrow % yblock != 0,
    )
    block = _coeff_block(m_stat)
    grid2 = (-(-m_stat // block),)
    _tle_stats_finalize_kernel[grid2](
        psum, psumsq, mean_f32, rstd_f32, m_stat, lrow, eps, block, n_chunks
    )


def _launch_stats(x, mean_f32, rstd_f32, m_stat, lrow, eps, geom, tl_dtype):
    """Bind `_tle_group_stats_kernel` once, then replay it flat."""
    sx, sy, sblocks = geom
    consts = (sx, sy, tl_dtype, lrow % sy != 0)
    launchers = _flat_launchers()

    def _bind():
        return _tle_group_stats_kernel[(sblocks,)](
            TensorDescriptor.from_tensor(x.view(m_stat, lrow), [sx, sy]),
            TensorDescriptor.from_tensor(mean_f32.view(m_stat), [sx]),
            TensorDescriptor.from_tensor(rstd_f32.view(m_stat), [sx]),
            lrow,
            eps,
            *consts,
        )

    if launchers is None:  # triton without the launcher cache: correct, slower
        _bind()
        return
    key = (m_stat, lrow, sblocks) + consts
    launch, stream = launchers.acquire(_tle_group_stats_kernel, key)
    if launch is None:
        launchers.bind(_tle_group_stats_kernel, key, _bind(), (sblocks,))
        return
    launch(
        stream,
        x.data_ptr(),
        m_stat,
        lrow,
        lrow,
        1,
        mean_f32.data_ptr(),
        m_stat,
        1,
        rstd_f32.data_ptr(),
        m_stat,
        1,
        lrow,
        eps,
    )


def _launch_coeff(
    mean_f32,
    rstd_f32,
    weight,
    bias,
    scale,
    shift,
    m_chan,
    C,
    group_size,
    group,
):
    """Bind `_group_affine_coeff_kernel` once, then replay it flat.

    `weight` / `bias` absent is a compile-time fact (HAS_W / HAS_B), so the
    stand-in pointer keeps the operand list the same length either way.
    """
    block = _coeff_block(m_chan)
    grid = (-(-m_chan // block),)
    has_w = weight is not None
    has_b = bias is not None
    w = weight if has_w else mean_f32
    b = bias if has_b else mean_f32
    consts = (block, has_w, has_b)
    launchers = _flat_launchers()

    def _bind():
        return _group_affine_coeff_kernel[grid](
            mean_f32,
            rstd_f32,
            w,
            b,
            scale,
            shift,
            m_chan,
            C,
            group_size,
            group,
            *consts,
        )

    if launchers is None:
        _bind()
        return
    # w.dtype / b.dtype belong in the key: mean/rstd/scale/shift are always fp32,
    # but weight and bias follow the INPUT dtype, and the compiled kernel loads
    # them at that width. Leaving them out let an fp32-compiled kernel be replayed
    # for an fp16 weight -- `y` came back off by up to 43 on 16 of 84 sweep cases
    # while mean/rstd stayed exact, since those never touch W or B.
    key = (m_chan, C, group_size, group, grid[0], w.dtype, b.dtype) + consts
    launch, stream = launchers.acquire(_group_affine_coeff_kernel, key)
    if launch is None:
        launchers.bind(_group_affine_coeff_kernel, key, _bind(), grid)
        return
    launch(
        stream,
        mean_f32.data_ptr(),
        rstd_f32.data_ptr(),
        w.data_ptr(),
        b.data_ptr(),
        scale.data_ptr(),
        shift.data_ptr(),
        m_chan,
        C,
        group_size,
        group,
    )


def _launch_stats_cast(mean_f32, rstd_f32, mean, rstd, m_stat, tl_dtype):
    """Bind `_stats_cast_kernel` once, then replay it flat."""
    block = _coeff_block(m_stat)
    grid = (-(-m_stat // block),)
    consts = (block, tl_dtype)
    launchers = _flat_launchers()

    def _bind():
        return _stats_cast_kernel[grid](mean_f32, rstd_f32, mean, rstd, m_stat, *consts)

    if launchers is None:
        _bind()
        return
    key = (m_stat, grid[0]) + consts
    launch, stream = launchers.acquire(_stats_cast_kernel, key)
    if launch is None:
        launchers.bind(_stats_cast_kernel, key, _bind(), grid)
        return
    launch(
        stream,
        mean_f32.data_ptr(),
        rstd_f32.data_ptr(),
        mean.data_ptr(),
        rstd.data_ptr(),
        m_stat,
    )


def _launch_affine(x, y, scale, shift, m_chan, HxW, geom, tl_dtype):
    """Bind `_tle_group_affine_kernel` once, then replay it flat."""
    ax, ay, ablocks = geom
    consts = (ax, ay, tl_dtype)
    launchers = _flat_launchers()

    def _bind():
        return _tle_group_affine_kernel[(ablocks,)](
            TensorDescriptor.from_tensor(x.view(m_chan, HxW), [ax, ay]),
            TensorDescriptor.from_tensor(y.view(m_chan, HxW), [ax, ay]),
            TensorDescriptor.from_tensor(scale, [ax]),
            TensorDescriptor.from_tensor(shift, [ax]),
            HxW,
            *consts,
        )

    if launchers is None:
        _bind()
        return
    key = (m_chan, HxW, ablocks) + consts
    launch, stream = launchers.acquire(_tle_group_affine_kernel, key)
    if launch is None:
        launchers.bind(_tle_group_affine_kernel, key, _bind(), (ablocks,))
        return
    launch(
        stream,
        x.data_ptr(),
        m_chan,
        HxW,
        HxW,
        1,
        y.data_ptr(),
        m_chan,
        HxW,
        HxW,
        1,
        scale.data_ptr(),
        m_chan,
        1,
        shift.data_ptr(),
        m_chan,
        1,
        HxW,
    )


def _tle_native_group_norm(
    input, y, weight, bias, mean, rstd, N, C, HxW, group, group_size, eps
):
    """Run group norm on tle; False (having touched nothing) if it does not fit.

    Two paths. When the group-row (group_size * HxW) fits LM, the FUSED kernel
    does the whole op in one launch, tle_layernorm-style: one core owns a
    group-row, reduces it core-local (no barrier), normalizes it in place -- one
    GM read, one GM write. When the row is too wide for LM (e.g. 32768 on
    (16, 8, 128, 128)) it cannot be resident, so we fall back to the split path:
    stats reduce (looped over the row) -> coefficient fold -> affine, three
    launches and the input read twice. When that stats reduce would run on a
    single cluster (few groups, long row), it is split into a two-stage
    partial+finalize reduce so the idle clusters share the column sweep.

    The earlier belief that a single kernel needed a cluster barrier (like the
    hand-written XDNN kernel's `sum_sm; sync_all(); groupsum_sm`) was wrong: a
    barrier is only needed to split ONE row across cores. Giving each core a whole
    group-row -- exactly what tle_layernorm does for a normalized instance --
    needs none. The barrier would only buy the wide-row tail, which is why that
    path stays on the split fallback rather than a compiler change.
    """
    if not _TLE_AVAILABLE:
        return False
    if input.dtype not in _TLE_TL_DTYPE:
        return False
    # `channel = stat_row * group_size` only holds when the groups tile C
    # exactly; a ragged last group would also make the (N * group, L) view lie
    # about the element count.
    if group * group_size != C:
        return False
    # A descriptor needs the last stride to be 1, and torch is free to give an
    # extent-1 dimension any stride, so both views need a real inner extent.
    if HxW < 2 or group_size * HxW < 2:
        return False
    if input.numel() < _TLE_MIN_NUMEL:
        return False

    tl_dtype = _TLE_TL_DTYPE[input.dtype]
    itemsize = input.element_size()
    m_stat = N * group
    lrow = group_size * HxW
    m_chan = N * C

    with torch_device_fn.device(input.device):
        if _TLE_FUSE_ENABLED and _tle_group_norm_fused(
            input,
            y,
            weight,
            bias,
            mean,
            rstd,
            m_stat,
            lrow,
            HxW,
            C,
            group_size,
            group,
            eps,
            tl_dtype,
        ):
            return True

    # Wide-group fallback: the row does not fit LM, so reduce it looped, fold the
    # coefficients, and apply them in a second read of the input.
    # The coefficient kernel reloads mean/rstd, so they have to be full
    # precision: groupnorm.py keeps the same fp32 scratch pair for the same
    # reason -- reloading an fp16 mean loses enough bits to fail
    # gems_assert_close. For fp32 input the scratch IS the output, so there is
    # neither an extra allocation nor a second write.
    if input.dtype is torch.float32:
        mean_f32, rstd_f32 = mean.view(m_stat), rstd.view(m_stat)
        narrow = False
    else:
        mean_f32 = torch.empty(m_stat, dtype=torch.float32, device=input.device)
        rstd_f32 = torch.empty(m_stat, dtype=torch.float32, device=input.device)
        narrow = True

    scale = torch.empty(m_chan, dtype=torch.float32, device=input.device)
    shift = torch.empty(m_chan, dtype=torch.float32, device=input.device)

    # mean/rstd are two fp32 result writes beside the tile; scale/shift are two
    # fp32 reads. Eight bytes a row either way.
    sgeom = _tle_row_geom(m_stat, lrow, itemsize, 8, _TLE_STATS_GEOM)
    ageom = _tle_row_geom(m_chan, HxW, itemsize, 8, _TLE_AFFINE_GEOM)

    # A wide row with few groups collapses the single-stage stats grid to one or
    # two clusters; split its column reduce so the other clusters do work too.
    s2geom = None
    if _TLE_STATS2_ENABLED and lrow >= _TLE_STATS2_MIN_LROW:
        s2geom = _tle_stats2_geom(m_stat, lrow, itemsize)

    with torch_device_fn.device(input.device):
        if s2geom is not None:
            _launch_stats_2stage(
                input, mean_f32, rstd_f32, m_stat, lrow, eps, s2geom, tl_dtype
            )
        else:
            _launch_stats(input, mean_f32, rstd_f32, m_stat, lrow, eps, sgeom, tl_dtype)
        _launch_coeff(
            mean_f32, rstd_f32, weight, bias, scale, shift, m_chan, C, group_size, group
        )
        _launch_affine(input, y, scale, shift, m_chan, HxW, ageom, tl_dtype)
        if narrow:
            _launch_stats_cast(
                mean_f32,
                rstd_f32,
                mean.view(m_stat),
                rstd.view(m_stat),
                m_stat,
                tl_dtype,
            )
    logger.debug(
        "GEMS_KUNLUNXIN NATIVE_GROUP_NORM tle stats=%dx%d %s, "
        "affine=%dx%d tile=%dx%d grid=%d",
        m_stat,
        lrow,
        (
            "2stage tile=%dx%d chunks=%d grid=%d"
            % (s2geom[0], s2geom[1], s2geom[4], s2geom[2] * s2geom[4])
            if s2geom is not None
            else "tile=%dx%d grid=%d" % (sgeom[0], sgeom[1], sgeom[2])
        ),
        m_chan,
        HxW,
        ageom[0],
        ageom[1],
        ageom[2],
    )
    return True


@libentry()
@triton.jit(do_not_specialize=["eps"])
def native_group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    HW,
    num_groups,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    base = pid * num_elements
    ch_base = group * group_size

    sum_acc = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK_HW_SIZE):
        idx = off + tl.arange(0, BLOCK_HW_SIZE)
        m = idx < num_elements
        x = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32)
        sum_acc += x
        sumsq_acc += x * x

    mean = tl.sum(sum_acc) / num_elements
    var = tl.sum(sumsq_acc) / num_elements - mean * mean
    rstd = rsqrt(var + eps)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    for c in range(0, GROUP_SIZE):
        cbase = base + c * HW
        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + ch_base + c).to(tl.float32)
        if B is None:
            bias = 0.0
        else:
            bias = tl.load(B + ch_base + c).to(tl.float32)
        for off in range(0, HW, BLOCK_HW_SIZE):
            idx = off + tl.arange(0, BLOCK_HW_SIZE)
            m = idx < HW
            x = tl.load(X + cbase + idx, mask=m, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * weight + bias
            tl.store(Y + cbase + idx, y, mask=m)


def native_group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    """aten::native_group_norm on Kunlunxin.

    The generic flag_gems.ops.native_group_norm binds
    flag_gems.ops.groupnorm.group_norm at import time, so SpecOpRegistrar
    swapping flag_gems.group_norm never reached it and native_group_norm kept
    running the generic single-kernel giant-2D-tile implementation on XPU.
    That path miscompiles on the small tiles used by the accuracy matrix and
    hard-fails with `out of resource: uni_sram` for HxW >= 4096, so bind a
    vendor kernel here explicitly.

    Two implementations live below it: the tle.gpu pair (cluster DMA, three
    launches) for anything big enough to pay for the launches, and the fused
    GM-pointer kernel for everything smaller or outside tle's envelope.
    """
    # The test asserts on the GENERIC spelling ("GEMS NATIVE_GROUP_NORM"), which
    # "GEMS_KUNLUNXIN NATIVE_GROUP_NORM" does not contain -- that mismatch alone
    # failed all 18 accuracy cases at tests/test_group_norm.py:76, before any
    # number was compared. linalg_svd.py carries the vendor tag this way.
    logger.debug("GEMS NATIVE_GROUP_NORM (kunlunxin)")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    if _tle_native_group_norm(
        input, y, weight, bias, mean, rstd, N, C, HxW, group, group_size, eps
    ):
        return y, mean, rstd

    grid = (N * group,)
    num_elements = group_size * HxW
    block_hw = min(8192, max(1024, triton.next_power_of_2(num_elements)))
    with torch_device_fn.device(input.device):
        native_group_norm_kernel[grid](
            input,
            y,
            weight,
            bias,
            mean,
            rstd,
            group_size,
            HxW,
            group,
            eps,
            GROUP_SIZE=group_size,
            BLOCK_HW_SIZE=block_hw,
        )
    return y, mean, rstd
