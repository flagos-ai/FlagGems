import logging

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

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
# Program count the give-back loop aims for (one program per cluster).
_TLE_CLUSTERS = 8
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
# sm_gather_task/HANDOFF.md for the chain.
# Per-core element ceiling for the fused kernel's resident tile. Measured: at
# 512 elements/core the (16,16,1024) / (32,32,32,32) configs stopped compiling
# ("over the local-memory budget after pressure relief and vrf_budget
# escalation"), so 256 is the empirical cap rather than a derived budget.
_TLE_FUSE_EPC = 256
# WIDE segment budget: tile + two accumulator copies share the registers.
_TLE_FUSE_EPC_WIDE = 64
# Row-width ceiling. The wide-row segment loop has no residency limit, so this
# only guards against absurd widths (a mis-shaped view) rather than tuning.
_TLE_FUSE_WMAX = 1 << 16
# Wide-row two-pass loop threshold and segment width (elements). Above the
# threshold the m==1 fused kernel streams [1, RB] segments instead of holding
# the row; RB/64 is the per-core register footprint, so RB = 4096 -> 64 lanes.
_TLE_FUSE_LOOP_WMIN = 8192
_TLE_FUSE_LOOP_RB = 2048
# Row-block pipelining (hold-the-row path only). A program takes RITER row
# blocks instead of one and double-buffers them, so block i+1's GM read is in
# flight while block i is normalized -- the same DMA/compute overlap the
# hand-written XDNN kernel gets from its triple-buffered affine pass, but across
# ROW BLOCKS rather than column segments, so the input is still read exactly
# once (a column-segment two-pass has to re-read it).
# The lever it buys is per-core RESIDENCY: RITER row blocks at XBLOCK rows each
# do the work one block of XBLOCK*RITER rows would, at 1/RITER of the register
# footprint, without changing the program count. _TLE_FUSE_EPC_PIPE is the
# per-core element count the split aims for. It is a FLOOR, not a target, and
# measured: 64 gives mean 0.68x of aten on the core benchmark set, 32 gives
# 0.54x -- below 64 elements/core the tiles stop paying for their own loop
# (the (16,16,128) fp16 config blew up 40x at 32), so the halving stops there.
_TLE_FUSE_EPC_PIPE = 64
_TLE_FUSE_RITER_MAX = 8

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


def _npo2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


@triton.jit(
    do_not_specialize=["L", "eps"],
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
    HW: tl.constexpr,
    eps,
    C: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
    XBLOCK: tl.constexpr,
    WT: tl.constexpr,
    CBLK: tl.constexpr,
    RB: tl.constexpr,
    WIDE: tl.constexpr,
    RITER: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    CDT: tl.constexpr,
    TAIL: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    SM_STATS: tl.constexpr,
):
    """Whole group norm for one tile of group-rows, the tle_layernorm way.

    A program owns XBLOCK contiguous groups. With XBLOCK >= 2 the rows
    core-tile whole and the reduce is core-local (no barrier); with XBLOCK == 1
    the [1, WT] tile is LargeN -- the row splits across ALL 64 cores and the
    reduce goes cross-core through the standard smem+barrier path, which is what
    lets the widest rows (group_size * HxW = 32768) fuse at all. mean/rstd are
    computed and the row is normalized, one kernel, one launch.

    WIDE rows (WT >= the loop threshold) hold too many per-core columns to sit
    in registers at once ("kernel stack over budget / vrf escalation"), so they
    take a TWO-PASS SEGMENT LOOP inside the same kernel: reduce segment by
    segment, then normalize segment by segment, over one [XBLOCK, RB] tile. The
    input is read from GM twice, but the launch count stays ONE and XBLOCK stays
    free to grow -- which is what keeps the program count down (a per-row
    program grid on a wide shape queues 16 waves on 8 clusters).

    The per-channel weight/bias is the one thing layernorm does not have: within
    a group-row, column j belongs to channel `group*group_size + j // HxW`, and
    the group differs per row. Weight/bias are staged into cluster SHARED memory
    (smem, not per-core lmem) and indexed by that computed channel map with
    `local_ptr`. The scope matters: an LM `local_ptr` DROPS its index and hands
    back the core's own contiguous slice (that is how the `a` tile is
    addressed), so an LM weight buffer would ignore `ch` and multiply by the
    wrong lane -- measured as y off by 4.8 while mean/rstd stayed exact. Only
    the SM `local_ptr` honours an arbitrary index. Either way it stays off the
    GM-gather path that hangs the cluster.
    """
    pid = tl.program_id(0)
    row0 = pid * XBLOCK
    rid = tl.arange(0, XBLOCK)
    rows = tl.broadcast_to(rid[:, None], (XBLOCK, WT))
    cols = tl.broadcast_to(tl.arange(0, WT)[None, :], (XBLOCK, WT))

    if WIDE:
        # ---- two-pass segment loop over one [XBLOCK, RB] tile ----
        # The residency here is the SEGMENT, not the row: [XBLOCK, RB]/64
        # elements per core. That is what lets XBLOCK grow on wide rows even
        # though the row cannot be held -- and XBLOCK is what keeps the program
        # count (and with it the wave count on 8 clusters) down. The tile
        # [XBLOCK, RB] core-tiles as LargeN when XBLOCK < 64, so each row's
        # segment reduce is cross-core across RB's column cores.
        srows = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, RB))
        scols = tl.broadcast_to(tl.arange(0, RB)[None, :], (XBLOCK, RB))
        buf = tle.gpu.alloc(
            [XBLOCK, RB], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
        )
        buf_ptr = tle.gpu.local_ptr(buf, (srows, scols))

        # Deferred merge: accumulate the segments ELEMENT-WISE and run the
        # cross-core reduce ONCE, after the loop, instead of once per segment.
        # A per-segment `tl.sum` has to merge across the row's column cores
        # every iteration (smem write + cluster barrier + read back), and that
        # cost is what made the wide rows 5-10x slower than the split path.
        # The 2-D accumulators cost registers instead -- [XBLOCK, RB] each,
        # which is what the WIDE segment budget is sized for.
        acc = tl.zeros([XBLOCK, RB], tl.float32)
        acc_sq = tl.zeros([XBLOCK, RB], tl.float32)
        for coff in tl.range(0, WT, RB):
            # Only a segment that runs past L can be short, and only then is the
            # buffer worth clearing -- every other iteration overwrites it whole.
            # Zero-fill rather than `tl.where` on the loaded value: sum.py:217
            # records that the mask "returns wrong numbers in bf16 and blows the
            # LM budget in f16 at every tile size", and it also costs a live
            # [XBLOCK, RB] f32 beside two f32 accumulators that are already what
            # caps this leg (_TLE_FUSE_EPC_WIDE = 64, a third of the row budget).
            if TAIL:
                if coff + RB > L:
                    tl.store(buf_ptr, tl.zeros([XBLOCK, RB], IN_DTYPE))
            tle.gpu.copy(x_desc, buf, [XBLOCK, RB], [row0, coff])
            xv = tl.load(buf_ptr).to(tl.float32)
            acc += xv
            acc_sq += xv * xv
        mean = tl.sum(acc, 1) / L
        var = tl.maximum(tl.sum(acc_sq, 1) / L - mean * mean, 0.0)
        rstd = rsqrt(var + eps)

        if HAS_W or HAS_B:
            grp = (row0 + rid) % num_groups
            if HAS_W:
                w_smem = tle.gpu.alloc(
                    [CBLK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.smem
                )
                tle.gpu.copy(W, w_smem, [C], [0])
            if HAS_B:
                b_smem = tle.gpu.alloc(
                    [CBLK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.smem
                )
                tle.gpu.copy(B, b_smem, [C], [0])
        for coff in tl.range(0, WT, RB):
            tle.gpu.copy(x_desc, buf, [XBLOCK, RB], [row0, coff])
            xv = tl.load(buf_ptr).to(CDT)
            yv = (xv - mean[:, None].to(CDT)) * rstd[:, None].to(CDT)
            if HAS_W or HAS_B:
                ch = grp[:, None] * group_size + ((coff + scols) // HW)
                if TAIL:
                    ch = tl.minimum(ch, CBLK - 1)
                if HAS_W:
                    yv = yv * tl.load(tle.gpu.local_ptr(w_smem, (ch,))).to(CDT)
                if HAS_B:
                    yv = yv + tl.load(tle.gpu.local_ptr(b_smem, (ch,))).to(CDT)
            tl.store(buf_ptr, yv.to(IN_DTYPE))
            tle.gpu.copy(buf, y_desc, [XBLOCK, RB], [row0, coff])
        # mean/rstd leave through CLUSTER-SHARED SM, not a per-core LM strip. The
        # LM route makes all 64 cores issue a sub-cache-line s_lm2gm into the same
        # GM line and they serialise on it: measured +9.6us for a [16] f32 row and
        # +22us for f16, against +0.1-0.8us once one core sends the whole row
        # (sm_gather_task/strip_store_cost.py). SM honours a local_ptr index, so
        # every core can drop its lane in, and copy_l2g out of an smem buffer
        # coalesces the writeback into >= 64-byte chunks. SM_STATS is off for
        # bf16: XPU3 cannot select a bf16 store into SM (the mirror of the bf16
        # SM load gap), and unlike the load side an f16 container does not help
        # -- InstCombine in the O3 pass before llc folds the bitcast away -- so
        # that dtype stays on the LM strip until the XPU LLVM backend grows the
        # pattern.
        m_lmem = tle.gpu.alloc(
            [XBLOCK],
            dtype=OUT_DTYPE,
            layout=None,
            scope=tle.gpu.smem if SM_STATS else tle.gpu.lmem,
        )
        r_lmem = tle.gpu.alloc(
            [XBLOCK],
            dtype=OUT_DTYPE,
            layout=None,
            scope=tle.gpu.smem if SM_STATS else tle.gpu.lmem,
        )
        tl.store(tle.gpu.local_ptr(m_lmem, (rid,)), mean.to(OUT_DTYPE))
        tl.store(tle.gpu.local_ptr(r_lmem, (rid,)), rstd.to(OUT_DTYPE))
        tle.gpu.copy(m_lmem, mean_desc, [XBLOCK], [row0])
        tle.gpu.copy(r_lmem, rstd_desc, [XBLOCK], [row0])
    else:
        # Weight/bias staging is common to both legs below and must happen ONCE,
        # before any pipelined prefetch: the smem copy carries its own
        # mfence(7)+barrier pair (the scope ignores `sync=`), and that fence
        # drains any in-flight DMA -- staging after a prefetch would cancel the
        # overlap it was issued for. Both buffers are always allocated so the
        # block helper can take them unconditionally; only the present ones are
        # filled (the launcher hands a dummy tensor for the absent one).
        w_smem = tle.gpu.alloc([CBLK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.smem)
        b_smem = tle.gpu.alloc([CBLK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.smem)
        if HAS_W:
            tle.gpu.copy(W, w_smem, [C], [0])
        if HAS_B:
            tle.gpu.copy(B, b_smem, [C], [0])
        if RITER > 1:
            # ---- double-buffered row blocks -------------------------------
            # One program takes RITER blocks of XBLOCK rows and runs them two
            # per beat against two alternating tiles, so block i+1's GM read is
            # in flight while block i is normalized and block i-1 is written
            # back. Three things here are forced rather than chosen:
            #   * the 2x unroll is MANDATORY -- a memdesc handle cannot be
            #     selected by a runtime `it % 2`, so the two buffers have to be
            #     two allocs and the loop two half beats;
            #   * `dma_wait()` is a global, non-counting fence, so it goes AFTER
            #     the compute it is meant to overlap; fencing right after the
            #     prefetch re-serializes the loop;
            #   * only the row tiles are double-buffered. The [XBLOCK] mean/rstd
            #     strips are per-beat (m0/m1) because their copy-out is in
            #     flight across the beat boundary too.
            base = pid * (XBLOCK * RITER)
            PAIRS: tl.constexpr = RITER // 2
            LASTROW: tl.constexpr = (RITER - 1) * XBLOCK
            a0 = tle.gpu.alloc(
                [XBLOCK, WT], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            a1 = tle.gpu.alloc(
                [XBLOCK, WT], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            m0 = tle.gpu.alloc(
                [XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            r0 = tle.gpu.alloc(
                [XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            m1 = tle.gpu.alloc(
                [XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            r1 = tle.gpu.alloc(
                [XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            a0_ptr = tle.gpu.local_ptr(a0, (rows, cols))
            a1_ptr = tle.gpu.local_ptr(a1, (rows, cols))
            m0_ptr = tle.gpu.local_ptr(m0, (rid,))
            r0_ptr = tle.gpu.local_ptr(r0, (rid,))
            m1_ptr = tle.gpu.local_ptr(m1, (rid,))
            r1_ptr = tle.gpu.local_ptr(r1, (rid,))
            # The pad is re-zeroed before EVERY refill, not once: the tiles are
            # reused each beat and normalized in place, so the previous beat's y
            # is sitting in those columns. Each zero sits immediately before the
            # async copy that refills that buffer -- the DMA lowering fences
            # before its transfer, so the store is visible, and the preceding
            # `dma_wait` has already retired that buffer's write-back.
            if TAIL:
                tl.store(a0_ptr, tl.zeros([XBLOCK, WT], IN_DTYPE))
            tle.gpu.copy(x_desc, a0, [XBLOCK, WT], [base, 0], sync=False)
            tle.gpu.dma_wait()
            for it in range(0, PAIRS):
                b0 = base + (2 * it) * XBLOCK
                b1 = b0 + XBLOCK
                # The last prefetch runs past the program's blocks; clamp it
                # onto the final block (one redundant read, no wrong result).
                b2 = base + tl.minimum((2 * it + 2) * XBLOCK, LASTROW)
                # ---- half beat A: prefetch b1 -> a1, compute a0 (block b0) --
                if TAIL:
                    tl.store(a1_ptr, tl.zeros([XBLOCK, WT], IN_DTYPE))
                tle.gpu.copy(x_desc, a1, [XBLOCK, WT], [b1, 0], sync=False)
                ch0 = ((b0 + rid) % num_groups)[:, None] * group_size + cols // HW
                if TAIL:
                    ch0 = tl.minimum(ch0, CBLK - 1)
                x0 = tl.load(a0_ptr)
                xf0 = x0.to(tl.float32)
                mean0 = tl.sum(xf0, 1) / L
                var0 = tl.maximum(tl.sum(xf0 * xf0, 1) / L - mean0 * mean0, 0.0)
                rstd0 = rsqrt(var0 + eps)
                y0 = (x0.to(CDT) - mean0[:, None].to(CDT)) * rstd0[:, None].to(CDT)
                if HAS_W:
                    y0 = y0 * tl.load(tle.gpu.local_ptr(w_smem, (ch0,))).to(CDT)
                if HAS_B:
                    y0 = y0 + tl.load(tle.gpu.local_ptr(b_smem, (ch0,))).to(CDT)
                tl.store(a0_ptr, y0.to(IN_DTYPE))
                tl.store(m0_ptr, mean0.to(OUT_DTYPE))
                tl.store(r0_ptr, rstd0.to(OUT_DTYPE))
                tle.gpu.copy(a0, y_desc, [XBLOCK, WT], [b0, 0], sync=False)
                tle.gpu.copy(m0, mean_desc, [XBLOCK], [b0], sync=False)
                tle.gpu.copy(r0, rstd_desc, [XBLOCK], [b0], sync=False)
                tle.gpu.dma_wait()
                # ---- half beat B: prefetch b2 -> a0, compute a1 (block b1) --
                if TAIL:
                    tl.store(a0_ptr, tl.zeros([XBLOCK, WT], IN_DTYPE))
                tle.gpu.copy(x_desc, a0, [XBLOCK, WT], [b2, 0], sync=False)
                ch1 = ((b1 + rid) % num_groups)[:, None] * group_size + cols // HW
                if TAIL:
                    ch1 = tl.minimum(ch1, CBLK - 1)
                x1 = tl.load(a1_ptr)
                xf1 = x1.to(tl.float32)
                mean1 = tl.sum(xf1, 1) / L
                var1 = tl.maximum(tl.sum(xf1 * xf1, 1) / L - mean1 * mean1, 0.0)
                rstd1 = rsqrt(var1 + eps)
                y1 = (x1.to(CDT) - mean1[:, None].to(CDT)) * rstd1[:, None].to(CDT)
                if HAS_W:
                    y1 = y1 * tl.load(tle.gpu.local_ptr(w_smem, (ch1,))).to(CDT)
                if HAS_B:
                    y1 = y1 + tl.load(tle.gpu.local_ptr(b_smem, (ch1,))).to(CDT)
                tl.store(a1_ptr, y1.to(IN_DTYPE))
                tl.store(m1_ptr, mean1.to(OUT_DTYPE))
                tl.store(r1_ptr, rstd1.to(OUT_DTYPE))
                tle.gpu.copy(a1, y_desc, [XBLOCK, WT], [b1, 0], sync=False)
                tle.gpu.copy(m1, mean_desc, [XBLOCK], [b1], sync=False)
                tle.gpu.copy(r1, rstd_desc, [XBLOCK], [b1], sync=False)
                tle.gpu.dma_wait()
        else:
            a_lmem = tle.gpu.alloc(
                [XBLOCK, WT], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
            )
            # Cluster-shared SM, for the coalescing reason documented in the WIDE
            # leg above: a [XBLOCK] LM strip writeback costs ~10us (f32) / ~22us
            # (f16) in cross-cluster cache-line contention, an SM one ~0.1-0.8us.
            m_lmem = tle.gpu.alloc(
                [XBLOCK],
                dtype=OUT_DTYPE,
                layout=None,
                scope=tle.gpu.smem if SM_STATS else tle.gpu.lmem,
            )
            r_lmem = tle.gpu.alloc(
                [XBLOCK],
                dtype=OUT_DTYPE,
                layout=None,
                scope=tle.gpu.smem if SM_STATS else tle.gpu.lmem,
            )
            a_ptr = tle.gpu.local_ptr(a_lmem, (rows, cols))
            if TAIL:
                tl.store(a_ptr, tl.zeros([XBLOCK, WT], IN_DTYPE))
            tle.gpu.copy(x_desc, a_lmem, [XBLOCK, WT], [row0, 0])
            ch = ((row0 + rid) % num_groups)[:, None] * group_size + cols // HW
            if TAIL:
                ch = tl.minimum(ch, CBLK - 1)
            x = tl.load(a_ptr)
            xf = x.to(tl.float32)
            mean = tl.sum(xf, 1) / L
            var = tl.maximum(tl.sum(xf * xf, 1) / L - mean * mean, 0.0)
            rstd = rsqrt(var + eps)
            y = (x.to(CDT) - mean[:, None].to(CDT)) * rstd[:, None].to(CDT)
            if HAS_W:
                y = y * tl.load(tle.gpu.local_ptr(w_smem, (ch,))).to(CDT)
            if HAS_B:
                y = y + tl.load(tle.gpu.local_ptr(b_smem, (ch,))).to(CDT)
            tl.store(a_ptr, y.to(IN_DTYPE))
            tl.store(tle.gpu.local_ptr(m_lmem, (rid,)), mean.to(OUT_DTYPE))
            tl.store(tle.gpu.local_ptr(r_lmem, (rid,)), rstd.to(OUT_DTYPE))
            tle.gpu.copy(a_lmem, y_desc, [XBLOCK, WT], [row0, 0])
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
    """ONE tle kernel for the whole operator; False only when it cannot run.

    Every shape the aten semantics can produce is covered here:
    * xblock >= 2 rows per program -- RowTiled, reduce core-local;
    * xblock == 1 -- LargeN m == 1: the [1, WT] tile splits the row across all
      64 cores and the reduce goes cross-core;
    * wide rows (WT >= the loop threshold) -- the same kernel streams [1, RB]
      segments in a two-pass loop (reduce, then normalize), so no register or
      LM residency limit applies to the row width.
    The only declines left are hard ones: a row wider than WMAX even for the
    segment loop, or a per-core slice that still will not fit.
    """
    # Pad the row width up to the core count: the m == 1 LargeN tiling needs
    # WT % coreNum == 0, and npo2(L) < 64 (tiny single-group rows) would leave
    # the core-tiling pass unable to engage. The padding columns are handled
    # by the TAIL machinery (masked out of the reduce, clamped out of the
    # gather, never written back).
    WT = max(_npo2(L), _TLE_CORE_NUM)
    if WT > _TLE_FUSE_WMAX:
        return False
    # Segment width for rows too wide to hold. Deciding it FIRST also decides
    # what the xblock growth below budgets against: a WIDE row's resident tile
    # is [xblock, RB], not [xblock, WT]. RB is capped so the per-core segment
    # footprint (RB/coreNum elements) leaves room for the accumulators, and
    # stays >= coreNum so the [xblock, RB] tile core-tiles.
    rb = min(WT, _TLE_FUSE_LOOP_RB)
    wide = WT >= _TLE_FUSE_LOOP_WMIN
    resident = rb if wide else WT
    # xblock growth: keep the core's slice of the RESIDENT tile plus the two
    # [xblock] fp32 accumulators inside the LM element budget, and stop when
    # more rows per program would no longer cut the program count.
    # _TLE_FUSE_EPC is the MEASURED per-core element ceiling for this kernel
    # (256; larger values compiled "over the local-memory budget" on the
    # (16,16,1024)/(32,32,32,32) shapes, so this is empirical, not a model).
    # The resident width is what it applies to, which is the whole point of
    # deciding RB first: a wide row budgets [xblock, RB] instead of
    # [xblock, WT], so xblock can still grow there.
    # The WIDE branch keeps two [XBLOCK, RB] fp32 accumulators beside the
    # segment tile, so its per-core register budget is roughly a third of the
    # hold-the-row budget (see the deferred-merge comment in the kernel).
    epc = _TLE_FUSE_EPC_WIDE if wide else _TLE_FUSE_EPC
    xblock = 1
    while xblock * 2 <= m_grp and (xblock * 2) * resident // _TLE_CORE_NUM <= epc:
        xblock *= 2
    # Give the grid back toward one program per cluster when rows are plentiful.
    while (
        xblock > 1
        and -(-m_grp // xblock) < _TLE_CLUSTERS
        and -(-m_grp // (xblock >> 1)) > -(-m_grp // xblock)
    ):
        xblock >>= 1
    # The core-tiling pass must ENGAGE for the SM weight/bias gather to be
    # correct (a no-op pass leaves the default layout, whose gather/store slot
    # counts disagree with the copy drain plan). resident is a power of two
    # >= coreNum, so the [xblock, resident] tile always tiles.
    grid = (-(-m_grp // xblock),)

    # Row-block pipelining: halve xblock and double the blocks per program, which
    # leaves the PROGRAM COUNT (and with it the wave count on 8 clusters)
    # untouched while cutting the per-core resident footprint in half and putting
    # block i+1's GM read in flight behind block i's compute. Two hard gates:
    # divisibility (the pipelined loop has no tail-block protection), and a
    # xblock floor of 2 -- xblock == 1 makes the [1, WT] tile LargeN, whose row
    # reduce goes cross-core through smem plus a cluster barrier, and paying that
    # barrier RITER times per program is exactly the trade this is trying to
    # avoid. Only the hold-the-row leg pipelines: the WIDE leg's residency is
    # already the segment, not the row.
    riter = 1
    if not wide:
        while (
            xblock > 2
            and riter * 2 <= _TLE_FUSE_RITER_MAX
            and (xblock // 2) * resident // _TLE_CORE_NUM >= _TLE_FUSE_EPC_PIPE
            and m_grp % ((xblock // 2) * (riter * 2)) == 0
        ):
            xblock //= 2
            riter *= 2
        grid = (m_grp // (xblock * riter),) if riter > 1 else grid

    has_w = weight is not None
    has_b = bias is not None
    dummy = (
        None
        if has_w and has_b
        else torch.empty(C, dtype=input.dtype, device=input.device)
    )
    w_t = weight if has_w else dummy
    b_t = bias if has_b else dummy
    # Affine compute dtype: the native input dtype for ALL of fp16/fp32/bf16.
    # bf16 needs no explicit f32 detour -- triton/tle-dtype-convert inserts the
    # bf16<->f32 conversion implicitly (XPU3 has no bf16 vector op). Only the
    # reduce is pinned to f32 for accumulation precision.
    cdt = tl_dtype

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
        rb,
        wide,
        riter,
        tl_dtype,
        tl_dtype,
        cdt,
        WT != L,
        has_w,
        has_b,
        tl_dtype != tl.bfloat16,
        isCloseCoreTiling=False,
    )
    logger.debug(
        "GEMS_KUNLUNXIN NATIVE_GROUP_NORM tle FUSED m_grp=%d L=%d tile=%dx%d "
        "wide=%d riter=%d grid=%d",
        m_grp,
        L,
        xblock,
        WT,
        wide,
        riter,
        grid[0],
    )
    return True


def _tle_native_group_norm(
    input, y, weight, bias, mean, rstd, N, C, HxW, group, group_size, eps
):
    """Run group norm as ONE tle kernel; False (untouched) when tle cannot.

    On KL3 the fused kernel is the operator: one launch, one kernel, every
    shape the aten semantics can produce. group * group_size == C is an aten
    invariant (PyTorch's group_norm rejects C % group != 0 at the ATen layer),
    so no ragged-group gate is needed here.
    """
    if input.dtype not in _TLE_TL_DTYPE:
        return False
    tl_dtype = _TLE_TL_DTYPE[input.dtype]
    m_stat = N * group
    lrow = group_size * HxW

    with torch_device_fn.device(input.device):
        if _tle_group_norm_fused(
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

    # The fused kernel covers every shape the aten semantics can produce: wide
    # rows go through the in-kernel two-pass segment loop, single rows through
    # the m == 1 LargeN tiling. A decline here would mean an unsupported dtype
    # or an absurd row width -- both are hard errors worth surfacing rather
    # than silently falling back to a kernel that does the same math slower.
    raise RuntimeError(
        "kunlunxin native_group_norm: tle kernel declined "
        f"(N={N} C={C} HxW={HxW} group={group} dtype={input.dtype} "
        f"L={group_size * HxW})"
    )


def native_group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    """aten::native_group_norm on Kunlunxin -- ONE tle.gpu kernel.

    The generic flag_gems.ops.native_group_norm binds
    flag_gems.ops.groupnorm.group_norm at import time, so SpecOpRegistrar
    swapping flag_gems.group_norm never reached it and native_group_norm kept
    running the generic single-kernel giant-2D-tile implementation on XPU.
    That path miscompiles on the small tiles used by the accuracy matrix and
    hard-fails with `out of resource: uni_sram` for HxW >= 4096, so bind a
    vendor kernel here explicitly.

    There is deliberately no GM-pointer fallback: like sum.py, this vendor
    operator is written for the KL3 tle pipeline and only exists there. A second
    GM implementation would be a slower duplicate of the same math with none of
    the DMA -- it previously existed only to serve shapes the tle kernel could
    not tile, and with the m == 1 LargeN tiling and the in-kernel wide-row
    segment loop there are no such shapes left.
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
    # Returned in the INPUT dtype, like the generic op and groupnorm.py, because
    # tests/test_group_norm.py asserts the container dtype through
    # gems_assert_close. That used to cost ~20us of a 30us fp16/bf16 kernel (the
    # narrow writeback scaled with the ELEMENT size, so 2-byte stats were twice
    # as expensive as 4-byte ones); the SM writeback below removes the whole
    # penalty for fp16/fp32, and bf16 still pays it only because its SM store
    # cannot be selected yet.
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    _tle_native_group_norm(
        input, y, weight, bias, mean, rstd, N, C, HxW, group, group_size, eps
    )
    return y, mean, rstd
