"""CTA-local split-U BF16 NT with static double-buffer slots.

A[M,K], physical B[N,K], C[M,N], contiguous BF16. Four compute waves
own four K32 partitions of the SAME output tile. No global split workspace.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store, mfma
from triton.experimental.gluon.language.hcu.raw_window import raw_warp_group_id
from triton.language import range as pipeline_range
from triton.runtime.jit import constexpr_function

from flag_gems.utils import libentry


@constexpr_function
def producer_layout(rows, row_warp_stride):
    assert rows >= 16 and rows & (rows - 1) == 0
    assert row_warp_stride in (1, 4)
    registers = [[0, 0, 1], [0, 0, 2], [0, 0, 4]]
    bit = 16
    while bit < rows:
        registers.append([0, bit, 0]); bit *= 2
    lane_rows = (1, 2) if row_warp_stride == 4 else (4, 8)
    warp_rows = (4, 8) if row_warp_stride == 4 else (1, 2)
    return gl.DistributedLinearLayout(
        registers, [[0, 0, 8], [0, 0, 16], [1, 0, 0], [2, 0, 0],
                    [0, lane_rows[0], 0], [0, lane_rows[1], 0]],
        [[0, warp_rows[0], 0], [0, warp_rows[1], 0]], [], [4, rows, 32])


@constexpr_function
def partial_layout(bm, bn):
    # Actual rank3 HCU LEGACY, transposed=False, native four-accumulator group.
    # Physical offset bits: r0,r1,lane0..5,remaining r,split0,split1.
    bases = [[0, 0, 4], [0, 0, 8], [0, 1, 0], [0, 2, 0],
             [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]]
    bit = 16
    while bit < bn:
        bases.append([0, 0, bit]); bit *= 2
    bit = 16
    while bit < bm:
        bases.append([0, bit, 0]); bit *= 2
    bases += [[1, 0, 0], [2, 0, 0]]
    # This interval is the whole allocation: no interior C padding.
    return gl.PaddedSharedLayout([[4 * bm * bn, 1]], bases, [], [4, bm, bn])


@g.jit
def input_coords(D: gl.constexpr, L: gl.constexpr):
    sp = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    k = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return row[None, :, None], sp[:, None, None] * 32 + k[None, None, :]


@libentry()
@g.jit
def _small_nt_splitu(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                     BM: gl.constexpr, BN: gl.constexpr,
                     PADDED: gl.constexpr = True,
                     ROW_WARP_STRIDE: gl.constexpr = 4,
                     DOUBLE_BUFFER: gl.constexpr = True):
    gl.static_assert(M > 0 and N > 0 and K >= 128 and K % 128 == 0)
    gl.static_assert(BM >= 16 and BN >= 16 and BM & (BM - 1) == 0 and BN & (BN - 1) == 0)
    gl.static_assert(BM * BN <= 4096 and BM + BN <= 128)
    gl.static_assert(ROW_WARP_STRIDE == 1 or ROW_WARP_STRIDE == 4)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and B.dtype.element_ty == gl.bfloat16)
    gl.static_assert(C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(2 * M * K < 2147483648 and 2 * N * K < 2147483648 and 2 * M * N < 2147483648)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], False, [4, 1, 1],
                                     tiles_per_warp=[1, 1, 1], mmac_layout=1)
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    pa: gl.constexpr = producer_layout(BM, ROW_WARP_STRIDE)
    pb: gl.constexpr = producer_layout(BN, ROW_WARP_STRIDE)
    if PADDED:
        sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [4, BM, 32], [2, 0, 1])
        sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [4, BN, 32], [2, 0, 1])
    else:
        sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[4 * BM * 32, 1]], [4, BM, 32], [2, 0, 1])
        sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[4 * BN * 32, 1]], [4, BN, 32], [2, 0, 1])
    rm, ka = input_coords(BM, pa)
    rn, kb = input_coords(BN, pb)
    row_m = gl.program_id(0) * BM
    col_n = gl.program_id(1) * BN
    abase = A + row_m.to(gl.int64) * K
    bbase = B + col_n.to(gl.int64) * K
    ao = (rm * K + ka).to(gl.int32)
    bo = (rn * K + kb).to(gl.int32)
    acc = gl.full([4, BM, BN], 0, gl.float32, mma)
    if DOUBLE_BUFFER:
        # Leading allocation dimension is an ordinary buffer dimension; shared
        # layout itself remains rank3, and index(slot) returns [4,D,32].
        aa = gl.allocate_shared_memory(gl.bfloat16, [2, 4, BM, 32], sa)
        bb = gl.allocate_shared_memory(gl.bfloat16, [2, 4, BN, 32], sb)
        first_a = buffer_load(abase, ao, row_m + rm < M, other=0)
        first_b = buffer_load(bbase, bo, col_n + rn < N, other=0)
        aa.index(0).store(first_a)
        bb.index(0).store(first_b)
        # Two explicit constant-slot phases. Two source pairs preserve the
        # previous four-K128-chunk unroll hint (2 pairs instead of4 singles).
        for q in pipeline_range(0, K // 128, 2, loop_unroll_factor=2):
            # q is even: publish slot0/q; retire every read of q-1/slot1.
            gl.thread_barrier()
            ar0 = aa.index(0).load(da)
            br0 = bb.index(0).permute([0, 2, 1]).load(db)
            have_phase1 = q + 1 < K // 128
            next_a0 = buffer_load(abase + (q + 1) * 128, ao,
                                  (row_m + rm < M) & have_phase1, other=0)
            next_b0 = buffer_load(bbase + (q + 1) * 128, bo,
                                  (col_n + rn < N) & have_phase1, other=0)
            acc = mfma(ar0, br0, acc)
            if have_phase1:
                aa.index(1).store(next_a0)
                bb.index(1).store(next_b0)
                # Uniform q/Q branch: ALL waves take this CTA or none do.
                # Publish slot1/q+1 and retire current slot0/q reads before
                # that slot is reused for q+2. No weakened or partial wait.
                gl.thread_barrier()
                ar1 = aa.index(1).load(da)
                br1 = bb.index(1).permute([0, 2, 1]).load(db)
                have_next_pair = q + 2 < K // 128
                next_a1 = buffer_load(abase + (q + 2) * 128, ao,
                                      (row_m + rm < M) & have_next_pair, other=0)
                next_b1 = buffer_load(bbase + (q + 2) * 128, bo,
                                      (col_n + rn < N) & have_next_pair, other=0)
                acc = mfma(ar1, br1, acc)
                if have_next_pair:
                    aa.index(0).store(next_a1)
                    bb.index(0).store(next_b1)
        aa._keep_alive()
        bb._keep_alive()
    else:
        aa = gl.allocate_shared_memory(gl.bfloat16, [4, BM, 32], sa)
        bb = gl.allocate_shared_memory(gl.bfloat16, [4, BN, 32], sb)
        for q in pipeline_range(0, K // 128, loop_unroll_factor=4):
            av = buffer_load(abase + q * 128, ao, row_m + rm < M, other=0)
            bv = buffer_load(bbase + q * 128, bo, col_n + rn < N, other=0)
            aa.store(av)
            bb.store(bv)
            # All waves: publish the same complete K128 tile.
            gl.thread_barrier()
            ar = aa.load(da)
            br = bb.permute([0, 2, 1]).load(db)
            acc = mfma(ar, br, acc)
            # All waves: retire LDS reads before this allocation is overwritten.
            gl.thread_barrier()
        aa._keep_alive()
        bb._keep_alive()

    pc: gl.constexpr = partial_layout(BM, BN)
    cs = gl.allocate_shared_memory(gl.float32, [4, BM, BN], pc)
    cs.store(acc)
    gl.thread_barrier()
    wave = raw_warp_group_id(1)
    if wave == 0:
        # batch=1 makes the MFMA warp bits redundant. All C coordinates live
        # in wave0, with exactly the same native register/lane mapping.
        c0 = cs.slice(0, 1, 0).load(mma)
        c1 = cs.slice(1, 1, 0).load(mma)
        c2 = cs.slice(2, 1, 0).load(mma)
        c3 = cs.slice(3, 1, 0).load(mma)
        value = ((c0 + c1) + (c2 + c3)).reshape([BM, BN])
        cl: gl.constexpr = gl.SliceLayout(0, mma)
        value = gl.convert_layout(value, cl)
        mi = gl.arange(0, BM, layout=gl.SliceLayout(1, cl))
        ni = gl.arange(0, BN, layout=gl.SliceLayout(0, cl))
        co = (mi[:, None] * N + ni[None, :]).to(gl.int32)
        cbase = C + row_m.to(gl.int64) * N + col_n.to(gl.int64)
        buffer_store(value.to(gl.bfloat16), cbase, co,
                     (row_m + mi[:, None] < M) & (col_n + ni[None, :] < N))
