# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""CTA-local BF16 split-K with explicit input and loop-schedule profiles.

INPUT_LAYOUT: 0 = K256, physical B[N,K]; 1 = K256, physical B[K,N];
2 = cooperative K128, physical B[N,K]. LOOP_SCHEDULE: 0 = ordinary,
1 = peeled final iteration, 2 = rotated K32 reads, 3 = multiple row tiles.
All profiles retain the same ordered four-partial FP32 output reduction.
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
def producer_layout(rows, kind=0, row_warp_stride=4):
    if kind == 2:
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
    elif kind == 1:
        assert rows >= 8 and rows & (rows - 1) == 0
        # Eight adjacent N values per thread -> candidate 16-byte global vector.
        registers = [[0, 1, 0], [0, 2, 0], [0, 4, 0]]
        lanes = []
        bit = 8
        while bit < rows:
            if len(lanes) < 6:
                lanes.append([0, bit, 0])
            else:
                registers.append([0, bit, 0])
            bit *= 2
        bit = 1
        while bit < 64:
            if len(lanes) < 6:
                lanes.append([0, 0, bit])
            else:
                registers.append([0, 0, bit])
            bit *= 2
        return gl.DistributedLinearLayout(
            registers, lanes, [[1, 0, 0], [2, 0, 0]], [], [4, rows, 64])
    else:
        assert rows >= 8 and rows & (rows - 1) == 0
        registers = [[0, 0, 1], [0, 0, 2], [0, 0, 4]]
        bit = 8
        while bit < rows:
            registers.append([0, bit, 0]); bit *= 2
        # lane K8/K16/K32 then row1/2/4; wave bits select the K64 partition.
        return gl.DistributedLinearLayout(
            registers, [[0, 0, 8], [0, 0, 16], [0, 0, 32],
                        [0, 1, 0], [0, 2, 0], [0, 4, 0]],
            [[1, 0, 0], [2, 0, 0]], [], [4, rows, 64])


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
def input_coords(D: gl.constexpr, L: gl.constexpr, WIDTH: gl.constexpr):
    sp = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    k = gl.arange(0, WIDTH, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return row[None, :, None], sp[:, None, None] * WIDTH + k[None, None, :]


@g.jit
def store_partials(acc, BM: gl.constexpr, BN: gl.constexpr):
    pc: gl.constexpr = partial_layout(BM, BN)
    cs = gl.allocate_shared_memory(gl.float32, [4, BM, BN], pc)
    cs.store(acc)
    return cs


@g.jit
def write_output_tile(cs, C, row_m, col_n, M: gl.constexpr, N: gl.constexpr,
                      BM: gl.constexpr, BN: gl.constexpr, MMA: gl.constexpr):
    # Keep the original grouping; do not reassociate the four FP32 partials.
    c0 = cs.slice(0, 1, 0).load(MMA)
    c1 = cs.slice(1, 1, 0).load(MMA)
    c2 = cs.slice(2, 1, 0).load(MMA)
    c3 = cs.slice(3, 1, 0).load(MMA)
    value = ((c0 + c1) + (c2 + c3)).reshape([BM, BN])
    cl: gl.constexpr = gl.SliceLayout(0, MMA)
    value = gl.convert_layout(value, cl)
    mi = gl.arange(0, BM, layout=gl.SliceLayout(1, cl))
    ni = gl.arange(0, BN, layout=gl.SliceLayout(0, cl))
    co = (mi[:, None] * N + ni[None, :]).to(gl.int32)
    cbase = C + row_m.to(gl.int64) * N + col_n.to(gl.int64)
    buffer_store(value.to(gl.bfloat16), cbase, co,
                 (row_m + mi[:, None] < M) & (col_n + ni[None, :] < N))


@libentry()
@g.jit
def _mm_local_splitk_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BM: gl.constexpr = 32, BN: gl.constexpr = 32,
    INPUT_LAYOUT: gl.constexpr = 0, LOOP_SCHEDULE: gl.constexpr = 0,
    M_TILES: gl.constexpr = 1, PADDED: gl.constexpr = True,
    ROW_WARP_STRIDE: gl.constexpr = 4, DOUBLE_BUFFER: gl.constexpr = True,
):
    gl.static_assert(INPUT_LAYOUT == 0 or INPUT_LAYOUT == 1 or INPUT_LAYOUT == 2)
    gl.static_assert(LOOP_SCHEDULE >= 0 and LOOP_SCHEDULE <= 3)
    gl.static_assert(INPUT_LAYOUT == 0 or LOOP_SCHEDULE == 0)
    gl.static_assert(LOOP_SCHEDULE == 3 or M_TILES == 1)
    gl.static_assert(M > 0 and N > 0)
    gl.static_assert(BM >= 16 and BN >= 16 and BM & (BM - 1) == 0 and BN & (BN - 1) == 0)
    gl.static_assert(2 * M * K < 2147483648 and 2 * N * K < 2147483648 and 2 * M * N < 2147483648)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and B.dtype.element_ty == gl.bfloat16)
    gl.static_assert(C.dtype.element_ty == gl.bfloat16)
    if INPUT_LAYOUT == 2:
        gl.static_assert(K >= 128 and K % 128 == 0)
        gl.static_assert(BM * BN <= 4096 and BM + BN <= 128)
        gl.static_assert(ROW_WARP_STRIDE == 1 or ROW_WARP_STRIDE == 4)
    else:
        gl.static_assert(K >= 256 and K % 256 == 0)
        if LOOP_SCHEDULE == 3:
            # 4*(M_TILES*BM+BN)*(64+8)*2 fits the 64KiB input allocation.
            gl.static_assert(M_TILES > 0 and M_TILES * BM + BN <= 112)
        else:
            gl.static_assert(BM + BN <= 112 and BM * BN <= 4096)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], False, [4, 1, 1],
                                     tiles_per_warp=[1, 1, 1], mmac_layout=1)
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    if INPUT_LAYOUT == 2:
        pa: gl.constexpr = producer_layout(BM, 2, ROW_WARP_STRIDE)
        pb: gl.constexpr = producer_layout(BN, 2, ROW_WARP_STRIDE)
        if PADDED:
            sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [4, BM, 32], [2, 0, 1])
            sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [4, BN, 32], [2, 0, 1])
        else:
            sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[4 * BM * 32, 1]], [4, BM, 32], [2, 0, 1])
            sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[4 * BN * 32, 1]], [4, BN, 32], [2, 0, 1])
    else:
        pa: gl.constexpr = producer_layout(BM)
        if INPUT_LAYOUT == 1:
            pb: gl.constexpr = producer_layout(BN, 1)
            sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BN, 8]], [4, BN, 64], [1, 2, 0])
        else:
            pb: gl.constexpr = producer_layout(BN)
            sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BN, 64], [2, 1, 0])
        sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BM, 64], [2, 1, 0])
    WIDTH: gl.constexpr = 32 if INPUT_LAYOUT == 2 else 64
    rm, ka = input_coords(BM, pa, WIDTH)
    rn, kb = input_coords(BN, pb, WIDTH)
    if LOOP_SCHEDULE == 3:
        row_m=gl.program_id(0)*(M_TILES*BM)
        col_n=gl.program_id(1)*BN
        bbase=B+col_n.to(gl.int64)*K
        bo=(rn*K+kb).to(gl.int32)
        bmask=col_n+rn<N
        aa=()
        av=()
        masks=()
        bases=()
        acc=()
        ao=(rm*K+ka).to(gl.int32)
        for i in gl.static_range(0,M_TILES):
            base=A+(row_m+i*BM).to(gl.int64)*K
            mask=row_m+i*BM+rm<M
            aa=aa+(gl.allocate_shared_memory(gl.bfloat16,[4,BM,64],sa),)
            av=av+(buffer_load(base,ao,mask,other=0),)
            masks=masks+(mask,)
            bases=bases+(base,)
            acc=acc+(gl.full([4,BM,BN],0,gl.float32,mma),)
        bb=gl.allocate_shared_memory(gl.bfloat16,[4,BN,64],sb)
        bv=buffer_load(bbase,bo,bmask,other=0)
        for q in pipeline_range(0,K//256,loop_unroll_factor=1):
            for i in gl.static_range(0,M_TILES):
                aa[i].store(av[i])
            bb.store(bv)
            br=bb.permute([0,2,1]).load(db)
            have_next=q+1<K//256
            next_av=()
            next_acc=()
            for i in gl.static_range(0,M_TILES):
                ar=aa[i].load(da)
                next_av=next_av+(buffer_load(bases[i]+(q+1)*256,ao,masks[i]&have_next,other=0),)
                next_acc=next_acc+(mfma(ar,br,acc[i]),)
            next_b=buffer_load(bbase+(q+1)*256,bo,bmask&have_next,other=0)
            av=next_av
            bv=next_b
            acc=next_acc
        for i in gl.static_range(0,M_TILES):
            aa[i]._keep_alive()
        bb._keep_alive()
        cs = ()
        for i in gl.static_range(0, M_TILES):
            cs = cs + (store_partials(acc[i], BM, BN),)
    else:
        row_m = gl.program_id(0) * BM
        col_n = gl.program_id(1) * BN
        abase = A + row_m.to(gl.int64) * K
        if INPUT_LAYOUT == 1:
            bbase = B + col_n.to(gl.int64)
        else:
            bbase = B + col_n.to(gl.int64) * K
        ao = (rm * K + ka).to(gl.int32)
        if INPUT_LAYOUT == 1:
            bo = (rn + kb * N).to(gl.int32)
        else:
            bo = (rn * K + kb).to(gl.int32)
        if INPUT_LAYOUT == 2:
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

        else:
            if M % BM == 0:
                am = gl.full([4, BM, 64], True, gl.int1, pa)
            else:
                am = row_m + rm < M
            if N % BN == 0:
                bm = gl.full([4, BN, 64], True, gl.int1, pb)
            else:
                bm = col_n + rn < N
            acc = gl.full([4, BM, BN], 0, gl.float32, mma)
            aa = gl.allocate_shared_memory(gl.bfloat16, [4, BM, 64], sa)
            bb = gl.allocate_shared_memory(gl.bfloat16, [4, BN, 64], sb)
            av = buffer_load(abase, ao, am, other=0)
            bv = buffer_load(bbase, bo, bm, other=0)
            if LOOP_SCHEDULE == 2:
                aa.store(av)
                bb.store(bv)
                ar0 = aa.slice(0, 32, 2).load(da)
                br0 = bb.slice(0, 32, 2).permute([0, 2, 1]).load(db)
                if K // 256 > 1:
                    av_next = buffer_load(abase + 256, ao, am, other=0)
                    bv_next = buffer_load(bbase + 256, bo, bm, other=0)
                    for q in pipeline_range(0, K // 256 - 2, loop_unroll_factor=1):
                        ar1 = aa.slice(32, 32, 2).load(da)
                        br1 = bb.slice(32, 32, 2).permute([0, 2, 1]).load(db)
                        acc = mfma(ar0, br0, acc)
                        aa.store(av_next)
                        bb.store(bv_next)
                        ar0 = aa.slice(0, 32, 2).load(da)
                        br0 = bb.slice(0, 32, 2).permute([0, 2, 1]).load(db)
                        av_next = buffer_load(abase + (q + 2) * 256, ao, am, other=0)
                        bv_next = buffer_load(bbase + (q + 2) * 256, bo, bm, other=0)
                        acc = mfma(ar1, br1, acc)
                    # Final producer step: no unused future global load is issued.
                    ar1 = aa.slice(32, 32, 2).load(da)
                    br1 = bb.slice(32, 32, 2).permute([0, 2, 1]).load(db)
                    acc = mfma(ar0, br0, acc)
                    aa.store(av_next)
                    bb.store(bv_next)
                    ar0 = aa.slice(0, 32, 2).load(da)
                    br0 = bb.slice(0, 32, 2).permute([0, 2, 1]).load(db)
                    acc = mfma(ar1, br1, acc)
                # Consume the final tile, including the single-tile K256 boundary.
                ar1 = aa.slice(32, 32, 2).load(da)
                br1 = bb.slice(32, 32, 2).permute([0, 2, 1]).load(db)
                acc = mfma(ar0, br0, acc)
                acc = mfma(ar1, br1, acc)
            elif LOOP_SCHEDULE == 1:
                for q in pipeline_range(0, K // 256 - 1, loop_unroll_factor=1):
                    aa.store(av)
                    bb.store(bv)
                    # No explicit A/B CTA: compiler must prove wave-private ownership
                    # and supply per-wave LDS waits, or conservatively retain full CTA.
                    ar = aa.load(da)
                    br = bb.permute([0, 2, 1]).load(db)
                    next_a = buffer_load(abase + (q + 1) * 256, ao, am, other=0)
                    next_b = buffer_load(bbase + (q + 1) * 256, bo, bm, other=0)
                    acc = mfma(ar, br, acc)
                    av = next_a
                    bv = next_b
                # Last compute iteration has no consumer for another input prefetch.
                # Keep exactly the same input ownership and ordered MFMA accumulation.
                aa.store(av)
                bb.store(bv)
                ar = aa.load(da)
                br = bb.permute([0, 2, 1]).load(db)
                acc = mfma(ar, br, acc)
            else:
                for q in pipeline_range(0, K // 256, loop_unroll_factor=1):
                    aa.store(av)
                    bb.store(bv)
                    # No explicit A/B CTA: compiler must prove wave-private ownership
                    # and supply per-wave LDS waits, or conservatively retain full CTA.
                    ar = aa.load(da)
                    br = bb.permute([0, 2, 1]).load(db)
                    have_next = q + 1 < K // 256
                    next_a = buffer_load(abase + (q + 1) * 256, ao, am & have_next, other=0)
                    if INPUT_LAYOUT == 1:
                        next_b = buffer_load(bbase + (q + 1) * 256 * N, bo, bm & have_next, other=0)
                    else:
                        next_b = buffer_load(bbase + (q + 1) * 256, bo, bm & have_next, other=0)
                    acc = mfma(ar, br, acc)
                    av = next_a
                    bv = next_b
            aa._keep_alive()
            bb._keep_alive()
        cs = store_partials(acc, BM, BN)
    gl.thread_barrier()
    wave = raw_warp_group_id(1)
    if wave == 0:
        if LOOP_SCHEDULE == 3:
            for i in gl.static_range(0, M_TILES):
                write_output_tile(cs[i], C, row_m + i * BM, col_n, M, N, BM, BN, mma)
        else:
            write_output_tile(cs, C, row_m, col_n, M, N, BM, BN, mma)
