# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""CTA-local BF16 NT with K256 tiles and wave-private padded input storage.

A[M,K], physical B[N,K], C[M,N], contiguous BF16. Four compute waves
own four K64 partitions of the SAME output tile. No global split workspace.
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
def producer_layout(rows):
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
def input_coords(D: gl.constexpr, L: gl.constexpr):
    sp = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    k = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return row[None, :, None], sp[:, None, None] * 64 + k[None, None, :]


@libentry()
@g.jit
def _small_nt_splitu_k256(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                           BM: gl.constexpr = 32, BN: gl.constexpr = 32):
    gl.static_assert(M > 0 and N > 0 and K >= 256 and K % 256 == 0)
    gl.static_assert(BM >= 16 and BN >= 16 and BM & (BM - 1) == 0 and BN & (BN - 1) == 0)
    gl.static_assert(BM + BN <= 112 and BM * BN <= 4096)
    gl.static_assert(2 * M * K < 2147483648 and 2 * N * K < 2147483648 and 2 * M * N < 2147483648)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and B.dtype.element_ty == gl.bfloat16)
    gl.static_assert(C.dtype.element_ty == gl.bfloat16)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], False, [4, 1, 1],
                                     tiles_per_warp=[1, 1, 1], mmac_layout=1)
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    pa: gl.constexpr = producer_layout(BM)
    pb: gl.constexpr = producer_layout(BN)
    sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BM, 64], [2, 1, 0])
    sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BN, 64], [2, 1, 0])
    rm, ka = input_coords(BM, pa)
    rn, kb = input_coords(BN, pb)
    row_m = gl.program_id(0) * BM
    col_n = gl.program_id(1) * BN
    abase = A + row_m.to(gl.int64) * K
    bbase = B + col_n.to(gl.int64) * K
    ao = (rm * K + ka).to(gl.int32)
    bo = (rn * K + kb).to(gl.int32)
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
    for q in pipeline_range(0, K // 256, loop_unroll_factor=1):
        aa.store(av)
        bb.store(bv)
        # No explicit A/B CTA: compiler must prove wave-private ownership
        # and supply per-wave LDS waits, or conservatively retain full CTA.
        ar = aa.load(da)
        br = bb.permute([0, 2, 1]).load(db)
        have_next = q + 1 < K // 256
        next_a = buffer_load(abase + (q + 1) * 256, ao, am & have_next, other=0)
        next_b = buffer_load(bbase + (q + 1) * 256, bo, bm & have_next, other=0)
        acc = mfma(ar, br, acc)
        av = next_a
        bv = next_b
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
