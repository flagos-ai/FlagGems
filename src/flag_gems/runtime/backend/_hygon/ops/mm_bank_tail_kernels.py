"""Selected FP64 bank phases with a shortened final K panel.

Only selected workbook configurations use this kernel. Full K blocks load
valid wrapped rows/columns without a boundary mask; output stores still mask
the original coordinates. Partial K blocks remain masked.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout

from .mm_mmac_swizzle import compute, load_layout, parts, rev, store_acc


@g.jit
def load_banked_tail(
    A,
    B,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    pm,
    pn,
    step,
    FULL: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
    SA_VEC: gl.constexpr,
    SA_MAX: gl.constexpr,
    SB_VEC: gl.constexpr,
    SB_MAX: gl.constexpr,
    SA_PER: gl.constexpr,
    SB_PER: gl.constexpr,
):
    WN: gl.constexpr = WARPS // WM
    LD: gl.constexpr = BlockedLayout([2], [64], [WARPS], [0])
    if TRANS:
        ax = gl.arange(0, WM * TM * 64 * PREFETCH, layout=LD)
        rr = (pm * 16 * TM * WM + ax % (16 * TM * WM)) % M
        kk = step * 4 * PREFETCH + ax // (16 * TM * WM)
        if FULL:
            av = gl.load(A + rr * AM + kk * AK)
        elif FULL:
            av = gl.load(A + rr * AM + kk * AK, rr < M, 0)
        else:
            av = gl.load(A + rr * AM + kk * AK, kk < K, 0)
        bx = gl.arange(0, WN * TN * 64 * PREFETCH, layout=LD)
        kk = step * 4 * PREFETCH + bx % (4 * PREFETCH)
        cc = (pn * 16 * TN * WN + bx // (4 * PREFETCH)) % N
        if FULL:
            bv = gl.load(B + kk * BK + cc * BN)
        elif FULL:
            bv = gl.load(B + kk * BK + cc * BN, cc < N, 0)
        else:
            bv = gl.load(B + kk * BK + cc * BN, kk < K, 0)
        av = gl.permute(gl.reshape(av, (4 * PREFETCH, 16 * TM * WM)), (1, 0))
        bv = gl.permute(gl.reshape(bv, (16 * TN * WN, 4 * PREFETCH)), (1, 0))
    else:
        ax = gl.arange(0, WM * TM * 64 * PREFETCH, layout=LD)
        rr = (pm * 16 * TM * WM + ax // (4 * PREFETCH)) % M
        kk = step * 4 * PREFETCH + ax % (4 * PREFETCH)
        if FULL:
            av = gl.load(A + rr * AM + kk * AK)
        elif FULL:
            av = gl.load(A + rr * AM + kk * AK, rr < M, 0)
        else:
            av = gl.load(A + rr * AM + kk * AK, kk < K, 0)
        bx = gl.arange(0, WN * TN * 64 * PREFETCH, layout=LD)
        kk = step * 4 * PREFETCH + bx // (16 * TN * WN)
        cc = (pn * 16 * TN * WN + bx % (16 * TN * WN)) % N
        if FULL:
            bv = gl.load(B + kk * BK + cc * BN)
        elif FULL:
            bv = gl.load(B + kk * BK + cc * BN, cc < N, 0)
        else:
            bv = gl.load(B + kk * BK + cc * BN, kk < K, 0)
        av = gl.reshape(av, (16 * TM * WM, 4 * PREFETCH))
        bv = gl.reshape(bv, (4 * PREFETCH, 16 * TN * WN))
    SA: gl.constexpr = (
        SwizzledSharedLayout(SA_VEC, SA_PER, SA_MAX, [1, 0])
        if SWIZZLE
        else SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    SB: gl.constexpr = (
        SwizzledSharedLayout(SB_VEC, SB_PER, SB_MAX, [1, 0])
        if SWIZZLE
        else SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    LA2: gl.constexpr = load_layout(TM, TN, WARPS, WM, PREFETCH, 0)
    LB2: gl.constexpr = load_layout(TM, TN, WARPS, WM, PREFETCH, 1)
    LA: gl.constexpr = BlockedLayout(
        [1, 1, TM, 1, PREFETCH, 1],
        [1, 1, 1, 16, 1, 4],
        [WN, WM, 1, 1, 1, 1],
        [3, 5, 4, 2, 1, 0],
    )
    LB: gl.constexpr = BlockedLayout(
        [1, 1, TN, PREFETCH, 1, 1],
        [1, 1, 1, 1, 4, 16],
        [WN, WM, 1, 1, 1, 1],
        [5, 4, 3, 2, 1, 0],
    )
    sa = gl.allocate_shared_memory(
        gl.float64, (16 * TM * WM, 4 * PREFETCH), SA, value=av
    )
    sb = gl.allocate_shared_memory(
        gl.float64, (4 * PREFETCH, 16 * TN * WN), SB, value=bv
    )
    av = sa.load(LA2)
    bv = sb.load(LB2)
    av = gl.reshape(av, (1, WM, TM, 16, PREFETCH, 4))
    av = gl.convert_layout(av, LA)
    av = (
        av.to(gl.uint64, bitcast=True)
        | gl.full((WN, WM, TM, 16, PREFETCH, 4), 0, gl.uint64, LA)
    ).to(gl.float64, bitcast=True)
    av = gl.reshape(av, (WARPS, TM, 16, PREFETCH, 4))
    av = gl.permute(av, (0, 4, 2, 1, 3))
    av = gl.reshape(av, (WARPS * 64, TM * PREFETCH))
    bv = gl.permute(gl.reshape(bv, (PREFETCH, 4, WN, TN, 16)), (2, 3, 0, 1, 4))
    bv = gl.reshape(bv, (WN, 1, TN, PREFETCH, 4, 16))
    bv = gl.convert_layout(bv, LB)
    bv = (
        bv.to(gl.uint64, bitcast=True)
        | gl.full((WN, WM, TN, PREFETCH, 4, 16), 0, gl.uint64, LB)
    ).to(gl.float64, bitcast=True)
    bv = gl.reshape(bv, (WARPS, TN, PREFETCH, 64))
    bv = gl.permute(bv, (0, 3, 1, 2))
    bv = gl.reshape(bv, (WARPS * 64, TN * PREFETCH))
    ap = parts(av, WARPS, TM * PREFETCH)
    bp = parts(bv, WARPS, TN * PREFETCH)
    aa = ()
    bb = ()
    L1: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    for ki in gl.static_range(PREFETCH):
        for i in gl.static_range(TM):
            aa += (
                gl.convert_layout(
                    gl.reshape(
                        ap[rev(i * PREFETCH + ki, TM * PREFETCH)], (WARPS * 64,)
                    ),
                    L1,
                ),
            )
        for j in gl.static_range(TN):
            bb += (
                gl.convert_layout(
                    gl.reshape(
                        bp[rev(j * PREFETCH + ki, TN * PREFETCH)], (WARPS * 64,)
                    ),
                    L1,
                ),
            )
    return (aa, bb)


@g.jit
def load_tail_direct(
    A,
    B,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    pm,
    pn,
    step,
    FULL: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
    SA_VEC: gl.constexpr,
    SA_MAX: gl.constexpr,
    SB_VEC: gl.constexpr,
    SB_MAX: gl.constexpr,
    SA_PER: gl.constexpr,
    SB_PER: gl.constexpr,
):
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    x = gl.arange(0, WARPS * 64, layout=L)
    lane = x % 64
    warp = x // 64
    kk = step * 4 + lane // 16
    aa = ()
    bb = ()
    for i in gl.static_range(TM):
        rr = (pm * 16 * TM * WM + (warp % WM) * 16 * TM + i * 16 + lane % 16) % M
        aa += (gl.load(A + rr * AM + kk * AK, kk < K, 0),)
    for j in gl.static_range(TN):
        cc = (
            pn * 16 * TN * (WARPS // WM) + (warp // WM) * 16 * TN + j * 16 + lane % 16
        ) % N
        bb += (gl.load(B + kk * BK + cc * BN, kk < K, 0),)
    return aa, bb


@g.jit
def mmac_banked_tail(
    A,
    B,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    GROUP: gl.constexpr,
    PIPE: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
    SA_VEC: gl.constexpr,
    SA_MAX: gl.constexpr,
    SB_VEC: gl.constexpr,
    SB_MAX: gl.constexpr,
    SA_PER: gl.constexpr,
    SB_PER: gl.constexpr,
):
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    nm = gl.cdiv(M, 16 * TM * WM)
    nn = gl.cdiv(N, 16 * TN * (WARPS // WM))
    if GROUP == 1:
        p = gl.program_id(0) + gl.program_id(1) * nm
        pm = p // nn
        pn = p % nn
    else:
        p = gl.program_id(0) + gl.program_id(1) * nm
        group = p // (GROUP * nn)
        size = gl.minimum(nm - group * GROUP, GROUP)
        pm = group * GROUP + p % (GROUP * nn) % size
        pn = p % (GROUP * nn) // size
    acc = ()
    for i in gl.static_range(TM * TN * 4):
        acc += (gl.full((WARPS * 64,), 0, gl.float64, L),)
    steps: gl.constexpr = K // (4 * PREFETCH)
    if PIPE and steps > 0:
        aa, bb = load_banked_tail(
            A,
            B,
            M,
            N,
            K,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            pm,
            pn,
            0,
            True,
            SWIZZLE,
            TRANS,
            SA_VEC,
            SA_MAX,
            SB_VEC,
            SB_MAX,
            SA_PER,
            SB_PER,
        )
        for step in range(steps - 1):
            na, nb = load_banked_tail(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                pm,
                pn,
                step + 1,
                True,
                SWIZZLE,
                TRANS,
                SA_VEC,
                SA_MAX,
                SB_VEC,
                SB_MAX,
                SA_PER,
                SB_PER,
            )
            acc = compute(aa, bb, acc, TM, TN, PREFETCH)
            aa = na
            bb = nb
        acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    else:
        for step in range(steps):
            aa, bb = load_banked_tail(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                pm,
                pn,
                step,
                True,
                SWIZZLE,
                TRANS,
                SA_VEC,
                SA_MAX,
                SB_VEC,
                SB_MAX,
                SA_PER,
                SB_PER,
            )
            acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    if K % (4 * PREFETCH) != 0:
        if K % (4 * PREFETCH) <= 4:
            aa, bb = load_tail_direct(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                1,
                pm,
                pn,
                steps * PREFETCH,
                False,
                SWIZZLE,
                TRANS,
                SA_VEC,
                SA_MAX,
                SB_VEC,
                SB_MAX,
                SA_PER,
                SB_PER,
            )
            acc = compute(aa, bb, acc, TM, TN, 1)
        else:
            aa, bb = load_banked_tail(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                pm,
                pn,
                steps,
                False,
                SWIZZLE,
                TRANS,
                SA_VEC,
                SA_MAX,
                SB_VEC,
                SB_MAX,
                SA_PER,
                SB_PER,
            )
            acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    if TRANS:
        x = gl.arange(0, WARPS * 64, layout=L)
        lane = x % 64
        warp = x // 64
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                for z in gl.static_range(4):
                    rr = pm * 16 * TM * WM + warp % WM * 16 * TM + i * 16 + lane % 16
                    cc = (
                        pn * 16 * TN * (WARPS // WM)
                        + warp // WM * 16 * TN
                        + j * 16
                        + lane // 16
                        + z * 4
                    )
                    v = acc[(i * TN + j) * 4 + z]
                    if M % (16 * TM * WM) == 0 and N % (16 * TN * (WARPS // WM)) == 0:
                        gl.store(C + rr + cc * M, v)
                    else:
                        gl.store(C + rr + cc * M, v, (rr < M) & (cc < N))
    else:
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                store_acc(acc, C, pm, pn, M, N, TM, TN, WARPS, WM, i, j, False)
