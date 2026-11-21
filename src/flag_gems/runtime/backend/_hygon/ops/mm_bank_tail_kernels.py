"""Tail-bank configurations using the repaired native FP64 MMAC lane map.

The shared loader masks partial reduction/output panels and keeps the same
lane ordering as the accumulator and transposed stores.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

from .mm_mmac_swizzle import compute, load, store_acc


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
    # The FP64 MMAC repair defines an explicit native lane mapping.  Bank
    # specializations must use the same loader as compute/store_acc; the old
    # cooperative-remap helpers no longer exist and are not precision-safe.
    return load(
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
        FULL,
        SWIZZLE,
        TRANS,
    )


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
    # Consume each MMAC bank before loading the next; a second live bank
    # can overlap the fixed VGPR operands under the HCU compiler.
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
