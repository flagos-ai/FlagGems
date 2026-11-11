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

import triton
import triton.language as tl


@triton.jit
def unpack64(x):
    bits = x.to(tl.uint64, bitcast=True)
    return bits.to(tl.uint32), (bits >> 32).to(tl.uint32)


@triton.jit
def pack64(lo, hi):
    return (lo.to(tl.uint64) | (hi.to(tl.uint64) << 32)).to(tl.float64, bitcast=True)


@triton.constexpr_function
def slot_asm(slot, total, tn, ki):
    start = slot * 8
    tmp = total * 8 + ki * 2 * (total // tn + tn)
    areg = tmp + 2 * (slot // tn)
    breg = tmp + 2 * (total // tn) + 2 * (slot % tn)
    return (
        f"v_mmac_16x16x4_f64 v[{start}:{start + 7}], "
        f"v[{areg}:{areg + 1}], v[{breg}:{breg + 1}], v[{start}:{start + 7}]"
    )


@triton.constexpr_function
def slot_constraints(slot, total, tn, ki):
    start = slot * 8
    tmp = total * 8 + ki * 2 * (total // tn + tn)
    areg = tmp + 2 * (slot // tn)
    breg = tmp + 2 * (total // tn) + 2 * (slot % tn)
    return ",".join(
        [f"={{v{start + i}}}" for i in range(8)]
        + [f"{{v{reg}}}" for reg in (areg, areg + 1, breg, breg + 1)]
        + [str(i) for i in range(8)]
    )


@triton.jit
def mmac(
    a,
    b,
    c0,
    c1,
    c2,
    c3,
    SLOT: tl.constexpr = 0,
    TOTAL: tl.constexpr = 1,
    NT: tl.constexpr = 1,
    KI: tl.constexpr = 0,
):
    al, ah = unpack64(a)
    bl, bh = unpack64(b)
    l0, h0 = unpack64(c0)
    l1, h1 = unpack64(c1)
    l2, h2 = unpack64(c2)
    l3, h3 = unpack64(c3)
    o0, o1, o2, o3, o4, o5, o6, o7 = tl.inline_asm_elementwise(
        asm=slot_asm(SLOT, TOTAL, NT, KI),
        constraints=slot_constraints(SLOT, TOTAL, NT, KI),
        args=[al, ah, bl, bh, l0, h0, l1, h1, l2, h2, l3, h3],
        dtype=(
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
        ),
        is_pure=True,
        pack=1,
    )
    return pack64(o0, o1), pack64(o2, o3), pack64(o4, o5), pack64(o6, o7)


@triton.jit
def mmac_tiled(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    SPLIT: tl.constexpr,
    PREFETCH: tl.constexpr,
    GROUP: tl.constexpr,
):
    tl.static_assert(SPLIT == 1)
    nm = tl.cdiv(M, 16 * TM * WM)
    nn = tl.cdiv(N, 16 * TN * (WARPS // WM))
    p = tl.program_id(0) + tl.program_id(1) * nm
    g = p // (GROUP * nn)
    gs = tl.minimum(nm - g * GROUP, GROUP)
    pm = g * GROUP + (p % (GROUP * nn)) % gs
    pn = (p % (GROUP * nn)) // gs
    acc = ()
    for i in tl.static_range(TM * TN * 4):
        acc += (tl.full((WARPS * 64,), 0, tl.float64),)
    chunk = tl.cdiv(tl.cdiv(K, 4 * PREFETCH), SPLIT) * 4 * PREFETCH
    for step in range(tl.cdiv(chunk, 4 * PREFETCH)):
        aa, bb = cooperative_load(
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
            chunk,
            step,
        )
        for ki in tl.static_range(PREFETCH):
            for i in tl.static_range(TM):
                for j in tl.static_range(TN):
                    acc = consume_acc(aa, bb, acc, ki, i, j, TM, TN)
    for i in tl.static_range(TM):
        for j in tl.static_range(TN):
            store_acc(acc, C, pm, pn, M, N, WARPS, WM, TM, TN, i, j)


@triton.jit
def consume_acc(
    aa,
    bb,
    acc,
    KI: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    off: tl.constexpr = (TILE_M * TN + TILE_N) * 4
    ai: tl.constexpr = KI * TM + TILE_M
    bi: tl.constexpr = KI * TN + TILE_N
    c0, c1, c2, c3 = mmac(
        aa[ai],
        bb[bi],
        acc[off],
        acc[off + 1],
        acc[off + 2],
        acc[off + 3],
        SLOT=TILE_M * TN + TILE_N,
        TOTAL=TM * TN,
        NT=TN,
        KI=KI,
    )
    return acc[:off] + (c0, c1, c2, c3) + acc[off + 4 :]


@triton.jit
def store_acc(
    acc,
    C,
    pm,
    pn,
    M: tl.constexpr,
    N: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TI: tl.constexpr,
    TJ: tl.constexpr,
):
    off: tl.constexpr = (TI * TN + TJ) * 4
    vals = tl.interleave(
        tl.interleave(acc[off], acc[off + 2]), tl.interleave(acc[off + 1], acc[off + 3])
    )
    x = tl.arange(0, WARPS * 256)
    warp = x // 256
    row = x % 256 // 16
    col = x % 16
    value = tl.reshape(
        tl.trans(tl.reshape(vals, (WARPS, 4, 16, 4)), (0, 2, 3, 1)), (WARPS * 256,)
    )
    rr = pm * (16 * TM * WM) + (warp % WM) * 16 * TM + TI * 16 + row
    cc = pn * (16 * TN * (WARPS // WM)) + (warp // WM) * 16 * TN + TJ * 16 + col
    if M % (16 * TM * WM) == 0 and N % (16 * TN * (WARPS // WM)) == 0:
        tl.store(C + rr * N + cc, value)
    else:
        tl.store(C + rr * N + cc, value, (rr < M) & (cc < N))


@triton.jit
def cooperative_load(
    A,
    B,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    PREFETCH: tl.constexpr,
    pm,
    pn,
    chunk: tl.constexpr,
    step,
):
    # MMAC consumes row/column lane % 16 and reduction lane // 16.
    # Keep that mapping explicit across warp replication and FP64 bit packing.
    x = tl.arange(0, WARPS * 64)
    lane = x % 64
    warp = x // 64
    aa = ()
    bb = ()
    for ki in tl.static_range(PREFETCH):
        kk = step * 4 * PREFETCH + ki * 4 + lane // 16
        for i in tl.static_range(TM):
            rr = pm * 16 * TM * WM + (warp % WM) * 16 * TM + i * 16 + lane % 16
            aa += (tl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0),)
        for j in tl.static_range(TN):
            cc = (
                pn * 16 * TN * (WARPS // WM)
                + (warp // WM) * 16 * TN
                + j * 16
                + lane % 16
            )
            bb += (tl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0),)
    return aa, bb
