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
def slot_asm(slot, total, tn):
    start = slot * 8
    tmp = total * 8
    areg = tmp + 2 * (slot // tn)
    breg = tmp + 2 * (total // tn) + 2 * (slot % tn)
    return f"v_mmac_16x16x4_f64 v[{start}:{start+7}], v[{areg}:{areg+1}], v[{breg}:{breg+1}], v[{start}:{start+7}]"


@triton.constexpr_function
def slot_constraints(slot, total, tn):
    start = slot * 8
    tmp = total * 8
    areg = tmp + 2 * (slot // tn)
    breg = tmp + 2 * (total // tn) + 2 * (slot % tn)
    return ",".join(
        [f"={{v{start+i}}}" for i in range(8)]
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
):
    al, ah = unpack64(a)
    bl, bh = unpack64(b)
    l0, h0 = unpack64(c0)
    l1, h1 = unpack64(c1)
    l2, h2 = unpack64(c2)
    l3, h3 = unpack64(c3)
    o0, o1, o2, o3, o4, o5, o6, o7 = tl.inline_asm_elementwise(
        asm=slot_asm(SLOT, TOTAL, NT),
        constraints=slot_constraints(SLOT, TOTAL, NT),
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
    UNROLL: tl.constexpr,
):
    ids = tl.arange(0, WARPS * 64)
    lane = ids % 64
    warp = ids // 64
    rbase = tl.program_id(0) * (16 * TM * WM) + (warp % WM) * 16 * TM
    nbase = tl.program_id(1) * (16 * TN * (WARPS // WM)) + (warp // WM) * 16 * TN
    acc = ()
    for i in tl.static_range(TM * TN * 4):
        acc += (tl.full((WARPS * 64,), 0, tl.float64),)
    chunk = tl.cdiv(tl.cdiv(K, 4), SPLIT) * 4
    for step in tl.range(tl.cdiv(chunk, 4), loop_unroll_factor=UNROLL):
        kk = tl.program_id(2) * chunk + step * 4 + lane // 16
        aa = ()
        bb = ()
        for i in tl.static_range(TM):
            rr = rbase + i * 16 + lane % 16
            aa += (tl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0),)
        for j in tl.static_range(TN):
            cc = nbase + j * 16 + lane % 16
            bb += (tl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0),)
        for i in tl.static_range(TM):
            for j in tl.static_range(TN):
                acc = update_acc(aa[i], bb[j], acc, i, j, TN, TM)
    for i in tl.static_range(TM):
        for j in tl.static_range(TN):
            for z in tl.static_range(4):
                rr = rbase + i * 16 + lane % 16
                cc = nbase + j * 16 + lane // 16 + z * 4
                tl.store(
                    C + tl.program_id(2) * M * N + rr * N + cc,
                    acc[(i * TN + j) * 4 + z],
                    (rr < M) & (cc < N),
                )


@triton.jit
def update_acc(
    a,
    b,
    acc,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TN: tl.constexpr,
    TM: tl.constexpr,
):
    off: tl.constexpr = (TILE_M * TN + TILE_N) * 4
    c0, c1, c2, c3 = mmac(
        a,
        b,
        acc[off],
        acc[off + 1],
        acc[off + 2],
        acc[off + 3],
        SLOT=TILE_M * TN + TILE_N,
        TOTAL=TM * TN,
        NT=TN,
    )
    return acc[:off] + (c0, c1, c2, c3) + acc[off + 4 :]
