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

"""FP64 MMAC with explicit native lane operands and transposed computation.

Load each reduction bank in the lane mapping consumed by MMAC and finish its
accumulation before advancing the bank. This avoids compiler-dependent FP64
cooperative remaps and overlapping fixed-register operand banks.
"""

import triton
import triton.experimental.gluon.language.amd._layouts as amd_layouts
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

# The pinned HCU frontend asks for these names during blocked-layout inference.
# Some HCU wheels expose the equivalent layouts only under their AMD names.
if not hasattr(amd_layouts, "HCUMFMALayout"):
    amd_layouts.HCUMFMALayout = amd_layouts.AMDMFMALayout
if not hasattr(amd_layouts, "HCUWMMALayout"):
    amd_layouts.HCUWMMALayout = amd_layouts.AMDWMMALayout


@g.jit
def unpack64(x):
    bits = x.to(gl.uint64, bitcast=True)
    return bits.to(gl.uint32), (bits >> 32).to(gl.uint32)


@g.jit
def pack64(lo, hi):
    return (lo.to(gl.uint64) | (hi.to(gl.uint64) << 32)).to(gl.float64, bitcast=True)


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


@g.jit
def mmac(
    a,
    b,
    c0,
    c1,
    c2,
    c3,
    SLOT: gl.constexpr = 0,
    TOTAL: gl.constexpr = 1,
    NT: gl.constexpr = 1,
    KI: gl.constexpr = 0,
):
    al, ah = unpack64(a)
    bl, bh = unpack64(b)
    l0, h0 = unpack64(c0)
    l1, h1 = unpack64(c1)
    l2, h2 = unpack64(c2)
    l3, h3 = unpack64(c3)
    o0, o1, o2, o3, o4, o5, o6, o7 = gl.inline_asm_elementwise(
        asm=slot_asm(SLOT, TOTAL, NT, KI),
        constraints=slot_constraints(SLOT, TOTAL, NT, KI),
        args=[al, ah, bl, bh, l0, h0, l1, h1, l2, h2, l3, h3],
        dtype=(
            gl.uint32,
            gl.uint32,
            gl.uint32,
            gl.uint32,
            gl.uint32,
            gl.uint32,
            gl.uint32,
            gl.uint32,
        ),
        is_pure=True,
        pack=1,
    )
    return pack64(o0, o1), pack64(o2, o3), pack64(o4, o5), pack64(o6, o7)


@g.jit
def consume(
    aa,
    bb,
    acc,
    ki: gl.constexpr,
    i: gl.constexpr,
    j: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
):
    off: gl.constexpr = (i * TN + j) * 4
    c0, c1, c2, c3 = mmac(
        aa[ki * TM + i],
        bb[ki * TN + j],
        acc[off],
        acc[off + 1],
        acc[off + 2],
        acc[off + 3],
        SLOT=i * TN + j,
        TOTAL=TM * TN,
        NT=TN,
        KI=ki,
    )
    return acc[:off] + (c0, c1, c2, c3) + acc[off + 4 :]


@g.jit
def store_acc(
    acc,
    C,
    pm,
    pn,
    M: gl.constexpr,
    N: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    I: gl.constexpr,
    J: gl.constexpr,
    TRANS: gl.constexpr,
):
    off: gl.constexpr = (I * TN + J) * 4
    value = gl.join(
        gl.join(acc[off], acc[off + 2]), gl.join(acc[off + 1], acc[off + 3])
    )
    value = gl.reshape(value, (WARPS, 4, 16, 4))
    value = gl.permute(value, (0, 2, 3, 1))
    value = gl.reshape(value, (WARPS, 16, 16))
    if TRANS:
        value = gl.permute(value, (0, 2, 1))
    value = gl.reshape(value, (WARPS * 256,))
    L: gl.constexpr = BlockedLayout([2], [64], [WARPS], [0])
    value = gl.convert_layout(value, L)
    x = gl.arange(0, WARPS * 256, layout=L)
    w = x // 256
    rr = (
        pm * 16 * TM * WM
        + (w % WM) * 16 * TM
        + I * 16
        + (x % 16 if TRANS else x % 256 // 16)
    )
    cc = (
        pn * 16 * TN * (WARPS // WM)
        + (w // WM) * 16 * TN
        + J * 16
        + (x % 256 // 16 if TRANS else x % 16)
    )
    offset = cc * M + rr if TRANS else rr * N + cc
    gl.store(C + offset, value, (rr < M) & (cc < N))


@g.jit
def load(
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
):
    # MMAC consumes row/column lane % 16 and reduction lane // 16.
    # Keep that mapping explicit across warp replication and FP64 bit packing.
    x = gl.arange(0, WARPS * 64, layout=BlockedLayout([1], [64], [WARPS], [0]))
    lane = x % 64
    warp = x // 64
    aa = ()
    bb = ()
    for ki in gl.static_range(PREFETCH):
        kk = step * 4 * PREFETCH + ki * 4 + lane // 16
        for i in gl.static_range(TM):
            rr = pm * 16 * TM * WM + (warp % WM) * 16 * TM + i * 16 + lane % 16
            aa += (gl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0),)
        for j in gl.static_range(TN):
            cc = (
                pn * 16 * TN * (WARPS // WM)
                + (warp // WM) * 16 * TN
                + j * 16
                + lane % 16
            )
            bb += (gl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0),)
    return aa, bb


@g.jit
def compute(aa, bb, acc, TM: gl.constexpr, TN: gl.constexpr, PREFETCH: gl.constexpr):
    for ki in gl.static_range(PREFETCH):
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                acc = consume(aa, bb, acc, ki, i, j, TM, TN)
    return acc


@g.jit
def mmac_tiled(
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
):
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    nm = gl.cdiv(M, 16 * TM * WM)
    nn = gl.cdiv(N, 16 * TN * (WARPS // WM))
    p = gl.program_id(0) + gl.program_id(1) * nm
    group = p // (GROUP * nn)
    size = gl.minimum(nm - group * GROUP, GROUP)
    pm = group * GROUP + (p % (GROUP * nn)) % size
    pn = (p % (GROUP * nn)) // size
    acc = ()
    for i in gl.static_range(TM * TN * 4):
        acc += (gl.full((WARPS * 64,), 0, gl.float64, L),)
    steps: gl.constexpr = K // (4 * PREFETCH)
    # A second live prefetch bank can overlap the fixed VGPR MMAC operands
    # under the HCU compiler. Load and consume each bank before advancing K.
    for step in range(steps):
        aa, bb = load(
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
        )
        acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    if K % (4 * PREFETCH) != 0:
        aa, bb = load(
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
        )
        acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    if TRANS:
        x = gl.arange(0, WARPS * 64, layout=L)
        lane = x % 64
        warp = x // 64
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                for z in gl.static_range(4):
                    rr = pm * 16 * TM * WM + (warp % WM) * 16 * TM + i * 16 + lane % 16
                    cc = (
                        pn * 16 * TN * (WARPS // WM)
                        + (warp // WM) * 16 * TN
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
