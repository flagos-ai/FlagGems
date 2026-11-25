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

"""Fused gfx936 addmm with a shared-memory ring pipeline and 16x16x16 MMAC."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

from ..bmm.ring_core import _RING_CONSTRAINTS, _RING_OUTPUT_DTYPES, ring_asm

_RING_F16 = ring_asm("_addmm_ring_${:uid}")
_RING_BF16 = ring_asm("_addmm_ring_${:uid}", bf16=True)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_ring"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    use_cuda_graph=True,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_ring_kernel(
    A,
    B,
    C,
    Bias,
    alpha,
    beta,
    IM: tl.constexpr,
    IN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BF16: tl.constexpr,
    LOGK: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_M == 256 and BLOCK_SIZE_N == 256 and BLOCK_SIZE_K == 16)
    t = tl.arange(0, 512)
    lane = t % 64
    wave = t // 64
    wm = wave // 2
    wn = wave % 2
    gm = tl.cdiv(M, 256)
    gn = tl.cdiv(N, 256)
    pid = tl.program_id(0)
    first = pid // (GROUP_SIZE_M * gn) * GROUP_SIZE_M
    gs = tl.minimum(GROUP_SIZE_M, gm - first)
    m0 = (first + pid % (GROUP_SIZE_M * gn) % gs) * 256
    n0 = pid % (GROUP_SIZE_M * gn) // gs * 256
    shared = tle.gpu.alloc((32768,), tl.uint16, nv_mma_shared_layout=False)
    base = tle.gpu.local_ptr(shared, (0,)).to(tl.uint64).to(tl.uint32)
    lg: tl.constexpr = LOGK
    stride: tl.constexpr = 1 << lg
    ratio: tl.constexpr = K // stride
    rest: tl.constexpr = K - ratio * stride
    aptr = (A + m0 * K).to(tl.uint64) - 16
    bptr = B.to(tl.uint64)
    ar = ((t & 7) >> 1) + 32 * ((t & 63) >> 3) + 4 * (t >> 6)
    ak = (t & 1) * 8
    bk = (t >> 5 & 1) * 2 + (t >> 6 & 1) + 4 * (t >> 7)
    bn = (t & 31) * 8
    vb = tl.where(n0 + bn < N, (bk * N + n0 + bn) * 2, -2147483648).to(tl.uint32)
    ma = base + wave * 1024 | (wave & 3) << 16
    mb = base + 32768 + wave * 1024 | (wave & 1) << 18
    ap = (lane & 3) * 16 + ((lane & 15) >> 2) * 520 + (lane >> 4) * 4 + wm * 128
    ap2 = ap + 64
    ap2 = tl.where(ap2 >= 512 + ((lane & 15) >> 2) * 512, ap2 - 512, ap2)
    bp = (
        (lane & 3) * 8
        + (lane >> 2 & 1) * 544
        + (lane >> 3 & 1) * 256
        + (lane >> 4) * 1024
        + wn * 128
    )
    bp3 = tl.where((wn == 1) & (lane >> 2 & 3 == 3), bp - 512, bp)
    if BF16:
        asm: tl.constexpr = _RING_BF16
    else:
        asm: tl.constexpr = _RING_F16
    acc = tl.inline_asm_elementwise(
        asm,
        _RING_CONSTRAINTS,
        [
            aptr.to(tl.uint32),
            (aptr >> 32 | stride * 2 << 16 | 1073741824).to(tl.uint32),
            ((M - m0) * ratio).to(tl.uint32),
            bptr.to(tl.uint32),
            (bptr >> 32).to(tl.uint32),
            tl.full((), K * N * 2, tl.uint32),
            ma,
            mb,
            tl.full((), 32 * N, tl.uint32),
            tl.full((), K // 16, tl.uint32),
            base + ap * 2,
            base + ap2 * 2,
            base + 32768 + bp * 2,
            base + 32768 + bp3 * 2,
            wave,
            ar * ratio,
            (ar * rest + ak + 8) * 2,
            vb,
        ],
        dtype=_RING_OUTPUT_DTYPES,
        is_pure=False,
        pack=1,
    )
    row = m0 + wm * 64 + lane // 16
    col = n0 + wn * 128 + lane % 16 * 2
    pair = tl.arange(0, 2)
    for mi in tl.static_range(0, 4):
        for ni in tl.static_range(0, 4):
            for ri in tl.static_range(0, 4):
                rr = row + mi * 16 + ri * 4
                cc = col[:, None] + ni * 32 + pair[None, :]
                values = tl.join(
                    acc[mi * 32 + ni * 8 + ri], acc[mi * 32 + ni * 8 + ri + 4]
                )
                mask = (rr[:, None] < M) & (cc < N)
                if not ALPHA_ONE:
                    values = values * alpha
                if not BETA_ZERO:
                    bias = tl.load(Bias + rr[:, None] * IM + cc * IN, mask, 0).to(
                        tl.float32
                    )
                    if not BETA_ONE:
                        bias = bias * beta
                    values = values + bias
                tl.store(C + rr[:, None] * CM + cc * CN, values, mask)
