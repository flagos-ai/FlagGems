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

"""TLE shared-memory BMM with the validated indexed-load ring pipeline."""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.utils import libentry

from .ring_core import _RING_CONSTRAINTS, _RING_OUTPUT_DTYPES, ring_asm

_RING_F16 = ring_asm("_bmm_ring")
_RING_BF16 = ring_asm("_bmm_ring", bf16=True)


@libentry()
@triton.jit
def ring_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    BF16: tl.constexpr,
    LOGK: tl.constexpr,
):
    t = tl.arange(0, 512)
    lane = t % 64
    wave = t // 64
    wm = wave // 2
    wn = wave % 2
    gm = tl.cdiv(M, 256)
    gn = tl.cdiv(N, 256)
    pid = tl.program_id(0)
    first = pid // (GROUP * gn) * GROUP
    gs = tl.minimum(GROUP, gm - first)
    m0 = (first + pid % (GROUP * gn) % gs) * 256
    n0 = pid % (GROUP * gn) // gs * 256
    batch = tl.program_id(1)
    shared = tle.gpu.alloc((32768,), tl.uint16, nv_mma_shared_layout=False)
    base = tle.gpu.local_ptr(shared, (0,)).to(tl.uint64).to(tl.uint32)
    lg: tl.constexpr = LOGK
    stride: tl.constexpr = 1 << lg
    ratio: tl.constexpr = K // stride
    rest: tl.constexpr = K - ratio * stride
    aptr = (A + batch * M * K + m0 * K).to(tl.uint64) - 16
    bptr = (B + batch * K * N).to(tl.uint64)
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
                offset = batch.to(tl.int64) * M * N + rr[:, None].to(tl.int64) * N + cc
                tl.store(C + offset, values, (rr[:, None] < M) & (cc < N))


def launch_ring(a, b, c, group=1):
    batch, m, k = a.shape
    n = b.shape[2]
    return ring_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256), batch)](
        a,
        b,
        c,
        m,
        n,
        k,
        group,
        a.dtype == torch.bfloat16,
        min(k.bit_length() - 1, 12),
        num_warps=8,
        num_stages=1,
        allow_flush_denorm=True,
        enable_fp_fusion=True,
    )
