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

"""Layer 2 candidate: basic INT8 QK, TLE BF16 matrix API (SQMMA on MUSA).

Installed MUSA adapter requires shared A/B, FP32 accumulator, pending=0.
The API is named wgmma but this source does not emit NVIDIA instructions.
Only tested target backends are documented; no universal compatibility claim.
"""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.fused.sage_attention_math import probability_sum


@triton.jit(do_not_specialize=["NQ", "NK"])
def _tle_matrix_attention(
    Q,
    K,
    V,
    QS,
    KS,
    O,
    H: tl.constexpr,
    NQ,
    NK,
    NQS: tl.constexpr,
    NKS: tl.constexpr,
    VZ: tl.constexpr,
    VN: tl.constexpr,
    VH: tl.constexpr,
    SM: tl.constexpr,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 128,
    D: tl.constexpr = 128,
    PRECISE_DIV: tl.constexpr = False,
    SUM_MODE: tl.constexpr = 0,
    ORDERED_PV: tl.constexpr = False,
    PV_BK: tl.constexpr = 8,
):
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + (bh * NQ + m[:, None]) * D + d[None, :], m[:, None] < NQ, 0)
    qs = tl.load(QS + bh * NQS + m // 16 * 8 + m % 8, m < NQ, 0.0)
    # Installed TLE copy's masked global-to-shared contract writes zero to all
    # false-mask destinations; no uninitialized tail is consumed by PV.
    if not ORDERED_PV:
        pb = tle.gpu.alloc((BM, BN), tl.bfloat16)
        vb = tle.gpu.alloc((BN, D), tl.bfloat16)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    sm = tl.full((), SM, tl.float32) * 1.4426950408889634
    for block in range(tl.cdiv(NK, BN)):
        kn = block * BN + n
        k = tl.load(K + (bh * NK + kn[None, :]) * D + d[:, None], kn[None, :] < NK, 0)
        if not ORDERED_PV:
            tle.gpu.copy(
                V + b * VZ + kn[:, None] * VN + h * VH + d[None, :],
                vb,
                [BN, D],
                mask=kn[:, None] < NK,
            )
        ks = tl.load(KS + bh * NKS + kn // 128 * 4 + (kn % 8) // 2, kn < NK, 0.0)
        raw = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        scores = (raw * (qs[:, None] * ks[None, :])) * sm
        scores = tl.where(kn[None, :] < NK, scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_maximum)
        probability = tl.exp2(scores - new_maximum[:, None])
        denominator = tl.fma(
            denominator, alpha, probability_sum(probability, BM, BN, SUM_MODE)
        )
        if ORDERED_PV:
            temporary = tl.zeros((BM, D), tl.float32)
            sn = tl.arange(0, PV_BK)
            for sub in range(tl.cdiv(tl.minimum(BN, NK - block * BN), PV_BK)):
                pn = sub * PV_BK + sn
                pp = tl.gather(
                    probability.to(tl.bfloat16),
                    tl.broadcast_to(pn[None, :], (BM, PV_BK)),
                    1,
                )
                vn = block * BN + pn
                vv = tl.load(
                    V + b * VZ + vn[:, None] * VN + h * VH + d[None, :],
                    vn[:, None] < NK,
                    0,
                )
                temporary = tl.dot(
                    pp.to(tl.float32),
                    vv.to(tl.float32),
                    temporary,
                    input_precision="ieee",
                )
        else:
            pb.store(probability.to(tl.bfloat16))
            temporary = tle.gpu.wgmma(pb, vb, out_dtype=tl.float32)
            temporary = tle.gpu.wgmma_wait(0, temporary)
        accumulator = tl.fma(accumulator, alpha[:, None], temporary)
        maximum = new_maximum
    if PRECISE_DIV:
        value = tl.div_rn(accumulator, denominator[:, None])
    else:
        value = accumulator / denominator[:, None]
    tl.store(
        O + ((b * NQ + m[:, None]) * H + h) * D + d[None, :], value, m[:, None] < NQ
    )


def core(
    z,
    output,
    sm_scale=None,
    bm=64,
    warps=4,
    stages=1,
    precise_div=False,
    sum_mode=0,
    ordered_pv=False,
    pv_bk=8,
):
    qi, qs, ki, ks, v, _ = z
    b, h, nq, d = qi.shape
    return _tle_matrix_attention[(triton.cdiv(nq, bm), b * h)](
        qi,
        ki,
        v,
        qs,
        ks,
        output,
        h,
        nq,
        ki.shape[2],
        qs.shape[-1],
        ks.shape[-1],
        *v.stride()[:3],
        d**-0.5 if sm_scale is None else sm_scale,
        BM=bm,
        BN=128,
        D=d,
        PRECISE_DIV=precise_div,
        SUM_MODE=sum_mode,
        ORDERED_PV=ordered_pv,
        PV_BK=pv_bk,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
