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

"""Exact bounded FP4-to-FP8 math for SM90 (Gluon, Triton 3.7.1).

Callers retain storage addressing, grouping, output allocation and fallback.
Eligibility must be recomputed for every replay. This module is imported lazily
only by the matched Hopper implementation; it changes no FP4 storage format.
"""

import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma

FP4_FP8_SCORE_ABI = 2


@triton.jit
def prepare_fp4_query(Q, Q8E, QS, OK):
    rg = tl.program_id(0) * 128 + tl.arange(0, 128)
    head = rg // 4
    block = rg % 4
    k = tl.arange(0, 32)
    q = tl.load(Q + head[:, None] * 128 + block[:, None] * 32 + k[None, :]).to(
        tl.float32
    )
    mx = tl.max(tl.abs(q), 1)
    exp = ((mx.to(tl.int32, bitcast=True) >> 23) & 255) - 127 - 7
    exp = tl.where(mx == 0, 0, tl.maximum(tl.minimum(exp, 120), -120))
    scale = tl.exp2(exp.to(tl.float32))
    norm = q * tl.exp2(-exp.to(tl.float32))[:, None]
    packed = norm.to(tl.float8e4nv)
    # Multiples of 16 bound a 32-product FP8 partial to at most 13 bits.
    units = norm / 16.0
    exact = tl.sum((units != units.to(tl.int32).to(tl.float32)).to(tl.int32), 1) == 0
    e = tl.reshape(exp, (32, 4))
    ok = tl.reshape(exact & (exp >= -32) & (exp <= 32), (32, 4))
    okhead = (tl.sum(ok.to(tl.int32), 1) == 4) & (tl.max(e, 1) - tl.min(e, 1) <= 4)
    tl.store(Q8E + head[:, None] * 128 + block[:, None] * 32 + k[None, :], packed)
    tl.store(QS + rg, scale)
    tl.store(OK + tl.program_id(0) * 32 + tl.arange(0, 32), okhead.to(tl.int32))


@g.jit
def e2m1x4_to_fp8(packed_words):
    """Map four packed nibbles to exact E4M3 bytes, canonicalizing -0."""
    return gl.inline_asm_elementwise(
        """{
                    .reg .b32 idx,mag,sgn,nz,tmp;
                    and.b32 idx,$1,0x7777;
                    prmt.b32 mag,0x3c383000,0x4c484440,idx;
                    and.b32 idx,$1,0x8888;
                    shr.u32 idx,idx,1;
                    prmt.b32 sgn,0,0x80808080,idx;
                    shr.u32 nz,mag,4;
                    shr.u32 tmp,mag,6;
                    or.b32 nz,nz,tmp;
                    and.b32 nz,nz,0x01010101;
                    shl.b32 nz,nz,7;
                    and.b32 sgn,sgn,nz;
                    or.b32 $0,mag,sgn;
                    }""",
        constraints="=r,r,r,r,r",
        args=[packed_words],
        dtype=gl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@g.jit
def block_scaled_fp8_dot(
    k_shared,
    q_shared,
    key_scale,
    query_scale,
    shape: gl.constexpr,
    layout: gl.constexpr,
):
    """One 32-element dot under the bounded SM90 contract.

    Query values must be the multiples-of-16 values accepted by prepare_fp4_query;
    K values must be exact unscaled E2M1. Across the four blocks, the caller
    checks K scale exponent span <= 4 and encoded exponents in [64, 191].
    Both conditions are runtime guards, not assumptions about captured data.
    """
    part = warpgroup_mma(
        k_shared,
        q_shared,
        gl.full(shape, 0, gl.float32, layout),
        use_acc=False,
        max_num_imprecise_acc=32,
    )
    return part * (key_scale[:, None] * query_scale[None, :])


@g.jit
def fp8_weighted_head_score(acc, weights, block_l: gl.constexpr):
    """Preserve BF16 dot, ReLU/product and 32-head sum rounding."""
    score = gl.maximum(acc.to(gl.bfloat16).to(gl.float32), 0.0)
    score = (score * weights[None, :]).to(gl.bfloat16).to(gl.float32)
    return (
        gl.sum(gl.reshape(score, (block_l, 2, 32)), 2)
        .permute((1, 0))
        .to(gl.bfloat16)
        .to(gl.float32)
    )


@g.jit
def normal_ue8m0_scale(encoded):
    """Exact normal powers of two for UE8M0 encodings 1..254.

    The caller checks the stricter native key range [64, 191] before calling.
    Encoding 0 (a float32 subnormal scale) and 255 are outside this helper ABI.
    """
    return (encoded.to(gl.int32) << 23).to(gl.float32, bitcast=True)
