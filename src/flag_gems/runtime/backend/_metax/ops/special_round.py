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

"""special_round: elementwise round to a given number of decimals.

Matches torch.special.round(x, decimals=d) semantics (verified on target):
  - round half to even (banker's rounding) via libdevice nearbyint,
  - scaling by 10^d computed in the tensor dtype,
  - for d > 0: r = nearbyint(x * mult) / mult, mult = <tensor dtype>(10^d)
  - for d < 0: r = nearbyint(x / mult) * mult, mult = <tensor dtype>(10^(-d))
  - division uses libdevice div_rn (correctly-rounded f32 division); the
    flagtree backend rewrites plain `a / b` into a * rcp(b) (approximate),
    which breaks bit-exactness with torch's IEEE division.
  - fp16/bf16 inputs are computed in fp32 (exact promotion) and re-rounded
    through the narrow dtype after each arithmetic op to mirror torch's
    Half/BFloat16 operator semantics; libdevice nearbyint has no fp16/bf16
    variant on this backend, so rounding itself is done in fp32.

Self-contained Triton implementation; no framework fallback for the result.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _exact_div(a, b, IS_F64: tl.constexpr):
    if IS_F64:
        return a / b
    return tl.extra.cuda.libdevice.div_rn(a, b)


@triton.jit
def _narrow_roundtrip(x, IS_F16: tl.constexpr, IS_BF16: tl.constexpr):
    # Round an fp32-computed value through the input narrow dtype and back,
    # mirroring torch Half/BFloat16 arithmetic (each op rounds to the dtype).
    if IS_F16:
        return x.to(tl.float16).to(tl.float32)
    if IS_BF16:
        return x.to(tl.bfloat16).to(tl.float32)
    return x


@triton.jit
def _special_round_kernel(
    a_ptr,
    o_ptr,
    mult,
    n_elements,
    DECIMALS: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_F16: tl.constexpr,
    IS_BF16: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < n_elements
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    else:
        # n % BLOCK == 0 specialization: no bounds mask, pure streaming.
        x = tl.load(a_ptr + offs)
    if IS_F64:
        mult = mult.to(tl.float64)
    elif IS_F16 or IS_BF16:
        x = x.to(tl.float32)
    if DECIMALS > 0:
        x = _narrow_roundtrip(x * mult, IS_F16, IS_BF16)
    elif DECIMALS < 0:
        x = _narrow_roundtrip(_exact_div(x, mult, IS_F64), IS_F16, IS_BF16)
    r = tl.extra.cuda.libdevice.nearbyint(x)
    if DECIMALS > 0:
        r = _narrow_roundtrip(_exact_div(r, mult, IS_F64), IS_F16, IS_BF16)
    elif DECIMALS < 0:
        r = _narrow_roundtrip(r * mult, IS_F16, IS_BF16)
    if MASKED:
        tl.store(o_ptr + offs, r, mask=mask)
    else:
        tl.store(o_ptr + offs, r)


@triton.jit
def _special_round_strided_kernel(
    a_ptr,
    o_ptr,
    mult,
    n_elements,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    IT0,
    IT1,
    IT2,
    IT3,
    IT4,
    IT5,
    IT6,
    IT7,
    OT0,
    OT1,
    OT2,
    OT3,
    OT4,
    OT5,
    OT6,
    OT7,
    RANK: tl.constexpr,
    DECIMALS: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_F16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sizes = (S0, S1, S2, S3, S4, S5, S6, S7)
    istr = (IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7)
    ostr = (OT0, OT1, OT2, OT3, OT4, OT5, OT6, OT7)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    rem = offs
    a_off = tl.zeros_like(offs)
    o_off = tl.zeros_like(offs)
    for k in tl.static_range(RANK):
        d = rem % sizes[k]
        rem = rem // sizes[k]
        a_off += d * istr[k]
        o_off += d * ostr[k]
    x = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    if IS_F64:
        mult = mult.to(tl.float64)
    elif IS_F16 or IS_BF16:
        x = x.to(tl.float32)
    if DECIMALS > 0:
        x = _narrow_roundtrip(x * mult, IS_F16, IS_BF16)
    elif DECIMALS < 0:
        x = _narrow_roundtrip(_exact_div(x, mult, IS_F64), IS_F16, IS_BF16)
    r = tl.extra.cuda.libdevice.nearbyint(x)
    if DECIMALS > 0:
        r = _narrow_roundtrip(_exact_div(r, mult, IS_F64), IS_F16, IS_BF16)
    elif DECIMALS < 0:
        r = _narrow_roundtrip(r * mult, IS_F16, IS_BF16)
    tl.store(o_ptr + o_off, r, mask=mask)


_BLOCK_F32 = 2048
_BLOCK_NARROW = 4096
_BLOCK_SMALL = 1024
_NUM_WARPS = 4
_NUM_STAGES = 1
_SMALL_N = 1 << 15
_MID_N = 1 << 25


def special_round(A, *, decimals=0):
    if isinstance(decimals, torch.Tensor):
        decimals = int(decimals.item())
    else:
        decimals = int(decimals)

    out = torch.empty_like(A)
    n = A.numel()
    if n == 0:
        return out

    if decimals == 0:
        mult = 1.0
    else:
        p = decimals if decimals > 0 else -decimals
        mult = float(torch.tensor(10.0**p, dtype=A.dtype).item())

    is_f64 = A.dtype == torch.float64
    is_f16 = A.dtype == torch.float16
    is_bf16 = A.dtype == torch.bfloat16

    if A.is_contiguous():
        # Block size scales with problem size and dtype:
        #   - tiny tensors: small blocks so many short threads cover the whole
        #     tensor (large blocks serialize long per-thread load->round->store
        #     chains and lose latency hiding);
        #   - fp16/bf16 need >=8 elems/thread (BLOCK>=2048 at 4 warps) for
        #     128-bit memory transactions; BLOCK=4096 is best at 2^30 scale;
        #   - fp32 reaches 128-bit at 4 elems/thread; BLOCK=4096 wins on the
        #     2^30 workload while BLOCK=2048 wins at mid scale.
        if n < _SMALL_N:
            # Tiny tensors: few elems/thread wins (round-2 evidence: 8-16
            # elems/thread regressed tiny workloads to 6.9-7.9us; round 5:
            # 2/thread put bf16 on the 6.144us floor; round 6: 1/thread put
            # fp32 BELOW the floor at 5.888us). f64 keeps 4/thread.
            if is_f64:
                block = _BLOCK_SMALL
            else:
                block = 256
        elif n < _MID_N:
            block = _BLOCK_F32
        else:
            block = _BLOCK_NARROW
        masked = (n % block) != 0
        grid = (triton.cdiv(n, block),)
        _special_round_kernel[grid](
            A,
            out,
            mult,
            n,
            DECIMALS=decimals,
            IS_F64=is_f64,
            IS_F16=is_f16,
            IS_BF16=is_bf16,
            MASKED=masked,
            BLOCK=block,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
    else:
        rank = A.dim()
        assert rank <= 8, f"unsupported rank {rank}"
        sz = list(A.shape) + [1] * (8 - rank)
        ist = list(A.stride()) + [0] * (8 - rank)
        ost = list(out.stride()) + [0] * (8 - rank)
        BLOCK = 512
        grid = (triton.cdiv(n, BLOCK),)
        _special_round_strided_kernel[grid](
            A,
            out,
            mult,
            n,
            sz[0],
            sz[1],
            sz[2],
            sz[3],
            sz[4],
            sz[5],
            sz[6],
            sz[7],
            ist[0],
            ist[1],
            ist[2],
            ist[3],
            ist[4],
            ist[5],
            ist[6],
            ist[7],
            ost[0],
            ost[1],
            ost[2],
            ost[3],
            ost[4],
            ost[5],
            ost[6],
            ost[7],
            RANK=rank,
            DECIMALS=decimals,
            IS_F64=is_f64,
            IS_F16=is_f16,
            IS_BF16=is_bf16,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out
