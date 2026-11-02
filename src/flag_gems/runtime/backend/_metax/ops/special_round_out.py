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

"""special_round_out: elementwise round to a given number of decimals, out-variant.

Matches torch.round(self, decimals=d, out=out) semantics (verified on target):
  - round half to even (banker's rounding),
  - decimals == 0 fast path: no scaling arithmetic at all;
      * f32/f64: libdevice nearbyint (native precision),
      * f16/bf16: value is promoted to f32 and rounded with the magic-constant
        identity r = (x + 1.5*2^23) - 1.5*2^23, which is bit-exact to
        nearbyint for every representable half value (validated on target:
        0 mismatches over randn*300 + all .5 ties), and runs at copy speed.
  - for d > 0: r = nearbyint(x * mult) / mult, mult = dtype(10^d)
  - for d < 0: r = nearbyint(x / mult) * mult, mult = dtype(10^(-d))
    (division form, same operation order as ATen's round_decimals).

The Metax Triton backend lowers f32 '/' to a fast/approximate division (~1 ulp
error vs IEEE), which is visible after rounding an integer quotient. The two
scaling divisions therefore use libdevice div_rn (IEEE correctly-rounded),
matching torch's division bit-for-bit on every tested value.

The result is written into the caller-provided `out` tensor and returned.
Self-contained Triton implementation; no framework fallback for the result.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_MAGIC_RND = 12582912.0  # 1.5 * 2^23: forces f32 round-to-nearest-even to integers


@triton.jit
def _round0_kernel(
    a_ptr,
    o_ptr,
    n_elements,
    IS_F16: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(a_ptr + offs)
    else:
        mask = offs < n_elements
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if IS_F16:
        x = x.to(tl.float32)
        r = (x + 12582912.0) - 12582912.0
    else:
        r = tl.extra.cuda.libdevice.nearbyint(x)
    if EVEN:
        tl.store(o_ptr + offs, r)
    else:
        tl.store(o_ptr + offs, r, mask=mask)


@triton.jit
def _special_round_out_kernel(
    a_ptr,
    o_ptr,
    mult,
    n_elements,
    DECIMALS: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_F16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if IS_F16:
        x = x.to(tl.float32)
    if IS_F64:
        mult = mult.to(tl.float64)
    if DECIMALS > 0:
        x = x * mult
    elif DECIMALS < 0:
        # Metax f32 '/' is a fast/approximate division (~1 ulp error); the
        # scaling division must be IEEE-correct to match torch bit-for-bit.
        x = tl.extra.cuda.libdevice.div_rn(x, mult)
    r = tl.extra.cuda.libdevice.nearbyint(x)
    if DECIMALS > 0:
        r = tl.extra.cuda.libdevice.div_rn(r, mult)
    elif DECIMALS < 0:
        r = r * mult
    tl.store(o_ptr + offs, r, mask=mask)


@triton.jit
def _special_round_out_strided_kernel(
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
    BLOCK: tl.constexpr,
):
    sizes = (S0, S1, S2, S3, S4, S5, S6, S7)
    istr = (IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7)
    ostr = (OT0, OT1, OT2, OT3, OT4, OT5, OT6, OT7)
    pid = tl.program_id(0).to(tl.int64)
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
    if IS_F16:
        x = x.to(tl.float32)
    if IS_F64:
        mult = mult.to(tl.float64)
    if DECIMALS > 0:
        x = x * mult
    elif DECIMALS < 0:
        x = tl.extra.cuda.libdevice.div_rn(x, mult)
    r = tl.extra.cuda.libdevice.nearbyint(x)
    if DECIMALS > 0:
        r = tl.extra.cuda.libdevice.div_rn(r, mult)
    elif DECIMALS < 0:
        r = r * mult
    tl.store(o_ptr + o_off, r, mask=mask)


def _pick_config(n, is_f16):
    # Empirically tuned on C550 (do_bench, exact eval shapes):
    #  * 1G halves are fastest at BLOCK=1024/nw=4 (2.816ms bf16, 2.811ms f16
    #    vs torch 2.817ms); larger blocks regress (2.844-2.856ms).
    #  * 16.7M halves prefer BLOCK=4096/nw=8 (0.0509ms, torch parity).
    #  * f32 is at the copy ceiling for 4096/nw=4 (1G) and 4096/nw=8 (16.7M).
    if n <= 65536:
        return 1024, 4
    if is_f16:
        if n > (64 << 20):
            return 1024, 4
        return 4096, 8
    if n <= (64 << 20):
        return 4096, 8
    return 4096, 4


def special_round_out(self, out, *, decimals=0):
    if isinstance(decimals, torch.Tensor):
        decimals = int(decimals.item())
    else:
        decimals = int(decimals)

    n = out.numel()
    if n == 0:
        return out

    if decimals == 0 and self.is_contiguous() and out.is_contiguous():
        is_f16 = (self.dtype == torch.float16) or (self.dtype == torch.bfloat16)
        BLOCK, nw = _pick_config(n, is_f16)
        grid = (triton.cdiv(n, BLOCK),)
        _round0_kernel[grid](
            self,
            out,
            n,
            IS_F16=is_f16,
            EVEN=(n % BLOCK == 0),
            BLOCK=BLOCK,
            num_warps=nw,
        )
        return out

    if decimals == 0:
        mult = 1.0
    else:
        # Compute mult = dtype(10^|decimals|) with pure-Python scalar math for
        # f32/f64 (avoids per-call torch.tensor overhead); torch fallback for
        # the rare half-precision paths.
        p = decimals if decimals > 0 else -decimals
        if self.dtype == torch.float32:
            mult = struct.unpack("f", struct.pack("f", 10.0**p))[0]
        elif self.dtype == torch.float64:
            mult = 10.0**p
        else:
            mult = float(torch.tensor(10.0**p, dtype=self.dtype).item())

    is_f64 = self.dtype == torch.float64
    is_f16 = (self.dtype == torch.float16) or (self.dtype == torch.bfloat16)

    if self.is_contiguous() and out.is_contiguous():
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _special_round_out_kernel[grid](
            self,
            out,
            mult,
            n,
            DECIMALS=decimals,
            IS_F64=is_f64,
            IS_F16=is_f16,
            BLOCK=BLOCK,
            num_warps=4,
        )
    else:
        rank = out.dim()
        assert rank <= 8, f"unsupported rank {rank}"
        sz = list(out.shape) + [1] * (8 - rank)
        ist = list(self.stride()) + [0] * (8 - rank)
        ost = list(out.stride()) + [0] * (8 - rank)
        BLOCK = 512
        grid = (triton.cdiv(n, BLOCK),)
        _special_round_out_strided_kernel[grid](
            self,
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
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out
