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

import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import libentry
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Integer views keep every memory access same-width as the fp operand so we can
# do pure sign-bit manipulation without an in-kernel bitcast (which trips
# TritonXPUDtypeConvert on bf16/f16). Same approach as the merged copysign_
# in-place kernel; this file extends it to the out-variant.
_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def _get_uint_dtype(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return tl.core.get_int_dtype(num_bits, False)


@tl.constexpr
def _get_sign_bit_mask(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


# Reduced buffer_size_limit: the bf16 path widens `other` to fp32 (doubling the
# temp footprint) and additionally allocates a uint32 view + mask, which
# overflows XPU uni_sram at larger shapes under the default pointwise config.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def copysign_func(input, other):
    # Compute magnitude of input, apply sign of other. Do all work in fp32:
    # bf16 in-place output otherwise miscompiles under TritonXPUDtypeConvert
    # (both the bitcast path and native-bf16 arithmetic + tl.where + negate
    # trip the pass). fp32 intermediate + explicit final cast mirrors log2_.
    inp_f32 = input.to(tl.float32)
    oth_f32 = other.to(tl.float32)
    abs_val = tl.abs(inp_f32)
    signed = tl.where(oth_f32 < 0.0, -abs_val, abs_val)
    return signed.to(input.dtype)


# In-place copysign_ fast path (XPU): integer bit-domain copysign.
# Measured on XPU 3 the float compare+select body is ~8-10x slower than
# ATen on large fp16/bf16 tensors, while a pure 2-op integer body
# ((abs bits of a) ^ (sign bit of b)) costs ~3x less. bf16 must widen to
# fp32 bits first (native u16 bit path overflows uni_sram / trips
# TritonXPUDtypeConvert); fp32 uses the native-width u32 path. The XOR
# form keeps the payload at two ALU ops; measured identical to AND+OR.
@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def copysign_bit_func(input, other):
    if input.dtype == tl.float16:
        ua = input.to(tl.uint16, bitcast=True)
        ub = other.to(tl.uint16, bitcast=True)
        r = (ua & 0x7FFF) ^ (ub & 0x8000)
        return r.to(input.dtype, bitcast=True)
    elif input.dtype == tl.bfloat16:
        # bf16 widening keeps bf16 bits in the HIGH half of u32, so the
        # fp32 bit pattern of the result is already exact; the final
        # value conversion to bf16 is lossless.
        ua = input.to(tl.float32).to(tl.uint32, bitcast=True)
        ub = other.to(tl.float32).to(tl.uint32, bitcast=True)
        r = (ua & 0x7FFFFFFF) ^ (ub & 0x80000000)
        v = r.to(tl.float32, bitcast=True)
        return v.to(input.dtype)
    else:
        ua = input.to(tl.uint32, bitcast=True)
        ub = other.to(tl.uint32, bitcast=True)
        r = (ua & 0x7FFFFFFF) ^ (ub & 0x80000000)
        return r.to(input.dtype, bitcast=True)


# Out-variant of the merged _copysign_inplace_kernel (copysign_.py): pure
# sign-bit manipulation on integer views, out = (|a| bits) | (sign bit of b).
# A fixed small CTA count with large contiguous tiles keeps the kernel
# memory-bound instead of launch-bound (the pointwise path launches tens of
# thousands of tiny CTAs on large tensors).
@libentry()
@triton.jit(do_not_specialize=["num_tasks"])
def _copysign_kernel(
    A, B, OUT, num_tasks, TILE: tl.constexpr, TILES_PER_CTA: tl.constexpr, ONE_TILE: tl.constexpr
):
    ity = A.type.element_ty
    num_bits: tl.constexpr = ity.primitive_bitwidth
    # signed-safe constants: the sign-bit-only value is -(1<<(w-1));
    # clear_mask = all bits except the sign bit.
    sign_mask: tl.constexpr = -(1 << (num_bits - 1))
    clear_mask: tl.constexpr = (1 << (num_bits - 1)) - 1

    pid = tl.program_id(0)
    if ONE_TILE:
        tid = pid * TILE + tl.arange(0, TILE)
        mask = tid < num_tasks
        a_bits = tl.load(A + tid, mask=mask)
        b_bits = tl.load(B + tid, mask=mask)
        out_bits = (a_bits & clear_mask) | (b_bits & sign_mask)
        tl.store(OUT + tid, out_bits, mask=mask)
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, TILES_PER_CTA):
            tile_id = pid + j * num_ctas
            tid = tile_id * TILE + tl.arange(0, TILE)
            mask = tid < num_tasks
            a_bits = tl.load(A + tid, mask=mask)
            b_bits = tl.load(B + tid, mask=mask)
            out_bits = (a_bits & clear_mask) | (b_bits & sign_mask)
            tl.store(OUT + tid, out_bits, mask=mask)


def _copysign_run(input, other, out):
    num_tasks = input.numel()
    if num_tasks == 0:
        return out
    ity = _INT_VIEW[input.element_size()]
    a = input.view(ity)
    b = (
        other.view(ity)
        if other.dtype == input.dtype
        else other.to(input.dtype).view(ity)
    )
    o = out.view(ity)
    num_ctas = 12
    num_tiles = num_ctas
    tile = triton.next_power_of_2(triton.cdiv(num_tasks, num_tiles))
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    _copysign_kernel[(num_ctas, 1, 1)](
        a,
        b,
        o,
        num_tasks,
        TILE=tile,
        TILES_PER_CTA=tiles_per_cta,
        ONE_TILE=tiles_per_cta == 1,
    )
    return out


def copysign(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN")
    if out is None:
        if other.shape != input.shape:
            # other may broadcast (scalar or different shape): the fixed-CTA
            # kernel indexes both operands by the same flat id, so arbitrary
            # other shapes go to the pointwise path.
            return copysign_func(input, other)
        out = torch.empty_like(input)
    return copysign_out(input, other, out=out)


def copysign_out(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_OUT")
    if out is None:
        if other.shape != input.shape:
            return copysign_func(input, other)
        out = torch.empty_like(input)
    if other.shape != input.shape:
        # out= variant with broadcast: the caller-sized out is filled by the
        # pointwise path, which broadcasts `other` to the result shape.
        copysign_func(input, other, out0=out)
        return out
    if not (input.is_contiguous() and other.is_contiguous() and out.is_contiguous()):
        # non-contiguous: compute into contiguous temps then copy into out
        a_c = input.contiguous()
        b_c = other if other.dtype == input.dtype else other.to(input.dtype)
        b_c = b_c.contiguous()
        out_c = torch.empty_like(a_c)
        _copysign_run(a_c, b_c, out_c)
        out.copy_(out_c)
        return out
    return _copysign_run(input, other, out)


def copysign_(input, other):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_")
    copysign_bit_func(input, other, out0=input)
    return input
