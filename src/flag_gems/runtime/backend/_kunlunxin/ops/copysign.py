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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


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


def copysign(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN")
    return copysign_func(input, other)


def copysign_out(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_OUT")
    if out is None:
        return copysign_func(input, other)
    copysign_bit_func(input, other, out0=out)
    return out


def copysign_(input, other):
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_")
    copysign_bit_func(input, other, out0=input)
    return input
