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

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    # ALWAYS_BOOL output -> keep isCloseMemoryAsync at its DEFAULT (True = async
    # copy closed). Opening async (=False) on a bool op re-introduces the
    # ConvertTritonXPUToLLVM giant-struct explosion on top of buffer_size_limit
    # (same trap as logical_or/and/xor).
    #
    # buffer_size_limit=4096 + unroll_num=16 keeps the per-unroll chunk at
    # 4096/16 == 256 elements (identical to the previous 2048/8 == 256, so the
    # monolithic-struct explosion is still avoided) but doubles the in-flight
    # buffer, which measurably raises large-shape DMA throughput: on XPU the
    # big shapes ([4096,4096], [1024,65536], [64,64,4096]) get a stable ~8%
    # latency drop across fp16/fp32/bf16 vs the old unroll_num=8 / default
    # (2048) buffer. Small shapes stay launch-floor bound (unchanged).
    #
    # kunlunAutoGrid=True for the single-CTA config: the 1-CTA path (numel
    # <= 131072) is faster for tiny shapes, and >= 16384 elements is covered
    # by the multi-CTA config below (hybrid dispatch, see logical_not()).
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=4096,
)

config_multi_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=False,
    unroll_num=16,
    buffer_size_limit=4096,
)

# Shapes below this numel keep the 1-CTA (kunlunAutoGrid=True) kernel; larger
# shapes switch to the constant 12-CTA kernel. A/B measured 2026-10-19 on the
# benchmark matrix: (64,64)/(64,64,1) and (1024,1) are 2-5x FASTER at 1-CTA
# (5.7/5.5/5.3us vs 10.6/10.6/26.3us) while (1024,16), (64,64,16) and every
# >= 262144-element shape are 1.2-2.2x faster at 12-CTA (6.8->5.5us,
# 11.7->6.1us, 155->126us @16.7M for fp16).
_MULTI_CTA_MIN_NUMEL = 16384


@triton.jit
def _logical_not_body(x):
    # `logical_not(x)` == `(x == 0)`. For float inputs a bitcast-icmp body is
    # used instead of an fp compare: XPU has no fast per-element `x != 0` in fp,
    # a floating-point compare (cmpf) lowered to i1 costs ~2x a plain icmp
    # (measured on XPU 7, 2026-08-13: [4096,4096] fp16 ~0.49ms for
    # `not x.to(tl.int1)` vs ~0.24ms for the bitcast-icmp body; [1024,65536]
    # 1.60ms vs 0.64ms; same ratio for fp32/bf16; re-verified 2026-10-19 on
    # XPU 4: fp16-domain `x == 0.0` is ~2.0x slower than the icmp body).
    #   u = bitcast(x, <same-width int>)     -- fp16 -> i16 (no widening!)
    #   (u & signmask) == 0                  -- true only for +/-0.0
    # All other bit patterns, incl. NaN/inf/subnormals, are non-zero after the
    # sign mask -> False. Result matches torch.logical_not bit-exactly for all
    # float edge cases (NaN, +/-inf, -0.0, subnormals) verified on-device.
    #
    # fp16 (and only fp16) stays in 16-bit arithmetic: the i16 bitcast-icmp is
    # ~1.8-2.4x faster than the fp32-upcast round trip (micro-benchmark
    # 2026-10-19: 16.7M elems 671us -> 373us; 67M 2633us -> 1465us), simply
    # because the per-element fp16->fp32 conversion is skipped. bf16 must keep
    # the fp32 upcast: the same-width `x.to(tl.int16, bitcast=True)` fails to
    # compile on this backend (TritonXPUDtypeConvert: "Cannot bitcast
    # data-type of size 32 to data-type of size 16"), and fp32 cannot bitcast
    # down to i16 at all. The integer/bool paths use a direct integer compare.
    if tl.constexpr(x.dtype.is_fp32()):
        u = x.to(tl.int32, bitcast=True)
        return (u & 0x7FFFFFFF) == 0
    elif tl.constexpr(x.dtype.is_fp16()):
        u = x.to(tl.int16, bitcast=True)
        return (u & 0x7FFF) == 0
    elif tl.constexpr(x.dtype.is_bf16()):
        u = x.to(tl.float32).to(tl.int32, bitcast=True)
        return (u & 0x7FFFFFFF) == 0
    else:
        return x == 0


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_not_func(x):
    return _logical_not_body(x)


# Duplicate of logical_not_func with the constant 12-CTA config. The two
# functions must be distinct @triton.jit functions (distinct source text) so
# the in-process codegen cache can hold both variants side by side.
@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=config_multi_)
@triton.jit
def logical_not_func_multi_cta(x):
    return _logical_not_body(x)


def logical_not(A):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_NOT")
    if A.numel() < _MULTI_CTA_MIN_NUMEL:
        return logical_not_func(A)
    return logical_not_func_multi_cta(A)


def logical_not_(A):
    # In-place variant was NOT overridden -> fell back to the generic
    # ops/logical_not.py path with the default pointwise_dynamic config (tile 512,
    # buffer_size_limit=2048, no unroll/autoGrid) -> catastrophic (38-161ms on
    # large shapes, speedup ~0.001-0.005). Reuse the now-tuned logical_not_func
    # with out0=A (same recipe as bitwise_not_), same hybrid dispatch.
    logger.debug("GEMS_KUNLUNXIN LOGICAL_NOT_")
    if A.numel() < _MULTI_CTA_MIN_NUMEL:
        logical_not_func(A, out0=A)
    else:
        logical_not_func_multi_cta(A, out0=A)
    return A
