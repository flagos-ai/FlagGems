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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# signbit output is a pure sign-bit decision, so the fast body extracts the
# IEEE sign bit through an int bitcast instead of an fp compare
# (`x < 0.0` lowers to fp compare -> i1, which the sgn/sign family measured
# ~2.8x slower than the pure integer ALU path on XPU). The formula
# `(i >> (bits-1)) & 1 != 0` is exact for +-0 / +-inf / NaN(sign) /
# subnormals / all two's-complement int inputs: the MSB is set iff the value
# is negative (IEEE sign bit). The vectorized 1D-tile DMA codegen path
# (CodeGenConfig, prefer_1d_tile) is ~450x faster than the generic masked
# pointwise codegen on numel > 65536 for this op (fp16 (4096,4096):
# 52.2ms -> 0.112ms, probe see harness/solution/performance/
# signbit_xpu3_20260817.md). bf16 must widen via fp32 (direct int16 bitcast
# fails the XPU TritonXPUDtypeConvert pass in the vectorized path, same
# known limitation as the sign/sgn `_sign_bit` family).
_SIGNBIT_CONFIG = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    buffer_size_limit=4096,
    unroll_num=8,
)


@triton.jit
def _signbit_body(x):
    # IEEE sign-bit extraction through int views; exact for every float bit
    # pattern (the raw MSB is the sign bit: -0.0/-inf/-NaN all set it).
    if tl.constexpr(x.dtype.is_fp32()):
        xi = x.to(tl.int32, bitcast=True)
        return (xi >> 31) & 1 != 0
    elif tl.constexpr(x.dtype.is_fp16()):
        xi = x.to(tl.int16, bitcast=True)
        return (xi >> 15) & 1 != 0
    elif tl.constexpr(x.dtype.is_bf16()):
        # bf16 must widen to fp32 (u16/int16 bitcast fails the TritonXPU
        # dtype-convert pass in the vectorized path on this backend)
        u = x.to(tl.float32).to(tl.uint32, bitcast=True)
        return ((u >> 31).to(tl.int32)) & 1 != 0
    elif tl.constexpr(x.dtype.is_fp64()):
        xi = x.to(tl.int64, bitcast=True)
        return (xi >> 63) & 1 != 0
    else:
        # integers / other dtypes: direct compare (exact for ints)
        return x < 0


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_SIGNBIT_CONFIG)
@triton.jit
def _signbit_func(x):
    return _signbit_body(x)


def signbit(A):
    logger.debug("GEMS_KUNLUNXIN SIGNBIT")
    if A.numel() == 0:
        return torch.empty_like(A, dtype=torch.bool)
    return _signbit_func(A)


def signbit_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN SIGNBIT_OUT")
    if out is None:
        return signbit(A)
    if A.numel() == 0:
        return out
    _signbit_func(A, out0=out)
    return out
