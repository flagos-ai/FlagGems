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

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# x == +inf  <=>  IEEE bits == {sign=0, exp=all-ones, mant=0}; the integer
# equality below is exact for every float bit pattern (NaN / -inf / +-0 /
# subnormal all differ from the single +inf pattern). It replaces the
# generic `isinf(x_fp32) & (x_fp32 > 0)` body (libdevice isinf extern call
# + fp compare on the masked pointwise codegen), which measured ~50-200 ms
# on numel > 65536 while the vectorized 1D-tile bit-pattern path is
# ~0.11 ms (probe see harness/solution/performance/
# isposinf_xpu6_20260817.md). bf16 must widen via fp32 (direct int16 bitcast
# fails the XPU TritonXPUDtypeConvert pass in the vectorized path, same
# known limitation as the signbit `_sign_bit` family).
_ISPOSINF_CONFIG = CodeGenConfig(
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
def _isposinf_body(x):
    # x == +inf: int view equality with the single +inf bit pattern.
    if tl.constexpr(x.dtype.is_fp32()):
        xi = x.to(tl.int32, bitcast=True)
        return xi == 0x7F800000
    elif tl.constexpr(x.dtype.is_fp16()):
        xi = x.to(tl.int16, bitcast=True)
        return xi == 0x7C00
    elif tl.constexpr(x.dtype.is_bf16()):
        # bf16 must widen to fp32 (u16/int16 bitcast fails the TritonXPU
        # dtype-convert pass in the vectorized path on this backend)
        u = x.to(tl.float32).to(tl.uint32, bitcast=True)
        return u == 0x7F800000
    elif tl.constexpr(x.dtype.is_fp64()):
        xi = x.to(tl.int64, bitcast=True)
        return xi == 0x7FF0000000000000
    else:
        # integers / other dtypes: widen and compare (finite -> False)
        return x.to(tl.float32) == float("inf")


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_ISPOSINF_CONFIG)
@triton.jit
def _isposinf_func(x):
    return _isposinf_body(x)


def isposinf(A):
    logger.debug("GEMS_KUNLUNXIN ISPOSINF")
    if A.numel() == 0:
        return torch.empty_like(A, dtype=torch.bool)
    return _isposinf_func(A)