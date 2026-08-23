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

from flag_gems.utils import tl_extra_shim

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_isinf = tl_extra_shim.isinf

# NOTE (kunlunxin/XPU): isinf lowers to the extern/libdevice call
# `xpu::isinf(f32)` (tt.extern_elementwise, symbol _ZN3xpu5isinfEf). On large
# shapes the op is bandwidth-bound and this CodeGenConfig (kunlunAutoGrid +
# 1d-tile + unroll) beats the DEFAULT pointwise config (0.58 vs 0.43 avg
# speedup). unroll_num was tuned on the extern call: 8 -> 0.579, 16 -> 0.612,
# 32 -> 0.510 (>=32 over-unrolls the extern call and scalarizes it, same failure
# mode as erf's buffer/unroll tuned config). unroll_num=16 lowers the large-shape
# gems latency ~10% ([4096,4096] 0.148->0.133ms, [1024,65536] 0.550->0.492ms)
# with no small-shape regression. Do NOT raise it further.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isinf_func(x):
    return _isinf(x.to(tl.float32))


def isinf(A):
    logger.debug("GEMS_KUNLUNXIN ISINF")
    return isinf_func(A)
