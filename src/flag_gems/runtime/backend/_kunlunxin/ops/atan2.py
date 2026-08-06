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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_atan2 = tl_extra_shim.atan2

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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def atan2_kernel(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    result = _atan2(x_f32, y_f32)

    # XPU atan2 returns zero for atan2(+/-0, negative), losing the quadrant.
    x_bits = x_f32.to(tl.int32, bitcast=True)
    y_bits = y_f32.to(tl.int32, bitcast=True)
    signed_pi = tl.where(x_bits < 0, -3.141592653589793, 3.141592653589793)
    negative_y = (y_f32 < 0.0) | ((y_f32 == 0.0) & (y_bits < 0))
    return tl.where((x_f32 == 0.0) & negative_y, signed_pi, result)


def atan2(input, other):
    logger.debug("GEMS_KUNLUNXIN ATAN2")
    return atan2_kernel(input, other)


def atan2_out(input, other, out):
    logger.debug("GEMS_KUNLUNXIN ATAN2_OUT")
    return atan2_kernel(input, other, out0=out)
