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

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


_atan2 = tl_extra_shim.atan2

# Without an explicit CodeGenConfig, pointwise_dynamic specializes the kernel
# per input shape on XPU -> per-shape recompile -> IR explosion
# (acos_kernel_kernel recompiled ~1269x across ~1053 modules, 1.6GB IR dump).
# kunlunAutoGrid=True + prefer_1d_tile + bounded tile makes the kernel
# shape-independent so it compiles ONCE. Mirrors cos/tan/abs/sgn_.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit()
def acos_kernel(x):
    # atan2 is more accurate than the XPU acos intrinsic for fp16/bf16 inputs.
    x_f32 = x.to(tl.float32)
    inside = (x_f32 >= -1.0) & (x_f32 <= 1.0)
    sine = tl.sqrt(tl.maximum(1.0 - x_f32 * x_f32, 0.0))
    result = _atan2(sine, x_f32)
    return tl.where(inside, result, float("nan"))


def acos(x):
    logger.debug("GEMS_KUNLUNXIN ACOS")
    y = acos_kernel(x)
    return y
