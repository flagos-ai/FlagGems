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

# Kunlunxin (XPU) override of sinh / sinh_.
#
# The generic `flag_gems.ops.sinh` uses pointwise_dynamic without an explicit
# CodeGenConfig, so on XPU it specializes the kernel per input shape (per-shape
# recompile) and runs with the default codegen knobs (slow path). Following the
# established Kunlunxin pointwise recipe (cosh / log1p / log2 / mish), the kernel
# is recompiled with an explicit bounded 1D-tile CodeGenConfig:
# kunlunAutoGrid=True + prefer_1d_tile + unroll_num=8 + buffer_size_limit=4096.
#
# Formula: sinh(x) = (exp(x) - exp(-x)) * 0.5, fp32 intermediate. Large |x|
# naturally overflows to +/-inf (exp(x) - exp(-x) = inf - 0 or 0 - inf), matching
# torch.sinh for the large-value stability test.
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
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def sinh_func(x):
    x32 = x.to(tl.float32)
    return (0.5 * (tl.exp(x32) - tl.exp(-x32))).to(x.dtype)


def sinh(A):
    logger.debug("GEMS_KUNLUNXIN SINH")
    return sinh_func(A)


def sinh_(A):
    logger.debug("GEMS_KUNLUNXIN SINH_")
    sinh_func(A, out0=A)
    return A
