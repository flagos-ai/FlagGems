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

# Kunlunxin (XPU) override of mish / mish_.
#
# The generic `flag_gems.ops.mish` uses pointwise_dynamic without an explicit
# CodeGenConfig, so on XPU it specializes the kernel per input shape (per-shape
# recompile) and runs with the default codegen knobs (slow path). Following the
# established Kunlunxin pointwise recipe (log1p / log2 / silu / mish_backward),
# the kernel is recompiled with an explicit bounded 1D-tile CodeGenConfig:
# kunlunAutoGrid=True + prefer_1d_tile + unroll_num=8 + buffer_size_limit=4096.
#
# Perf: the naive x*tanh(log1p(exp(x))) pays 3 transcendentals (exp + log +
# tanh), which on XPU is compute-bound and lands ~0.78x on large shapes. Using
# the algebraically exact identity (y = exp(x)):
#     tanh(log1p(exp(x))) = ((y+1)^2 - 1) / ((y+1)^2 + 1) = (y^2 + 2y)/(y^2+2y+2)
# leaves a single exp + one division, pushing large-shape speedup above 1.3x.
# Guard x > 20 (mish(x) -> x) to avoid exp overflow / inf-inf NaN.
#
# isCloseVectorization=True (vectorization CLOSED) to keep the div/exp pipeline
# numerically correct on bf16 (same class of vectorized-miscompile documented in
# the sibling log1p.py / log2.py). Kernel is fp32-intermediate, cast back.
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
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def mish_func(x):
    # mish(x) = x * tanh(softplus(x)); with y = exp(x):
    #   tanh(log1p(exp(x))) = (y^2 + 2y) / (y^2 + 2y + 2)
    x_fp32 = x.to(tl.float32)
    y = tl.exp(x_fp32)
    num = y * y + 2.0 * y
    result = x_fp32 * (num / (num + 2.0))
    return tl.where(x_fp32 > 20.0, x_fp32, result).to(x.dtype)


def mish(A):
    logger.debug("GEMS_KUNLUNXIN MISH")
    return mish_func(A)


def mish_(A):
    logger.debug("GEMS_KUNLUNXIN MISH_")
    return mish_func(A, out0=A)
