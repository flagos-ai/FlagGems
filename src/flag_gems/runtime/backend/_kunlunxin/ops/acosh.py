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


# Without an explicit CodeGenConfig, pointwise_dynamic specializes the kernel
# per input shape on XPU -> per-shape recompile -> IR explosion, and the default
# tiny tile no-unroll codegen underutilizes the XPU badly (baseline ~0.019-0.83x
# torch, with 16M-element shapes running at ~42ms vs ~0.79ms native).
# kunlunAutoGrid=True + prefer_1d_tile + bounded tile makes the kernel
# shape-independent so it compiles ONCE and covers large tensors. Mirrors
# acos/asin/logaddexp2. isCloseVectorization stays False (vec OPEN) as this is
# a log/sqrt transcendental kernel.
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
def acosh_kernel(x):
    # acosh(x) = log(x + sqrt(x*x - 1)); domain [1, inf), NaN for x < 1.
    x_f32 = x.to(tl.float32)
    sqrt_term = tl.sqrt(x_f32 * x_f32 - 1.0)
    return tl.log(x_f32 + sqrt_term)


def acosh(A):
    logger.debug("GEMS_KUNLUNXIN ACOSH")
    return acosh_kernel(A)


def acosh_(A):
    logger.debug("GEMS_KUNLUNXIN ACOSH_")
    acosh_kernel(A, out0=A)
    return A
