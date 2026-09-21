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

# Kunlunxin (XPU) override of special_log1p / special_log1p.out.
#
# The generic codegen emits no XPU launch tuning and leaves the memory-bound
# log1p kernel slow; this override moves the op onto the vendor
# pointwise_dynamic with a tuned CodeGenConfig (unroll_num=16,
# kunlunAutoGrid=False). isCloseVectorization=True is required: with
# vectorization open the vectorized log miscompiles bf16 (~1.6% of elements
# off by exactly +ln(2)).
import logging
import math

import torch
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
    kunlunAutoGrid=False,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def special_log1p_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


def special_log1p(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P")
    if isinstance(A, torch.Tensor):
        return special_log1p_func(A)
    return math.log1p(A)


def special_log1p_out(A, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P_OUT")
    return special_log1p_func(A, out0=out)
