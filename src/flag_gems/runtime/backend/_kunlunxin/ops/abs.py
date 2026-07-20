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

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Custom config with memory async enabled for better memory throughput
_abs_config = CodeGenConfig(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,  # Enable memory async for better overlap
)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=_abs_config)
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logger.debug("GEMS_KUNLUNXIN ABS")
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS_KUNLUNXIN ABS_")
    abs_func(A, out0=A)
    return A
