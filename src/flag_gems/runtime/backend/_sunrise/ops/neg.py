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

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

config = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=(65535, 65535, 65535),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
    # num_warps=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config)
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logger.debug("GEMS_SUNRISE NEG")
    return neg_func(A)


def neg_(A):
    logger.debug("GEMS_SUNRISE NEG_")
    return neg_func(A, out0=A)
