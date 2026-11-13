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

"""GCU400 专属 ``_foreach_clamp_min_.List``（in-place）。

公共 ``foreach_binary_list`` 用二维 grid 承载 tensor 列表，GCU400 的
``grid.y`` 上限 255 在大 shape 下会 ``out of resource``。这里对浮点连续
列表改走一维固定 grid + 核内跨步的 maximum kernel；非浮点 / 非连续 /
广播不等长等形态仍委托公共 ``_apply_list`` 保持 dtype policy。
"""

import logging

from flag_gems.ops._foreach_binary import _apply_list

from ._foreach_flat import FALLBACK, MODE_LIST, OP_MAX, run_binary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；均为编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


def _foreach_clamp_min__List(self, other):
    logger.debug("GEMS_GCU400 FOREACH_CLAMP_MIN__LIST")

    def fallback():
        return _apply_list("clamp_min", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MAX,
        MODE_LIST,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result
