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

"""expm1 候选 v1：按算子覆盖 CodeGenConfig.max_grid_size。

与第 15 项 expm1_out 共用同一语义，只是 outplace（省略 out0）。
沿用零分支模板：不手写 kernel，只把 ENFLAME 默认
`max_grid_size=(12,1,1)` 放大到 `(1024,1,1)` 重新生成 pointwise_dynamic
包装函数，保持 `INT_TO_FLOAT` promotion 不变。

基线复用 `base_eoN`：与 expm1_out 同一次 pytest 进程采集、时段完全一致，
日志里含 expm1 的 3 轮 15 case，按 op_name 过滤即可。
不修改全局 CODEGEN_COFIGS，符合 skill「pointwise_dynamic 必须特化」要求。
"""

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

_BASE = get_codegen_config()
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=_OPT_CONFIG)
@triton.jit
def _expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


def expm1(A):
    return _expm1_func(A)
