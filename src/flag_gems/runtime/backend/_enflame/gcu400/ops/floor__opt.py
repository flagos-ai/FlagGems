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

"""floor 候选 v5：按算子覆盖 CodeGenConfig.max_grid_size（零 Python 分支）。

前 4 版都是「手写 1D kernel + 小 shape 分档」，但纯生产路径在微秒级小
shape 上始终 p10≈0.90：override 与生产路径的 dispatch 层级不同，任何额外
Python 判断都会造成 0.3~0.9us 的偏差。

本版改为直接复用 pointwise_dynamic 的 codegen 路径，只把 ENFLAME 默认
`max_grid_size=(12,1,1)` 放大到 `(1024,1,1)`：
- codegen 内部 `num_ctas = min(max_grid_size, num_tiles)`，小 shape 自动仍用
  少量 CTA，调用链与公共实现完全一致，零额外开销；
- 大 shape 的 CTA 数从 12 提升到 1024，消除 grid 并行度瓶颈。
探针实测 2^30 `16.38ms -> 5.90ms` (2.78x)、2^28 `4.12 -> 1.51ms` (2.73x)，
且各 shape 各 grid 均 `exact=True`。

不修改全局 CODEGEN_COFIGS，只影响本算子生成的包装函数，对其它算子无副作用。
"""

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

_BASE = get_codegen_config()

# 1024 在 2^24 / 2^28 / 2^30 三档均为最优或次优，且远低于 grid.x 上限 65535
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_OPT_CONFIG)
@triton.jit
def _floor_func(x):
    return tl.floor(x.to(tl.float32)).to(x.dtype)


def floor(A):
    return _floor_func(A)
