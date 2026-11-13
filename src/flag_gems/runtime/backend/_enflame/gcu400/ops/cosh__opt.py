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

"""cosh 候选 v1：outplace 变体复用第 29 项已落地的优化 jit 对象。

公共 `cosh = cosh_func(A)`，与 `cosh_out` / `cosh_` 共用同一个
`pointwise_dynamic` 包装对象（`INT_TO_FLOAT` promotion）。
第 29 项落地的 `cosh_out__opt.py` 中 `_cosh_func` 已按算子把
`max_grid_size` 从 ENFLAME 默认 `(12,1,1)` 放大到 `(1024,1,1)`，
本项**直接 import 复用、零重复 jit 对象**，host 与生产 `cosh`
逐字同构。

不修改任何 TRACKED 文件，也不改全局 CODEGEN_COFIGS。
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.cosh_out__opt import _cosh_func

__all__ = ["cosh"]


def cosh(A):
    return _cosh_func(A)
