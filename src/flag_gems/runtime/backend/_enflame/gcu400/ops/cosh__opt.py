"""cosh 候选 v1：outplace 变体复用第 29 项已落地的优化 jit 对象。

公共 `cosh = cosh_func(A)`，与 `cosh_out` / `cosh_` 共用同一个
`pointwise_dynamic` 包装对象（`INT_TO_FLOAT` promotion）。
第 29 项落地的 `cosh_out__opt.py` 中 `_cosh_func` 已按算子把
`max_grid_size` 从 ENFLAME 默认 `(12,1,1)` 放大到 `(1024,1,1)`，
本项**直接 import 复用、零重复 jit 对象**，host 与生产 `cosh`
逐字同构。

不修改任何 TRACKED 文件，也不改全局 CODEGEN_COFIGS。
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.cosh_out__opt import (
    _cosh_func,
)

__all__ = ["cosh"]


def cosh(A):
    return _cosh_func(A)
