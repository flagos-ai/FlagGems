"""cosh_ 候选 v1：inplace 变体复用第 29 项已落地的优化 jit 对象。

公共 `cosh_ = cosh_func(A, out0=A)`，与 `cosh_out` 共用同一个
`pointwise_dynamic` 包装对象（`INT_TO_FLOAT` promotion）。
第 29 项落地的 `cosh_out__opt.py` 中 `_cosh_func` 已按算子把
`max_grid_size` 从 ENFLAME 默认 `(12,1,1)` 放大到 `(1024,1,1)`，
本项**直接 import 复用、零重复 jit 对象**，host 仅 `out0=A`，
与生产 `cosh_` 逐字同构。

不修改任何 TRACKED 文件，也不改全局 CODEGEN_COFIGS。
"""
from flag_gems.runtime.backend._enflame.gcu400.ops.cosh_out__opt import (
    _cosh_func,
)

__all__ = ["cosh_"]


def cosh_(A):
    return _cosh_func(A, out0=A)
