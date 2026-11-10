"""expm1_ 候选 v1：复用 expm1__opt 已按算子覆盖 config 的 jit 对象。

生产 `expm1_(A)` 与 `expm1(A)` 共用 `expm1_func`，唯一差别是
`out0=A` 做 in-place 写回。第 17 项落地的 `expm1__opt.py` 已把
`max_grid_size=(12,1,1)->(1024,1,1)` 并保持 `INT_TO_FLOAT`
promotion（expm1 纯生产 3 轮 PASS 3.220x），本项直接 import 复用，
不重复生成 jit 对象、零 Python 分支。

`out0=A` 保证 in-place 契约；整型输入经 promotion 输出浮点，
与公共实现一致。
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.expm1__opt import (
    _expm1_func,
)

__all__ = ["expm1_"]


def expm1_(A):
    return _expm1_func(A, out0=A)
