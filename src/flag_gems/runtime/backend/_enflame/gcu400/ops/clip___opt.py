"""clip_ 候选 v1：复用 clip__opt 三个已按算子覆盖 config 的 jit 对象。

生产 `clip_(A, mini, maxi)` 与 `clip()` 共用 `clip_func` /
`clip_func_min` / `clip_func_max` 三个 pointwise_dynamic 对象，
唯一差别是传 `out0=A` 做 in-place 写回。第 16 项已把这三个对象
按算子覆盖 `max_grid_size=(12,1,1)->(1024,1,1)` 并验证
（clip 纯生产 3 轮 PASS 3.178x），本项直接复用，不重复生成 jit 对象。

host 分支与生产逐字同构（零额外 Python 判断），保留 ValueError 语义；
`out0=A` 保证 in-place 契约（返回值就是 A 本体）。
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.clip__opt import (
    clip_func,
    clip_func_max,
    clip_func_min,
)

__all__ = ["clip_"]


def clip_(A, mini=None, maxi=None):
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clip_func_min(A, mini, out0=A)
    else:
        return clip_func(A, mini, maxi, out0=A)
