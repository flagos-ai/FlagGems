"""expm1_out 候选 v1：按算子覆盖 CodeGenConfig.max_grid_size。

沿用第 11~13 项验证的零分支模板：不手写 kernel，只把 ENFLAME 默认
`max_grid_size=(12,1,1)` 放大到 `(1024,1,1)` 重新生成 pointwise_dynamic
包装函数。codegen 内部 `num_ctas=min(grid,num_tiles)` 使小 shape 调用链
与公共实现一致、无微秒级回归，大 shape 并行度提升。

保持生产原有 `INT_TO_FLOAT` promotion 与 out 变体语义
（`expm1_func(A, out0=out)`）。生产 `expm1_out(A, out)` 的 out 是必传参数，
本实现与之对齐、零 Python 分支；注意不能显式传 `out0=None`
（codegen 会取 `.dtype` 报错，省略参数才是自动分配路径）。
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


def expm1_out(A, out):
    return _expm1_func(A, out0=out)
