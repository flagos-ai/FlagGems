"""less_equal_scalar 候选 v1：按算子覆盖 CodeGenConfig.max_grid_size。

沿用第 11 项 floor 验证的零分支模板：不手写 kernel，只把 ENFLAME 默认
`max_grid_size=(12,1,1)` 放大到 `(1024,1,1)` 重新生成 pointwise_dynamic
包装函数。codegen 内部 `num_ctas=min(grid,num_tiles)` 使小 shape 调用链
与公共实现一致、无微秒级回归，大 shape 并行度提升。

保持生产原有的 `is_tensor=[True, False]` 与 `ALWAYS_BOOL` promotion 语义
（输出 bool，标量 B 与 x 的 fp32 比较）。不修改全局 CODEGEN_COFIGS。
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


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_OPT_CONFIG,
)
@triton.jit
def _less_equal_func_scalar(x, y):
    return x.to(tl.float32) <= y


def less_equal_scalar(A, B):
    return _less_equal_func_scalar(A, B)
