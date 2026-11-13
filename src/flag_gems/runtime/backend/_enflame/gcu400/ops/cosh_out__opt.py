"""cosh_out 候选 v1：按算子覆盖 CodeGenConfig.max_grid_size。

公共 `cosh_func` 是 pointwise_dynamic 生成的一元算子
（`promotion_methods=[(0, "INT_TO_FLOAT")]`，kernel 内 `x.to(tl.float32)`
后 `0.5*(exp(x)+exp(-x))`）。ENFLAME 默认 `max_grid_size=(12,1,1)`
让大 shape 只有 12 个 CTA。沿用本批已 PASS 的零分支模板：
只把 `max_grid_size` 放大到 `(1024,1,1)` 重新生成包装函数，
kernel 主体与公共实现逐字同构；codegen 内部
`num_ctas=min(grid,num_tiles)` 使小 shape 调用链与公共实现一致。

自包含（不 import 其它候选文件），便于 override 从独立目录加载。
不修改全局 CODEGEN_COFIGS。
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
    promotion_methods=[(0, "INT_TO_FLOAT")],
    config=_OPT_CONFIG,
)
@triton.jit
def _cosh_func(x):
    x = x.to(tl.float32)
    return 0.5 * (tl.exp(x) + tl.exp(-x))


def cosh_out(A, out):
    return _cosh_func(A, out0=out)
