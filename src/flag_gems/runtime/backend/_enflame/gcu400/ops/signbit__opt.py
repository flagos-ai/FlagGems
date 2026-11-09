"""signbit 候选 v1：按算子覆盖 CodeGenConfig.max_grid_size。

公共 `signbit_func` 是 pointwise_dynamic 生成的一元算子
（`promotion_methods=[(0, "ALWAYS_BOOL")]`，输出 bool）。ENFLAME 默认
`max_grid_size=(12,1,1)` 使大 shape 只有 12 个 CTA。沿用本批已在
8 个 pointwise 算子上 PASS 的零分支模板：只把 `max_grid_size`
放大到 `(1024,1,1)` 重新生成包装函数；codegen 内部
`num_ctas=min(grid,num_tiles)` 使小 shape 调用链与公共实现一致。

kernel 主体与公共实现逐字同构（fp32/fp16/bf16/fp64 分别 bitcast 到
对应整数看符号位，其它 dtype 用 `x<0`）；不修改全局 CODEGEN_COFIGS。
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
    promotion_methods=[(0, "ALWAYS_BOOL")],
    config=_OPT_CONFIG,
)
@triton.jit
def _signbit_func(x):
    if tl.constexpr(x.dtype.is_fp32()):
        xi32 = x.to(tl.int32, bitcast=True)
        return xi32 < 0
    elif tl.constexpr(x.dtype.is_fp16()):
        xi16 = x.to(tl.int16, bitcast=True)
        return xi16 < 0
    elif tl.constexpr(x.dtype.is_bf16()):
        xi16 = x.to(tl.int16, bitcast=True)
        return xi16 < 0
    elif tl.constexpr(x.dtype.is_fp64()):
        xi64 = x.to(tl.int64, bitcast=True)
        return xi64 < 0
    else:
        return x < 0


def signbit(A):
    return _signbit_func(A)
