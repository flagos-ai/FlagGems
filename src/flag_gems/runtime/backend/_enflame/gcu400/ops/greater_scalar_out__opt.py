"""greater_scalar_out 候选 v1：按算子覆盖 CodeGenConfig.max_grid_size。

沿用零分支 config 覆盖模板：只把 ENFLAME 默认 `max_grid_size=(12,1,1)`
放大到 `(1024,1,1)` 重新生成 pointwise_dynamic 包装函数。

保持生产语义：`greater_func_scalar(A, B, out0=out)`（`ALWAYS_BOOL`
promotion 输出 bool，out 与输入 dtype 不同）；out=None 时 codegen
省略 out0 自动分配。**注意不能显式传 out0=None**（codegen 会取
`.dtype` 报 `NoneType` 错误，见第 15 项 expm1_out 教训），因此保留
与生产逐字同构的 if 分支，不引入额外判断。

基线复用 `base_gsN`：与第 13 项 greater_scalar 同一次 pytest 进程采集，
日志含 greater_scalar_out 的 3 轮 15 case，时段一致。
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


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_OPT_CONFIG,
)
@triton.jit
def _greater_func_scalar(x, y):
    return x.to(tl.float32) > y


def greater_scalar_out(A, B, *, out=None):
    if out is None:
        return _greater_func_scalar(A, B)
    _greater_func_scalar(A, B, out0=out)
    return out
