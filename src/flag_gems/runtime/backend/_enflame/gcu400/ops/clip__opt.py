"""clip 候选 v1：三个 jit 对象统一按算子覆盖 CodeGenConfig.max_grid_size。

生产 `clip()` 按 mini/maxi 是否为 None 分派到三个不同的
pointwise_dynamic 对象（clip_func / clip_func_min / clip_func_max），
三者都受 ENFLAME 默认 `max_grid_size=(12,1,1)` 限制。
本版保持 host 分支结构与生产逐字同构，只把三个 jit 对象的 config
换成 `(1024,1,1)`，不引入任何额外 Python 判断。

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
    is_tensor=[True, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=_OPT_CONFIG,
)
@triton.jit
def clip_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_OPT_CONFIG,
)
@triton.jit
def clip_func_min(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_OPT_CONFIG,
)
@triton.jit
def clip_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clip(A, mini=None, maxi=None):
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi)
    elif maxi is None:
        return clip_func_min(A, mini)
    else:
        return clip_func(A, mini, maxi)
