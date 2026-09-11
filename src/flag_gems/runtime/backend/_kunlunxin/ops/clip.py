import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# clip is torch's alias of clamp (scalar min/max). The generic ops/clip.py uses
# bare pointwise_dynamic (no config) -> XPU judges the store discrete
# (lm2gm offsetState=-1) with a tiny 256/512 tile -> ~0.001 speedup on large
# shapes (56ms on 4096^2, 2200ms on [10000,65536]). clip is a pure memory-bound
# 1-in/1-out elementwise op, so reuse clamp_max's proven recipe: prefer_1d_tile
# contiguous block DMA + vec OPEN (isCloseVectorization=False, the bandwidth
# lever for 1-in ops) + unroll 8. Drop the generic's x.to(tl.float32): min/max
# is exact in native dtype and clamp (XPU-verified) omits it.
clip_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=clip_config,
)
@triton.jit
def clip_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x))


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=clip_config,
)
@triton.jit
def clip_func_min(x, mini):
    return tl.maximum(mini, x)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=clip_config,
)
@triton.jit
def clip_func_max(x, maxi):
    return tl.minimum(maxi, x)


def clip(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLIP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi)
    elif maxi is None:
        return clip_func_min(A, mini)
    else:
        return clip_func(A, mini, maxi)


def clip_(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLIP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clip_func_min(A, mini, out0=A)
    else:
        return clip_func(A, mini, maxi, out0=A)
