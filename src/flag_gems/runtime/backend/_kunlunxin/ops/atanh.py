import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def atanh_func(x):
    x32 = x.to(tl.float32)
    return (0.5 * tl.log((1.0 + x32) / (1.0 - x32))).to(x.dtype)


def atanh(A):
    return atanh_func(A)


def atanh_(A):
    atanh_func(A, out0=A)
    return A
