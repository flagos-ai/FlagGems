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
def arccosh_func(x):
    x32 = x.to(tl.float32)
    return tl.log(x32 + tl.sqrt(x32 - 1.0) * tl.sqrt(x32 + 1.0)).to(x.dtype)


def arccosh_(A):
    arccosh_func(A, out0=A)
    return A
