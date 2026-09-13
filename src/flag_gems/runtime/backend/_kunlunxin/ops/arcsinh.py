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
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def arcsinh_func(x):
    x32 = x.to(tl.float32)
    return tl.log(x32 + tl.sqrt(x32 * x32 + 1.0)).to(x.dtype)


def arcsinh(A):
    return arcsinh_func(A)


def arcsinh_(A):
    arcsinh_func(A, out0=A)
    return A


def arcsinh_out(A, out):
    return arcsinh_func(A, out0=out)
