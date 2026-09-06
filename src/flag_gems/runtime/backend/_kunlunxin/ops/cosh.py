import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Same XPU-tuned codegen config as the hyperbolic siblings (sinh/acosh/asinh_):
# isCloseVectorization=True + buffer_size_limit=4096 + kunlunAutoGrid +
# unroll_num=8 route memory through the XPU close-vectorized path. A/B on the
# full benchmark matrix (12 shapes x {fp16, fp32}) shows mean speedup
# 1.69x -> 2.27x (gems latency -7%..-48% per cell) vs the previous
# isCloseVectorization=False variant; see solution/cosh/.
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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def cosh_func(x):
    x32 = x.to(tl.float32)
    return (0.5 * (tl.exp(x32) + tl.exp(-x32))).to(x.dtype)


def cosh(A):
    return cosh_func(A)


def cosh_(A):
    cosh_func(A, out0=A)
    return A


def cosh_out(A, out):
    return cosh_func(A, out0=out)
