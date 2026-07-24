import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
#         = scale * where(x > 0, x, alpha * (exp(x) - 1))
# i.e. elu(x, alpha, scale, input_scale=1). Neither selu nor selu_ was overridden
# by kunlunxin, so both fell to the generic ops/selu.py / ops/selu_.py hand-written
# kernel with a fixed BLOCK_SIZE=1024 + grid=cdiv(N,1024) -> launch-bound on XPU for
# large N (IR ir-selu-dev3.log). Fix: rewrite as tuned pointwise_dynamic mirroring
# sibling transcendental exp.py (isCloseVectorization=True + kunlunAutoGrid +
# unroll_num=8) -> bounded contiguous tiles, no launch explosion.
_ALPHA = tl.constexpr(1.6732632423543772848170429916717)
_SCALE = tl.constexpr(1.0507009873554804934193349852946)

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
def selu_func(x):
    x_fp32 = x.to(tl.float32)
    return _SCALE * tl.where(x_fp32 > 0, x_fp32, _ALPHA * (tl.exp(x_fp32) - 1.0))


def selu(A):
    logger.debug("GEMS_KUNLUNXIN SELU")
    return selu_func(A)


def selu_(A):
    logger.debug("GEMS_KUNLUNXIN SELU_")
    return selu_func(A, out0=A)
