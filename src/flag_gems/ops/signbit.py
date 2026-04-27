import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def signbit_func(x):
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
    logger.debug("GEMS SIGNBIT")
    return signbit_func(A)


def signbit_out(A, *, out=None):
    logger.debug("GEMS SIGNBIT_OUT")
    if out is None:
        return signbit_func(A)
    signbit_func(A, out0=out)
    return out
