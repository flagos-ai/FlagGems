import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def copysign_func(input, other):
    # Magnitude of input, sign of other
    abs = tl.abs(input)
    # Check sign bit of other (bitcast to int and check MSB)
    if tl.constexpr(input.dtype.is_fp32()):
        other_i = other.to(tl.int32, bitcast=True)
    elif tl.constexpr(input.dtype.is_fp16() or input.dtype.is_bf16()):
        other_i = other.to(tl.int16, bitcast=True)
    else:
        other_i = other.to(tl.int64, bitcast=True)
    return tl.where(other_i < 0, -abs, abs)


def copysign(input, other, *, out=None):
    logger.debug("GEMS COPYSIGN")
    return copysign_func(input, other)


def copysign_out(input, other, *, out=None):
    logger.debug("GEMS COPYSIGN_OUT")
    if out is None:
        return copysign_func(input, other)
    copysign_func(input, other, out0=out)
    return out
