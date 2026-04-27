import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def _get_uint_dtype(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return tl.core.get_int_dtype(num_bits, False)


@tl.constexpr
def _get_sign_bit_mask(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def copysign_func(input, other):
    # Magnitude of input, sign of other
    abs_val = tl.abs(input)
    # Check sign bit of other (bitcast to unsigned int and check MSB)
    num_bits: tl.constexpr = input.dtype.primitive_bitwidth
    uint_dtype = _get_uint_dtype(num_bits)
    sign_bit_mask: tl.constexpr = _get_sign_bit_mask(num_bits)
    other_u = other.to(uint_dtype, bitcast=True)
    # Extract sign bit and check if it's set
    return tl.where((other_u & sign_bit_mask) != 0, -abs_val, abs_val)


def copysign(input, other, *, out=None):
    logger.debug("GEMS COPYSIGN")
    return copysign_func(input, other)


def copysign_out(input, other, *, out=None):
    logger.debug("GEMS COPYSIGN_OUT")
    if out is None:
        return copysign_func(input, other)
    copysign_func(input, other, out0=out)
    return out
