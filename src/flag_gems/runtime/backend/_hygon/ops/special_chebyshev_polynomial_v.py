import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_FAST_MAX_DEGREE = tl.constexpr(4)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def chebyshev_polynomial_v_func(x, n):
    x_fp32 = x.to(tl.float32)
    two_x = x_fp32 + x_fp32

    # V_0(x) = 1, V_1(x) = 2x - 1, and
    # V_k(x) = 2x * V_{k-1}(x) - V_{k-2}(x).
    v_km2 = 1.0 + 0.0 * x_fp32
    v_km1 = two_x - 1.0
    result = tl.where(n == 0, v_km2, 0.0)
    result = tl.where(n == 1, v_km1, result)

    for k in tl.static_range(2, _FAST_MAX_DEGREE + 1):
        v_k = two_x * v_km1 - v_km2
        result = tl.where(n == k, v_k, result)
        v_km2 = v_km1
        v_km1 = v_k

    if tl.max(n, axis=0) > _FAST_MAX_DEGREE:
        acos_x = tl_extra_shim.acos(x_fp32)
        numerator = tl_extra_shim.cos((n + 0.5) * acos_x)
        denominator = tl_extra_shim.cos(0.5 * acos_x)
        fallback = numerator / denominator
        result = tl.where(n > _FAST_MAX_DEGREE, fallback, result)

    return result.to(x.dtype)


def special_chebyshev_polynomial_v(x, n):
    logger.debug("GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_V")
    return chebyshev_polynomial_v_func(x, n)
