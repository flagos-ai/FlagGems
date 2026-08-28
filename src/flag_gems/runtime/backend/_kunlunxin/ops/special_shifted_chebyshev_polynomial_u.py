# Copyright 2026 FlagOS Contributors
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def shifted_chebyshev_polynomial_u_kernel(x, n):
    z = x.to(tl.float32) * 2.0 - 1.0
    ni = n.to(tl.int32)
    u0 = z * 0.0 + 1.0
    u1 = 2.0 * z
    result = tl.where(ni == 0, u0, u1)
    prev, cur = u0, u1
    for k in tl.static_range(2, 10):
        nxt = 2.0 * z * cur - prev
        result = tl.where(ni == k, nxt, result)
        prev, cur = cur, nxt
    return result.to(x.dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n):
    z = x.to(tl.float32) * 2.0 - 1.0
    ni = n
    u0 = z * 0.0 + 1.0
    u1 = 2.0 * z
    result = tl.where(ni == 0, u0, u1)
    prev, cur = u0, u1
    for k in tl.static_range(2, 10):
        nxt = 2.0 * z * cur - prev
        result = tl.where(ni == k, nxt, result)
        prev, cur = cur, nxt
    return result.to(x.dtype)


def special_shifted_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_U")
    if x.dtype != torch.float32:
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if not isinstance(n, torch.Tensor):
        return shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n)
    return shifted_chebyshev_polynomial_u_kernel(x, n)


def special_shifted_chebyshev_polynomial_u_(x, n):
    if not isinstance(n, torch.Tensor):
        return shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n, out0=x)
    return shifted_chebyshev_polynomial_u_kernel(x, n, out0=x)
