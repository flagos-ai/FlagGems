# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _hermite_hn(x, n):
    # Physicist's Hermite polynomial recurrence, identical to ATen's
    # hermite_polynomial_h_forward:
    #   H_0(x) = 1, H_1(x) = 2x,
    #   H_{k+1}(x) = 2x * H_k(x) - 2k * H_{k-1}(x)
    # Evaluated in the input precision (no float32 upcast) so the result dtype
    # matches x. `tl.fma` mirrors the FMA contraction of ATen's compiled loop
    # `r = (x + x) * q - k * p`, which is essential in float32: for n >= 5 the
    # recurrence is ill-conditioned (terms ~100x the result), so any deviation
    # from ATen's exact rounding is amplified to ~1e-5 relative and exceeds the
    # 1.3e-6 rtol. This also reproduces the closed form exactly (H_2..H_9) at
    # ~1/6 of the FLOPs.
    # n is the degree (int32 tensor lowered per-lane, or an int scalar).
    two_x = x + x
    h_km1 = 1.0  # H_0(x)
    h_k = two_x  # H_1(x)
    result = tl.where(n == 0, h_km1, h_k)
    for k in tl.static_range(1, 9):
        h_kp1 = tl.fma(two_x, h_k, (-2.0 * k) * h_km1)
        h_km1 = h_k
        h_k = h_kp1
        result = tl.where(n == (k + 1), h_k, result)
    return result


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _hermite_tensor_tensor(x, n):
    return _hermite_hn(x, n.to(tl.int32)).to(x.dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _hermite_tensor_scalar(x, n):
    return _hermite_hn(x, n.to(tl.int32)).to(x.dtype)


def special_hermite_polynomial_h(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_HERMITE_POLYNOMIAL_H")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Unsupported dtype {x.dtype}")

    if isinstance(n, torch.Tensor):
        n = n.to(device=x.device, dtype=torch.int32)
        if torch.any((n < 0) | (n > 9)).item():
            raise ValueError("special_hermite_polynomial_h only supports n in [0, 9]")
        return _hermite_tensor_tensor(x, n)

    n_int = int(n)
    if n_int < 0 or n_int > 9:
        raise ValueError(
            f"special_hermite_polynomial_h only supports n in [0, 9], got n={n}"
        )
    return _hermite_tensor_scalar(x, n_int)
