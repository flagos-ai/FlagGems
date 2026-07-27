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
def _hermite_hn(x, n_int):
    xf = x.to(tl.float32)
    h0 = 1.0
    h1 = 2.0 * xf
    x2 = xf * xf
    h2 = 4.0 * x2 - 2.0
    x3 = x2 * xf
    h3 = 8.0 * x3 - 12.0 * xf
    x4 = x2 * x2
    h4 = 16.0 * x4 - 48.0 * x2 + 12.0
    x5 = x4 * xf
    h5 = 32.0 * x5 - 160.0 * x3 + 120.0 * xf
    x6 = x3 * x3
    h6 = 64.0 * x6 - 480.0 * x4 + 720.0 * x2 - 120.0
    x7 = x6 * xf
    h7 = 128.0 * x7 - 1344.0 * x5 + 3360.0 * x3 - 1680.0 * xf
    x8 = x4 * x4
    h8 = 256.0 * x8 - 3584.0 * x6 + 13440.0 * x4 - 13440.0 * x2 + 1680.0
    x9 = x8 * xf
    h9 = 512.0 * x9 - 9216.0 * x7 + 48384.0 * x5 - 80640.0 * x3 + 30240.0 * xf

    result = h0
    result = tl.where(n_int == 1, h1, result)
    result = tl.where(n_int == 2, h2, result)
    result = tl.where(n_int == 3, h3, result)
    result = tl.where(n_int == 4, h4, result)
    result = tl.where(n_int == 5, h5, result)
    result = tl.where(n_int == 6, h6, result)
    result = tl.where(n_int == 7, h7, result)
    result = tl.where(n_int == 8, h8, result)
    result = tl.where(n_int == 9, h9, result)
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
