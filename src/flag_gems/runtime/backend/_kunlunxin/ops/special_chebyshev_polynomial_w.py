# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

_MAX_POLY_DEGREE = tl.constexpr(10)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _chebyshev_polynomial_w(x, n):
    x_f32 = x.to(tl.float32)
    n_f32 = n.to(tl.float32)

    wkm2 = x_f32 * 0.0 + 1.0
    wkm1 = 2.0 * x_f32 + 1.0
    result = tl.where(n_f32 < 0.5, wkm2, wkm1)

    for k in tl.static_range(2, _MAX_POLY_DEGREE):
        wk = 2.0 * x_f32 * wkm1 - wkm2
        result = tl.where(tl.abs(n_f32 - k) < 0.5, wk, result)
        wkm2, wkm1 = wkm1, wk

    return result


def special_chebyshev_polynomial_w(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_W")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"special_chebyshev_polynomial_w only supports float32/float64, got {x.dtype}"
        )
    if not isinstance(n, torch.Tensor):
        n = torch.empty((), dtype=torch.int64, device=x.device).fill_(n)
    else:
        n = n.to(device=x.device)
    return _chebyshev_polynomial_w(x, n)


def special_chebyshev_polynomial_w_out(x, n, out):
    result = special_chebyshev_polynomial_w(x, n)
    out.copy_(result)
    return out
