# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import math

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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _chebyshev_polynomial_v(x, n):
    x_f32 = x.to(tl.float32)
    n_f32 = n.to(tl.float32)
    vkm2 = 1.0
    vkm1 = 2.0 * x_f32 - 1.0
    result = tl.where(n_f32 < -0.5, 0.0, tl.where(n_f32 < 0.5, vkm2, vkm1))

    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 2.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 3.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 4.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 5.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 6.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 7.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 8.0) < 0.5, vk, result)
    vkm2, vkm1 = vkm1, vk
    vk = 2.0 * x_f32 * vkm1 - vkm2
    result = tl.where(tl.abs(n_f32 - 9.0) < 0.5, vk, result)
    return result.to(x.dtype)


def special_chebyshev_polynomial_v(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_V")
    if isinstance(n, torch.Tensor):
        finite = torch.isfinite(n)
        degree = torch.trunc(n) if n.is_floating_point() else n
        if torch.any(finite & (degree >= 10)).item():
            raise NotImplementedError(
                "Kunlunxin special_chebyshev_polynomial_v supports degrees below 10"
            )
        n = torch.where(finite, degree, -1.0)
    elif not math.isfinite(n):
        n = -1.0
    else:
        n = math.trunc(n)
        if n >= 10:
            raise NotImplementedError(
                "Kunlunxin special_chebyshev_polynomial_v supports degrees below 10"
            )
    return _chebyshev_polynomial_v(x, n)
