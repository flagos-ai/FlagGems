# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
def _shifted_chebyshev_polynomial_w(x, n):
    # Shifted Chebyshev polynomial of the second kind (torch reference):
    #   W_0(x) = 1
    #   W_1(x) = 4x - 1
    #   W_k(x) = (4x - 2) * W_{k-1}(x) - W_{k-2}(x)
    # torch semantics: float n is truncated to int (C-style cast); n < 0 yields 0.
    x_f32 = x.to(tl.float32)
    n_int = n.to(tl.int32)

    wkm2 = x_f32 * 0.0 + 1.0  # W_0
    wkm1 = 4.0 * x_f32 - 1.0  # W_1
    result = tl.where(n_int == 0, wkm2, tl.where(n_int == 1, wkm1, x_f32 * 0.0))

    for k in tl.static_range(2, _MAX_POLY_DEGREE + 1):
        wk = tl.fma(4.0 * x_f32 - 2.0, wkm1, -wkm2)
        result = tl.where(n_int == k, wk, result)
        wkm2, wkm1 = wkm1, wk

    result = tl.where(n_int < 0, x_f32 * 0.0, result)
    return result.to(x.dtype)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def _shifted_chebyshev_polynomial_w_scalar_n(x, n):
    x_f32 = x.to(tl.float32)
    n_int = n.to(tl.int32)

    coeff = 4.0 * x_f32 - 2.0
    wkm2 = x_f32 * 0.0 + 1.0  # W_0
    wkm1 = 4.0 * x_f32 - 1.0  # W_1
    result = tl.where(n_int == 0, wkm2, tl.where(n_int == 1, wkm1, x_f32 * 0.0))

    for k in tl.static_range(2, _MAX_POLY_DEGREE + 1):
        wk = tl.fma(coeff, wkm1, -wkm2)
        result = tl.where(n_int == k, wk, result)
        wkm2, wkm1 = wkm1, wk

    result = tl.where(n_int < 0, x_f32 * 0.0, result)
    return result.to(x.dtype)


def special_shifted_chebyshev_polynomial_w(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_W")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"special_shifted_chebyshev_polynomial_w only supports float32/float64, got {x.dtype}"
        )
    if isinstance(n, torch.Tensor):
        n = n.to(device=x.device)
        if torch.any(n > 10).item():
            raise ValueError(
                "n must be <= 10, got values up to {}.".format(int(n.max().item()))
            )
        return _shifted_chebyshev_polynomial_w(x, n)
    if n > 10:
        raise ValueError(f"n must be <= 10, got {n}")
    return _shifted_chebyshev_polynomial_w_scalar_n(x, n)