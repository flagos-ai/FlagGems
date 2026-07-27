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


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def _shifted_v(x, n):
    xf = x.to(tl.float32)
    nf = n.to(tl.int32)
    coefficient = 4.0 * xf - 2.0
    v0 = xf * 0.0 + 1.0
    v1 = 4.0 * xf - 3.0
    result = tl.where(nf == 0, v0, tl.where(nf == 1, v1, 0.0))
    v2 = coefficient * v1 - v0
    v3 = coefficient * v2 - v1
    v4 = coefficient * v3 - v2
    v5 = coefficient * v4 - v3
    v6 = coefficient * v5 - v4
    v7 = coefficient * v6 - v5
    v8 = coefficient * v7 - v6
    v9 = coefficient * v8 - v7
    v10 = coefficient * v9 - v8
    v11 = coefficient * v10 - v9
    v12 = coefficient * v11 - v10
    v13 = coefficient * v12 - v11
    v14 = coefficient * v13 - v12
    v15 = coefficient * v14 - v13
    result = tl.where(nf == 2, v2, result)
    result = tl.where(nf == 3, v3, result)
    result = tl.where(nf == 4, v4, result)
    result = tl.where(nf == 5, v5, result)
    result = tl.where(nf == 6, v6, result)
    result = tl.where(nf == 7, v7, result)
    result = tl.where(nf == 8, v8, result)
    result = tl.where(nf == 9, v9, result)
    result = tl.where(nf == 10, v10, result)
    result = tl.where(nf == 11, v11, result)
    result = tl.where(nf == 12, v12, result)
    result = tl.where(nf == 13, v13, result)
    result = tl.where(nf == 14, v14, result)
    result = tl.where(nf == 15, v15, result)
    return result.to(x.dtype)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def _shifted_v_scalar_n(x, n):
    xf = x.to(tl.float32)
    coefficient = 4.0 * xf - 2.0
    v0 = xf * 0.0 + 1.0
    v1 = 4.0 * xf - 3.0
    result = tl.where(n == 0, v0, tl.where(n == 1, v1, 0.0))
    v2 = coefficient * v1 - v0
    v3 = coefficient * v2 - v1
    v4 = coefficient * v3 - v2
    v5 = coefficient * v4 - v3
    v6 = coefficient * v5 - v4
    v7 = coefficient * v6 - v5
    v8 = coefficient * v7 - v6
    v9 = coefficient * v8 - v7
    v10 = coefficient * v9 - v8
    v11 = coefficient * v10 - v9
    v12 = coefficient * v11 - v10
    v13 = coefficient * v12 - v11
    v14 = coefficient * v13 - v12
    v15 = coefficient * v14 - v13
    result = tl.where(n == 2, v2, result)
    result = tl.where(n == 3, v3, result)
    result = tl.where(n == 4, v4, result)
    result = tl.where(n == 5, v5, result)
    result = tl.where(n == 6, v6, result)
    result = tl.where(n == 7, v7, result)
    result = tl.where(n == 8, v8, result)
    result = tl.where(n == 9, v9, result)
    result = tl.where(n == 10, v10, result)
    result = tl.where(n == 11, v11, result)
    result = tl.where(n == 12, v12, result)
    result = tl.where(n == 13, v13, result)
    result = tl.where(n == 14, v14, result)
    result = tl.where(n == 15, v15, result)
    return result.to(x.dtype)


def special_shifted_chebyshev_polynomial_v(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_V")
    if x.dtype != torch.float32:
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if not isinstance(n, torch.Tensor):
        return _shifted_v_scalar_n(x, n)
    return _shifted_v(x, n)
