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

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import CodeGenConfig, pointwise_dynamic

logger = logging.getLogger(__name__)

# benchmark/test_special_xlogy.py 的原地用例经 aten.xlogy_ 分发到这里。
# 通用 flag_gems.ops.xlogy 的 codegen 配置在 GCU400 上 tile 过大，按实测
# 与 special_xlogy 使用同一组特化配置，不改动全局配置。
_XLOGY_INPLACE_CONFIG = CodeGenConfig(4096, (1024, 1, 1), 4, True, prefer_1d_tile=False)


@triton.jit
def _xlogy_inplace_compute(x, y):
    # aten 语义：y 为 NaN 时返回 NaN；x 为 0 时返回 0；其余返回 x * log(y)。
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    y_is_nan = y_fp32 != y_fp32
    x_is_zero = x_fp32 == 0.0
    # x 为 0 且 y 非 NaN 时把 y 换成 1，log(1)=0，乘积自然为 0；
    # y 为 NaN 时保留 NaN，0 * NaN 仍为 NaN，语义不变。
    safe_y = tl.where(x_is_zero & ~y_is_nan, 1.0, y_fp32)
    return x_fp32 * tl.log(safe_y)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_XLOGY_INPLACE_CONFIG,
)
@triton.jit
def xlogy_func(x, y):
    return _xlogy_inplace_compute(x, y)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=_XLOGY_INPLACE_CONFIG,
)
@triton.jit
def xlogy_func_tensor_scalar(x, y):
    return _xlogy_inplace_compute(x, y)


def xlogy_(A, B):
    logger.debug("GEMS XLOGY_")
    return xlogy_func(A, B, out0=A)


def xlogy_tensor_scalar_(A, B):
    logger.debug("GEMS XLOGY__TENSOR_SCALAR")
    return xlogy_func_tensor_scalar(A, B, out0=A)
