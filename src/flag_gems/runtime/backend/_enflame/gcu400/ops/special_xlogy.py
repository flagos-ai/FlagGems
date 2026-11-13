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

# GCU400 默认的 65536 元素 tile 只配 48 个 CTA，单 CTA 寄存器压力过大；
# libdevice 的 extern __ef_logf 又比 Triton 核心 tl.log 慢很多。按实测定制
# 本算子的 codegen 配置，不改动全局 CODEGEN_COFIGS。
# 注意：不能在本模块 import codegen_config_utils，pointwise_dynamic 会把模块的
# import 原样复制进生成代码，而相对导入在生成的 gcu400.ops.* 路径下无法解析。
_XLOGY_CONFIG = CodeGenConfig(4096, (1024, 1, 1), 4, True, prefer_1d_tile=False)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=_XLOGY_CONFIG)
@triton.jit
def special_xlogy_func(x, y):
    # xlogy(x, y) = x * log(y)，但 x == 0 时必须返回 0（y 为 0/负数/inf 都一样），
    # 唯一的例外是 y 为 NaN：此时无论 x 取值都要返回 NaN。
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    y_is_nan = y_fp32 != y_fp32
    x_is_zero = x_fp32 == 0.0
    # 把「x 为 0 且 y 非 NaN」的输入换成 1：log(1) = 0，乘积自然是 0；
    # 既省掉末尾两层 tl.where，也避免对 0/负数走 log 的慢速异常分支。
    # y 为 NaN 时保留 NaN，0 * NaN 仍是 NaN，语义不变。
    safe_y = tl.where(x_is_zero & ~y_is_nan, 1.0, y_fp32)
    return x_fp32 * tl.log(safe_y)


def special_xlogy(A, B):
    logger.debug("GEMS_ENFLAME SPECIAL_XLOGY")
    return special_xlogy_func(A, B)


def special_xlogy_(A, B):
    logger.debug("GEMS_ENFLAME SPECIAL_XLOGY_")
    return special_xlogy_func(A, B, out0=A)
