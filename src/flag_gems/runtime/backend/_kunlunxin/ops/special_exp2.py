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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_single_cta = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

config_multi_cta_fp32 = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=16,
)

config_multi_cta_16bit = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=32,
)

config_many_cta_16bit = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    num_ctas_1d=64,
    unroll_num=32,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_single_cta)
@triton.jit
def _special_exp2_func_single(x):
    LN2 = 0.69314718056
    return tl.exp(x.to(tl.float32) * LN2)


# The scalar fns need distinct source text so the pointwise codegen cache keys
# do not collide inside one process (same trick as _kunlunxin/ops/log.py).
# All four LN2 literals round to the same fp32 constant, so the math is
# bit-identical for every variant.
@pointwise_dynamic(
    promotion_methods=[(0, "INT_TO_FLOAT")], config=config_multi_cta_fp32
)
@triton.jit
def _special_exp2_func_multi_fp32(x):
    LN2 = 6.9314718056e-01
    return tl.exp(x.to(tl.float32) * LN2)


@pointwise_dynamic(
    promotion_methods=[(0, "INT_TO_FLOAT")], config=config_multi_cta_16bit
)
@triton.jit
def _special_exp2_func_multi_16bit(x):
    LN2 = 0.6931471805599453
    return tl.exp(x.to(tl.float32) * LN2)


@pointwise_dynamic(
    promotion_methods=[(0, "INT_TO_FLOAT")], config=config_many_cta_16bit
)
@triton.jit
def _special_exp2_func_many_16bit(x):
    LN2 = 0.69314718055994531
    return tl.exp(x.to(tl.float32) * LN2)


# numel window that prefers the one-CTA geometry (probe on the official matrix:
# 1024 -> 5.7us vs 15.9us, 4096 -> equal, 16384 -> 8.7us vs 6.4us).
_SINGLE_CTA_MAX_NUMEL = 4096
# numel window that prefers the 64-CTA geometry (16-bit dtypes only).
_MANY_CTA_MIN_NUMEL = 524288
_MANY_CTA_MAX_NUMEL = 2097152
_IS_16BIT = (torch.float16, torch.bfloat16)


def special_exp2(x):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_EXP2")
    numel = x.numel()
    if numel <= _SINGLE_CTA_MAX_NUMEL:
        return _special_exp2_func_single(x)
    if x.dtype in _IS_16BIT:
        if _MANY_CTA_MIN_NUMEL < numel <= _MANY_CTA_MAX_NUMEL:
            return _special_exp2_func_many_16bit(x)
        return _special_exp2_func_multi_16bit(x)
    return _special_exp2_func_multi_fp32(x)
