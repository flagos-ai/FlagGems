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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        (0, "DEFAULT"),
        (0, "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def _frexp_func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)

    is_nan = x_fp32 != x_fp32
    is_inf = abs_x == float("inf")
    is_zero = x_fp32 == 0.0
    is_special = is_nan | is_inf | is_zero

    bits = x_fp32.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    exponent = biased_exp - 126

    mant_abs_bits = (bits & 0x007FFFFF) | (126 << 23)
    mant_abs = mant_abs_bits.to(tl.float32, bitcast=True)
    mantissa = tl.where(bits < 0, -mant_abs, mant_abs)

    mantissa = tl.where(is_special, x_fp32, mantissa)
    exponent = tl.where(is_special, 0, exponent)

    return mantissa.to(x.dtype), exponent


def frexp(A):
    logger.debug("GEMS_KUNLUNXIN FREXP")

    if not A.is_floating_point():
        raise RuntimeError(
            f"frexp(): expected a floating-point tensor, but got {A.dtype}"
        )

    if A.dtype == torch.float64:
        raise RuntimeError("FlagGems frexp currently does not support float64")

    mantissa = torch.empty_like(A)
    exponent = torch.empty_like(A, dtype=torch.int32)

    return _frexp_func(
        A,
        out0=mantissa,
        out1=exponent,
    )
