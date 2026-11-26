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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        (0, "DEFAULT"),
        # The actual dtype of the second output is determined by the
        # preallocated int32 output tensor passed through out1.
        (0, "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def _frexp_kernel(x):
    # Decompose every input into (mantissa, exponent) with
    #     x = mantissa * 2 ** exponent,  |mantissa| in [0.5, 1.0)
    # Special values: frexp(+-0/+-inf/nan) == (x, 0).
    #
    # Everything below works on the IEEE-754 bit pattern with integer ops.
    # Floating-point transcendentals are unusable here: in a loop-free
    # micro-kernel this backend lowers tl.log2 / tl.exp2 to natural log / exp
    # (KERNEL_OPT_EXPERIENCE 2.2 #20), and FP arithmetic flushes subnormal
    # inputs to zero (DAZ/FTZ), so x * 2**24 cannot be used to normalize them.
    x_fp32 = x.to(tl.float32)
    bits = x_fp32.to(tl.int32, bitcast=True)

    exp_field = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    sign = (bits >> 31) << 31

    # Zero / infinity / NaN keep their original bits and get exponent 0.
    # NOTE: the obvious integer idioms (bits & 0x7FFFFFFF) == 0 and
    # (bits << 1) == 0 are rewritten by the compiler into a floating-point
    # zero test on this backend, which flushes subnormals to zero (DAZ), so a
    # nonzero subnormal would be misclassified as zero. Checking the exponent
    # and fraction fields separately survives.
    is_zero = (exp_field == 0) & (frac == 0)
    is_nonfinite = exp_field == 0xFF
    is_special = is_zero | is_nonfinite

    # Normal values: only the biased exponent changes (to 126, unbiased -1).
    norm_bits = frac | 0x3F000000 | sign
    norm_exp = exp_field - 126

    # Subnormals: x = frac * 2**-149. Locate L = floor(log2(frac)) with a
    # branchless integer binary search, then build the normalized mantissa
    # x * 2**-exponent = frac * 2**-(L + 1) in [0.5, 1).
    sh = frac
    lg = frac - frac
    c = sh >= 65536
    lg += tl.where(c, 16, 0)
    sh = tl.where(c, sh >> 16, sh)
    c = sh >= 256
    lg += tl.where(c, 8, 0)
    sh = tl.where(c, sh >> 8, sh)
    c = sh >= 16
    lg += tl.where(c, 4, 0)
    sh = tl.where(c, sh >> 4, sh)
    c = sh >= 4
    lg += tl.where(c, 2, 0)
    sh = tl.where(c, sh >> 2, sh)
    c = sh >= 2
    lg += tl.where(c, 1, 0)

    sub_bits = ((frac << (23 - lg)) & 0x7FFFFF) | 0x3F000000 | sign
    sub_exp = lg - 148

    is_subnormal = exp_field == 0
    out_bits = tl.where(is_subnormal, sub_bits, norm_bits)
    out_e = tl.where(is_subnormal, sub_exp, norm_exp)
    out_bits = tl.where(is_special, bits, out_bits)
    out_e = tl.where(is_special, 0, out_e)

    # mantissa keeps the input dtype, exponent is always int32.
    mantissa = out_bits.to(tl.float32, bitcast=True)
    return mantissa.to(x.dtype), out_e.to(tl.int32)


def frexp(A):
    logger.debug("GEMS_KUNLUNXIN FREXP")

    if not A.is_floating_point():
        raise RuntimeError(
            f"frexp(): expected a floating-point tensor, but got {A.dtype}"
        )

    # The decomposition is carried out in float32; float64 cannot be
    # represented exactly on this backend in any case.
    if A.dtype == torch.float64:
        raise RuntimeError("FlagGems frexp currently does not support float64")

    mantissa = torch.empty_like(A)
    exponent = torch.empty_like(A, dtype=torch.int32)

    return _frexp_kernel(A, out0=mantissa, out1=exponent)
