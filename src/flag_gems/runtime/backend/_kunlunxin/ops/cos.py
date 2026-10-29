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

# cos is NOT bandwidth bound on P800: at 4096x4096 a pure copy costs 0.066 ms
# and `x + 1.0` costs 0.064 ms, but `tl.cos` costs 0.172 ms.  So ~0.11 ms is the
# transcendental itself, and every CodeGenConfig knob (unroll 0-32, warps,
# interleave, dtypeconvert) measured flat.  Two levers actually move it:
#
# 1. Tile width.  The launcher derives the tile as
#    cdiv(buffer_size_limit * 64, element_size) capped at 32768, then rewrites an
#    exactly-32768 tile down to 16384 to avoid a device hang (_BAD_TILE_SIZE_1D
#    in pointwise_dynamic).  Tile 24576 is the fastest tile and is reached with
#    buffer 768 for 2-byte and 1536 for 4-byte elements, hence two configs.
#
# 2. Replacing tl.cos with a polynomial, for fp16 only (see below).
_BASE_CFG = dict(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
)

config_narrow_ = CodeGenConfig(buffer_size_limit=768, **_BASE_CFG)
config_wide_ = CodeGenConfig(buffer_size_limit=1536, **_BASE_CFG)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_narrow_)
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_wide_)
@triton.jit
def cos_func_wide(x):
    return tl.cos(x.to(tl.float32))


# fp16-only fast path.  Argument reduction uses the magic-number round
# q = (x/2pi + 1.5*2^23) - 1.5*2^23, which is pure FMA: tl.floor and an int cast
# both measured 0.29-0.38 ms here, which is what killed a Cephes-style pi/4 +
# quadrant-select version.  Reducing to the full period [-pi, pi] then lets a
# single even degree-6 polynomial in r^2 cover everything (fitted max error
# 1.1e-8), with no quadrant selection at all.
#
# The reduction is only exact while |x/2pi| stays under 2^22, so accuracy falls
# apart above |x| ~ 1e6 -- fp32/bf16 therefore keep tl.cos, which is accurate to
# |x| ~ 1.3e7.  fp16's largest finite value is 65504, two decades inside the safe
# band, so for fp16 this is exact over the whole domain: measured max error
# 2.44e-4 across the full +-65504 range, identical to both torch and tl.cos
# (that figure is just fp16's own 1 ulp at the output).
#
# A tl.where guard was tried and is far slower (0.55 ms) because both arms are
# evaluated; a tile-level scalar `if` on tl.max(tl.abs(x)) does not compile on
# this backend (out of resource: uni_sram), in a hand-written kernel too.
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_narrow_)
@triton.jit
def cos_func_poly(x):
    f = x.to(tl.float32)
    q = (f * 0.15915494309189535 + 12582912.0) - 12582912.0
    # 2pi subtracted in two chunks (Cody-Waite) so q * term stays exact
    r = f - q * 6.28125
    r = r - q * 0.0019353071795864769
    s = r * r
    p = 1.7247786959948514e-09
    p = p * s + -2.707983290059827e-07
    p = p * s + 2.4769973252653524e-05
    p = p * s + -0.001388780823741278
    p = p * s + 0.0416664903108346
    p = p * s + -0.49999989198743056
    p = p * s + 0.9999999892330095
    return p


def _cos_kernel(A):
    # The polynomial is only used where its reduction is exact over the whole
    # dtype, which is fp16 alone.
    if A.dtype == torch.float16:
        return cos_func_poly
    # Otherwise pick whichever config derives a 24576-element tile.  The tile
    # follows the widest of the input and output element sizes, and an integer
    # input is promoted to fp32 on output, so only the 2-byte floats are narrow.
    if A.dtype is torch.bfloat16:
        return cos_func
    return cos_func_wide


def cos(A):
    logger.debug("GEMS_KUNLUNXIN COS")
    return _cos_kernel(A)(A)


def cos_(A):
    logger.debug("GEMS_KUNLUNXIN COS_")
    _cos_kernel(A)(A, out0=A)
    return A
