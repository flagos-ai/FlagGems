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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def sinc_func(x):
    # sinc(x) = sin(pi*x)/(pi*x), sinc(0) = 1.
    #
    # Shape of this kernel is pinned by three measured constraints:
    #  1. `isCloseVectorization=True` (the previous setting) silently
    #     miscompiles `tl.sin`: maxabs err 1.51e-01 (fp32) / 3.63e-01 (fp16)
    #     on (4096, 4096) instead of the dtype-rounding floor. Disabled.
    #  2. `tl.sin` is not trustworthy for tiny arguments when the tile is
    #     narrow (0-d / (1,) / (6,) / (64,) launches): sin(pi*1e-8) returns 0,
    #     so a bare `sin(px)/px` body yields 0.0 instead of 1.0 (abs err 1.0,
    #     60 tolerance violations over a 64-element sweep). The series branch
    #     below covers |pi*x| < 0.5, which removes the sin call from the whole
    #     region where it misbehaves; worst observed err is then 2.4e-08.
    #  3. An integer-reduction body built on `tl.floor` (nearest + parity) is
    #     exact but costs 4.60 ms on (4096, 4096) fp16 vs 0.81 ms here -- the
    #     two `math.floor` soft externs alone are ~87% of that kernel.
    x_f32 = x.to(tl.float32)
    px = 3.141592653589793 * x_f32
    px2 = px * px
    # sin(px)/px truncated at u^10/11!, exact to ~4e-14 for |px| < 0.5
    series = 1.0 + px2 * (
        -1.0 / 6.0
        + px2
        * (
            1.0 / 120.0
            + px2 * (-1.0 / 5040.0 + px2 * (1.0 / 362880.0 - px2 / 39916800.0))
        )
    )
    return tl.where(px2 < 0.25, series, tl.sin(px) / px)


def sinc(A):
    logger.debug("GEMS_KUNLUNXIN SINC")
    return sinc_func(A)


def sinc_(A):
    logger.debug("GEMS_KUNLUNXIN SINC_")
    sinc_func(A, out0=A)
    return A


def special_sinc(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SINC")
    return sinc_func(A)
