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

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_bessel_y1_func(x):
    # bessel_y1 oscillates sharply near its zeros, so float32 truncation cannot
    # meet float64's rtol=1e-7. Route by input dtype: keep double precision for
    # float64, otherwise compute in float32.
    inp_dtype = x.type.element_ty
    if inp_dtype == tl.float64:
        return tl_extra_shim.y1(x.to(tl.float64)).to(x.dtype)
    y = tl_extra_shim.y1(x.to(tl.float32))
    # The Kunlunxin libdevice `y1` returns NaN for x == +-0.0, while
    # PyTorch/fdlibm define Y1(+-0) = -inf. Rewrite only those boundary lanes;
    # all other lanes stay on the native (and numerically correct) path.
    y = tl.where(x == 0.0, float("-inf"), y)
    return y.to(x.dtype)


def special_bessel_y1(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_BESSEL_Y1")
    return special_bessel_y1_func(A)
