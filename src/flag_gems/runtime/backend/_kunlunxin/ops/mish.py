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

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_tanh = tl_extra_shim.tanh

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def mish_func(x):
    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    # softplus guard (x>20): keeps softplus(x) = x, avoiding exp overflow for
    # large x (same form as _kunlunxin mish_backward). The tanh guard is
    # required in addition: the XPU libdevice tanh is exp-based, so with a
    # large finite x (e.g. 1e5) or +inf it returns nan (inf/inf) whereas
    # torch's mish saturates to x. tanh_sat = 1.0 is bit-identical to
    # _tanh(x) for every x > 20 in fp32 (tanh >= ~1-1e-9 rounds to 1.0f).
    x_fp32 = x.to(tl.float32)
    softplus = tl.where(x_fp32 > 20.0, x_fp32, tl.log(1.0 + tl.exp(x_fp32)))
    tanh_sp = tl.where(x_fp32 > 20.0, 1.0, _tanh(softplus))
    return (x_fp32 * tanh_sp).to(x.dtype)


def mish(A):
    logger.debug("GEMS_KUNLUNXIN MISH")
    return mish_func(A)


def mish_(A):
    logger.debug("GEMS_KUNLUNXIN MISH_")
    mish_func(A, out0=A)
    return A
