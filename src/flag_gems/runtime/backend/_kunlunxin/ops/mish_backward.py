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

# Kunlunxin (XPU) override of mish_backward.
#
# The generic `flag_gems.ops.mish_backward` uses pointwise_dynamic without an
# explicit CodeGenConfig, so on XPU it specializes the kernel per input shape
# (per-shape recompile) and runs with the default codegen knobs. Following the
# established Kunlunxin pointwise recipe (acos / asin / atanh / trunc etc.),
# the same kernel body is recompiled with an explicit bounded 1D-tile
# CodeGenConfig: kunlunAutoGrid=True + prefer_1d_tile + unroll_num=8 +
# buffer_size_limit=4096. Kernel body / math unchanged (zero correctness risk).
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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def mish_backward_func(grad_output, input):
    x = input.to(tl.float32)
    grad = grad_output.to(tl.float32)
    softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tanh_sp = _tanh(softplus)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    derivative = tanh_sp + x * sigmoid * (1.0 - tanh_sp * tanh_sp)
    return grad * derivative


def mish_backward(grad_output, input):
    logger.debug("GEMS_KUNLUNXIN MISH_BACKWARD")
    return mish_backward_func(grad_output, input)
