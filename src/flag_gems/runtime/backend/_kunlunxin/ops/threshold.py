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
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def threshold_kernel(self, threshold, value):
    return tl.where(self > threshold, self, value)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    # grad_input = grad_output where self > threshold else 0.
    # The old form `tl.where(self > threshold, grad_output, 0)` compiles a
    # data-dependent select-against-a-zero-constant on XPU that pins gems latency
    # far above the memory floor: 4096^2 fp16 0.46ms / fp32 0.40ms / bf16 0.48ms
    # (fp16 SLOWER than fp32 -> not memory-bound). A plain add/mul of the two
    # tensors runs at the memory floor (~0.10ms fp16), so the select-with-zero is
    # the cost, not the compare or the memory traffic. Rewriting the select as a
    # multiply by the boolean mask keeps the tensor-op fast path: 4096^2 fp16
    # 0.46->0.32, fp32 0.40->0.28, bf16 0.48->0.43 (avg gems speedup 0.172->0.221).
    return grad_output * (self > threshold)


def threshold(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD")
    output = threshold_kernel(self, threshold, value)
    return output


def threshold_(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_")
    threshold_kernel(self, threshold, value, out0=self)
    return self


def threshold_backward(grad_output, self, threshold):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_BACKWARD")
    grad_input = threshold_backward_kernel(grad_output, self, threshold)
    return grad_input
