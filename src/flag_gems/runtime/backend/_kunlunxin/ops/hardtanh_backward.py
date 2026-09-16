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
    kunlunAutoGrid=True,
    unroll_num=4,
)


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def hardtanh_backward_func(grad_output, self, min_val, max_val):
    # hardtanh: y = clamp(x, min_val, max_val)
    # gradient: 1 when min_val < x < max_val (strict, matches ATen
    # hardtanh_backward), 0 on/outside the bounds.
    grad_output_fp32 = grad_output.to(tl.float32)
    self_fp32 = self.to(tl.float32)
    # 0.0 if x <= min_val, +inf (-> 1.0) if x > min_val; mirrored for max_val.
    p = tl.maximum(0.0, (self_fp32 - min_val) * 1.0e30)
    q = tl.maximum(0.0, (max_val - self_fp32) * 1.0e30)
    in_range = tl.minimum(1.0, p) * tl.minimum(1.0, q)
    result = grad_output_fp32 * in_range
    return result.to(grad_output.dtype)


def hardtanh_backward(grad_output, self, min_val, max_val):
    logger.debug("GEMS_KUNLUNXIN HARDTANH_BACKWARD")
    if grad_output.numel() == 0:
        return grad_output
    return hardtanh_backward_func(grad_output, self, float(min_val), float(max_val))
