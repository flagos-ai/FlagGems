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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    derivative = tl.where(self_fp32 < 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
    return grad_output * derivative


@triton.jit
def log_sigmoid_backward_contiguous_kernel(
    grad_output,
    self,
    buffer,
    grad_input,
    n_elements,
    HAS_BUFFER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad = tl.load(grad_output + offsets, mask=mask)
    inp = tl.load(self + offsets, mask=mask).to(tl.float32)
    if HAS_BUFFER:
        z = tl.load(buffer + offsets, mask=mask).to(tl.float32)
        derivative = tl.where(inp < 0.0, 1.0, z) / (1.0 + z)
    else:
        derivative = tl.sigmoid(-inp)
    result = grad * derivative
    tl.store(grad_input + offsets, result, mask=mask)


def _can_use_contiguous_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (grad_input is None or grad_input.is_contiguous())
    )


def _launch_contiguous_kernel(grad_output, self, buffer, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    # On T-Head PPU, recomputing sigmoid is faster than the extra buffer read
    # for every supported floating-point dtype.
    has_buffer = False
    with torch_device_fn.device(self.device):
        log_sigmoid_backward_contiguous_kernel[grid](
            grad_output,
            self,
            buffer,
            grad_input,
            n_elements,
            HAS_BUFFER=has_buffer,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS LOG_SIGMOID BACKWARD")

    del buffer
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS LOG_SIGMOID BACKWARD OUT")

    if _can_use_contiguous_kernel(grad_output, self, grad_input):
        return _launch_contiguous_kernel(grad_output, self, buffer, grad_input)
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
