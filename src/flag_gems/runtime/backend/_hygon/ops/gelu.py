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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
tanh = tl_extra_shim.tanh

_CONTIGUOUS_BLOCK_SIZE = 2048
_CONTIGUOUS_NUM_WARPS = 8
_INT32_MAX = torch.iinfo(torch.int32).max


@triton.jit
def _gelu_none(x):
    x_fp32 = x.to(tl.float32)
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    return 0.5 * x_fp32 * (1 + erf(x_fp32 * scale))


@triton.jit
def _gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    x_sq = x_fp32 * x_fp32
    return 0.5 * x_fp32 * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * x_sq)))


@triton.jit
def _gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    scaled_x = scale1 * x_fp32
    dydx = scale2 * x_fp32 * exp(-(scaled_x * scaled_x)) + 0.5 * erf(scaled_x) + 0.5
    return dydx * dy


@triton.jit
def _gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    x_sq = x_fp32 * x_fp32
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * x_sq))
    dydx = 0.5 * x_fp32 * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x_sq)
    ) + 0.5 * (1 + tanh_out)
    return dydx * dy


@triton.jit
def _gelu_forward_contiguous_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    use_tanh: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    if use_tanh:
        output = _gelu_tanh(x)
    else:
        output = _gelu_none(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _gelu_backward_contiguous_kernel(
    input_ptr,
    grad_output_ptr,
    grad_input_ptr,
    n_elements,
    use_tanh: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    if use_tanh:
        grad_input = _gelu_backward_tanh(x, grad_output)
    else:
        grad_input = _gelu_backward_none(x, grad_output)
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    return _gelu_none(x)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    return _gelu_tanh(x)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    return _gelu_backward_none(x, dy)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    return _gelu_backward_tanh(x, dy)


def _can_use_contiguous_path(tensor):
    return tensor.is_contiguous() and tensor.numel() <= _INT32_MAX


def _launch_contiguous_forward(A, output, approximate):
    n_elements = A.numel()
    if n_elements == 0:
        return output
    grid = (triton.cdiv(n_elements, _CONTIGUOUS_BLOCK_SIZE),)
    with torch_device_fn.device(A.device.index):
        _gelu_forward_contiguous_kernel[grid](
            A,
            output,
            n_elements,
            use_tanh=approximate == "tanh",
            BLOCK_SIZE=_CONTIGUOUS_BLOCK_SIZE,
            num_warps=_CONTIGUOUS_NUM_WARPS,
        )
    return output


def _launch_contiguous_backward(self, grad_output, approximate):
    grad_input = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input
    grid = (triton.cdiv(n_elements, _CONTIGUOUS_BLOCK_SIZE),)
    with torch_device_fn.device(self.device.index):
        _gelu_backward_contiguous_kernel[grid](
            self,
            grad_output,
            grad_input,
            n_elements,
            use_tanh=approximate == "tanh",
            BLOCK_SIZE=_CONTIGUOUS_BLOCK_SIZE,
            num_warps=_CONTIGUOUS_NUM_WARPS,
        )
    return grad_input


def gelu(A, *, approximate="none"):
    logger.debug("GEMS_HYGON GELU")
    if _can_use_contiguous_path(A):
        return _launch_contiguous_forward(A, torch.empty_like(A), approximate)
    if approximate == "tanh":
        return gelu_tanh(A)
    return gelu_none(A)


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_HYGON GELU_BACKWARD")
    if (
        _can_use_contiguous_path(self)
        and grad_output.is_contiguous()
        and grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
    ):
        return _launch_contiguous_backward(self, grad_output, approximate)
    if approximate == "tanh":
        return gelu_backward_tanh(self, grad_output)
    return gelu_backward_none(self, grad_output)


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_HYGON GELU_")
    if _can_use_contiguous_path(A):
        return _launch_contiguous_forward(A, A, approximate)
    if approximate == "tanh":
        return gelu_tanh(A, out0=A)
    return gelu_none(A, out0=A)
