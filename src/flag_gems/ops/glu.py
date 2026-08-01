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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("glu"),
    key=["N_BUCKET", "D"],
)
@triton.jit
def _glu_forward_kernel(
    input_ptr,
    output_ptr,
    N,
    N_BUCKET,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    rows = offsets // D
    columns = offsets % D
    input_offsets = rows * (2 * D) + columns
    a = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    b = tl.load(input_ptr + input_offsets + D, mask=mask, other=0.0)
    if input_ptr.dtype.element_ty == tl.float64:
        sigmoid_b = 1.0 / (1.0 + exp(-b))
    else:
        sigmoid_b = 1.0 / (1.0 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b
    tl.store(output_ptr + offsets, result, mask=mask)


def _can_use_triton_glu(input_tensor: torch.Tensor, dim: int) -> bool:
    """Return whether the contiguous Triton kernel can process the input."""
    if input_tensor.ndim == 0:
        return False

    normalized_dim = dim if dim >= 0 else dim + input_tensor.ndim
    if normalized_dim != input_tensor.ndim - 1:
        return False

    last_dim = input_tensor.shape[-1]
    return (
        input_tensor.device.type not in ("cpu", "meta")
        and input_tensor.layout == torch.strided
        and input_tensor.is_contiguous()
        and input_tensor.numel() > 0
        and input_tensor.is_floating_point()
        and last_dim % 2 == 0
    )


def _glu_triton_forward(input_tensor: torch.Tensor) -> torch.Tensor:
    """Launch the contiguous pure Triton GLU kernel."""
    last_dim = input_tensor.shape[-1]
    half_dim = last_dim // 2
    output = torch.empty(
        (*input_tensor.shape[:-1], half_dim),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    numel = output.numel()
    # Reuse one autotune result for nearby sizes instead of every row count.
    numel_bucket = triton.next_power_of_2(numel)
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input_tensor.device):
        _glu_forward_kernel[grid](
            input_tensor,
            output,
            numel,
            numel_bucket,
            D=half_dim,
        )
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b

    return result


@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ]
)
@triton.jit
def glu_backward_kernel(grad_output, a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    db = grad_output.to(tl.float32) * a * sigmoid_b * (1.0 - sigmoid_b)

    return da, db


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GLU FORWARD")
    if _can_use_triton_glu(self, dim):
        return _glu_triton_forward(self)

    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)

    return out


def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS GLU BACKWARD")
    # Recreate a and b
    a, b = torch.chunk(self, 2, dim=dim)
    grad_input = torch.empty_like(self, memory_format=torch.contiguous_format)
    grad_a, grad_b = torch.chunk(grad_input, 2, dim=dim)
    glu_backward_kernel(grad_output, a, b, out0=grad_a, out1=grad_b)

    return grad_input
