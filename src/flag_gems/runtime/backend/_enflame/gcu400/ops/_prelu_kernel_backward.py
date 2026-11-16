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

import flag_gems

logger = logging.getLogger(__name__)

# Small-table path: pass only cumulative row counts and reconstruct time step and in-batch index inside the kernel.
_TILE_ELEMENTS = 8192
_MAX_GRID_PROGRAMS = 48
_NUM_WARPS = 4
_NUM_STAGES = (
    0  # 0 means keep the backend default; do not set an extra software pipeline depth
)
_FUSED_SELECT = False  # Whether to fuse two comparisons into one comparison plus arithmetic selection


@triton.jit
def _prelu_kernel_backward_kernel(
    grad_output_ptr,
    x_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    n_elements,
    C,
    BLOCK_SIZE: tl.constexpr,
    IS_SCALAR_WEIGHT: tl.constexpr,
    FUSED_SELECT: tl.constexpr,
):
    # Fixed program count + grid-stride: large shapes no longer expand the grid by element count, avoiding the GCU
    # grid.x limit of 65535
    program_id = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    element_range = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + element_range
        mask = offsets < n_elements

        grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Fixed program count + grid-stride: large shapes no longer expand the grid by element count, avoiding the GCU
        # grid.x limit of 65535
        if IS_SCALAR_WEIGHT:
            weight = tl.load(weight_ptr)
        else:
            channel_index = offsets % C
            weight = tl.load(weight_ptr + channel_index, mask=mask, other=0.0)

        # grad_input = grad_output if x > 0 else grad_output * weight
        # grad_weight = grad_output * x if x < 0 else 0
        if FUSED_SELECT:
            # Scalar weight needs only one scalar load; per-channel weight is gathered by the last dimension index
            is_positive = x > 0
            grad_input = grad_output * tl.where(is_positive, 1.0, weight)
            grad_weight = grad_output * tl.where(is_positive, 0.0, tl.minimum(x, 0.0))
        else:
            grad_input = tl.where(x > 0, grad_output, grad_output * weight)
            grad_weight = tl.where(x < 0, grad_output * x, 0.0)

        tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
        tl.store(grad_weight_ptr + offsets, grad_weight, mask=mask)


def _prelu_kernel_backward(*args, **kwargs):
    logger.debug("GEMS_ENFLAME_GCU400 _PRELU_KERNEL_BACKWARD")

    # Extract inputs
    if len(args) >= 3:
        grad_output, x, weight = args[0], args[1], args[2]
    else:
        grad_output = kwargs.get("grad_output")
        x = kwargs.get("self")
        weight = kwargs.get("weight")

    if grad_output is None or x is None or weight is None:
        raise ValueError(
            "_prelu_kernel_backward expects (grad_output, self, weight) as arguments."
        )

    if (
        grad_output.device.type != flag_gems.device
        or x.device.type != flag_gems.device
        or weight.device.type != flag_gems.device
    ):
        raise RuntimeError(
            f"_prelu_kernel_backward: all tensors must be "
            f"{flag_gems.device} tensors for Triton kernel."
        )

    # Ensure dtype match
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if grad_output.dtype != x.dtype:
        grad_output = grad_output.to(dtype=x.dtype)

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return grad_input, grad_weight

    # Determine C (last dimension size)
    ndim = x.dim()
    if weight.numel() == 1:
        # Scalar weight - broadcast to all elements
        C = 1
    else:
        if ndim == 0:
            raise AssertionError("Non-scalar weight provided for a 0-dim input.")
        # Weight should match the last dimension
        C = x.shape[-1]
        if weight.numel() != C:
            raise AssertionError(
                f"Weight numel ({weight.numel()}) must equal last dimension size ({C})."
            )

    # Make sure C is at least 1 to avoid div/mod by zero in kernel math
    C = max(int(C), 1)

    # One sign comparison drives both outputs, removing the second comparison of x < 0
    BLOCK_SIZE = _TILE_ELEMENTS
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (min(num_blocks, _MAX_GRID_PROGRAMS),)
    is_scalar_weight = weight.numel() == 1

    launch_options = {"num_warps": _NUM_WARPS}
    if _NUM_STAGES > 0:
        launch_options["num_stages"] = _NUM_STAGES

    _prelu_kernel_backward_kernel[grid](
        grad_output,
        x,
        weight,
        grad_input,
        grad_weight,
        n_elements,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_SCALAR_WEIGHT=is_scalar_weight,
        FUSED_SELECT=_FUSED_SELECT,
        **launch_options,
    )

    return grad_input, grad_weight
