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
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs). The generic implementation launches
# cdiv(num_elements, BLOCK_SIZE) CTAs which exceeds the 65535 grid.x
# hardware limit for large tensors (2**28 elements -> 262144 CTAs ->
# OutOfResources). Striding from a fixed grid keeps every launch within
# the limit regardless of tensor size.
_GRID_SIZE = 48


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _amp_foreach_non_finite_check_and_unscale_kernel(
    inp_ptr,
    inv_scale_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # In-place scale: each CTA loads a tile, scales finite values by the
    # scalar inv_scale (non-finite values pass through unchanged) and
    # stores back to the same addresses. Math is identical to the generic
    # kernel; only the launch strategy differs (fixed grid + grid-stride,
    # and the scale scalar is hoisted out of the tile loop).
    pid = ext.program_id(0)
    scale = tl.load(inv_scale_ptr)
    scale_fp32 = scale.to(tl.float32)
    num_tile = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        inp_fp32 = inp.to(tl.float32)
        # Check for non-finite values (inf or nan) using float32
        is_non_finite = ~tl_extra_shim.finitef(inp_fp32)
        # Scale the values: only finite values are scaled
        # Non-finite values stay as-is (in original dtype)
        scaled_fp32 = tl.where(is_non_finite, inp_fp32, inp_fp32 * scale_fp32)
        scaled = scaled_fp32.to(inp.dtype)
        tl.store(inp_ptr + offsets, scaled, mask=mask)


def _amp_foreach_non_finite_check_and_unscale_(
    tensors: List[torch.Tensor],
    found_inf: torch.Tensor,
    inv_scale: torch.Tensor,
):
    """
    Check for non-finite values in tensors and unscale them.

    For each tensor in the list:
    - Scale finite values by inv_scale
    - Non-finite values (inf, nan) remain unchanged

    If any tensor has non-finite values, set found_inf to 1.0.
    """
    logger.debug("GEMS_ENFLAME AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE")

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected list or tuple of tensors, got {type(tensors)}")

    if len(tensors) == 0:
        return

    # Ensure inv_scale is a float32 tensor as PyTorch expects
    inv_scale = inv_scale.to(dtype=torch.float32)

    # Process each tensor in the list (in-place: no temporary + copy back)
    for tensor in tensors:
        if not tensor.is_floating_point():
            # Skip non-floating point tensors
            continue

        num_elements = tensor.numel()
        if num_elements == 0:
            continue

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Tuned on GCU400: BLOCK=32768 + num_warps=4 + num_stages=4.
        block_size = 32768
        num_tile = triton.cdiv(num_elements, block_size)
        grid = (min(_GRID_SIZE, num_tile), 1, 1)
        with torch_device_fn.device(tensor.device):
            _amp_foreach_non_finite_check_and_unscale_kernel[grid](
                tensor,
                inv_scale,
                num_elements,
                BLOCK_SIZE=block_size,
                GRID_SIZE=min(_GRID_SIZE, num_tile),
                NUM_STAGES=4,
                num_warps=4,
            )

    # Check if any tensor has non-finite values
    # This needs to be done after scaling since scaling might produce inf from overflow
    for tensor in tensors:
        if not tensor.is_floating_point():
            continue

        if tensor.is_complex():
            # Check real and imaginary parts separately
            if not torch.all(torch.isfinite(tensor.real)) or not torch.all(
                torch.isfinite(tensor.imag)
            ):
                found_inf.fill_(1.0)
                return
        else:
            if not torch.all(torch.isfinite(tensor)):
                found_inf.fill_(1.0)
                return
