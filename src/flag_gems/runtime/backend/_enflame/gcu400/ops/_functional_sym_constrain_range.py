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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs). The generic implementation launches
# cdiv(n_elements, BLOCK_SIZE) CTAs which exceeds the 65535 grid.x
# hardware limit for large tensors. Striding from a fixed grid keeps
# every launch within the limit regardless of input size.
_GRID_SIZE = 48


@libentry()
@triton.jit
def _functional_sym_constrain_range_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = ext.program_id(0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, x, mask=mask)


def _functional_sym_constrain_range(*args, **kwargs):
    logger.debug("GEMS_ENFLAME FUNCTIONAL_SYM_CONSTRAIN_RANGE")
    # Get the dep_token tensor to determine output device and properties
    tensor_arg = None
    for a in args:
        if isinstance(a, torch.Tensor):
            tensor_arg = a
            break
    if tensor_arg is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensor_arg = v
                break

    if tensor_arg is not None:
        # Return a copy of the dep_token tensor
        # The actual constraint validation is handled by PyTorch's symbolic tracing
        if tensor_arg.is_contiguous():
            n_elements = tensor_arg.numel()
            if n_elements > 0:
                output = torch.empty_like(tensor_arg)
                block_size = 32768
                num_tile = triton.cdiv(n_elements, block_size)
                grid = (min(_GRID_SIZE, num_tile), 1, 1)
                with torch_device_fn.device(tensor_arg.device):
                    _functional_sym_constrain_range_kernel[grid](
                        tensor_arg,
                        output,
                        n_elements,
                        BLOCK_SIZE=block_size,
                        GRID_SIZE=min(_GRID_SIZE, num_tile),
                        NUM_STAGES=4,
                        num_warps=4,
                    )
                return output
        # Fallback for non-contiguous or empty tensors
        return tensor_arg.clone()

    # Fallback: return args[0] if it's a tensor
    return args[0] if len(args) > 0 else None
