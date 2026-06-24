# Copyright 2026, The FlagOS Contributors.
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
#
# Adapted for the Kunlunxin backend from the generic KernelGen clone op.
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .copy import copy_slice

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def _clone_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def clone(inp: torch.Tensor, memory_format: Optional[torch.memory_format] = None):
    """
    Returns a copy of the input tensor.

    Args:
        inp: The input tensor to clone.
        memory_format: The desired memory format of the returned tensor.
            Default is torch.preserve_format.
    """
    logger.debug("GEMS_KUNLUNXIN CLONE")

    if memory_format is None:
        memory_format = torch.preserve_format

    n_elements = inp.numel()
    if n_elements == 0:
        # Handle empty tensors
        if memory_format == torch.contiguous_format:
            return torch.empty_like(inp, memory_format=torch.contiguous_format)
        else:
            return torch.empty_strided(
                inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
            )

    # Fast path: contiguous tensor, use Triton kernel
    if memory_format == torch.preserve_format and inp.is_contiguous(
        memory_format=torch.preserve_format
    ):
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )
        # Flatten both for contiguous memory access
        src = inp.flatten()
        dst = out.flatten()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(inp.device):
            _clone_kernel[grid](src, dst, n_elements, BLOCK_SIZE=1024)
        return out

    if memory_format == torch.contiguous_format:
        # Make the result contiguous
        out = torch.empty_like(inp, memory_format=torch.contiguous_format)
        if inp.is_contiguous():
            # flatten() returns a view for a contiguous tensor, so the flat
            # kernel can copy element-by-element safely.
            src = inp.flatten()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            with torch_device_fn.device(inp.device):
                _clone_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
        else:
            # For a non-contiguous tensor, flatten() would itself trigger a
            # contiguous()/clone(), recursing back into this op indefinitely,
            # and a plain copy_ redispatches to a native aten kernel that is
            # unavailable on this backend. Use the strided-aware Triton kernel
            # (the same path contiguous()/slice_scatter use) instead.
            copy_slice(inp, out0=out)
        return out

    # Fallback: non-contiguous preserve or other memory formats
    if memory_format == torch.preserve_format:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )
    else:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
        out = out.to(memory_format=memory_format)

    # copy_ would redispatch a non-contiguous gather to a native aten kernel
    # that is missing on this backend; use the strided Triton copy instead.
    copy_slice(inp, out0=out)
    return out
