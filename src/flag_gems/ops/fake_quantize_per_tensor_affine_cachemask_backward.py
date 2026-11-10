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

logger = logging.getLogger(__name__)

_MAX_GRID_X = 65535
# An empty kernel costs a fixed launch plus a per-program term on this backend,
# so the win is few, large programs: one program covers the whole tensor up to
# 128K elements (256K for very large tensors, where the grid would otherwise
# exceed 256 programs).
_BLOCK_MIN = 1024
_BLOCK_MAX = 131072
_BLOCK_MAX_HUGE = 262144


def _pick_block(n):
    block = 1 << max(10, (n - 1).bit_length())
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    if (n + block - 1) // block > 256:
        block = _BLOCK_MAX_HUGE
    while (n + block - 1) // block > _MAX_GRID_X:
        block *= 2
    return block


def _pick_warps(block, nblocks):
    if block <= 16384:
        return 2
    if block <= 32768:
        return 4
    if nblocks <= 2:
        return 8
    if nblocks <= 4:
        return 4
    return 2


@triton.jit
def _cachemask_bwd_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    EVEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        g = tl.load(grad_ptr + offsets)
        m = tl.load(mask_ptr + offsets)
        out = (g.to(tl.float32) * m.to(tl.float32)).to(g.dtype)
        tl.store(out_ptr + offsets, out)
    else:
        valid = offsets < n_elements
        g = tl.load(grad_ptr + offsets, mask=valid, other=0.0)
        m = tl.load(mask_ptr + offsets, mask=valid, other=0)
        out = (g.to(tl.float32) * m.to(tl.float32)).to(g.dtype)
        tl.store(out_ptr + offsets, out, mask=valid)


def fake_quantize_per_tensor_affine_cachemask_backward(grad, mask):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD")
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output

    # A bool mask is byte-addressed by the kernel, so it is viewed rather than
    # converted: the value 0/1 is exactly what the multiply needs.
    kernel_mask = mask.view(torch.uint8) if mask.dtype == torch.bool else mask

    block_size = _pick_block(n_elements)
    nblocks = (n_elements + block_size - 1) // block_size
    with torch_device_fn.device(grad.device):
        _cachemask_bwd_kernel[(nblocks,)](
            grad,
            kernel_mask,
            output,
            n_elements,
            EVEN=n_elements % block_size == 0,
            BLOCK_SIZE=block_size,
            num_warps=_pick_warps(block_size, nblocks),
        )
    return output
