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

from flag_gems.ops.flip import flip as default_flip
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _can_use_triton_kernel(x: torch.Tensor) -> bool:
    """Check if we can use our triton kernel for this tensor."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa":
        return False
    if x.numel() <= 1:
        return False
    if not x.is_contiguous():
        return False
    return True


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
    ],
    key=["inner_size"],
)
@triton.jit
def _flip_block_kernel(
    x_ptr,
    out_ptr,
    shape_ptr,
    strides_ptr,
    flip_mask_ptr,
    inner_size,
    outer_size,
    split: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flip kernel with correct multi-dimensional indexing.

    Each program processes multiple blocks (grid-stride loop).
    Each block has inner_size contiguous elements.
    The block_id is decoded to multi-dim indices, each dim independently flipped if needed.

    Args:
        x_ptr: source tensor data pointer
        out_ptr: output tensor data pointer
        shape_ptr: shape of the leading dims [0:split]
        strides_ptr: strides of the leading dims [0:split]
        flip_mask_ptr: boolean mask indicating which dims to flip [0:split]
        inner_size: number of contiguous elements per block (trailing non-flipped dims)
        outer_size: number of blocks (product of leading dims)
        split: number of leading dimensions
        BLOCK_SIZE: tile size for processing inner elements
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Grid-stride loop over outer blocks
    for block_id in range(pid, outer_size, num_programs):
        # Decode block_id to multi-dimensional index and compute source offset
        # Row-major layout: decode from the last dimension backwards
        remaining = block_id
        src_offset = 0

        for dim in range(split - 1, -1, -1):
            dim_size = tl.load(shape_ptr + dim)
            dim_stride = tl.load(strides_ptr + dim)
            flip_dim = tl.load(flip_mask_ptr + dim)

            # Extract index for this dimension (in output layout)
            idx = remaining % dim_size
            remaining = remaining // dim_size

            # Apply flip if needed for this dimension
            src_idx = tl.where(flip_dim, dim_size - 1 - idx, idx)

            # Accumulate source offset
            src_offset += src_idx * dim_stride

        dst_base = block_id * inner_size

        # Process inner elements in tiles
        offsets = tl.arange(0, BLOCK_SIZE)
        for inner_start in range(0, inner_size, BLOCK_SIZE):
            idx = inner_start + offsets
            mask = idx < inner_size
            val = tl.load(x_ptr + src_offset + idx, mask=mask, other=0.0)
            tl.store(
                out_ptr + dst_base + idx, val.to(out_ptr.dtype.element_ty), mask=mask
            )


def _compute_split_point(shape: tuple, strides: tuple, dims: tuple, ndim: int) -> int:
    """Find the split point where trailing dims are no longer flipped.

    Returns the first index (from the end) where a dim is flipped.
    If no trailing dims are flipped, returns ndim (all trailing dims non-flipped).
    If the last dim is flipped, returns ndim - 1.

    We need to find the largest k such that dims ndim-k, ndim-k+1, ..., ndim-1
    are all NOT flipped.
    """
    flip_set = set(dim if dim >= 0 else ndim + dim for dim in dims)
    split = ndim
    for i in range(ndim - 1, -1, -1):
        if i in flip_set:
            break
        split = i
    return split


def flip(x: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS FLIP")

    if not _can_use_triton_kernel(x):
        return default_flip(x, dims)

    ndim = x.dim()
    shape = x.shape
    strides = x.stride()

    # Normalize dims to positive
    flip_set = set(dim if dim >= 0 else ndim + dim for dim in dims)

    # Validate dims
    for dim in flip_set:
        if dim < 0 or dim >= ndim:
            return default_flip(x, dims)

    # Check if any dim actually needs flipping
    active_flip = any(
        dim in flip_set and shape[dim] > 1 and strides[dim] != 0 for dim in range(ndim)
    )
    if not active_flip:
        return x.clone()

    # Find how many trailing dims are NOT flipped (can be treated as inner contiguous)
    split = _compute_split_point(shape, strides, dims, ndim)

    # Only use our optimized kernel when there are trailing non-flipped dims
    # AND the inner block size is large enough to benefit from coalesced access.
    # For 1D cases and cases where all dims (including innermost) are flipped,
    # fall back to the generic implementation which works well for those patterns.
    if split == 0 or split == ndim:
        # All dims flipped or no dims flipped: use generic (works well for 1D/2D)
        return default_flip(x, dims)

    # There are trailing non-flipped dims: use block-based kernel
    # Inner size = product of trailing non-flipped dims
    inner_size = 1
    for i in range(split, ndim):
        inner_size *= shape[i]

    # Outer size = product of leading dims
    outer_size = 1
    for i in range(0, split):
        outer_size *= shape[i]

    # Prepare shape, strides, and flip_mask for leading dims
    leading_shape = torch.tensor(shape[:split], dtype=torch.int32, device=x.device)
    leading_strides = torch.tensor(strides[:split], dtype=torch.int32, device=x.device)
    leading_flip_mask = torch.tensor(
        [i in flip_set for i in range(split)], dtype=torch.bool, device=x.device
    )

    out = torch.empty_like(x)

    # Launch kernel: one CTA per outer block, with grid-stride loop
    # Use a reasonable grid size to balance parallelism and overhead
    max_grid = min(outer_size, 65536)
    grid = lambda meta: (max_grid,)
    with torch_device_fn.device(x.device):
        _flip_block_kernel[grid](
            x,
            out,
            leading_shape,
            leading_strides,
            leading_flip_mask,
            inner_size,
            outer_size,
            split,
        )
    return out
