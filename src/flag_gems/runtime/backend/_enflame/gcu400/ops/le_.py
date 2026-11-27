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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# One wave of 48 CTAs with a wide tile is the fastest streaming shape on
# GCU400 (same geometry the native pointwise kernels use); the generic
# code-generated launcher streams at roughly a quarter of that bandwidth.
_MAX_GRID = 48
_SMALL_BLOCK = 8192
_LARGE_BLOCK = 65536
_LARGE_TILE_ELEMENTS = 1 << 27


@triton.jit
def _le_tensor_inplace_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """`a <= b` for two same-shape dense tensors, written back into `a`."""
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(a_ptr + offsets, (a <= b).to(a.dtype), mask=mask)


@triton.jit
def _le_scalar_inplace_kernel(a_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    """`a <= scalar` for a dense tensor, written back into `a`."""
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        tl.store(a_ptr + offsets, (a <= scalar).to(a.dtype), mask=mask)


def _pick_block_size(n_elements: int) -> int:
    if n_elements >= _LARGE_TILE_ELEMENTS:
        return _LARGE_BLOCK
    # Small tensors pick the power of two from the element count: handing a 4096-element call
    # the 65536-wide tile idles 16x the lane time (measured 4.1us -> 7.3us).
    tile_size = 1 << max(8, (n_elements - 1).bit_length())
    return min(_LARGE_BLOCK, tile_size)


def _pick_tensor_block_size(n_elements: int, element_size: int) -> int:
    """Block for the two-operand in-place kernel.

    16-bit operands stream three 2-byte accesses per element and measured slower
    with the 65536 lane tile (77.3us -> 79.3us at 4096^2); the 4-byte stream
    wants the wide tile (145.0us -> 125.5us on the same shape).
    """
    if element_size <= 2 and n_elements < _LARGE_TILE_ELEMENTS:
        return _SMALL_BLOCK
    return _pick_block_size(n_elements)


def _can_use_fast_path(A, B) -> bool:
    """Dense, same-shape, same-dtype operands only; everything else keeps the
    generic broadcasting launcher."""
    return (
        isinstance(B, torch.Tensor)
        and A.shape == B.shape
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.numel() > 0
    )


def _launch_le_tensor(A, B):
    n_elements = A.numel()
    block_size = _pick_tensor_block_size(n_elements, A.element_size())
    grid = (min(triton.cdiv(n_elements, block_size), _MAX_GRID),)
    _le_tensor_inplace_kernel[grid](A, B, n_elements, BLOCK_SIZE=block_size)


def _launch_le_scalar(A, scalar):
    n_elements = A.numel()
    block_size = _pick_block_size(n_elements)
    grid = (min(triton.cdiv(n_elements, block_size), _MAX_GRID),)
    _le_scalar_inplace_kernel[grid](A, scalar, n_elements, BLOCK_SIZE=block_size)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func(x, y):
    # Compare in the native dtype: promotion to fp32 only widens the vectors
    # (and loses precision for 64-bit integers) without changing the result.
    return x <= y


def le_(A, B):
    logger.debug("GEMS_ENFLAME LE_")
    if _can_use_fast_path(A, B):
        _launch_le_tensor(A, B)
        return A
    le_func(A, B, out0=A)
    return A


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func_scalar(x, y):
    return x <= y


def le_scalar_(A, B):
    logger.debug("GEMS_ENFLAME LE_SCALAR_")
    if isinstance(B, (int, float, bool)) and A.is_contiguous() and A.numel() > 0:
        _launch_le_scalar(A, B)
        return A
    le_func_scalar(A, B, out0=A)
    return A
