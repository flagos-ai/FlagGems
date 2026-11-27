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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# The shared pointwise codegen caps a single block at 65536 elements, so a medium-sized tensor expands to
# only 1~16 CTAs, leaving most of the 48 compute cores idle. When the element count falls in the window below, switch to
# the
# flat grid-stride kernel (shallow blocks + wide grid) to gain roughly 2x performance; outside the window,
# very small sizes (dominated by launch overhead) and large sizes (DMA tiling is better) still go back to the shared
# implementation.
_FLAT_PATH_MIN_ELEMENTS = 8 * 1024
_FLAT_PATH_MAX_ELEMENTS = 512 * 1024
_FLAT_BLOCK_SMALL = 4096
_FLAT_BLOCK_MEDIUM = 8192
_FLAT_BLOCK_SMALL_LIMIT = 128 * 1024
_FLAT_NUM_WARPS = 2
_MAX_GRID_X = 65535

# Above the window the flat kernel keeps winning for 16-bit tensors, which are
# pure DMA streams: a 32768-element tile per CTA with 2 warps takes the 16.7M
# element cases from 2.15x to 2.60x. fp32 falls back to the shared tiling - the
# wide tile regresses there (2.35x -> 2.05x at 536M elements).
_FLAT_LARGE_BLOCK_16BIT = 32768
_FLAT_LARGE_WARPS_16BIT = 2
# The wide-tile path keeps one CTA per resident SPC slot and walks its share of
# blocks with the grid-stride loop; letting the grid grow to n/BLOCK instead
# spawns thousands of CTAs and costs ~20%.
_NUM_SIPS = 24
_FLAT_LARGE_GRID = 2 * _NUM_SIPS

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _sin_flat_kernel(
    x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    """Elementwise sin over a flattened, contiguous buffer (grid-stride loop)."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        # A masked load/store costs an implicit select per lane on GCU, so the
        # fully covered case runs without predicates at all.
        if EVEN:
            values = tl.load(x_ptr + offsets)
        else:
            values = tl.load(x_ptr + offsets, mask=offsets < n_elements)
        result = tl.sin(values.to(tl.float32))
        if EVEN:
            tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty))
        else:
            tl.store(
                out_ptr + offsets,
                result.to(out_ptr.dtype.element_ty),
                mask=offsets < n_elements,
            )


def _can_use_flat_path(tensor):
    if not tensor.is_contiguous() or tensor.dtype not in _SUPPORTED_DTYPES:
        return False
    n_elements = tensor.numel()
    if n_elements <= _FLAT_PATH_MIN_ELEMENTS:
        return False
    if n_elements <= _FLAT_PATH_MAX_ELEMENTS:
        return True
    return tensor.element_size() <= 2


def _sin_flat(x, out):
    n_elements = x.numel()
    if n_elements <= _FLAT_PATH_MAX_ELEMENTS:
        block = (
            _FLAT_BLOCK_SMALL
            if n_elements <= _FLAT_BLOCK_SMALL_LIMIT
            else _FLAT_BLOCK_MEDIUM
        )
        num_warps = _FLAT_NUM_WARPS
        grid_cap = _MAX_GRID_X
    else:
        block = _FLAT_LARGE_BLOCK_16BIT
        num_warps = _FLAT_LARGE_WARPS_16BIT
        grid_cap = _FLAT_LARGE_GRID
    even = n_elements % block == 0
    grid = (min(triton.cdiv(n_elements, block), grid_cap),)
    _sin_flat_kernel[grid](
        x.view(-1),
        out.view(-1),
        n_elements,
        BLOCK=block,
        EVEN=even,
        num_warps=num_warps,
    )
    return out


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sin_func(x):
    return tl.sin(x.to(tl.float32))


def sin(A):
    logger.debug("GEMS_ENFLAME SIN")
    if _can_use_flat_path(A):
        return _sin_flat(A, torch.empty_like(A))
    return sin_func(A)


def sin_(A):
    logger.debug("GEMS_ENFLAME SIN_")
    if _can_use_flat_path(A):
        return _sin_flat(A, A)
    sin_func(A, out0=A)
    return A
