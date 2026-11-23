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

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _sin_flat_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Elementwise sin over a flattened, contiguous buffer (grid-stride loop)."""
    pid = tl.program_id(0)
    span = tl.num_programs(0) * BLOCK
    for start in tl.range(pid * BLOCK, n_elements, span):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask)
        result = tl.sin(values.to(tl.float32))
        tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def _can_use_flat_path(tensor):
    if not tensor.is_contiguous() or tensor.dtype not in _SUPPORTED_DTYPES:
        return False
    n_elements = tensor.numel()
    return _FLAT_PATH_MIN_ELEMENTS < n_elements <= _FLAT_PATH_MAX_ELEMENTS


def _sin_flat(x, out):
    n_elements = x.numel()
    block = (
        _FLAT_BLOCK_SMALL
        if n_elements <= _FLAT_BLOCK_SMALL_LIMIT
        else _FLAT_BLOCK_MEDIUM
    )
    grid = (min(triton.cdiv(n_elements, block), _MAX_GRID_X),)
    _sin_flat_kernel[grid](
        x.view(-1),
        out.view(-1),
        n_elements,
        BLOCK=block,
        num_warps=_FLAT_NUM_WARPS,
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
