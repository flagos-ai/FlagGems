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

"""GCU400 ``_foreach_zero`` / ``_foreach_zero_``: one launch per tensor group.

The original local copy launched one flat 1-D Triton kernel per tensor to
sidestep the generic executor's two GCU400 blockers (grid.y hard cap 255
and the torch_gcu int64 transfer truncation predating the
``TORCH_GCU_ENABLE_INT64_AND_UINT64`` default).  Sixteen launches at ~11us
of host dispatch each measured 168-197us regardless of shape (official bench
core, 16-tensor list, every dtype), i.e. 0.72-0.89x out-of-place and
0.22-0.35x in-place against the native per-tensor ``zero_`` baseline.

The single-launch route this file now takes is the one the unary foreach
family shipped on this device (``gcu400/ops/_foreach_unary.py``
``_launch_large_block_group``, 2026-11-16): the pointer/numel table rides
``_device_meta``'s memoised int64 buffer (zero H2D on a steady-state call,
requires ``TORCH_GCU_ENABLE_INT64_AND_UINT64=1`` like every other batched
foreach op here), the launch replays the cached launcher through
``_foreach_reduction._launch``, and the grid is 2-D ``(NT, chunks)`` with
the chunk axis grid-strided so ``grid.y`` never needs to exceed the 255
hardware cap even for giant tensors -- BLOCK stays inside the DSM budget
instead of doubling.

A zero never reads its output, so the kernel is pure store: no input
pointer row in the metadata, no load, and the alignment split plus the
mask-free full-chunk path are what let the compiler emit wide vector
stores (``foreach_unary_kernel`` documents the measured 2.2x cost of losing
them).
"""

import logging
from typing import Any, Dict, List, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import _device_meta, check_tensor_list, tl_dtype

from ._foreach_reduction import _launch

logger = logging.getLogger(__name__)

# GCU400 grid limits (enflame backend launch validation): grid.x <= 65535,
# grid.y <= 255.  The batched kernel keeps grid.x = NT and strides grid.y,
# so neither cap is ever violated.
_MAX_GRID_X = 65535
_MAX_GRID_Y = 255

# Store vector per program, chosen like the unary executor's light rows:
# 256 KiB live per CTA for every itemsize, inside the 917504B DSM budget.
_BLOCK_16 = 131072
_BLOCK_32 = 65536

# Historical flat per-tensor kernel: still the engine behind the
# non-contiguous staging fallback, where any dtype (including ones
# ``tl_dtype`` cannot name) goes through a dense buffer and a copy back.
_BASE_BLOCK = 131072


@triton.jit
def _foreach_zero_kernel(out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    tl.store(
        out_ptr + offsets,
        tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _foreach_zero_list_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero every contiguous tensor of one group in a single launch.

    ``meta_ptr`` layout, all int64: ``[NT output pointers..., NT element
    counts...]``.  The grid is ``(NT, min(chunks, 255))``: axis 0 selects the
    tensor, axis 1 strides the chunk range so tensors larger than one grid
    fill still complete within the launch.

    Chunk offsets are a single ``int64`` product (``chunk.to(tl.int64) *
    BLOCK``) followed by the tile vector -- the safe addressing shape for
    element offsets past 2**31, per the compiler-trap table.
    """
    t = tl.program_id(0)
    out_v = tl.load(meta_ptr + t)
    n_elements = tl.load(meta_ptr + NT + t)

    zeros = tl.zeros((BLOCK,), dtype=DT)
    if (out_v & 15) == 0:
        out_ptr = tl.multiple_of(out_v.to(tl.pointer_type(DT)), 16)
    else:
        out_ptr = out_v.to(tl.pointer_type(DT))

    chunk = tl.program_id(1)
    ny = tl.num_programs(1)
    while chunk.to(tl.int64) * BLOCK < n_elements:
        offset = chunk.to(tl.int64) * BLOCK
        if offset + BLOCK <= n_elements:
            idx = offset + tl.arange(0, BLOCK)
            tl.store(out_ptr + idx, zeros)
        else:
            idx = offset + tl.arange(0, BLOCK)
            mask = idx < n_elements
            tl.store(out_ptr + idx, zeros, mask=mask)
        chunk += ny


def _zero_group_launch(work: List[torch.Tensor]) -> None:
    """Zero one ``(device, dtype)`` contiguous group with one launch.

    BLOCK follows the unary executor's small-tensor cap (never above the
    next power of two over the data, floor 1024); grid.y is the chunk count
    clamped to the hardware cap because the kernel strides whatever it is
    handed.  Lists longer than the grid.x cap are sliced, each slice with
    its own metadata row block.
    """
    numels = [t.numel() for t in work]
    max_numel = max(numels)
    if max_numel == 0:
        return
    dtype = work[0].dtype
    block = _BLOCK_16 if dtype.itemsize <= 2 else _BLOCK_32
    block = min(block, max(1024, 1 << (max_numel - 1).bit_length()))
    grid_y = min(-(-max_numel // block), _MAX_GRID_Y)
    dt = tl_dtype(dtype)
    for start in range(0, len(work), _MAX_GRID_X):
        part = work[start : start + _MAX_GRID_X]
        meta = _device_meta(
            [t.data_ptr() for t in part] + numels[start : start + _MAX_GRID_X],
            part[0].device,
        )
        _launch(
            _foreach_zero_list_kernel,
            (len(part), grid_y),
            (len(part), dt, block),
            (meta, len(part), dt, block),
            (meta,),
            part[0].device,
        )


def _zero_one(t: torch.Tensor, block: int, num_warps: int) -> None:
    """Zero one tensor in place, staging non-contiguous views.

    ATen's ``_foreach_zero_`` accepts any view (a strided slice must be
    zeroed in the original storage), so a non-contiguous target is zeroed
    through a dense buffer and copied back, mirroring the element-wise
    executor's behaviour for gappy views.
    """
    n = t.numel()
    if n == 0:
        return
    if not t.is_contiguous():
        staged = torch.empty_like(t, memory_format=torch.contiguous_format)
        _zero_one(staged, block, num_warps)
        t.copy_(staged)
        return
    # Grow BLOCK so the flat grid stays within the GCU400 grid.x limit.
    while triton.cdiv(n, block) > _MAX_GRID_X:
        block *= 2
    _foreach_zero_kernel[(triton.cdiv(n, block),)](t, n, block, num_warps=num_warps)


def _zero_list_fast(tensors: List[torch.Tensor]) -> None:
    """Zero a whole list; one launch per ``(device, dtype)`` group.

    Contiguous tensors in a dtype the batched kernel can name go to the
    single-launch path; everything else (non-contiguous views, complex
    inputs) keeps the per-tensor staging fallback.
    """
    groups: Dict[Tuple[Any, torch.dtype], List[torch.Tensor]] = {}
    fallback: List[torch.Tensor] = []
    for tensor in tensors:
        if tensor.is_contiguous():
            try:
                tl_dtype(tensor.dtype)
            except NotImplementedError:
                fallback.append(tensor)
            else:
                groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        else:
            fallback.append(tensor)
    for work in groups.values():
        _zero_group_launch(work)
    for tensor in fallback:
        _zero_one(tensor, _BASE_BLOCK, 4)


def _foreach_zero(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    logger.debug("GEMS_GCU400 _FOREACH_ZERO")
    tensors = check_tensor_list(self)
    _zero_list_fast(tensors)
    return tensors


def _foreach_zero_(self: Sequence[torch.Tensor]) -> None:
    logger.debug("GEMS_GCU400 _FOREACH_ZERO_")
    tensors = check_tensor_list(self)
    _zero_list_fast(tensors)
    return None


__all__ = ["_foreach_zero", "_foreach_zero_"]
