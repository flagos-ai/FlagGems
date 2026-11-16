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

"""Triton implementation of ``aten::_indices_copy`` for sparse COO tensors.

GCU400's sparse COO constructor currently falls back to SparseCPU, so ``self._indices()`` lives on CPU,
while Triton kernels can only access GCU pointers. Move the index buffer to the device first, let the Triton kernel
do the core copy, then write back according to the output tensor's device; these transfers are unavoidable
pre/post-processing.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Tunable constants replaced per optimization round; keep a single kernel definition to avoid cold recompilation from
# repeated specialization.
FLAT_BLOCK_SIZE = 2048
FLAT_NUM_WARPS = 4
FLAT_FIXED_GRID = 0
TWO_D_BLOCK_SIZE = 4096
TWO_D_NUM_WARPS = 2
REBIND_OUT_TO_SOURCE_DEVICE = True


@triton.jit
def _indices_copy_flat(
    src_ptr,
    dst_ptr,
    numel,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """1D copy of a contiguous index buffer; with FIXED_GRID>0 use a fixed grid plus a stride loop."""
    lane_offsets = tl.arange(0, BLOCK)
    num_tiles = tl.cdiv(numel, BLOCK)
    if FIXED_GRID > 0:
        for tile_id in tl.range(tl.program_id(0), num_tiles, FIXED_GRID):
            offsets = tile_id * BLOCK + lane_offsets
            mask = offsets < numel
            values = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, values, mask=mask)
    else:
        offsets = tl.program_id(0) * BLOCK + lane_offsets
        mask = offsets < numel
        values = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _indices_copy_2d(src_ptr, dst_ptr, nnz, s0, s1, BLOCK: tl.constexpr):
    """2D strided gather for non-contiguous index views, writing rows back into a compact output."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < nnz
    values = tl.load(src_ptr + row * s0 + cols * s1, mask=mask)
    tl.store(dst_ptr + row * nnz + cols, values, mask=mask)


def _current_gcu_device():
    """Return the current GCU device; fall back to default device 0 when current_device is unavailable."""
    device_index = (
        torch.gcu.current_device() if hasattr(torch.gcu, "current_device") else 0
    )
    return torch.device("gcu", device_index)


def _indices_copy_impl(self, out=None, *, caller_out=False):
    """Copy the sparse COO index buffer into a new buffer or the caller-provided ``out``.

    When the source index is on CPU, do one H2D first and let the Triton kernel perform the copy on device; if the
    caller's
    ``out`` is on a different device from the source index, rebind the same object to storage on the source device after
    write-back,
    so the ``out`` variant returns an object consistent with the comparison convention.
    """
    plain_indices = self._indices()
    sparse_dim, nnz = plain_indices.shape
    if out is None:
        out = torch.empty(
            (sparse_dim, nnz), dtype=torch.int64, device=plain_indices.device
        )
    if nnz == 0:
        return out

    gcu_device = _current_gcu_device()
    source = (
        plain_indices
        if plain_indices.device.type == "gcu"
        else plain_indices.to(gcu_device)
    )
    if out.device.type == "gcu":
        destination = out
    else:
        destination = torch.empty(
            (sparse_dim, nnz), dtype=torch.int64, device=gcu_device
        )

    with torch_device_fn.device(gcu_device):
        if source.is_contiguous():
            numel = sparse_dim * nnz
            block_size = FLAT_BLOCK_SIZE
            if FLAT_FIXED_GRID > 0:
                grid = (min(FLAT_FIXED_GRID, triton.cdiv(numel, block_size)),)
            else:
                grid = (triton.cdiv(numel, block_size),)
            _indices_copy_flat[grid](
                source,
                destination,
                numel,
                BLOCK=block_size,
                FIXED_GRID=FLAT_FIXED_GRID,
                num_warps=FLAT_NUM_WARPS,
            )
        else:
            stride_row, stride_col = source.stride()
            block_size = TWO_D_BLOCK_SIZE
            grid = (sparse_dim, triton.cdiv(nnz, block_size))
            _indices_copy_2d[grid](
                source,
                destination,
                nnz,
                stride_row,
                stride_col,
                BLOCK=block_size,
                num_warps=TWO_D_NUM_WARPS,
            )

    if destination is not out:
        out.copy_(destination.to(out.device))
    elif caller_out and out.device != plain_indices.device:
        # Environment difference between CPU sparse indices and a GCU out: keep the same out object, rebind its storage.
        # set_ rejects cross-device assignment (measured RuntimeError); assigning to .data switches device in place
        # while preserving object identity.
        if REBIND_OUT_TO_SOURCE_DEVICE:
            out.data = destination.to(plain_indices.device)
    return out


def _indices_copy(self: torch.Tensor) -> torch.Tensor:
    """``aten::_indices_copy``: return an independent copy of the index buffer."""
    logger.debug("GEMS_ENFLAME_GCU400 _INDICES_COPY")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_indices_copy only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    return _indices_copy_impl(self)


def _indices_copy_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """``aten::_indices_copy.out``: write the index buffer into the caller-provided ``out``."""
    logger.debug("GEMS_ENFLAME_GCU400 _INDICES_COPY_OUT")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_indices_copy only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    if out.dtype != torch.int64:
        raise RuntimeError(
            "Expected out tensor to have dtype torch.int64, but got "
            f"{out.dtype} instead"
        )
    sparse_dim, nnz = self._indices().shape
    if tuple(out.shape) != (sparse_dim, nnz):
        out.resize_((sparse_dim, nnz))
    return _indices_copy_impl(self, out=out, caller_out=True)
