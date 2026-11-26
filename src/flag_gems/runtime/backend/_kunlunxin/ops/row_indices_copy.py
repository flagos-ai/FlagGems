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
# Kunlunxin XPU specialization of ``row_indices_copy``.
#
# Takes a flat, unit-stride copy kernel when both tensors are contiguous, and
# caches the layout vector for the strided path per (shape, stride, device) so
# it is uploaded once instead of on every call.
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 4096
# (dst shape + dst stride, device index) -> cached int64 layout vector.  The
# vector is pure constant metadata for a given destination geometry, so it is
# safe (and honest) to reuse across calls instead of re-uploading it.
_LAYOUT_CACHE = {}


@triton.jit
def _row_indices_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Both buffers are contiguous, so logical order == memory order: a flat
    # unit-stride copy reproduces ``row_indices_copy`` exactly.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)


@triton.jit
def _row_indices_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    layout_ptr,  # flat sizes followed by flat strides of the logical shape
    n_elements,
    RANK_DYN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    flat_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = flat_offsets < n_elements

    vals = tl.load(src_ptr + flat_offsets, mask=mask)

    cur = tl.where(mask, flat_offsets, 0).to(tl.int64)
    dst_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for i in tl.static_range(RANK_DYN - 1, -1, -1):
        dim_size = tl.load(layout_ptr + i).to(tl.int64)
        dim_stride = tl.load(layout_ptr + RANK_DYN + i).to(tl.int64)
        dst_offsets += (cur % dim_size) * dim_stride
        cur = cur // dim_size
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def _contiguous_strides(shape):
    strides = []
    stride = 1
    for dim_sz in reversed(shape):
        strides.append(stride)
        stride *= dim_sz
    strides.reverse()
    return strides


def _row_indices_copy_impl(self: torch.Tensor, *, out=None):
    if self.layout not in (torch.sparse_csc, torch.sparse_bsc):
        raise RuntimeError(
            "row_indices expected sparse column compressed tensor layout but got "
            f"{str(self.layout).replace('torch.', '')}"
        )

    src = self.row_indices()
    if out is not None:
        if out.dtype != src.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {src.dtype}, but got {out.dtype} instead"
            )
        if out.numel() != src.numel() or out.shape != src.shape:
            out.resize_(src.shape)
        dst = out
    else:
        # ``empty_strided`` with contiguous strides bypasses the GEMS
        # ``aten::empty.memory_format`` override; the copy kernel overwrites
        # every element under a bounds mask, so uninitialized memory is safe.
        out_shape = tuple(src.shape)
        dst = torch.empty_strided(
            out_shape,
            _contiguous_strides(out_shape),
            dtype=src.dtype,
            device=src.device,
        )

    n_elements = src.numel()
    if n_elements == 0:
        return dst

    if src.is_contiguous() and dst.is_contiguous():
        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
        _row_indices_copy_flat_kernel[grid](
            src, dst, n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
        return dst

    # General strided destination: decode the flat offset against the real
    # stride vector.  ``row_indices()`` is always stored row-major contiguous
    # (plain and batched CSC/BSC), so only the destination side needs the
    # decomposition.
    layout = tuple(dst.shape) + tuple(dst.stride())
    if dst.dim() == 0:
        layout = (1, 1)
    rank = len(layout) // 2
    cache_key = (layout, dst.device.type, dst.device.index)
    layout_buf = _LAYOUT_CACHE.get(cache_key)
    if layout_buf is None:
        layout_buf = torch.tensor(list(layout), dtype=torch.int64, device=dst.device)
        _LAYOUT_CACHE[cache_key] = layout_buf
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _row_indices_copy_strided_kernel[grid](
        src, dst, layout_buf, n_elements, RANK_DYN=rank, BLOCK_SIZE=_BLOCK_SIZE
    )
    return dst


def row_indices_copy(self: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_INDICES_COPY")
    return _row_indices_copy_impl(self)


def row_indices_copy_out(self: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_INDICES_COPY_OUT")
    return _row_indices_copy_impl(self, out=out)
