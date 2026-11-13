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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _copy_storage_kernel(src, out, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out + offsets, tl.load(src + offsets, mask=mask), mask=mask)


@triton.jit
def _as_strided_owner_kernel(
    owners,
    sizes,
    strides,
    n_elements,
    storage_offset,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets.to(tl.int64)
    target = tl.full(offsets.shape, storage_offset, tl.int64)

    # Preload sizes and strides to avoid repeated global memory access
    for reverse_dim in tl.static_range(NDIM):
        dim = NDIM - reverse_dim - 1
        dim_size = tl.load(sizes + dim)
        dim_stride = tl.load(strides + dim)
        coord = remaining % dim_size
        target += coord * dim_stride
        remaining //= dim_size
    # ATen's TensorIterator copy keeps the first logical source element when
    # multiple source indices address the same destination storage slot.
    tl.atomic_min(owners + target, offsets.to(tl.int64), mask=mask)


@triton.jit
def _as_strided_scatter_direct_kernel(
    src,
    out,
    n_elements,
    storage_offset,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    # Non-overlapping view: every destination storage slot is written by at
    # most one logical source element, so the atomic owner table is unnecessary.
    # Sizes/strides stay scalar device values; keeping them in the index dtype
    # avoids promoting the whole coordinate decomposition to 64-bit division.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_INT64:
        remaining = offsets.to(tl.int64)
        target = tl.full(offsets.shape, storage_offset, tl.int64)
    else:
        remaining = offsets
        target = tl.full(offsets.shape, storage_offset, tl.int32)
    for reverse_dim in tl.static_range(NDIM):
        dim_size = SIZES[reverse_dim]
        dim_stride = STRIDES[reverse_dim]
        coord = remaining % dim_size
        target += coord * dim_stride
        remaining = remaining // dim_size
    values = tl.load(src + offsets, mask=mask)
    tl.store(out + target, values, mask=mask)


@triton.jit
def _as_strided_fused_kernel(
    self_storage,
    src,
    out,
    storage_numel,
    target_offset,
    SIZES: tl.constexpr,
    STRIDES: tl.constexpr,
    SRC_MULTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    # Single-pass scatter for a positive-stride injective view: each program
    # owns a tile of the *output storage* and copies the original value unless
    # that slot is addressed by the view. The inverse coordinate decomposition
    # is unique because each stride exceeds the span of all lower dimensions.
    output_indices = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_indices < storage_numel
    original_value = tl.load(self_storage + output_indices, mask=mask, other=0)
    remaining = output_indices - target_offset
    src_offset = tl.zeros_like(output_indices)
    is_view_slot = remaining >= 0
    for dim in tl.static_range(NDIM):
        dim_size = SIZES[dim]
        dim_stride = STRIDES[dim]
        if dim_size > 1:
            coord = remaining // dim_stride
            src_offset += coord * SRC_MULTS[dim]
            remaining -= coord * dim_stride
            is_view_slot &= (coord >= 0) & (coord < dim_size)
    is_view_slot &= remaining == 0
    # Keep the load unconditional with a clamped index: on GCU400 a predicate
    # that depends on the loop-carried `remaining` mis-selects lanes when the
    # innermost stride is 1, leaving stale values in scattered slots.
    safe_src_offset = tl.where(is_view_slot, src_offset, 0)
    scattered_value = tl.load(src + safe_src_offset, mask=mask, other=0)
    tl.store(
        out + output_indices,
        tl.where(is_view_slot, scattered_value, original_value),
        mask=mask,
    )


def _view_is_injective(size, stride):
    """Conservative sufficient test that an as-strided view has no aliasing.

    Walking dimensions from innermost to outermost, each dimension's stride
    must exceed the whole address span reachable by all lower dimensions; then
    distinct coordinates can never map to the same storage slot. Zero / negative
    strides are handled conservatively, so unsafe views keep the owner path.
    """
    lower_span = 0
    for dim in range(len(size) - 1, -1, -1):
        if size[dim] == 0:
            return True
        if size[dim] > 1:
            step = abs(stride[dim])
            if step <= lower_span:
                return False
            lower_span += (size[dim] - 1) * step
    return True


@triton.jit
def _as_strided_scatter_kernel(
    src,
    out,
    owners,
    sizes,
    strides,
    n_elements,
    storage_offset,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets.to(tl.int64)
    target = tl.full(offsets.shape, storage_offset, tl.int64)

    # Preload sizes and strides to avoid repeated global memory access
    for reverse_dim in tl.static_range(NDIM):
        dim = NDIM - reverse_dim - 1
        dim_size = tl.load(sizes + dim)
        dim_stride = tl.load(strides + dim)
        coord = remaining % dim_size
        target += coord * dim_stride
        remaining //= dim_size

    is_selected_writer = (
        tl.load(owners + target, mask=mask, other=n_elements) == offsets
    )
    values = tl.load(src + offsets, mask=mask)
    tl.store(out + target, values, mask=mask & is_selected_writer)


def as_strided_scatter(
    self: torch.Tensor,
    src: torch.Tensor,
    size,
    stride,
    storage_offset=None,
) -> torch.Tensor:
    """Clone ``self`` storage and scatter ``src`` through an as-strided view."""
    logger.debug("GEMS AS_STRIDED_SCATTER")
    if self.device != src.device or self.dtype != src.dtype:
        raise RuntimeError("self and src must have the same device and dtype")
    if self.layout != torch.strided or src.layout != torch.strided:
        raise RuntimeError("as_strided_scatter supports strided tensors only")
    if len(size) != len(stride):
        raise RuntimeError("mismatch in length of strides and shape")

    size = tuple(int(value) for value in size)
    stride = tuple(int(value) for value in stride)
    target_offset = (
        self.storage_offset() if storage_offset is None else int(storage_offset)
    )
    # Let ATen validate negative strides and storage bounds without performing a copy.
    torch.as_strided(self, size, stride, target_offset)
    expected_numel = math.prod(size)
    if src.numel() != expected_numel:
        raise RuntimeError("src size must match the requested as_strided view size")

    storage_numel = self.untyped_storage().nbytes() // self.element_size()
    storage_view = torch.as_strided(self, (storage_numel,), (1,), 0)
    out_storage = torch.empty(storage_numel, dtype=self.dtype, device=self.device)
    if storage_numel == 0:
        return torch.as_strided(
            out_storage, self.size(), self.stride(), self.storage_offset()
        )

    # Fast path: if the view covers the entire storage with no overlap,
    # we can directly copy without atomic tracking.
    # This is true when the view is contiguous and spans the storage.
    is_contiguous_view = len(size) > 0 and all(
        stride[i] == math.prod(size[i + 1 :]) for i in range(len(size))
    )
    covers_storage = (
        target_offset == 0 and expected_numel == storage_numel and is_contiguous_view
    )

    # A 1024-element tile for better parallelism and reduced launch overhead.
    block_size = 1024
    injective_view = _view_is_injective(size, stride)
    fused_view = (
        not covers_storage
        and expected_numel > 0
        and injective_view
        and all(step > 0 or size[i] == 1 for i, step in enumerate(stride))
    )
    with torch_device_fn.device(self.device):
        # The fused kernel writes every output slot itself, so only the other
        # paths need a separate kernel that preserves the original storage.
        if not covers_storage and not fused_view:
            _copy_storage_kernel[(triton.cdiv(storage_numel, block_size),)](
                storage_view,
                out_storage,
                storage_numel,
                BLOCK_SIZE=block_size,
            )

        if expected_numel:
            src = src.contiguous()

            if covers_storage:
                # Fast path: direct copy since src covers entire storage
                src_flat = src.flatten()
                grid = (triton.cdiv(expected_numel, block_size),)
                _copy_storage_kernel[grid](
                    src_flat,
                    out_storage,
                    expected_numel,
                    BLOCK_SIZE=block_size,
                )
            elif fused_view:
                src_mults = tuple(math.prod(size[i + 1 :]) for i in range(len(size)))
                fused_grid = (triton.cdiv(storage_numel, block_size),)
                _as_strided_fused_kernel[fused_grid](
                    storage_view,
                    src,
                    out_storage,
                    storage_numel,
                    target_offset,
                    tuple(size),
                    tuple(stride),
                    src_mults,
                    BLOCK_SIZE=block_size,
                    NDIM=len(size),
                )
            elif injective_view:
                # No aliasing: one scatter kernel replaces owner init + atomic
                # min + owner-checked scatter. Keep int64 indices only for
                # storages that cannot be addressed by 32-bit offsets.
                use_int64 = max(storage_numel, expected_numel) > (1 << 31)
                direct_grid = (triton.cdiv(expected_numel, block_size),)
                _as_strided_scatter_direct_kernel[direct_grid](
                    src,
                    out_storage,
                    expected_numel,
                    target_offset,
                    tuple(reversed(size)),
                    tuple(reversed(stride)),
                    BLOCK_SIZE=block_size,
                    NDIM=len(size),
                    USE_INT64=use_int64,
                    num_warps=2,
                )
            else:
                # Slow path: need atomic tracking for potential overlaps
                sizes = torch.tensor(size, dtype=torch.int64, device=self.device)
                strides = torch.tensor(stride, dtype=torch.int64, device=self.device)
                owners = torch.full(
                    (storage_numel,),
                    expected_numel,
                    dtype=torch.int64,
                    device=self.device,
                )
                grid = (triton.cdiv(expected_numel, block_size),)
                _as_strided_owner_kernel[grid](
                    owners,
                    sizes,
                    strides,
                    expected_numel,
                    target_offset,
                    BLOCK_SIZE=block_size,
                    NDIM=len(size),
                )
                _as_strided_scatter_kernel[grid](
                    src,
                    out_storage,
                    owners,
                    sizes,
                    strides,
                    expected_numel,
                    target_offset,
                    BLOCK_SIZE=block_size,
                    NDIM=len(size),
                )

    return torch.as_strided(
        out_storage, self.size(), self.stride(), self.storage_offset()
    )
