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

_BLOCK_SIZE = 512
_BLOCK_DIM0 = 32
_BLOCK_DIM1 = 32
_MAX_TILED_RANK = 4
_MAX_TRITON_OFFSET = torch.iinfo(torch.int32).max


@triton.jit
def _copy_kernel(
    input,
    out,
    input_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input + offsets * input_stride, mask=mask, other=0)
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def _transpose_copy_tiled_kernel(
    input,
    out,
    input_stride_dim0,
    input_stride_dim1,
    out_stride_dim0,
    out_stride_dim1,
    input_stride_other0,
    input_stride_other1,
    out_stride_other0,
    out_stride_other1,
    other_size1,
    dim0_size,
    dim1_size,
    tiles_dim1,
    tiles_per_slice,
    BLOCK_DIM0: tl.constexpr,
    BLOCK_DIM1: tl.constexpr,
):
    program_id = tl.program_id(0)
    slice_id = program_id // tiles_per_slice
    tile_id = program_id - slice_id * tiles_per_slice
    tile_dim0 = tile_id // tiles_dim1
    tile_dim1 = tile_id - tile_dim0 * tiles_dim1

    other_index0 = slice_id // other_size1
    other_index1 = slice_id - other_index0 * other_size1
    input_base = other_index0 * input_stride_other0 + other_index1 * input_stride_other1
    out_base = other_index0 * out_stride_other0 + other_index1 * out_stride_other1

    offsets_dim0 = tile_dim0 * BLOCK_DIM0 + tl.arange(0, BLOCK_DIM0)
    offsets_dim1 = tile_dim1 * BLOCK_DIM1 + tl.arange(0, BLOCK_DIM1)
    mask_dim0 = offsets_dim0 < dim0_size
    mask_dim1 = offsets_dim1 < dim1_size

    input_offsets = (
        input_base
        + offsets_dim0[:, None] * input_stride_dim0
        + offsets_dim1[None, :] * input_stride_dim1
    )
    values = tl.load(
        input + input_offsets,
        mask=mask_dim0[:, None] & mask_dim1[None, :],
        other=0,
    )

    out_offsets = (
        out_base
        + offsets_dim1[:, None] * out_stride_dim0
        + offsets_dim0[None, :] * out_stride_dim1
    )
    tl.store(
        out + out_offsets,
        tl.trans(values, 1, 0),
        mask=mask_dim1[:, None] & mask_dim0[None, :],
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    lower = -ndim if ndim > 0 else -1
    upper = ndim - 1 if ndim > 0 else 0
    if dim < lower or dim > upper:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{lower}, {upper}], but got {dim})"
        )
    if ndim == 0:
        return 0
    return dim % ndim


def _is_float8(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8_")


def _has_lazy_metadata(input: torch.Tensor) -> bool:
    is_neg = getattr(input, "is_neg", lambda: False)
    return input.is_conj() or is_neg()


def _can_use_triton(input: torch.Tensor) -> bool:
    if input.device.type == "cpu" or input.layout != torch.strided:
        return False
    if input.is_quantized or input.is_complex() or _is_float8(input.dtype):
        return False
    if _has_lazy_metadata(input):
        return False
    if input.numel() > _MAX_TRITON_OFFSET:
        return False
    max_input_offset = sum(
        (size - 1) * stride for size, stride in zip(input.shape, input.stride())
    )
    return max_input_offset <= _MAX_TRITON_OFFSET


def _fallback_transpose_copy(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return input.transpose(dim0, dim1).clone(memory_format=torch.contiguous_format)


def _launch_copy(input: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    input_stride = input.stride(0) if input.ndim == 1 else 1
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    with torch_device_fn.device(input.device):
        _copy_kernel[grid](
            input,
            out,
            input_stride,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return out


def _launch_tiled_transpose_copy(
    input: torch.Tensor,
    out: torch.Tensor,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    other_dims = [dim for dim in range(input.ndim) if dim not in (dim0, dim1)]
    other_sizes = [input.shape[dim] for dim in other_dims]
    input_other_strides = [input.stride(dim) for dim in other_dims]
    out_other_strides = [out.stride(dim) for dim in other_dims]

    while len(other_dims) < _MAX_TILED_RANK - 2:
        other_sizes.insert(0, 1)
        input_other_strides.insert(0, 0)
        out_other_strides.insert(0, 0)
        other_dims.insert(0, -1)

    dim0_size = input.shape[dim0]
    dim1_size = input.shape[dim1]
    tiles_dim0 = triton.cdiv(dim0_size, _BLOCK_DIM0)
    tiles_dim1 = triton.cdiv(dim1_size, _BLOCK_DIM1)
    tiles_per_slice = tiles_dim0 * tiles_dim1
    num_slices = other_sizes[0] * other_sizes[1]
    grid = (num_slices * tiles_per_slice,)

    with torch_device_fn.device(input.device):
        _transpose_copy_tiled_kernel[grid](
            input,
            out,
            input.stride(dim0),
            input.stride(dim1),
            out.stride(dim0),
            out.stride(dim1),
            input_other_strides[0],
            input_other_strides[1],
            out_other_strides[0],
            out_other_strides[1],
            other_sizes[1],
            dim0_size,
            dim1_size,
            tiles_dim1,
            tiles_per_slice,
            BLOCK_DIM0=_BLOCK_DIM0,
            BLOCK_DIM1=_BLOCK_DIM1,
        )
    return out


def transpose_copy(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return a contiguous copy with ``dim0`` and ``dim1`` swapped."""
    logger.debug("GEMS TRANSPOSE_COPY")

    normalized_dim0 = _normalize_dim(dim0, input.ndim)
    normalized_dim1 = _normalize_dim(dim1, input.ndim)

    out_shape = list(input.shape)
    if input.ndim > 0:
        out_shape[normalized_dim0], out_shape[normalized_dim1] = (
            out_shape[normalized_dim1],
            out_shape[normalized_dim0],
        )
    if input.numel() == 0:
        return torch.empty(out_shape, dtype=input.dtype, device=input.device)

    if not _can_use_triton(input):
        return _fallback_transpose_copy(input, normalized_dim0, normalized_dim1)

    if input.ndim <= 1:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        return _launch_copy(input, out)

    if normalized_dim0 == normalized_dim1:
        if input.is_contiguous():
            out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
            return _launch_copy(input, out)
        return _fallback_transpose_copy(input, normalized_dim0, normalized_dim1)

    if input.ndim <= _MAX_TILED_RANK:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        return _launch_tiled_transpose_copy(
            input, out, normalized_dim0, normalized_dim1
        )
    return _fallback_transpose_copy(input, normalized_dim0, normalized_dim1)
