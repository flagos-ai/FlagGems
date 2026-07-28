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
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_MAX_TRITON_ELEMENTS = torch.iinfo(torch.int32).max
_BLOCK_SIZE = 512
_BLOCK_M = 16
_BLOCK_N = 16


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _as_strided_copy_kernel(x):
    return x


@triton.jit
def _as_strided_copy_1d_kernel(
    input,
    out,
    input_stride_0,
    out_stride_0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    values = tl.load(input + offsets * input_stride_0, mask=mask)
    tl.store(out + offsets * out_stride_0, values, mask=mask)


@triton.jit
def _as_strided_copy_2d_kernel(
    input,
    out,
    input_stride_0,
    input_stride_1,
    out_stride_0,
    out_stride_1,
    dim_0,
    dim_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_m = offsets_m.to(tl.int64)[:, None]
    offsets_n = offsets_n.to(tl.int64)[None, :]
    mask = (offsets_m < dim_0) & (offsets_n < dim_1)
    input_offsets = offsets_m * input_stride_0 + offsets_n * input_stride_1
    out_offsets = offsets_m * out_stride_0 + offsets_n * out_stride_1
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(out + out_offsets, values, mask=mask)


@triton.jit
def _as_strided_copy_3d_kernel(
    input,
    out,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    dim_1,
    dim_2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    index_2 = offsets % dim_2
    tmp = offsets // dim_2
    index_1 = tmp % dim_1
    index_0 = tmp // dim_1
    input_offsets = (
        index_0 * input_stride_0 + index_1 * input_stride_1 + index_2 * input_stride_2
    )
    out_offsets = (
        index_0 * out_stride_0 + index_1 * out_stride_1 + index_2 * out_stride_2
    )
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(out + out_offsets, values, mask=mask)


def _is_float8(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8_")


def _has_lazy_metadata(tensor: torch.Tensor) -> bool:
    is_neg = getattr(tensor, "is_neg", lambda: False)
    return tensor.is_conj() or is_neg()


def _make_as_strided_view(
    input: torch.Tensor,
    size,
    stride,
    storage_offset,
) -> torch.Tensor:
    # Reuse PyTorch's view construction to match its validation and None-offset semantics.
    if storage_offset is None:
        return torch.as_strided(input, size, stride)
    return torch.as_strided(input, size, stride, storage_offset)


def _native_copy_(out: torch.Tensor, src: torch.Tensor):
    return torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, src, False)


def _fallback_as_strided_copy(input, size, stride, storage_offset=None):
    view = _make_as_strided_view(input, size, stride, storage_offset)
    out = torch.empty(tuple(size), dtype=input.dtype, device=input.device)
    if out.numel() != 0:
        # Call native copy_ directly so unsupported CUDA dtypes do not re-enter
        # FlagGems copy kernels through the composite as_strided_copy fallback.
        _native_copy_(out, view)
    return out


def _fallback_as_strided_copy_out(input, size, stride, storage_offset=None, *, out):
    view = _make_as_strided_view(input, size, stride, storage_offset)
    if (
        torch._C._is_alias_of(input, out)
        or has_internal_overlapping(out) != MemOverlap.No
    ):
        temp = torch.empty(tuple(size), dtype=input.dtype, device=input.device)
        if temp.numel() != 0:
            _native_copy_(temp, view)
        view = temp
    _native_copy_(out, view)
    return out


def _can_use_triton(input: torch.Tensor, out: torch.Tensor) -> bool:
    if input.layout != torch.strided or out.layout != torch.strided:
        return False
    if input.device != out.device or input.dtype != out.dtype:
        return False
    if input.is_quantized or out.is_quantized:
        return False
    if input.is_complex() or _is_float8(input.dtype):
        return False
    if out.numel() > _MAX_TRITON_ELEMENTS:
        return False
    return True


def _can_use_byte_triton(input: torch.Tensor, out: torch.Tensor) -> bool:
    if input.layout != torch.strided or out.layout != torch.strided:
        return False
    if input.device != out.device or input.dtype != out.dtype:
        return False
    if not _is_float8(input.dtype):
        return False
    if input.element_size() != 1 or out.element_size() != 1:
        return False
    if _has_lazy_metadata(input) or _has_lazy_metadata(out):
        return False
    if out.numel() > _MAX_TRITON_ELEMENTS:
        return False
    return True


def _launch_as_strided_copy(view: torch.Tensor, out: torch.Tensor):
    dim = view.dim()
    if dim == 0:
        _as_strided_copy_1d_kernel[(1,)](
            view,
            out,
            0,
            0,
            1,
            BLOCK_SIZE=1,
        )
    elif dim == 1:
        n_elements = view.numel()
        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
        _as_strided_copy_1d_kernel[grid](
            view,
            out,
            view.stride(0),
            out.stride(0),
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    elif dim == 2:
        dim_0, dim_1 = view.shape
        grid = (triton.cdiv(dim_0, _BLOCK_M), triton.cdiv(dim_1, _BLOCK_N))
        _as_strided_copy_2d_kernel[grid](
            view,
            out,
            view.stride(0),
            view.stride(1),
            out.stride(0),
            out.stride(1),
            dim_0,
            dim_1,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
        )
    elif dim == 3:
        n_elements = view.numel()
        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
        _as_strided_copy_3d_kernel[grid](
            view,
            out,
            view.stride(0),
            view.stride(1),
            view.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            view.shape[1],
            view.shape[2],
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    else:
        return _as_strided_copy_kernel(view, out0=out)
    return out


def _launch_byte_as_strided_copy(view: torch.Tensor, out: torch.Tensor):
    # Copy one-byte dtypes through uint8 views to avoid Triton fp8 scalar codegen.
    # The dtype-view API requires at least one logical dimension on some builds.
    byte_view = (
        view.reshape(1).view(torch.uint8) if view.dim() == 0 else view.view(torch.uint8)
    )
    byte_out = (
        out.reshape(1).view(torch.uint8) if out.dim() == 0 else out.view(torch.uint8)
    )
    _launch_as_strided_copy(byte_view, byte_out)
    return out


def as_strided_copy(input, size, stride, storage_offset=None):
    logger.debug("GEMS AS_STRIDED_COPY")
    if input.device.type != "cuda":
        view = _make_as_strided_view(input, size, stride, storage_offset)
        return view.clone(memory_format=torch.contiguous_format)

    out = torch.empty(size, dtype=input.dtype, device=input.device)
    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out

    view = _make_as_strided_view(input, size, stride, storage_offset)
    if _can_use_triton(view, out):
        return _launch_as_strided_copy(view, out)
    if _can_use_byte_triton(view, out):
        return _launch_byte_as_strided_copy(view, out)
    return _fallback_as_strided_copy(input, size, stride, storage_offset)


def as_strided_copy_out(input, size, stride, storage_offset=None, *, out):
    logger.debug("GEMS AS_STRIDED_COPY_OUT")
    if out.dtype != input.dtype:
        # Match PyTorch's strict out-dtype contract without measuring native fallback.
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )

    target_size = tuple(size)
    if tuple(out.shape) != target_size:
        out.resize_(target_size)

    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out

    if input.device.type != "cuda":
        view = _make_as_strided_view(input, size, stride, storage_offset)
        if (
            torch._C._is_alias_of(input, out)
            or has_internal_overlapping(out) != MemOverlap.No
        ):
            view = view.clone(memory_format=torch.contiguous_format)
        out.copy_(view)
        return out

    if (
        torch._C._is_alias_of(input, out)
        or has_internal_overlapping(out) != MemOverlap.No
    ):
        return _fallback_as_strided_copy_out(
            input, size, stride, storage_offset, out=out
        )

    view = _make_as_strided_view(input, size, stride, storage_offset)
    if _can_use_triton(view, out):
        return _launch_as_strided_copy(view, out)
    if _can_use_byte_triton(view, out):
        return _launch_byte_as_strided_copy(view, out)
    return _fallback_as_strided_copy_out(input, size, stride, storage_offset, out=out)
