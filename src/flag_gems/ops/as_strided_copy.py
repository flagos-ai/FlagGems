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


def _fallback_as_strided_copy(input, size, stride, storage_offset=None):
    return torch.ops.aten.as_strided_copy.default.redispatch(
        _FALLBACK_KEYSET,
        input,
        size,
        stride,
        storage_offset,
    )


def _fallback_as_strided_copy_out(input, size, stride, storage_offset=None, *, out):
    return torch.ops.aten.as_strided_copy.out.redispatch(
        _FALLBACK_KEYSET,
        input,
        size,
        stride,
        storage_offset,
        out=out,
    )


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


def as_strided_copy(input, size, stride, storage_offset=None):
    logger.debug("GEMS AS_STRIDED_COPY")
    if input.device.type != "cuda":
        view = _make_as_strided_view(input, size, stride, storage_offset)
        return view.clone(memory_format=torch.contiguous_format)

    out = torch.empty(size, dtype=input.dtype, device=input.device)
    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out
    if not _can_use_triton(input, out):
        return _fallback_as_strided_copy(input, size, stride, storage_offset)

    view = _make_as_strided_view(input, size, stride, storage_offset)
    return _launch_as_strided_copy(view, out)


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
        not _can_use_triton(input, out)
        or torch._C._is_alias_of(input, out)
        or has_internal_overlapping(out) != MemOverlap.No
    ):
        return _fallback_as_strided_copy_out(
            input, size, stride, storage_offset, out=out
        )

    view = _make_as_strided_view(input, size, stride, storage_offset)
    return _launch_as_strided_copy(view, out)
