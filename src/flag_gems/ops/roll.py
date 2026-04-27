import logging
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

IntOrInts = int | Sequence[int]
MAX_DIMS = 5


def roll(inp: torch.Tensor, shifts, dims=None) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    validate_inputs(inp, shifts, dims)
    if _can_use_triton(inp):
        if _can_use_flat_single_dim_triton(inp, dims):
            return _candidate_triton(inp, shifts, None)
        if _can_use_first_dim_triton(inp, dims):
            return _candidate_triton_first_dim(inp, shifts)
        if _can_use_last_dim_triton(inp, dims):
            return _candidate_triton_last_dim(inp, shifts)
        if dims is not None and not _is_empty_sequence(dims):
            dim_values = _as_tuple(dims)
            if len(dim_values) == 1:
                return _candidate_triton_single_dim(
                    inp,
                    _as_tuple(shifts)[0],
                    _canonicalize_dim(
                        dim_values[0],
                        inp.dim(),
                        allow_empty_wrap=inp.numel() == 0,
                    ),
                )
        return _candidate_triton(inp, shifts, dims)
    return _candidate_fallback(inp, shifts, dims)


def _candidate_triton(
    inp: torch.Tensor, shifts: IntOrInts, dims: IntOrInts | None = None
) -> torch.Tensor:
    shift_values = _as_tuple(shifts)

    if dims is None or _is_empty_sequence(dims):
        flattened = inp.reshape(-1).contiguous()
        out_flat = torch.empty_like(flattened)
        block = _select_flat_block(flattened)
        _launch_roll_flat_kernel(
            flattened,
            out_flat,
            shift_values[0] % max(flattened.numel(), 1),
            block=block,
        )
        return out_flat.reshape(inp.shape)

    return _candidate_triton_multi_dim(inp, shift_values, _as_tuple(dims))


def _candidate_triton_last_dim(inp: torch.Tensor, shifts: IntOrInts) -> torch.Tensor:
    shift = _as_tuple(shifts)[0] % inp.shape[-1]
    if shift == 0:
        return inp.contiguous().clone()

    out = torch.empty_like(inp)
    _launch_roll_last_dim_kernel(inp, out, shift)
    return out


def _candidate_triton_first_dim(inp: torch.Tensor, shifts: IntOrInts) -> torch.Tensor:
    shift = (_as_tuple(shifts)[0] % inp.shape[0]) * inp.stride(0)
    if shift == 0:
        return inp.contiguous().clone()

    out = torch.empty_like(inp)
    _launch_roll_flat_kernel(inp.reshape(-1), out.reshape(-1), shift, block=1024)
    return out


def _select_flat_block(inp: torch.Tensor) -> int:
    if inp.numel() <= 2048:
        return 128
    if inp.dtype is torch.float32 and inp.numel() >= (1 << 20):
        return 1024
    return 512


def _candidate_triton_single_dim(
    inp: torch.Tensor, shift: int, dim: int
) -> torch.Tensor:
    size = inp.size(dim)
    if size == 0:
        return inp.clone()

    shift %= size
    if shift == 0:
        return inp.clone()

    inp_contig = inp.contiguous()
    out = torch.empty_like(inp_contig)
    dim_stride = inp_contig.stride(dim)
    _launch_roll_single_dim_kernel(inp_contig, out, size, shift, dim_stride)
    return out


def _candidate_triton_multi_dim(
    inp: torch.Tensor, shifts: Sequence[int], dims: Sequence[int]
) -> torch.Tensor:
    if inp.numel() == 0:
        return inp.clone()

    effective_shifts = _normalize_roll_dims(inp.shape, shifts, dims)
    active_dims = [
        (dim, shift)
        for dim, (size, shift) in enumerate(zip(inp.shape, effective_shifts))
        if size and shift
    ]
    if not active_dims:
        return inp.contiguous().clone()

    if len(active_dims) == 1:
        dim, shift = active_dims[0]
        if inp.is_contiguous() and _can_use_first_dim_triton(inp, dim):
            return _candidate_triton_first_dim(inp, shift)
        if inp.is_contiguous() and _can_use_last_dim_triton(inp, dim):
            return _candidate_triton_last_dim(inp, shift)
        return _candidate_triton_single_dim(inp, shift, dim)

    inp_contig = inp.contiguous()
    out = torch.empty_like(inp_contig)
    sizes = [inp_contig.size(dim) for dim, _ in active_dims]
    strides = [inp_contig.stride(dim) for dim, _ in active_dims]
    active_shifts = [shift for _, shift in active_dims]
    _launch_roll_multi_dim_kernel(inp_contig, out, sizes, strides, active_shifts)
    return out


def _candidate_fallback(
    inp: torch.Tensor, shifts: IntOrInts, dims: IntOrInts | None = None
) -> torch.Tensor:
    shift_values = _as_tuple(shifts)

    if dims is None or _is_empty_sequence(dims):
        flattened = inp.reshape(-1)
        return _roll_along_dim(flattened, shift_values[0], 0).reshape(inp.shape)

    result = inp
    for shift, dim in zip(shift_values, _as_tuple(dims)):
        result = _roll_along_dim(
            result,
            shift,
            _canonicalize_dim(dim, inp.dim(), allow_empty_wrap=inp.numel() == 0),
        )
    return result


def validate_inputs(
    inp: torch.Tensor, shifts: IntOrInts, dims: IntOrInts | None = None
) -> None:
    if not isinstance(inp, torch.Tensor):
        raise TypeError("roll(): argument 'input' must be Tensor")
    if not _is_int_or_int_sequence(shifts):
        raise TypeError("roll(): argument 'shifts' must be int or tuple of ints")
    shift_count = 1 if isinstance(shifts, int) else len(shifts)
    if shift_count == 0:
        raise RuntimeError("`shifts` required")

    if dims is None or _is_empty_sequence(dims):
        if shift_count > 1:
            raise RuntimeError(
                f"shifts and dimensions must align. shifts: {shift_count}, dims:0"
            )
        return

    if not _is_int_or_int_sequence(dims):
        raise TypeError("roll(): argument 'dims' must be int or tuple of ints")
    dim_count = 1 if isinstance(dims, int) else len(dims)
    if shift_count != dim_count:
        raise RuntimeError("shifts and dimensions must align")


def _roll_along_dim(inp: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    size = inp.size(dim)
    if size == 0:
        return inp.clone(memory_format=torch.preserve_format)

    shift %= size
    if shift == 0:
        return inp.clone(memory_format=torch.preserve_format)

    split = size - shift
    return torch.cat(
        (inp.narrow(dim, split, shift), inp.narrow(dim, 0, split)), dim=dim
    )


def _canonicalize_dim(dim: int, ndim: int, allow_empty_wrap: bool = False) -> int:
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if allow_empty_wrap:
        return dim % ndim
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _as_tuple(value: IntOrInts) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _can_use_triton(inp: torch.Tensor) -> bool:
    return inp.is_cuda and inp.dim() <= MAX_DIMS and not inp.dtype.is_complex


def _can_use_first_dim_triton(inp: torch.Tensor, dims: IntOrInts | None) -> bool:
    if not _can_use_triton(inp) or not inp.is_contiguous() or inp.dim() <= 1:
        return False

    if isinstance(dims, int):
        dim = dims
    elif isinstance(dims, Sequence) and not isinstance(dims, int) and len(dims) == 1:
        dim = dims[0]
    else:
        return False

    return dim in {0, -inp.dim()} and inp.numel() >= (1 << 20)


def _can_use_flat_single_dim_triton(inp: torch.Tensor, dims: IntOrInts | None) -> bool:
    if not _can_use_triton(inp) or inp.dim() != 1 or inp.dtype is not torch.float32:
        return False

    if isinstance(dims, int):
        dim = dims
    elif isinstance(dims, Sequence) and not isinstance(dims, int) and len(dims) == 1:
        dim = dims[0]
    else:
        return False

    return dim in {0, -1}


def _can_use_last_dim_triton(inp: torch.Tensor, dims: IntOrInts | None) -> bool:
    if not _can_use_triton(inp) or not inp.is_contiguous() or inp.dim() == 0:
        return False

    if isinstance(dims, int):
        dim = dims
    elif isinstance(dims, Sequence) and not isinstance(dims, int) and len(dims) == 1:
        dim = dims[0]
    else:
        return False

    return dim in {-1, inp.dim() - 1} and inp.numel() >= (1 << 20)


def _normalize_roll_dims(
    shape: Sequence[int], shifts: Sequence[int], dims: Sequence[int]
) -> list[int]:
    ndim = len(shape)
    effective = [0] * ndim
    for shift, dim in zip(shifts, dims):
        canonical_dim = _canonicalize_dim(dim, ndim)
        effective[canonical_dim] += shift
    for index, size in enumerate(shape):
        if size:
            effective[index] %= size
    return effective


def _pad_left(values: Sequence[int], total: int, fill_value: int) -> list[int]:
    padded = [fill_value] * (total - len(values))
    padded.extend(int(value) for value in values)
    return padded


def _launch_roll_flat_kernel(
    inp: torch.Tensor, out: torch.Tensor, shift: int, block: int = 256
) -> None:
    if out.numel() == 0:
        return

    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
    _roll_flat_kernel[grid](inp, out, numel, shift, BLOCK=block)


def _launch_roll_last_dim_kernel(
    inp: torch.Tensor, out: torch.Tensor, shift: int
) -> None:
    if out.numel() == 0:
        return

    numel = out.numel()
    width = inp.shape[-1]
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
    _roll_last_dim_kernel[grid](inp, out, numel, width, shift, BLOCK=1024)


def _launch_roll_single_dim_kernel(
    inp: torch.Tensor,
    out: torch.Tensor,
    dim_size: int,
    shift: int,
    dim_stride: int,
) -> None:
    if out.numel() == 0:
        return

    numel = out.numel()
    block = 1024
    if inp.dtype is torch.float32 and numel <= (1 << 18):
        block = 512
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
    _roll_single_dim_kernel[grid](
        inp,
        out,
        numel,
        dim_size,
        shift,
        dim_stride,
        BLOCK=block,
    )


def _launch_roll_multi_dim_kernel(
    inp: torch.Tensor,
    out: torch.Tensor,
    sizes: Sequence[int],
    strides: Sequence[int],
    shifts: Sequence[int],
) -> None:
    if out.numel() == 0:
        return

    numel = out.numel()
    size_values = _pad_right(sizes, MAX_DIMS, 1)
    stride_values = _pad_right(strides, MAX_DIMS, 0)
    shift_values = _pad_right(shifts, MAX_DIMS, 0)
    block = 1024
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
    _roll_multi_dim_kernel[grid](
        inp,
        out,
        numel,
        size_values[0],
        stride_values[0],
        shift_values[0],
        size_values[1],
        stride_values[1],
        shift_values[1],
        size_values[2],
        stride_values[2],
        shift_values[2],
        size_values[3],
        stride_values[3],
        shift_values[3],
        size_values[4],
        stride_values[4],
        shift_values[4],
        DIMS=len(sizes),
        BLOCK=block,
    )


def _is_int_or_int_sequence(value: object) -> bool:
    if isinstance(value, int):
        return True
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, int) for item in value)


def _is_empty_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence) and not isinstance(value, int) and len(value) == 0
    )


def _pad_right(values: Sequence[int], total: int, fill_value: int) -> list[int]:
    padded = [int(value) for value in values]
    padded.extend([fill_value] * (total - len(padded)))
    return padded


@libentry()
@triton.jit
def _roll_flat_kernel(inp_ptr, out_ptr, numel, shift, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    split = numel - shift
    src_offsets = offsets + split
    src_offsets = tl.where(offsets < shift, src_offsets, offsets - shift)
    values = tl.load(inp_ptr + src_offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, values, mask=mask)


@libentry()
@triton.jit
def _roll_single_dim_kernel(
    inp_ptr,
    out_ptr,
    numel,
    dim_size,
    shift,
    dim_stride,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    dim_index = (offsets // dim_stride) % dim_size
    target_dim_index = (dim_index + shift) % dim_size
    target_offsets = offsets + (target_dim_index - dim_index) * dim_stride

    values = tl.load(inp_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + target_offsets, values, mask=mask)


@libentry()
@triton.jit
def _roll_last_dim_kernel(inp_ptr, out_ptr, numel, width, shift, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    column = offsets % width
    row_start = offsets - column
    source_column = (column + width - shift) % width
    values = tl.load(inp_ptr + row_start + source_column, mask=mask, other=0)
    tl.store(out_ptr + offsets, values, mask=mask)


@libentry()
@triton.jit
def _roll_multi_dim_kernel(
    inp_ptr,
    out_ptr,
    numel,
    size0,
    stride0,
    shift0,
    size1,
    stride1,
    shift1,
    size2,
    stride2,
    shift2,
    size3,
    stride3,
    shift3,
    size4,
    stride4,
    shift4,
    DIMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    source_offsets = offsets

    if DIMS >= 1:
        dim_index0 = (offsets // stride0) % size0
        source_dim_index0 = (dim_index0 + size0 - shift0) % size0
        source_offsets += (source_dim_index0 - dim_index0) * stride0
    if DIMS >= 2:
        dim_index1 = (offsets // stride1) % size1
        source_dim_index1 = (dim_index1 + size1 - shift1) % size1
        source_offsets += (source_dim_index1 - dim_index1) * stride1
    if DIMS >= 3:
        dim_index2 = (offsets // stride2) % size2
        source_dim_index2 = (dim_index2 + size2 - shift2) % size2
        source_offsets += (source_dim_index2 - dim_index2) * stride2
    if DIMS >= 4:
        dim_index3 = (offsets // stride3) % size3
        source_dim_index3 = (dim_index3 + size3 - shift3) % size3
        source_offsets += (source_dim_index3 - dim_index3) * stride3
    if DIMS >= 5:
        dim_index4 = (offsets // stride4) % size4
        source_dim_index4 = (dim_index4 + size4 - shift4) % size4
        source_offsets += (source_dim_index4 - dim_index4) * stride4

    values = tl.load(inp_ptr + source_offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, values, mask=mask)
