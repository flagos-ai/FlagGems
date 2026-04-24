import logging
from typing import List, Optional, Sequence, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _roll_single_dim_kernel(
    in_ptr,
    out_ptr,
    numel,
    dim_size,
    shift,
    inner_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    outer_stride = dim_size * inner_size
    outer_idx = offsets // outer_stride
    remainder = offsets % outer_stride
    dim_idx = remainder // inner_size
    inner_idx = remainder % inner_size

    src_dim_idx = (dim_idx - shift + dim_size) % dim_size
    src_offsets = outer_idx * outer_stride + src_dim_idx * inner_size + inner_idx

    values = tl.load(in_ptr + src_offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


def _canonicalize_dim(dim: int, ndim: int) -> int:
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _normalize_shifts_dims(
    input: torch.Tensor,
    shifts: Union[int, Sequence[int]],
    dims: Optional[Union[int, Sequence[int]]],
):
    if dims is None:
        if isinstance(shifts, int):
            return [shifts], [0], True
        shifts = list(shifts)
        if len(shifts) == 0:
            raise RuntimeError("`shifts` required")
        if len(shifts) == 1:
            return shifts, [0], True
        raise RuntimeError(
            f"shifts and dimensions must align. shifts: {len(shifts)}, dims:0"
        )

    dims_list = [dims] if isinstance(dims, int) else list(dims)
    if isinstance(shifts, int):
        shifts_list = [shifts]
    else:
        shifts_list = list(shifts)
        if len(shifts_list) == 0:
            raise RuntimeError("`shifts` required")

    if len(shifts_list) != len(dims_list):
        raise RuntimeError(
            f"shifts and dimensions must align. shifts: {len(shifts_list)}, dims:{len(dims_list)}"
        )

    dims_list = [_canonicalize_dim(dim, input.ndim) for dim in dims_list]
    return shifts_list, dims_list, False


def _collapse_shifts_dims(
    input: torch.Tensor, shifts: List[int], dims: List[int]
) -> tuple[List[int], List[int]]:
    merged_shifts = {}
    merged_dims = []
    for shift, dim in zip(shifts, dims):
        if dim not in merged_shifts:
            merged_dims.append(dim)
            merged_shifts[dim] = 0
        merged_shifts[dim] += shift

    normalized_shifts = []
    normalized_dims = []
    for dim in merged_dims:
        dim_size = input.shape[dim]
        shift = 0 if dim_size == 0 else merged_shifts[dim] % dim_size
        if shift != 0:
            normalized_shifts.append(shift)
            normalized_dims.append(dim)
    return normalized_shifts, normalized_dims


def _roll_single_dim(input: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    input_c = input if input.is_contiguous() else input.contiguous()
    dim_size = input_c.shape[dim]
    if input_c.numel() == 0 or dim_size <= 1:
        return input_c.clone()

    shift %= dim_size
    if shift == 0:
        return input_c.clone()

    inner_size = 1
    for axis in range(dim + 1, input_c.ndim):
        inner_size *= input_c.shape[axis]

    out = torch.empty_like(input_c)
    numel = input_c.numel()
    block = 1024
    grid = (triton.cdiv(numel, block),)

    with torch_device_fn.device(input.device):
        _roll_single_dim_kernel[grid](
            input_c,
            out,
            numel,
            dim_size,
            shift,
            inner_size,
            BLOCK=block,
        )
    return out


def roll(
    input: torch.Tensor,
    shifts: Union[int, List[int]],
    dims: Optional[Union[int, List[int]]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    shifts, dims, flatten = _normalize_shifts_dims(input, shifts, dims)
    if flatten:
        flat = input.contiguous().reshape(-1)
        return _roll_single_dim(flat, shifts[0], 0).reshape(input.shape)

    shifts, dims = _collapse_shifts_dims(input, shifts, dims)
    if len(shifts) == 0:
        return input.contiguous().clone()
    result = input
    for shift, dim in zip(shifts, dims):
        result = _roll_single_dim(result, shift, dim)
    return result
