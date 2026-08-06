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

import operator

import torch
import triton
import triton.language as tl

from flag_gems.ops.nonzero_static import nonzero_static as _nonzero_static
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_SMALL_INPUT_MAX_NUMEL = 16384
_MULTI_BLOCK_TILE_SIZE = 16384
_MULTI_BLOCK_COUNT_SIZE = 128
_MULTI_BLOCK_MAX_NUMEL = _MULTI_BLOCK_TILE_SIZE * _MULTI_BLOCK_COUNT_SIZE


def _check_int_arg(value, name):
    if isinstance(value, bool):
        raise TypeError(f"nonzero_static(): argument '{name}' must be int, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"nonzero_static(): argument '{name}' must be int, "
            f"not {type(value).__name__}"
        ) from exc


@libentry()
@triton.jit
def _nonzero_static_small_kernel(
    x_ptr,
    workspace_ptr,
    count_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    if IS_COMPLEX:
        real = tl.load(x_ptr + offsets * 2)
        imag = tl.load(x_ptr + offsets * 2 + 1)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + offsets) != 0

    valid = flags & (offsets < numel)
    rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    destination = tl.where(
        valid & (rank < size), rank.to(tl.int64), (size + offsets).to(tl.int64)
    )
    linear = offsets.to(tl.int64)

    if ndim == 1:
        tl.store(workspace_ptr + destination, linear)
    if ndim == 2:
        tl.store(workspace_ptr + destination * 2, linear // D1)
        tl.store(workspace_ptr + destination * 2 + 1, linear % D1)
    if ndim == 3:
        d12 = D1 * D2
        rem = linear % d12
        tl.store(workspace_ptr + destination * 3, linear // d12)
        tl.store(workspace_ptr + destination * 3 + 1, rem // D2)
        tl.store(workspace_ptr + destination * 3 + 2, rem % D2)
    if ndim == 4:
        d123 = D1 * D2 * D3
        d23 = D2 * D3
        rem = linear % d123
        tl.store(workspace_ptr + destination * 4, linear // d123)
        tl.store(workspace_ptr + destination * 4 + 1, rem // d23)
        tl.store(workspace_ptr + destination * 4 + 2, (rem % d23) // D3)
        tl.store(workspace_ptr + destination * 4 + 3, rem % D3)
    if ndim == 5:
        d1234 = D1 * D2 * D3 * D4
        d234 = D2 * D3 * D4
        d34 = D3 * D4
        rem = linear % d1234
        tl.store(workspace_ptr + destination * 5, linear // d1234)
        tl.store(workspace_ptr + destination * 5 + 1, rem // d234)
        tl.store(workspace_ptr + destination * 5 + 2, (rem % d234) // d34)
        tl.store(workspace_ptr + destination * 5 + 3, (rem % d34) // D4)
        tl.store(workspace_ptr + destination * 5 + 4, rem % D4)
    if ndim == 6:
        d12345 = D1 * D2 * D3 * D4 * D5
        d2345 = D2 * D3 * D4 * D5
        d345 = D3 * D4 * D5
        d45 = D4 * D5
        rem = linear % d12345
        tl.store(workspace_ptr + destination * 6, linear // d12345)
        tl.store(workspace_ptr + destination * 6 + 1, rem // d2345)
        tl.store(workspace_ptr + destination * 6 + 2, (rem % d2345) // d345)
        tl.store(workspace_ptr + destination * 6 + 3, (rem % d345) // d45)
        tl.store(workspace_ptr + destination * 6 + 4, (rem % d45) // D5)
        tl.store(workspace_ptr + destination * 6 + 5, rem % D5)

    tl.store(count_ptr, tl.sum(valid.to(tl.int32), axis=0).to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_fill_tail_kernel(
    workspace_ptr,
    count_ptr,
    fill_value,
    size: tl.constexpr,
    ndim: tl.constexpr,
):
    row = tl.program_id(0)
    valid_count = tl.minimum(tl.load(count_ptr), size)
    if row >= valid_count:
        for column in tl.static_range(0, ndim):
            tl.store(workspace_ptr + row * ndim + column, fill_value)


@libentry()
@triton.jit
def _nonzero_static_multiblock_count_kernel(
    x_ptr,
    counts_ptr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if IS_COMPLEX:
        real = tl.load(x_ptr + offsets * 2)
        imag = tl.load(x_ptr + offsets * 2 + 1)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + offsets) != 0
    valid = offsets < numel
    tl.store(counts_ptr + pid, tl.sum((flags & valid).to(tl.int32), axis=0).to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_multiblock_scan_kernel(
    counts_ptr,
    prefix_ptr,
    total_ptr,
    COUNT_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, COUNT_SIZE)
    counts = tl.load(counts_ptr + offsets)
    prefix = tl.cumsum(counts, axis=0) - counts
    tl.store(prefix_ptr + offsets, prefix)
    tl.store(total_ptr, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _nonzero_static_multiblock_write_kernel(
    x_ptr,
    counts_ptr,
    workspace_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COUNT_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    linear = pid * BLOCK_SIZE + offsets
    if IS_COMPLEX:
        real = tl.load(x_ptr + linear * 2)
        imag = tl.load(x_ptr + linear * 2 + 1)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + linear) != 0
    valid = (linear < numel) & flags
    local_rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    prior_counts = tl.load(counts_ptr + tl.arange(0, COUNT_SIZE))
    prefix = tl.sum(tl.where(tl.arange(0, COUNT_SIZE) < pid, prior_counts, 0), axis=0)
    selected = valid & (prefix + local_rank < size)
    destination = tl.where(
        selected,
        prefix + local_rank,
        size + pid * BLOCK_SIZE + offsets,
    ).to(tl.int64)

    if ndim == 1:
        c0 = linear
        tl.store(workspace_ptr + destination, c0)
    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(workspace_ptr + destination * 2, c0)
        tl.store(workspace_ptr + destination * 2 + 1, c1)
    if ndim == 3:
        d12 = D1 * D2
        rem = linear % d12
        tl.store(workspace_ptr + destination * 3, linear // d12)
        tl.store(workspace_ptr + destination * 3 + 1, rem // D2)
        tl.store(workspace_ptr + destination * 3 + 2, rem % D2)
    if ndim == 4:
        d123 = D1 * D2 * D3
        d23 = D2 * D3
        rem = linear % d123
        tl.store(workspace_ptr + destination * 4, linear // d123)
        tl.store(workspace_ptr + destination * 4 + 1, rem // d23)
        tl.store(workspace_ptr + destination * 4 + 2, (rem % d23) // D3)
        tl.store(workspace_ptr + destination * 4 + 3, rem % D3)
    if ndim == 5:
        d1234 = D1 * D2 * D3 * D4
        d234 = D2 * D3 * D4
        d34 = D3 * D4
        rem = linear % d1234
        tl.store(workspace_ptr + destination * 5, linear // d1234)
        tl.store(workspace_ptr + destination * 5 + 1, rem // d234)
        tl.store(workspace_ptr + destination * 5 + 2, (rem % d234) // d34)
        tl.store(workspace_ptr + destination * 5 + 3, (rem % d34) // D4)
        tl.store(workspace_ptr + destination * 5 + 4, rem % D4)
    if ndim == 6:
        d12345 = D1 * D2 * D3 * D4 * D5
        d2345 = D2 * D3 * D4 * D5
        d345 = D3 * D4 * D5
        d45 = D4 * D5
        rem = linear % d12345
        tl.store(workspace_ptr + destination * 6, linear // d12345)
        tl.store(workspace_ptr + destination * 6 + 1, rem // d2345)
        tl.store(workspace_ptr + destination * 6 + 2, (rem % d2345) // d345)
        tl.store(workspace_ptr + destination * 6 + 3, (rem % d345) // d45)
        tl.store(workspace_ptr + destination * 6 + 4, (rem % d45) // D5)
        tl.store(workspace_ptr + destination * 6 + 5, rem % D5)


@libentry()
@triton.jit
def _nonzero_static_multiblock_fill_tail_kernel(
    workspace_ptr,
    count_ptr,
    fill_value,
    size: tl.constexpr,
    ndim: tl.constexpr,
):
    row = tl.program_id(0)
    total = tl.minimum(tl.load(count_ptr), size)
    if row >= total:
        for column in tl.static_range(0, ndim):
            tl.store(workspace_ptr + row * ndim + column, fill_value)


def _multiblock_nonzero_static(input, size, fill_value, out):
    ndim = input.dim()
    numel = input.numel()
    if ndim == 0 or ndim > 6 or numel <= _SMALL_INPUT_MAX_NUMEL:
        return None
    num_blocks = triton.cdiv(numel, _MULTI_BLOCK_TILE_SIZE)
    if num_blocks > _MULTI_BLOCK_COUNT_SIZE:
        return None

    source = input.contiguous()
    padded_numel = num_blocks * _MULTI_BLOCK_TILE_SIZE
    padded = torch.zeros((padded_numel,), device=input.device, dtype=source.dtype)
    padded[:numel].copy_(source.reshape(-1))
    if source.is_complex():
        x = torch.view_as_real(padded).reshape(-1)
    else:
        x = padded
    workspace = torch.empty(
        (size + padded_numel, ndim), device=input.device, dtype=torch.int64
    )
    counts = torch.empty((_MULTI_BLOCK_COUNT_SIZE,), device=input.device, dtype=torch.int64)
    prefixes = torch.empty_like(counts)
    total = torch.empty((), device=input.device, dtype=torch.int64)
    shape = tuple(input.shape) + (1,) * (6 - ndim)
    with torch_device_fn.device(input.device):
        _nonzero_static_multiblock_count_kernel[(num_blocks,)](
            x, counts, numel, IS_COMPLEX=source.is_complex(), BLOCK_SIZE=_MULTI_BLOCK_TILE_SIZE
        )
        counts[num_blocks:].zero_()
        _nonzero_static_multiblock_scan_kernel[(1,)](
            counts, prefixes, total, COUNT_SIZE=_MULTI_BLOCK_COUNT_SIZE
        )
        _nonzero_static_multiblock_write_kernel[(num_blocks,)](
            x,
            counts,
            workspace,
            size,
            numel,
            ndim,
            *shape,
            IS_COMPLEX=source.is_complex(),
            BLOCK_SIZE=_MULTI_BLOCK_TILE_SIZE,
            COUNT_SIZE=_MULTI_BLOCK_COUNT_SIZE,
        )
        if size:
            _nonzero_static_multiblock_fill_tail_kernel[(size,)](
                workspace, total, fill_value, size, ndim
            )
    result = workspace[:size]
    if out is None:
        return result
    out.resize_((size, ndim))
    out.copy_(result)
    return out


def _small_nonzero_static(input, size, fill_value, out):
    ndim = input.dim()
    numel = input.numel()
    if ndim == 0 or ndim > 6 or numel > _SMALL_INPUT_MAX_NUMEL:
        return None

    block_size = triton.next_power_of_2(max(numel, 1))
    source = input.contiguous()
    padded = torch.zeros((block_size,), device=input.device, dtype=source.dtype)
    padded[:numel].copy_(source.reshape(-1))
    if source.is_complex():
        x = torch.view_as_real(padded).reshape(-1)
    else:
        x = padded

    workspace = torch.empty(
        (size + block_size, ndim), device=input.device, dtype=torch.int64
    )
    count = torch.empty((), device=input.device, dtype=torch.int64)
    shape = tuple(input.shape) + (1,) * (6 - ndim)
    with torch_device_fn.device(input.device):
        _nonzero_static_small_kernel[(1,)](
            x,
            workspace,
            count,
            size,
            numel,
            ndim,
            *shape,
            IS_COMPLEX=source.is_complex(),
            BLOCK_SIZE=block_size,
        )
        if size:
            _nonzero_static_fill_tail_kernel[(size,)](
                workspace, count, fill_value, size, ndim
            )

    result = workspace[:size]
    if out is None:
        return result
    out.resize_((size, ndim))
    out.copy_(result)
    return out


def nonzero_static(input: torch.Tensor, *, size: int, fill_value: int = -1):
    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")
    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")
    result = _small_nonzero_static(input, size, fill_value, out=None)
    if result is not None:
        return result
    result = _multiblock_nonzero_static(input, size, fill_value, out=None)
    if result is not None:
        return result
    return _nonzero_static(input, size=size, fill_value=fill_value)


def nonzero_static_out(
    input: torch.Tensor,
    *,
    size: int,
    fill_value: int = -1,
    out: torch.Tensor,
):
    if out.dtype != torch.int64:
        raise RuntimeError(
            f"Expected out tensor to have dtype torch.int64, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on {input.device}, but got {out.device} instead"
        )

    size = _check_int_arg(size, "size")
    fill_value = _check_int_arg(fill_value, "fill_value")
    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")
    result = _small_nonzero_static(input, size, fill_value, out=out)
    if result is not None:
        return result
    result = _multiblock_nonzero_static(input, size, fill_value, out=out)
    if result is not None:
        return result
    result = _nonzero_static(input, size=size, fill_value=fill_value)
    out.resize_((size, input.dim()))
    out.copy_(result)
    return out
