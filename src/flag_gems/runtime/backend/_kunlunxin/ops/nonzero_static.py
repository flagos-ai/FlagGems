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

# NOTE (2026-09-05, nonzero_static structural rewrite)
# ---------------------------------------------------------
# Old structure: count -> single-CTA scan(COUNT_SIZE=256) -> write(prefix recomputed
# via a full COUNT_SIZE masked reduction per block) -> fill(grid=size tiny CTAs).
# Bottlenecks found on the XPU backend:
#   1. The write kernel's data-dependent scatter store is the hard wall (~1.2ms @1M
#      elems regardless of sparsity): any non-uniform store address poisons the store
#      instruction, and clustered dummy destinations are catastrophic (181ms), so the
#      dummy destination MUST stay coalesced ("size + pid*TILE + lane").
#   2. fill tail used grid=size CTAs (one element each) -> ~1.2ms for size=4096.
#   3. num_blocks > 256 fell back to the generic path, whose *masked* scatter store is
#      both incorrect (wrong values) and catastrophically slow (241ms in do_bench).
# New structure:
#   count -> scan (single CTA, exclusive prefix array, O(1) lookup in write) ->
#   write (scatter the flat LINEAR index, unmasked to a coalesced dummy dest; the
#   int64 div/mod de-linearization is moved OUT of the scatter path) ->
#   de-linearize (dense, ndim>=2 only) -> batched guarded fill tail.
# This removes the masked scatter entirely (correctness), caps the prefix cost at
# O(1)/block, extends to num_blocks <= _MULTI_BLOCK_MAX_DIRECT and makes the fill a
# wide batched kernel.

_SMALL_INPUT_MAX_NUMEL = 8192
_MULTI_BLOCK_TILE_SIZE = 16384
_MULTI_BLOCK_MAX_DIRECT = 2048
_DELIN_BLOCK_SIZE = 1024
_FILL_BLOCK_SIZE = 1024


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
    total_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    load_mask = offsets < numel
    if IS_COMPLEX:
        real = tl.load(x_ptr + offsets * 2, mask=load_mask, other=0)
        imag = tl.load(x_ptr + offsets * 2 + 1, mask=load_mask, other=0)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + offsets, mask=load_mask, other=0) != 0

    valid = flags & (offsets < numel)
    rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    # unmasked scatter of the flat linear index, coalesced dummy dest
    destination = tl.where(
        valid & (rank < size), rank.to(tl.int64), (size + offsets).to(tl.int64)
    )
    tl.store(workspace_ptr + destination, offsets.to(tl.int64))
    tl.store(total_ptr, tl.sum(valid.to(tl.int32), axis=0).to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_count_kernel(
    x_ptr,
    counts_ptr,
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
    tl.store(
        counts_ptr + pid, tl.sum(flags.to(tl.int32), axis=0).to(tl.int64)
    )


@libentry()
@triton.jit
def _nonzero_static_scan_kernel(
    counts_ptr,
    prefix_ptr,
    total_ptr,
    num_blocks: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, PREFIX_BLOCK_SIZE)
    counts = tl.load(counts_ptr + offsets, mask=offsets < num_blocks, other=0)
    prefix = tl.cumsum(counts, axis=0) - counts  # exclusive
    tl.store(prefix_ptr + offsets, prefix, mask=offsets < num_blocks)
    tl.store(total_ptr, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _nonzero_static_write_kernel(
    x_ptr,
    prefix_ptr,
    workspace_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    load_mask = offsets < numel
    if IS_COMPLEX:
        real = tl.load(x_ptr + offsets * 2, mask=load_mask, other=0)
        imag = tl.load(x_ptr + offsets * 2 + 1, mask=load_mask, other=0)
        flags = (real != 0) | (imag != 0)
    else:
        flags = tl.load(x_ptr + offsets, mask=load_mask, other=0) != 0
    prefix = tl.load(prefix_ptr + pid).to(tl.int64)
    local_rank = tl.cumsum(flags.to(tl.int32), axis=0) - 1
    global_rank = prefix + local_rank.to(tl.int64)
    selected = flags & (global_rank < size)
    destination = tl.where(
        selected,
        global_rank,
        (size + offsets).to(tl.int64),  # coalesced dummy, within workspace (size+padded_numel)
    )
    tl.store(workspace_ptr + destination, offsets.to(tl.int64), mask=load_mask)


@libentry()
@triton.jit
def _nonzero_static_delinearize_kernel(
    workspace_ptr,
    total_ptr,
    out_ptr,
    size: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_rows = tl.minimum(tl.load(total_ptr), size)
    mask = row < valid_rows
    linear = tl.load(workspace_ptr + row, mask=mask, other=0)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + row * 2, c0, mask=mask)
        tl.store(out_ptr + row * 2 + 1, c1, mask=mask)
    if ndim == 3:
        d12 = D1 * D2
        rem = linear % d12
        tl.store(out_ptr + row * 3, linear // d12, mask=mask)
        tl.store(out_ptr + row * 3 + 1, rem // D2, mask=mask)
        tl.store(out_ptr + row * 3 + 2, rem % D2, mask=mask)
    if ndim == 4:
        d123 = D1 * D2 * D3
        d23 = D2 * D3
        rem = linear % d123
        tl.store(out_ptr + row * 4, linear // d123, mask=mask)
        tl.store(out_ptr + row * 4 + 1, rem // d23, mask=mask)
        tl.store(out_ptr + row * 4 + 2, (rem % d23) // D3, mask=mask)
        tl.store(out_ptr + row * 4 + 3, rem % D3, mask=mask)
    if ndim == 5:
        d1234 = D1 * D2 * D3 * D4
        d234 = D2 * D3 * D4
        d34 = D3 * D4
        rem = linear % d1234
        tl.store(out_ptr + row * 5, linear // d1234, mask=mask)
        tl.store(out_ptr + row * 5 + 1, rem // d234, mask=mask)
        tl.store(out_ptr + row * 5 + 2, (rem % d234) // d34, mask=mask)
        tl.store(out_ptr + row * 5 + 3, (rem % d34) // D4, mask=mask)
        tl.store(out_ptr + row * 5 + 4, rem % D4, mask=mask)
    if ndim == 6:
        d12345 = D1 * D2 * D3 * D4 * D5
        d2345 = D2 * D3 * D4 * D5
        d345 = D3 * D4 * D5
        d45 = D4 * D5
        rem = linear % d12345
        tl.store(out_ptr + row * 6, linear // d12345, mask=mask)
        tl.store(out_ptr + row * 6 + 1, rem // d2345, mask=mask)
        tl.store(out_ptr + row * 6 + 2, (rem % d2345) // d345, mask=mask)
        tl.store(out_ptr + row * 6 + 3, (rem % d345) // d45, mask=mask)
        tl.store(out_ptr + row * 6 + 4, (rem % d45) // D5, mask=mask)
        tl.store(out_ptr + row * 6 + 5, rem % D5, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_fill_tail_kernel(
    out_ptr,
    total_ptr,
    size: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FILL_VALUE: tl.constexpr,
):
    total_out = size * ndim
    valid_rows = tl.minimum(tl.load(total_ptr), size)
    tail_start = valid_rows * ndim
    pid = tl.program_id(0)
    offsets = tail_start + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out
    vals = tl.full((BLOCK_SIZE,), FILL_VALUE, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


def _small_nonzero_static(input, size, fill_value, out):
    ndim = input.dim()
    numel = input.numel()
    if ndim == 0 or ndim > 6 or numel > _SMALL_INPUT_MAX_NUMEL:
        return None
    if numel == 0:
        # empty input: all rows are fill_value (avoids the unreliable all-masked
        # tiny-tile kernel path on the XPU backend)
        if out is not None:
            out.resize_((size, ndim))
            out.fill_(fill_value)
            return out
        return torch.full((size, ndim), fill_value, dtype=torch.int64, device=input.device)

    # keep BLOCK >= 64 (XPU backend min reliable block size)
    block_size = triton.next_power_of_2(max(numel, 64))
    source = input.contiguous()
    padded = torch.zeros((block_size,), device=input.device, dtype=source.dtype)
    padded[:numel].copy_(source.reshape(-1))
    if source.is_complex():
        x = torch.view_as_real(padded).reshape(-1)
    else:
        x = padded

    workspace = torch.empty(
        (size + block_size,), device=input.device, dtype=torch.int64
    )
    total = torch.empty((), device=input.device, dtype=torch.int64)
    with torch_device_fn.device(input.device):
        _nonzero_static_small_kernel[(1,)](
            x,
            workspace,
            total,
            size,
            numel,
            IS_COMPLEX=source.is_complex(),
            BLOCK_SIZE=block_size,
        )
    shape = tuple(input.shape) + (1,) * (6 - ndim)
    return _finish_ndim(
        workspace, total, input, size, ndim, out, fill_value, shape
    )


def _multiblock_nonzero_static(input, size, fill_value, out):
    ndim = input.dim()
    numel = input.numel()
    if ndim == 0 or ndim > 6 or numel <= _SMALL_INPUT_MAX_NUMEL:
        return None
    num_blocks = triton.cdiv(numel, _MULTI_BLOCK_TILE_SIZE)
    if num_blocks > _MULTI_BLOCK_MAX_DIRECT:
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
        (size + padded_numel,), device=input.device, dtype=torch.int64
    )
    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)
    prefix = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)
    total = torch.empty((), device=input.device, dtype=torch.int64)
    shape = tuple(input.shape) + (1,) * (6 - ndim)
    prefix_block_size = 1 << (num_blocks - 1).bit_length()
    with torch_device_fn.device(input.device):
        _nonzero_static_count_kernel[(num_blocks,)](
            x, counts, IS_COMPLEX=source.is_complex(), BLOCK_SIZE=_MULTI_BLOCK_TILE_SIZE
        )
        _nonzero_static_scan_kernel[(1,)](
            counts, prefix, total, num_blocks=num_blocks, PREFIX_BLOCK_SIZE=prefix_block_size
        )
        _nonzero_static_write_kernel[(num_blocks,)](
            x, prefix, workspace, size, numel,
            IS_COMPLEX=source.is_complex(), BLOCK_SIZE=_MULTI_BLOCK_TILE_SIZE,
        )
    return _finish_ndim(
        workspace, total, input, size, ndim, out, fill_value, shape
    )


def _finish_ndim(workspace, total, input, size, ndim, out, fill_value, shape):
    """Build the (size, ndim) result from the flat linear workspace + fill tail."""
    if ndim == 1:
        result = workspace[:size].reshape(size, 1)
        fill_target = workspace
    else:
        result = torch.empty((size, ndim), device=input.device, dtype=torch.int64)
        with torch_device_fn.device(input.device):
            _nonzero_static_delinearize_kernel[
                (triton.cdiv(size, _DELIN_BLOCK_SIZE),)
            ](
                workspace,
                total,
                result,
                size,
                ndim,
                *shape,
                BLOCK_SIZE=_DELIN_BLOCK_SIZE,
            )
        fill_target = result
    with torch_device_fn.device(input.device):
        _nonzero_static_fill_tail_kernel[
            (triton.cdiv(size * ndim, _FILL_BLOCK_SIZE),)
        ](
            fill_target,
            total,
            size,
            ndim,
            BLOCK_SIZE=_FILL_BLOCK_SIZE,
            FILL_VALUE=fill_value,
        )
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
