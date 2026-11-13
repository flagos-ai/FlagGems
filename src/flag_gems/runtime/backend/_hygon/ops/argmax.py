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

import math
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops.argmax import argmax as _generic_argmax
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


_ZERO_LAUNCH_CONFIGS = {}


def _device_context(device):
    if device.index is None or torch_device_fn.current_device() == device.index:
        return nullcontext()
    return torch_device_fn.device(device)


@libentry(cache_binding=True)
@triton.jit
def argmax_zero_kernel(out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if N == BLOCK:
        tl.store(out + offsets, 0)
    else:
        tl.store(out + offsets, 0, mask=offsets < N)


def _zero_output(out, size):
    config = _ZERO_LAUNCH_CONFIGS.get(size)
    if config is None:
        block = min(256, triton.next_power_of_2(size))
        grid = triton.cdiv(size, block)
        if block >= 256:
            num_warps = 8
        elif block >= 64:
            num_warps = 4
        else:
            num_warps = 1
        config = (grid, block, num_warps)
        if len(_ZERO_LAUNCH_CONFIGS) >= 128:
            _ZERO_LAUNCH_CONFIGS.clear()
        _ZERO_LAUNCH_CONFIGS[size] = config
    grid, block, num_warps = config
    argmax_zero_kernel[(grid,)](out, size, block, num_warps=num_warps)
    return out


@triton.jit
def _ordered_key(values, valid):
    if values.dtype == tl.float16 or values.dtype == tl.bfloat16:
        values = values.to(tl.float32)
    bits = values.to(tl.uint32, bitcast=True)
    # Normalize signed zero so equal values keep the lowest logical index.
    bits = tl.where(values == 0, 0, bits)
    key = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    key = tl.where(valid, key, 0)
    return tl.where(valid & (values != values), 0xFFFFFFFF, key)


@libentry()
@triton.jit
def argmax_short_lastdim_kernel(
    inp,
    out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < N
    values = tl.load(inp + row.to(tl.int64) * N + col, mask=valid, other=0.0)
    key = _ordered_key(values, valid)
    _, index = tl.max(
        key, 0, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(out + row, index)


@libentry()
@triton.jit
def argmax_lastdim_kernel(
    inp,
    out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    sentinel: tl.constexpr = 0x7FFFFFFF
    max_key = tl.full((), 0, tl.uint32)
    max_index = tl.full((), sentinel, tl.int32)
    for start in tl.range(0, N, BLOCK):
        col = start + lane
        valid = col < N
        values = tl.load(
            inp + row.to(tl.int64) * N + col,
            mask=valid,
            other=0.0,
        )
        key = _ordered_key(values, valid)
        block_key = tl.max(key, 0)
        block_index = tl.min(
            tl.where((key == block_key) & valid, col, sentinel), 0
        )
        take = (block_key > max_key) | (
            (block_key == max_key) & (block_index < max_index)
        )
        max_key = tl.where(take, block_key, max_key)
        max_index = tl.where(take, block_index, max_index)
    tl.store(out + row, max_index)


@libentry()
@triton.jit
def argmax_small_noninner_kernel(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < M * K
    base = (offsets // K).to(tl.int64) * (N * K) + offsets % K
    max_key = tl.full((BLOCK,), 0, tl.uint32)
    max_index = tl.full((BLOCK,), 0, tl.int32)
    for index in tl.static_range(N):
        values = tl.load(inp + base + index * K, mask=valid, other=0.0)
        key = _ordered_key(values, valid)
        take = key > max_key
        max_key = tl.where(take, key, max_key)
        max_index = tl.where(take, index, max_index)
    tl.store(out + offsets, max_index, mask=valid)


@triton.jit
def argmax_global_block_kernel(inp, mid, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(inp + offsets, mask=mask, other=0.0)
    if values.dtype == tl.float16 or values.dtype == tl.bfloat16:
        values = values.to(tl.float32)
    bits = values.to(tl.uint32, bitcast=True)
    bits = tl.where(values == 0, 0, bits)
    key = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    key = tl.where(mask, key, 0)
    key = tl.where(mask & (values != values), 0xFFFFFFFF, key)
    block_key = tl.max(key, 0)
    block_index = tl.min(
        tl.where((key == block_key) & mask, offsets, 0x7FFFFFFF), 0
    )
    packed = (block_key.to(tl.uint64) << 32) | (
        0xFFFFFFFF - block_index.to(tl.uint64)
    )
    tl.store(mid + pid, packed)


@triton.jit
def argmax_global_final_kernel(mid, out, M, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    packed = tl.load(mid + offsets, mask=offsets < M, other=0)
    best = tl.max(packed, 0)
    index = 0xFFFFFFFF - (best & 0xFFFFFFFF)
    tl.store(out, index.to(tl.int64))


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    numel = inp.numel()
    rank = inp.ndim
    if dim is not None:
        if rank == 0:
            if dim not in (0, -1):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
                )
            return torch.zeros((), dtype=torch.int64, device=inp.device)
        if dim < -rank or dim >= rank:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {dim})"
            )
        dim %= rank
        shape = inp.shape
        if shape[dim] == 0:
            raise IndexError(
                f"argmax(): Expected reduction dim {dim} to have non-zero size."
            )

    if numel == 0:
        if dim is None:
            raise IndexError(
                "argmax(): Expected reduction dim to be specified for input.numel() == 0."
            )
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if dim is not None and shape[dim] == 1:
        if keepdim:
            out_shape = shape[:dim] + (1,) + shape[dim + 1 :]
        else:
            out_shape = shape[:dim] + shape[dim + 1 :]
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        with _device_context(inp.device):
            return _zero_output(out, numel)

    fast_dtype = inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not fast_dtype:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if dim is None:
        size = inp.numel()
        if not inp.is_contiguous() or size > 1 << 24:
            return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)
        block = 2048
        mid_size = triton.cdiv(size, block)
        mid = torch.empty((mid_size,), dtype=torch.uint64, device=inp.device)
        out_shape = [1] * inp.ndim if keepdim else []
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        with _device_context(inp.device):
            argmax_global_block_kernel[(mid_size,)](
                inp,
                mid,
                size,
                block,
                num_warps=4,
            )
            argmax_global_final_kernel[(1,)](
                mid,
                out,
                mid_size,
                triton.next_power_of_2(mid_size),
                num_warps=4,
            )
        return out

    shape = inp.shape
    n = shape[dim]
    out_shape = list(shape)
    out_shape[dim] = 1
    if not keepdim:
        del out_shape[dim]

    m = math.prod(shape[:dim])
    k = math.prod(shape[dim + 1 :])
    if not inp.is_contiguous() or n > 0x7FFFFFFF:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    if k != 1:
        if n <= 32 and m * k <= 1 << 20:
            block = 128
            with _device_context(inp.device):
                argmax_small_noninner_kernel[(triton.cdiv(m * k, block),)](
                    inp,
                    out,
                    m,
                    n,
                    k,
                    block,
                    num_warps=1,
                )
            return out
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if m < 64:
        return _generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    with _device_context(inp.device):
        if n <= 4096:
            if n <= 512:
                num_warps = 1
            elif n == 4096 and inp.dtype != torch.float32:
                num_warps = 2
            else:
                num_warps = 4
            argmax_short_lastdim_kernel[(m,)](
                inp,
                out,
                n,
                triton.next_power_of_2(n),
                num_warps=num_warps,
            )
        else:
            if n <= 131072:
                block = 4096
                num_warps = 4
            else:
                block = 8192
                num_warps = 8 if inp.dtype == torch.float32 else 16
            argmax_lastdim_kernel[(m,)](
                inp,
                out,
                n,
                block,
                num_warps=num_warps,
            )
    return out
