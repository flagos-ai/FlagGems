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

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _acos_flat_kernel(
    A_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(A_ptr + offsets, mask=mask, other=1.0)
    if x.dtype == tl.float64:
        y = libdevice.acos(x)
    else:
        y = libdevice.acos(x.to(tl.float32)).to(x.dtype)
    tl.store(A_ptr + offsets, y, mask=mask)


@triton.jit
def _acos_flat_nomask_kernel(
    A_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(A_ptr + offsets)
    if x.dtype == tl.float64:
        y = libdevice.acos(x)
    else:
        y = libdevice.acos(x.to(tl.float32)).to(x.dtype)
    tl.store(A_ptr + offsets, y)


@triton.jit
def _acos_strided_kernel(
    A_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    rem = offs
    addr = tl.zeros(offs.shape, dtype=tl.int64)
    for d in tl.static_range(len(SHAPE)):
        idx = rem % SHAPE[d]
        rem = rem // SHAPE[d]
        addr += idx * STRIDES[d]
    x = tl.load(A_ptr + addr, mask=mask, other=1.0)
    if x.dtype == tl.float64:
        y = libdevice.acos(x)
    else:
        y = libdevice.acos(x.to(tl.float32)).to(x.dtype)
    tl.store(A_ptr + addr, y, mask=mask)


def acos_(A):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous():
        if A.dtype == torch.float32:
            BLOCK_SIZE = 512
            num_warps = 8
        else:
            BLOCK_SIZE = 1024
            num_warps = 4
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        if n % BLOCK_SIZE == 0:
            _acos_flat_nomask_kernel[grid](
                A, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
            )
        else:
            _acos_flat_kernel[grid](A, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    else:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _acos_strided_kernel[grid](
            A, n, SHAPE=A.shape, STRIDES=A.stride(), BLOCK_SIZE=BLOCK_SIZE
        )
    return A
