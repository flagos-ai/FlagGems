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

logger = logging.getLogger(__name__)


def _logical_not_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x == 0, 1, 0).to(x.dtype)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
def _logical_not_u8_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = (x == 0).to(x.dtype)
    tl.store(x_ptr + offs, y, mask=mask)


def logical_not_(A):
    n = A.numel()
    if n == 0:
        return A
    if A.dtype == torch.bool:
        B = A.view(torch.uint8)
        if n <= 65536:
            BLOCK = 512
        elif n <= 134217728:
            BLOCK = 4096
        else:
            BLOCK = 2048
        grid = (triton.cdiv(n, BLOCK),)
        _logical_not_u8_kernel[grid](B, n, BLOCK=BLOCK)
    elif A.dtype == torch.int16:
        if n <= 67108864:
            BLOCK = 2048
        else:
            BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _logical_not_kernel[grid](A, n, BLOCK=BLOCK)
    elif A.dtype == torch.int32:
        if n <= 67108864:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _logical_not_kernel[grid](A, n, BLOCK=BLOCK)
        else:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _logical_not_kernel[grid](A, n, BLOCK=BLOCK, num_warps=8)
    else:
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _logical_not_kernel[grid](A, n, BLOCK=BLOCK)
    return A
