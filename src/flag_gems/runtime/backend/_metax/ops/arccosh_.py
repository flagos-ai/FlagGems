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


def _arccosh_even(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    x32 = x.to(tl.float32)
    z = x32 * x32 - 1.0
    # sqrt(z) via z * rsqrt(max(z, tiny)): SFU rsqrt, preserves NaN for x<1 and 0 at x==1
    s = z * tl.rsqrt(tl.maximum(z, 1e-30))
    y = tl.log(x32 + s)
    tl.store(x_ptr + offsets, y.to(x.dtype))


@triton.jit
def _arccosh_masked(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    x32 = x.to(tl.float32)
    z = x32 * x32 - 1.0
    s = z * tl.rsqrt(tl.maximum(z, 1e-30))
    y = tl.log(x32 + s)
    tl.store(x_ptr + offsets, y.to(x.dtype), mask=mask)


def arccosh_(A):
    n = A.numel()
    if n == 0:
        return A
    x = A.reshape(-1)
    if A.dtype == torch.float32:
        # fp32: BLOCK=1024 wins through 67M; 2048 for 1G streaming; 4 warps
        BLOCK_SIZE = 1024 if n <= (1 << 26) else 2048
        NUM_WARPS = 4
    elif n > (1 << 26):
        # 1G streaming fp16/bf16: BLOCK=512 with 1 warp wins (microbench
        # 3.164/3.593 vs 3.186/3.612ms for fp16/bf16)
        BLOCK_SIZE = 512
        NUM_WARPS = 1
    elif n <= 4096:
        # tiny tensors: 4 blocks of 1024 with 4 warps hide launch latency best
        BLOCK_SIZE = 1024
        NUM_WARPS = 4
    else:
        # mid-size fp16/bf16: BLOCK=1024 with 2 warps
        BLOCK_SIZE = 1024
        NUM_WARPS = 2
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    if n % BLOCK_SIZE == 0:
        _arccosh_even[grid](x, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    else:
        _arccosh_masked[grid](x, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return A
