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


@triton.jit
def _log10_inplace_kernel(A_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(A_ptr + offsets, mask=mask)
    # log10(x) = log2(x) * log10(2); compute in fp32 then round back
    y = tl.log2(x.to(tl.float32)) * 0.30102999566398119521
    y = y.to(x.dtype)
    tl.store(A_ptr + offsets, y, mask=mask)


@triton.jit
def _log10_inplace_kernel_cg(A_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(A_ptr + offsets, mask=mask, cache_modifier=".cg")
    y = tl.log2(x.to(tl.float32)) * 0.30102999566398119521
    y = y.to(x.dtype)
    tl.store(A_ptr + offsets, y, mask=mask, cache_modifier=".cg")


def log10_(A):
    x = A.view(-1)
    n = x.numel()
    if n == 0:
        return A
    dt = A.dtype
    if dt == torch.float32:
        grid = (triton.cdiv(n, 256),)
        _log10_inplace_kernel_cg[grid](x, n, BLOCK_SIZE=256, num_warps=4)
    elif dt == torch.float16:
        grid = (triton.cdiv(n, 512),)
        _log10_inplace_kernel_cg[grid](x, n, BLOCK_SIZE=512, num_warps=4)
    else:  # bfloat16 and other dtypes
        grid = (triton.cdiv(n, 1024),)
        _log10_inplace_kernel[grid](x, n, BLOCK_SIZE=1024, num_warps=4)
    return A
