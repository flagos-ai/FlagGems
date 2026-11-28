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


@triton.jit(do_not_specialize=["n_elements"])
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)); compute in fp32 for stability
    x_fp32 = x.to(tl.float32)
    y = x_fp32 / (1.0 + tl.exp(-x_fp32))
    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


def silu(x):
    # Raw-pointer elementwise kernel (NOT tl.make_block_ptr): the generic
    # @pointwise_dynamic path crashes the triton-shared CPU backend on
    # block-pointer lowering, so hand-roll the contiguous fast path like
    # sub.py's _sub_contiguous_kernel.
    if x.numel() == 0:
        return torch.empty_like(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out
