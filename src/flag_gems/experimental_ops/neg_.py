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
def neg__kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = -x
    tl.store(x_ptr + offsets, x, mask=mask)


def neg_(*args, **kwargs):
    # Retrieve input tensor (first positional or from kwargs)
    if len(args) >= 1:
        x = args[0]
    elif "input" in kwargs:
        x = kwargs["input"]
    elif "self" in kwargs:
        x = kwargs["self"]
    else:
        raise ValueError("neg_ expects a tensor as the first argument")

    if not isinstance(x, torch.Tensor):
        raise TypeError("neg_ expects a torch.Tensor")

    if x.numel() == 0:
        return x

    if not x.is_cuda:
        raise ValueError("neg_ Triton kernel requires a CUDA tensor")

    if not x.is_contiguous():
        raise ValueError("neg_ Triton kernel requires a contiguous tensor")

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    neg__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
