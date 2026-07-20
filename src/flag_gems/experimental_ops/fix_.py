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
def fix_(x_ptr, n_elements, DO_UPCAST: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if DO_UPCAST:
        x_work = x.to(tl.float32)
    else:
        x_work = x

    y_floor = tl.floor(x_work)
    y_ceil = tl.ceil(x_work)
    y_work = tl.where(x_work >= 0, y_floor, y_ceil)

    if DO_UPCAST:
        y = y_work.to(x.dtype)
    else:
        y = y_work

    tl.store(x_ptr + offsets, y, mask=mask)


# Keep reference to the Triton kernel before redefining the name for the Python wrapper
_fix_kernel = fix_


def fix_(*args, **kwargs):
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("fix_ expects a torch.Tensor as the first argument")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernel")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")

    # In-place fix_ does nothing for non-floating tensors
    if not x.is_floating_point():
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    # Upcast low-precision types for stable math
    do_upcast = x.dtype in (torch.float16, torch.bfloat16)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fix_kernel[grid](x, n_elements, DO_UPCAST=do_upcast, BLOCK_SIZE=BLOCK_SIZE)
    return x
