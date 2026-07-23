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
def log_(
    x_ptr,  # *Pointer* to input/output vector (in-place).
    n_elements,  # Number of elements.
    BLOCK_SIZE: tl.constexpr,  # Elements processed per program.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    y_f32 = tl.log(x_f32)
    y = y_f32.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper with the same name.
log__triton_kernel = log_


def log_(*args, **kwargs):
    x = args[0] if len(args) > 0 else kwargs.get("input", None)
    if x is None:
        raise ValueError("log_ expects a tensor as the first argument.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("log_ expects a torch.Tensor as input.")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if not x.is_floating_point():
        raise TypeError("log_ only supports floating point tensors.")
    if not x.is_contiguous():
        raise ValueError(
            "This log_ Triton implementation requires a contiguous tensor."
        )

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    log__triton_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
