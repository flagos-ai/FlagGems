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
def log10_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.log(x_fp32) * 0.4342944819032518  # 1 / ln(10)
    y = y_fp32.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper with the same name.
_log10__kernel = log10_


def log10_(*args, **kwargs):
    if len(args) == 0:
        raise TypeError(
            "log10_ expects at least one positional argument: a torch.Tensor."
        )
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("log10_ expects a torch.Tensor as its first argument.")
    if x.numel() == 0:
        return x
    if x.device.type != "cuda":
        # Fallback to PyTorch implementation for non-CUDA tensors
        return torch.log10_(x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        # Fallback to PyTorch for unsupported dtypes (e.g., float64, complex)
        return torch.log10_(x)

    BLOCK_SIZE = 1024
    if x.is_contiguous():
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _log10__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        buf = x.contiguous()
        n_elements = buf.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _log10__kernel[grid](buf, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(buf)

    return x
