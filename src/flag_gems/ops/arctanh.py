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
def arctanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)

    one = 1.0
    # atanh(x) = 0.5 * (log(1 + x) - log(1 - x))
    y_f32 = 0.5 * (tl.log(one + x_f32) - tl.log(one - x_f32))
    y = y_f32.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_arctanh(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert x.shape == out.shape, "Input and output shapes must match"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "Supported dtypes: float16, bfloat16, float32"

    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else torch.empty_like(out)

    n_elements = x_contig.numel()
    if n_elements == 0:
        if out_contig is not out:
            out.copy_(out_contig)
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    arctanh_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=1024)

    if out_contig is not out:
        out.copy_(out_contig)
    return out


def arctanh(x: torch.Tensor):
    out = torch.empty_like(x)
    _launch_arctanh(x, out)
    return out


def arctanh_out(x: torch.Tensor, out: torch.Tensor):
    _launch_arctanh(x, out)
    return out
