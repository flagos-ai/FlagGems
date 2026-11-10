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

# Zero-copy same-itemsize reinterpretations of the quantized storage.
_VIEW_DTYPE = {
    torch.qint8: torch.int8,
    torch.quint8: torch.uint8,
    torch.qint32: torch.int32,
}


@triton.jit(do_not_specialize=["zero_point"])
def _dequantize_full(
    x_ptr,
    scale,
    zero_point,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, (v - zero_point) * scale)


@triton.jit(do_not_specialize=["zero_point", "n_elements"])
def _dequantize_tail(
    x_ptr,
    scale,
    zero_point,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offs, (v - zero_point) * scale, mask=mask)


def dequantize(a):
    x = a.view(_VIEW_DTYPE[a.dtype])
    n_elements = x.numel()
    # new_empty inherits device/layout without building a device object.
    out = x.new_empty(x.shape, dtype=torch.float32)
    if n_elements == 0:
        return out

    # Wider block for large tensors: 16 int8 per thread at 256 threads per
    # program gives 128-bit loads and four 128-bit stores per thread.
    BLOCK = 4096 if n_elements > 131072 else 1024
    grid = ((n_elements + BLOCK - 1) // BLOCK,)
    scale = a.q_scale()
    zero_point = a.q_zero_point()
    if n_elements % BLOCK:
        _dequantize_tail[grid](x, scale, zero_point, out, n_elements, BLOCK)
    else:
        # Exact multiples need no tail predicate at all.
        _dequantize_full[grid](x, scale, zero_point, out, BLOCK)
    return out
