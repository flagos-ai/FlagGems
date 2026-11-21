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

from flag_gems.runtime import torch_device_fn


@triton.jit
def _mul_dense(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr, SCALAR: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + off, mask=off < N, other=0).to(tl.float32)
    if SCALAR:
        b = tl.load(B).to(tl.float32)
    else:
        b = tl.load(B + off, mask=off < N, other=0).to(tl.float32)
    tl.store(C + off, a * b, mask=off < N)


def mul(A, B, *, out=None):
    if (
        not isinstance(A, torch.Tensor)
        or not isinstance(B, torch.Tensor)
        or A.device.type != "npu"
        or B.device != A.device
        or A.dtype != B.dtype
        or A.dtype not in (torch.bfloat16, torch.float32)
        or not A.is_contiguous()
        or not B.is_contiguous()
        or (B.ndim != 0 and B.shape != A.shape)
        or A.numel() >= 2**31
        or A.is_neg()
        or B.is_neg()
        or out is not None
    ):
        from flag_gems.runtime.backend._ascend.ops.mul import mul as ascend_mul

        return ascend_mul(A, B, out=out)
    result = torch.empty_like(A)
    if A.numel():
        block = min(triton.next_power_of_2(A.numel()), 8192)
        with torch_device_fn.device(A.device):
            _mul_dense[(triton.cdiv(A.numel(), block),)](
                A, B, result, A.numel(), block, B.ndim == 0
            )
    return result
