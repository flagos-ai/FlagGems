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


def _gt_scalar_kernel(
    A_ptr, b, n_elements, BLOCK_SIZE: tl.constexpr, USE_I64: tl.constexpr
):
    pid = tl.program_id(0)
    if USE_I64:
        offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(A_ptr + offsets, mask=mask)
    res = tl.where(x > b, 1.0, 0.0)
    tl.store(A_ptr + offsets, res, mask=mask)


def gt_scalar_(A, B):
    b = B.item() if hasattr(B, "item") else B
    numel = A.numel()
    if numel == 0:
        return A
    BLOCK_SIZE = 1024
    # Metax C550: max 512 threads/CTA (64-thread warps). fp32 reaches the HBM
    # bandwidth plateau with 512 threads (2 elems/thread); fp16/bf16 saturate at
    # 256 threads (4 elems/thread) and regress badly at 512 threads.
    num_warps = 8 if A.dtype in (torch.float32, torch.float64) else 4
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    _gt_scalar_kernel[grid](
        A,
        b,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_I64=numel >= (1 << 31),
        num_warps=num_warps,
    )
    return A
