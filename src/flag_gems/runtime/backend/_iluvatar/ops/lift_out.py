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
def _lift_copy_kernel(
    A_ptr,
    Out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    CG_STORE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offs < numel
        val = tl.load(A_ptr + offs, mask=mask)
        if CG_STORE:
            tl.store(Out_ptr + offs, val, mask=mask, cache_modifier=".cg")
        else:
            tl.store(Out_ptr + offs, val, mask=mask)
    else:
        val = tl.load(A_ptr + offs)
        if CG_STORE:
            tl.store(Out_ptr + offs, val, cache_modifier=".cg")
        else:
            tl.store(Out_ptr + offs, val)


def lift_out(A, *, out=None):
    numel = A.numel()
    if numel == 0:
        return torch.empty_like(A) if out is None else out

    if out is None or out.shape != A.shape:
        out = torch.empty_like(A)

    if numel > 8192 and A.element_size() == 4:
        # 4-byte elements: 256 elems / 2 warps = 16 bytes per thread (128-bit)
        BLOCK_SIZE = 256
        num_warps = 2
        cg_store = False
    else:
        # 2-byte elements and tiny sizes: 2048 elems / 16 warps;
        # .cg stores (bypass L1) measured ~+0.9% for 2-byte streams
        BLOCK_SIZE = 2048
        num_warps = 16
        cg_store = A.element_size() != 4

    need_mask = (numel % BLOCK_SIZE) != 0
    grid = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _lift_copy_kernel[grid](
        A,
        out,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        NEED_MASK=need_mask,
        CG_STORE=cg_store,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
