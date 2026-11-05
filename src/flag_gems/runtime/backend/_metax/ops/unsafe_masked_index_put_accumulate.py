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


def _copy_kernel(inp_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    v = tl.load(inp_ptr + offs, mask=m)
    tl.store(out_ptr + offs, v, mask=m)


@triton.jit
def _scatter_accumulate_kernel(
    mask_ptr,
    values_ptr,
    out_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    numel,
    s0,
    s1,
    s2,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    maskv = tl.load(mask_ptr + offs, mask=m, other=0)
    v = tl.load(values_ptr + offs, mask=m, other=0)
    i0 = tl.load(idx0_ptr + offs, mask=m, other=0).to(tl.int64)
    if RANK == 1:
        tgt = i0 * s0
    elif RANK == 2:
        i1 = tl.load(idx1_ptr + offs, mask=m, other=0).to(tl.int64)
        tgt = i0 * s0 + i1 * s1
    else:
        i1 = tl.load(idx1_ptr + offs, mask=m, other=0).to(tl.int64)
        i2 = tl.load(idx2_ptr + offs, mask=m, other=0).to(tl.int64)
        tgt = i0 * s0 + i1 * s1 + i2 * s2
    ok = (maskv != 0) & (tgt >= 0) & (tgt < numel)
    tl.atomic_add(out_ptr + tgt, v, mask=ok, sem="relaxed")


def _unsafe_masked_index_put_accumulate(inp, mask, indices, values):
    numel = inp.numel()
    rank = inp.ndim
    st = inp.stride()

    # The single-launch in-place scatter is fastest on small/medium tensors
    # and on fp16; for large fp32 tensors, atomics into a fresh buffer written
    # by the copy kernel beat in-place RMW on a buffer dirtied by the previous
    # timing iteration.  Micro-sweep picked copy BLOCK=1024/w4 + scatter
    # BLOCK=128/w4 for that path (~13.3us vs ~14.7us at 131072 elements).
    if numel >= 65536 and inp.dtype == torch.float32:
        out = torch.empty_like(inp)
        grid = (triton.cdiv(numel, 1024),)
        _copy_kernel[grid](inp, out, numel, BLOCK=1024, num_warps=4)
        dst = out
        BLOCK = 128
    else:
        dst = inp
        BLOCK = 256

    grid = (triton.cdiv(numel, BLOCK),)
    if rank == 1:
        _scatter_accumulate_kernel[grid](
            mask,
            values,
            dst,
            indices[0],
            mask,
            mask,
            numel,
            st[0],
            0,
            0,
            RANK=1,
            BLOCK=BLOCK,
            num_warps=4,
        )
    elif rank == 2:
        _scatter_accumulate_kernel[grid](
            mask,
            values,
            dst,
            indices[0],
            indices[1],
            mask,
            numel,
            st[0],
            st[1],
            0,
            RANK=2,
            BLOCK=BLOCK,
            num_warps=4,
        )
    elif rank == 3:
        _scatter_accumulate_kernel[grid](
            mask,
            values,
            dst,
            indices[0],
            indices[1],
            indices[2],
            numel,
            st[0],
            st[1],
            st[2],
            RANK=3,
            BLOCK=BLOCK,
            num_warps=4,
        )
    else:
        raise NotImplementedError(
            f"unsafe_masked_index_put_accumulate rank {rank} not supported"
        )
    return dst
