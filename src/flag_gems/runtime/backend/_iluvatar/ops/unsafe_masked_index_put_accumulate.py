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

_MAX_RANK = 8
_FUSED_LIMIT = 4096
_COPY_BLOCK = 1024
_SCATTER_BLOCK_LARGE = 512
_SCATTER_BLOCK_SMALL = 128
_SCATTER_THRESHOLD = 32768


@triton.jit
def _copy_kernel(inp_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    v = tl.load(inp_ptr + offs, mask=m)
    tl.store(out_ptr + offs, v, mask=m)


@triton.jit
def _scatter_kernel(
    out_ptr,
    mask_ptr,
    val_ptr,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    M,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < M
    mask = tl.load(mask_ptr + offs, mask=m, other=0) != 0
    val = tl.load(val_ptr + offs, mask=mask, other=0.0)
    flat = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK >= 1:
        flat += tl.load(i0 + offs, mask=mask, other=0).to(tl.int32) * s0
    if RANK >= 2:
        flat += tl.load(i1 + offs, mask=mask, other=0).to(tl.int32) * s1
    if RANK >= 3:
        flat += tl.load(i2 + offs, mask=mask, other=0).to(tl.int32) * s2
    if RANK >= 4:
        flat += tl.load(i3 + offs, mask=mask, other=0).to(tl.int32) * s3
    if RANK >= 5:
        flat += tl.load(i4 + offs, mask=mask, other=0).to(tl.int32) * s4
    if RANK >= 6:
        flat += tl.load(i5 + offs, mask=mask, other=0).to(tl.int32) * s5
    if RANK >= 7:
        flat += tl.load(i6 + offs, mask=mask, other=0).to(tl.int32) * s6
    if RANK >= 8:
        flat += tl.load(i7 + offs, mask=mask, other=0).to(tl.int32) * s7
    tl.atomic_add(out_ptr + flat, val, mask=mask, sem="relaxed")


@triton.jit
def _fused_kernel(
    inp_ptr,
    out_ptr,
    mask_ptr,
    val_ptr,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    N,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    m = offs < N
    v = tl.load(inp_ptr + offs, mask=m)
    tl.store(out_ptr + offs, v, mask=m)
    tl.debug_barrier()
    mask = tl.load(mask_ptr + offs, mask=m, other=0) != 0
    val = tl.load(val_ptr + offs, mask=mask, other=0.0)
    flat = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK >= 1:
        flat += tl.load(i0 + offs, mask=mask, other=0).to(tl.int32) * s0
    if RANK >= 2:
        flat += tl.load(i1 + offs, mask=mask, other=0).to(tl.int32) * s1
    if RANK >= 3:
        flat += tl.load(i2 + offs, mask=mask, other=0).to(tl.int32) * s2
    if RANK >= 4:
        flat += tl.load(i3 + offs, mask=mask, other=0).to(tl.int32) * s3
    if RANK >= 5:
        flat += tl.load(i4 + offs, mask=mask, other=0).to(tl.int32) * s4
    if RANK >= 6:
        flat += tl.load(i5 + offs, mask=mask, other=0).to(tl.int32) * s5
    if RANK >= 7:
        flat += tl.load(i6 + offs, mask=mask, other=0).to(tl.int32) * s6
    if RANK >= 8:
        flat += tl.load(i7 + offs, mask=mask, other=0).to(tl.int32) * s7
    tl.atomic_add(out_ptr + flat, val, mask=mask, sem="relaxed")


@triton.jit
def _grid_kernel(
    inp_ptr,
    out_ptr,
    mask_ptr,
    val_ptr,
    flag_ptr,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    N,
    M,
    NB_COPY,
    RANK: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    SCATTER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NB_COPY:
        offs = pid * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
        m = offs < N
        v = tl.load(inp_ptr + offs, mask=m)
        tl.store(out_ptr + offs, v, mask=m)
        tl.atomic_add(flag_ptr, 1, sem="release")
    else:
        cnt = tl.atomic_add(flag_ptr, 0, sem="acquire")
        while cnt < NB_COPY:
            cnt = tl.atomic_add(flag_ptr, 0, sem="acquire")

    offs = pid * SCATTER_BLOCK + tl.arange(0, SCATTER_BLOCK)
    m = offs < M
    mask = tl.load(mask_ptr + offs, mask=m, other=0) != 0
    val = tl.load(val_ptr + offs, mask=mask, other=0.0)
    flat = tl.zeros([SCATTER_BLOCK], dtype=tl.int32)
    if RANK >= 1:
        flat += tl.load(i0 + offs, mask=mask, other=0).to(tl.int32) * s0
    if RANK >= 2:
        flat += tl.load(i1 + offs, mask=mask, other=0).to(tl.int32) * s1
    if RANK >= 3:
        flat += tl.load(i2 + offs, mask=mask, other=0).to(tl.int32) * s2
    if RANK >= 4:
        flat += tl.load(i3 + offs, mask=mask, other=0).to(tl.int32) * s3
    if RANK >= 5:
        flat += tl.load(i4 + offs, mask=mask, other=0).to(tl.int32) * s4
    if RANK >= 6:
        flat += tl.load(i5 + offs, mask=mask, other=0).to(tl.int32) * s5
    if RANK >= 7:
        flat += tl.load(i6 + offs, mask=mask, other=0).to(tl.int32) * s6
    if RANK >= 8:
        flat += tl.load(i7 + offs, mask=mask, other=0).to(tl.int32) * s7
    tl.atomic_add(out_ptr + flat, val, mask=mask, sem="relaxed")


def _strides(shape):
    strides = []
    prod = 1
    for s in reversed(shape):
        strides.insert(0, prod)
        prod *= s
    return strides


def run(inp, mask, indices, values):
    out = torch.empty_like(inp)
    N = inp.numel()
    M = mask.numel()

    rank = len(indices)
    strides = _strides(tuple(inp.shape))
    dummy = mask
    idx_args = list(indices) + [dummy] * (_MAX_RANK - rank)
    stride_args = list(strides) + [0] * (_MAX_RANK - rank)

    if N <= _FUSED_LIMIT and M == N:
        BLOCK = triton.next_power_of_2(N)
        nw = 1
        while nw * 32 < BLOCK:
            nw *= 2
        if nw > 16:
            nw = 16
        _fused_kernel[(1,)](
            inp,
            out,
            mask,
            values,
            *idx_args,
            *stride_args,
            N,
            RANK=rank,
            BLOCK=BLOCK,
            num_warps=nw,
        )
    else:
        flag = torch.zeros(1, dtype=torch.int32, device=inp.device)
        nb_copy = triton.cdiv(N, _COPY_BLOCK)
        sb = _SCATTER_BLOCK_LARGE if M >= _SCATTER_THRESHOLD else _SCATTER_BLOCK_SMALL
        grid = (nb_copy + triton.cdiv(M, sb),)
        _grid_kernel[grid](
            inp,
            out,
            mask,
            values,
            flag,
            *idx_args,
            *stride_args,
            N,
            M,
            nb_copy,
            RANK=rank,
            COPY_BLOCK=_COPY_BLOCK,
            SCATTER_BLOCK=sb,
            num_warps=8,
        )
    return out
