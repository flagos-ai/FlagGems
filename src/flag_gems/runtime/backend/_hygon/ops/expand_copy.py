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

_MAX_TERMS = 8


@triton.jit
def _copy_contig_kernel(
    src_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        tl.store(out_ptr + offs, tl.load(src_ptr + offs))
    else:
        mask = offs < numel
        tl.store(out_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton.jit
def _expand_copy_kernel(
    src_ptr,
    out_ptr,
    numel,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s_last,
    NP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    rem = tl.where(mask, offs, 0)
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    if NP > 0:
        c = rem // d0
        rem = rem - c * d0
        src_off += c * s0
    if NP > 1:
        c = rem // d1
        rem = rem - c * d1
        src_off += c * s1
    if NP > 2:
        c = rem // d2
        rem = rem - c * d2
        src_off += c * s2
    if NP > 3:
        c = rem // d3
        rem = rem - c * d3
        src_off += c * s3
    if NP > 4:
        c = rem // d4
        rem = rem - c * d4
        src_off += c * s4
    if NP > 5:
        c = rem // d5
        rem = rem - c * d5
        src_off += c * s5
    if NP > 6:
        c = rem // d6
        rem = rem - c * d6
        src_off += c * s6
    if NP > 7:
        c = rem // d7
        rem = rem - c * d7
        src_off += c * s7
    src_off += rem * s_last
    v = tl.load(src_ptr + src_off, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


def _as_size(sizes):
    if isinstance(sizes, (int, torch.SymInt)):
        return [int(sizes)]
    return [-1 if s is None else int(s) for s in sizes]


def expand_copy(self, size, implicit=False):
    logger.debug("GEMS_HYGON EXPAND_COPY")
    x = self

    # Fast path: identity mapping (contiguous source, unchanged shape) becomes a
    # plain flat copy with fully affine offsets, so both load and store vectorize.
    shape = x.shape
    if x.is_contiguous():
        try:
            identical = tuple(size) == shape
        except TypeError:
            identical = False
        if identical:
            numel = x.numel()
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
            if numel == 0:
                return out
            if numel < 2147483648:
                # 16 B per lane with 8 warps: half as many CTAs as the 4-warp shape.
                BLOCK = 8192 // x.element_size()
                grid = ((numel + BLOCK - 1) // BLOCK,)
                _copy_contig_kernel[grid](
                    x,
                    out,
                    numel,
                    BLOCK=BLOCK,
                    EVEN=(numel % BLOCK) == 0,
                    num_warps=8,
                )
                return out

    sizes = _as_size(size)
    nd = x.dim()
    n = len(sizes)
    if n < nd:
        raise RuntimeError(
            "expand(torch.{}, size={}): the number of sizes provided ({}) must be "
            "greater or equal to the number of dimensions in the tensor ({})".format(
                x.dtype, sizes, n, nd
            )
        )
    pad = n - nd
    if pad:
        a_sizes = [1] * pad + list(shape)
        a_strides = [0] * pad + list(x.stride())
    else:
        a_sizes = list(shape)
        a_strides = list(x.stride())

    if implicit:
        for i in range(n):
            if sizes[i] == -1:
                sizes[i] = a_sizes[i]

    for i in range(n):
        ai = a_sizes[i]
        si = sizes[i]
        if si != ai and ai != 1:
            raise RuntimeError(
                "The expanded size of the tensor ({}) must match the existing size ({}) "
                "at non-singleton dimension {}.  Target sizes: {}.  Tensor sizes: {}".format(
                    si, ai, i, sizes, a_sizes
                )
            )

    out = torch.empty(sizes, dtype=x.dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out

    # Right-aligned source strides: 0 for broadcast (size-1) and padding dims.
    st = [0 if a_sizes[i] == 1 else a_strides[i] for i in range(n)]

    # Output strides (divisors) for the row-major output index.
    od = [1] * n
    for i in range(n - 2, -1, -1):
        od[i] = od[i + 1] * sizes[i + 1]

    # Keep only dims whose coordinate can change the source address, then merge
    # adjacent dims that are contiguous in the source.
    groups = []
    for i in range(n):
        sz = sizes[i]
        if sz == 1:
            continue
        groups.append([sz, st[i], od[i]])
    terms = []
    for sz, s, o in groups:
        if terms and terms[-1][1] == s * sz:
            terms[-1][0] *= sz
            terms[-1][1] = s
            terms[-1][2] = o
        else:
            terms.append([sz, s, o])

    if terms:
        s_last = terms[-1][1]
        pairs = terms[:-1]
    else:
        s_last = 0
        pairs = []

    npairs = len(pairs)
    divs = [1] * _MAX_TERMS
    strs = [0] * _MAX_TERMS
    for j in range(npairs):
        divs[j] = pairs[j][2]
        strs[j] = pairs[j][1]

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _expand_copy_kernel[grid](
        x,
        out,
        numel,
        divs[0],
        divs[1],
        divs[2],
        divs[3],
        divs[4],
        divs[5],
        divs[6],
        divs[7],
        strs[0],
        strs[1],
        strs[2],
        strs[3],
        strs[4],
        strs[5],
        strs[6],
        strs[7],
        s_last,
        NP=npairs,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out
