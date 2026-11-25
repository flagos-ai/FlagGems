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

# gcu400 caps grid.x at 65535.  The public codegen uses BLOCK_SIZE <= 512 and
# launches cdiv(N, BLOCK_SIZE) programs, so a (200, 40999, 3) source needs
# 96092 programs and raises OutOfResources.  A persistent grid keeps the
# launch size bounded while a strided loop covers every element.
_MAX_PROGRAMS = 2048
_BLOCK = 1024


@triton.jit
def _index_copy_kernel(
    src,
    index,
    out,
    N,
    inner,
    src_dim,
    out_dim,
    BLOCK: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    stride = NUM_PROGRAMS * BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK, N, stride):
        offs = base + cols
        mask = offs < N
        outer = offs // (src_dim * inner)
        d = (offs // inner) % src_dim
        i = offs % inner
        # index is downcast to int32 on the host: gcu400 cannot issue a
        # 64-bit load from inside a Triton kernel.
        idx = tl.load(index + d, mask=mask, other=0).to(tl.int32)
        dst = outer * (out_dim * inner) + idx * inner + i
        val = tl.load(src + offs, mask=mask, other=0)
        tl.store(out + dst, val, mask=mask)


def _check(inp, dim, index, src):
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    return dim


def _scatter(out, dim, index, src):
    N = src.numel()
    if N == 0:
        return
    inner = 1
    for i in range(dim + 1, src.ndim):
        inner *= src.size(i)
    src_dim = src.size(dim)
    out_dim = out.size(dim)
    num_programs = min(triton.cdiv(N, _BLOCK), _MAX_PROGRAMS)
    _index_copy_kernel[(num_programs,)](
        src.contiguous(),
        index.to(torch.int32).contiguous(),
        out,
        N,
        inner,
        src_dim,
        out_dim,
        BLOCK=_BLOCK,
        NUM_PROGRAMS=num_programs,
        num_warps=4,
    )


def index_copy(inp, dim, index, src):
    logger.debug("GEMS_ENFLAME INDEX_COPY")
    dim = _check(inp, dim, index, src)
    out = inp.clone()
    _scatter(out, dim, index, src)
    return out


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS_ENFLAME INDEX_COPY_")
    dim = _check(inp, dim, index, src)
    _scatter(inp, dim, index, src)
    return inp
