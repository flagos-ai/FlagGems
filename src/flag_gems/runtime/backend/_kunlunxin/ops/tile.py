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


# Packed uint64 wide-vector block-DMA block (mirrors expand_copy._expand_flat_pack_kernel).
_PACK_BLOCK = 32768
# Per-element flat block for the odd-numel / non-packable leading-repeat fallback.
_FLAT_BLOCK = 4096
# Block for the general modulo-gather fallback (mirrors expand_copy._expand_gather_kernel).
_GATHER_BLOCK = 4096


@triton.jit
def _tile_leading_pack_kernel(
    src_ptr, dst_ptr, nwide, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    """Leading repeat through a uint64 view: out == [rep, src_numel], each output
    row is a contiguous byte-copy of the source block.  One lane moves 8//ELEM
    consecutive elements so the load/store hit the wide-vector block-DMA path."""
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    src64 = src_ptr.to(tl.pointer_type(tl.uint64))
    dst64 = dst_ptr.to(tl.pointer_type(tl.uint64))
    if NEED_MASK:
        mask = offs < nwide
        vals = tl.load(src64 + offs, mask=mask)
        tl.store(dst64 + pid_r * nwide + offs, vals, mask=mask)
    else:
        vals = tl.load(src64 + offs)
        tl.store(dst64 + pid_r * nwide + offs, vals)


@triton.jit
def _tile_leading_flat_kernel(
    src_ptr, dst_ptr, src_numel, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    """Per-element leading repeat fallback (odd numel or misaligned storage)."""
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < src_numel
        vals = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + pid_r * src_numel + offs, vals, mask=mask)
    else:
        vals = tl.load(src_ptr + offs)
        tl.store(dst_ptr + pid_r * src_numel + offs, vals)


@triton.jit
def _tile_gather_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """General multi-dim tile: decompose the flat output index into per-dim
    indices, apply the per-dim modulo against the source shape, and accumulate
    the contiguous source offset.  Correctness fallback for mixed repeats."""
    pid = tl.program_id(0)
    out_offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = out_offs
    src_off = tl.zeros([BLOCK], dtype=tl.int32)
    if NDIM >= 6:
        c = offs % s5
        offs = offs // s5
        src_off += (c % a5) * st5
    if NDIM >= 5:
        c = offs % s4
        offs = offs // s4
        src_off += (c % a4) * st4
    if NDIM >= 4:
        c = offs % s3
        offs = offs // s3
        src_off += (c % a3) * st3
    if NDIM >= 3:
        c = offs % s2
        offs = offs // s2
        src_off += (c % a2) * st2
    if NDIM >= 2:
        c = offs % s1
        offs = offs // s1
        src_off += (c % a1) * st1
    if NDIM >= 1:
        c = offs % s0
        src_off += (c % a0) * st0
    if NEED_MASK:
        mask = out_offs < n_elements
        vals = tl.load(src_ptr + src_off, mask=mask)
        tl.store(dst_ptr + out_offs, vals, mask=mask)
    else:
        vals = tl.load(src_ptr + src_off)
        tl.store(dst_ptr + out_offs, vals)


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN TILE")

    in_rank = inp.dim()
    dims_rank = len(dims)
    in_shape = list(inp.shape)
    dims_shape = list(dims)

    if dims_rank < in_rank:
        dims_shape = [1] * (in_rank - dims_rank) + dims_shape
    elif dims_rank > in_rank:
        in_shape = [1] * (dims_rank - in_rank) + in_shape

    is_empty = False
    out_shape = []
    for i in range(len(in_shape)):
        d = dims_shape[i]
        if d < 0:
            raise RuntimeError(
                "the number of repetitions per dimension out of range "
                "(expected to >= 0) but got {}".format(d)
            )
        if d == 0:
            is_empty = True
        out_shape.append(in_shape[i] * d)

    out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
    if is_empty:
        return out

    in_contig = inp.reshape(in_shape)
    if not in_contig.is_contiguous():
        in_contig = in_contig.contiguous()

    # Only the leading dimension is repeated (dims_shape[1:] all 1): the output
    # is `rep` contiguous copies of the source block.  This is the common case
    # (benchmark repeats dim 0) and maps to a no-div/no-mod block-DMA copy.
    if all(d == 1 for d in dims_shape[1:]):
        rep = dims_shape[0]
        src_numel = in_contig.numel()
        elem = in_contig.element_size()
        epl = {2: 4, 4: 2}.get(elem, 1)
        if (
            epl > 1
            and src_numel % epl == 0
            and in_contig.data_ptr() % 8 == 0
            and out.data_ptr() % 8 == 0
        ):
            nwide = src_numel // epl
            grid = (triton.cdiv(nwide, _PACK_BLOCK), rep)
            _tile_leading_pack_kernel[grid](
                in_contig,
                out,
                nwide,
                BLOCK=_PACK_BLOCK,
                NEED_MASK=(nwide % _PACK_BLOCK != 0),
                num_warps=32,
            )
        else:
            grid = (triton.cdiv(src_numel, _FLAT_BLOCK), rep)
            _tile_leading_flat_kernel[grid](
                in_contig,
                out,
                src_numel,
                BLOCK=_FLAT_BLOCK,
                NEED_MASK=(src_numel % _FLAT_BLOCK != 0),
                num_warps=4,
            )
        return out

    # General multi-dim repeat: flat modulo gather.
    r = len(in_shape)
    in_strides = list(in_contig.stride())
    out_shapes = tuple(out_shape) + (1,) * (6 - r)
    in_shapes = tuple(in_shape) + (1,) * (6 - r)
    strides = tuple(in_strides) + (0,) * (6 - r)
    n = out.numel()
    grid = (triton.cdiv(n, _GATHER_BLOCK),)
    _tile_gather_kernel[grid](
        in_contig,
        out,
        n,
        *out_shapes,
        *in_shapes,
        *strides,
        NDIM=r,
        BLOCK=_GATHER_BLOCK,
        NEED_MASK=(n % _GATHER_BLOCK != 0),
        num_warps=4,
    )
    return out
