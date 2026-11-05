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
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


BLOCK = 1024
MAX_NDIM = 8


@triton.jit
def _copy_storage_kernel(src_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)


@triton.jit
def _inverse_kernel(
    storage_ptr,
    src_ptr,
    out_ptr,
    n_storage,
    rows,
    cols,
    target_offset,
    src_offset,
    st0,
    st1,
    ss0,
    ss1,
    ls0,
    ls1,
    MODE: tl.constexpr,
    P2S0: tl.constexpr,
    P2S1: tl.constexpr,
    CONTIG_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One pass over the output storage: out[p] = src[...] if p is covered by the
    # as-strided view, else storage[p]. MODE 0: t = c*st1 (1D / single row).
    # MODE 1: t = r*st0 + c*st1 with st1 < st0 and (cols-1)*st1 < st0.
    # MODE 2: t = r*st0 + c*st1 with st1 > st0 and rows*st0 <= st1.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_storage
    t = offsets - target_offset
    tge0 = t >= 0

    if MODE == 0:
        if P2S1:
            c = t >> ls1
            rem = t & (st1 - 1)
        else:
            c = t // st1
            rem = t - c * st1
        r = tl.zeros([BLOCK], dtype=tl.int32)
        covered = tge0 & (rem == 0) & (c < cols)
    elif MODE == 1:
        if P2S0:
            q0 = t >> ls0
            m = t & (st0 - 1)
        else:
            q0 = t // st0
            m = t - q0 * st0
        if P2S1:
            c = m >> ls1
            rem = m & (st1 - 1)
        else:
            c = m // st1
            rem = m - c * st1
        r = q0
        covered = tge0 & (rem == 0) & (c < cols) & (r < rows)
    else:
        if P2S1:
            c = t >> ls1
            mm = t & (st1 - 1)
        else:
            c = t // st1
            mm = t - c * st1
        if P2S0:
            r = mm >> ls0
            rem = mm & (st0 - 1)
        else:
            r = mm // st0
            rem = mm - r * st0
        covered = tge0 & (rem == 0) & (r < rows) & (c < cols)

    if CONTIG_SRC:
        src_idx = r.to(tl.int64) * cols + c.to(tl.int64)
        v = tl.where(
            covered,
            tl.load(src_ptr + src_offset + src_idx, mask=mask & covered, other=0.0),
            tl.load(storage_ptr + offsets, mask=mask),
        )
    else:
        src_idx = r.to(tl.int64) * ss0 + c.to(tl.int64) * ss1
        v = tl.where(
            covered,
            tl.load(src_ptr + src_offset + src_idx, mask=mask & covered, other=0.0),
            tl.load(storage_ptr + offsets, mask=mask),
        )
    tl.store(out_ptr + offsets, v, mask=mask)


@triton.jit
def _scatter0d_kernel(src_ptr, out_ptr, target_offset, src_offset):
    tl.store(out_ptr + target_offset, tl.load(src_ptr + src_offset))


@triton.jit
def _scatter1d_kernel(
    src_ptr,
    out_ptr,
    n_elements,
    target_offset,
    src_offset,
    st0,
    ss0,
    CONTIG_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    target = (
        tl.zeros([BLOCK], dtype=tl.int64) + target_offset + offsets.to(tl.int64) * st0
    )
    if CONTIG_SRC:
        v = tl.load(src_ptr + src_offset + offsets, mask=mask)
    else:
        v = tl.load(src_ptr + src_offset + offsets.to(tl.int64) * ss0, mask=mask)
    tl.store(out_ptr + target, v, mask=mask)


@triton.jit
def _scatter2d_kernel(
    src_ptr,
    out_ptr,
    rows,
    cols,
    target_offset,
    src_offset,
    st0,
    st1,
    ss0,
    ss1,
    CONTIG_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.program_id(1)
    cmask = c < cols
    target = (
        tl.zeros([BLOCK], dtype=tl.int64)
        + target_offset
        + r.to(tl.int64) * st0
        + c.to(tl.int64) * st1
    )
    if CONTIG_SRC:
        v = tl.load(src_ptr + src_offset + r * cols + c, mask=cmask)
    else:
        v = tl.load(
            src_ptr + src_offset + r.to(tl.int64) * ss0 + c.to(tl.int64) * ss1,
            mask=cmask,
        )
    tl.store(out_ptr + target, v, mask=cmask)


@triton.jit
def _scatter_nd_kernel(
    src_ptr,
    out_ptr,
    n_elements,
    target_offset,
    src_offset,
    size_ptr,
    stride_ptr,
    src_stride_ptr,
    NDIM: tl.constexpr,
    CONTIG_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    remaining = offsets.to(tl.int64)
    target = tl.zeros([BLOCK], dtype=tl.int64) + target_offset
    for j in tl.static_range(NDIM):
        dim = NDIM - 1 - j
        s = tl.load(size_ptr + dim)
        st = tl.load(stride_ptr + dim)
        q = remaining // s
        r = remaining - q * s
        target += r * st
        remaining = q
    if CONTIG_SRC:
        v = tl.load(src_ptr + src_offset + offsets, mask=mask)
    else:
        rem2 = offsets.to(tl.int64)
        spos = tl.zeros([BLOCK], dtype=tl.int64)
        for j in tl.static_range(NDIM):
            dim = NDIM - 1 - j
            s = tl.load(size_ptr + dim)
            sst = tl.load(src_stride_ptr + dim)
            q = rem2 // s
            r = rem2 - q * s
            spos += r * sst
            rem2 = q
        v = tl.load(src_ptr + src_offset + spos, mask=mask)
    tl.store(out_ptr + target, v, mask=mask)


def _pick_inverse_mode(ndim, size, stride):
    if ndim == 0:
        return (0, 1, 1, 0, 1)
    if ndim == 1:
        if stride[0] > 0:
            return (0, 1, size[0], 0, stride[0])
        return (-1, 1, 1, 0, 1)
    if ndim == 2:
        rows, cols = size[0], size[1]
        st0, st1 = stride[0], stride[1]
        if st0 > 0 and st1 > 0:
            if rows == 1:
                return (0, rows, cols, st0, st1)
            if st1 < st0 and (cols - 1) * st1 < st0:
                return (1, rows, cols, st0, st1)
            if st1 > st0 and rows * st0 <= st1:
                return (2, rows, cols, st0, st1)
    return (-1, 1, 1, 0, 1)


def as_strided_scatter(self, src, size, stride, storage_offset=None):
    size = [int(s) for s in size]
    stride = [int(s) for s in stride]
    ndim = len(size)
    target_offset = (
        self.storage_offset() if storage_offset is None else int(storage_offset)
    )

    storage_numel = self.untyped_storage().nbytes() // self.element_size()
    dev = self.device
    out_storage = torch.empty(storage_numel, dtype=self.dtype, device=dev)

    expected_numel = 1
    for s in size:
        expected_numel *= s

    is_contiguous_view = ndim > 0
    for i in range(ndim):
        if stride[i] != math.prod(size[i + 1 :]):
            is_contiguous_view = False
            break
    covers_storage = (
        target_offset == 0 and expected_numel == storage_numel and is_contiguous_view
    )

    if covers_storage:
        if expected_numel > 0:
            _copy_storage_kernel[(triton.cdiv(expected_numel, BLOCK),)](
                src.reshape(-1),
                out_storage,
                expected_numel,
                BLOCK=BLOCK,
            )
    else:
        storage_view = torch.as_strided(self, (storage_numel,), (1,), 0)
        if expected_numel > 0:
            mode, rows, cols, st0, st1 = _pick_inverse_mode(ndim, size, stride)
            if mode >= 0:
                contig_src = src.is_contiguous()
                src_off = src.storage_offset()
                ss0 = src.stride()[0] if ndim > 0 else 0
                ss1 = src.stride()[1] if ndim > 1 else 0
                p2s0 = (mode >= 1) and (st0 > 0) and (st0 & (st0 - 1)) == 0
                p2s1 = (st1 > 0) and (st1 & (st1 - 1)) == 0
                ls0 = st0.bit_length() - 1 if p2s0 else 0
                ls1 = st1.bit_length() - 1 if p2s1 else 0
                _inverse_kernel[(triton.cdiv(storage_numel, BLOCK),)](
                    storage_view,
                    src,
                    out_storage,
                    storage_numel,
                    rows,
                    cols,
                    target_offset,
                    src_off,
                    st0,
                    st1,
                    ss0,
                    ss1,
                    ls0,
                    ls1,
                    MODE=mode,
                    P2S0=p2s0,
                    P2S1=p2s1,
                    CONTIG_SRC=contig_src,
                    BLOCK=BLOCK,
                )
            else:
                _copy_storage_kernel[(triton.cdiv(storage_numel, BLOCK),)](
                    storage_view,
                    out_storage,
                    storage_numel,
                    BLOCK=BLOCK,
                )
                contig_src = src.is_contiguous()
                src_off = src.storage_offset()
                if ndim == 0:
                    _scatter0d_kernel[(1,)](src, out_storage, target_offset, src_off)
                elif ndim == 1:
                    _scatter1d_kernel[(triton.cdiv(expected_numel, BLOCK),)](
                        src,
                        out_storage,
                        expected_numel,
                        target_offset,
                        src_off,
                        stride[0],
                        src.stride()[0],
                        CONTIG_SRC=contig_src,
                        BLOCK=BLOCK,
                    )
                elif ndim == 2 and size[0] <= 65535:
                    _scatter2d_kernel[(triton.cdiv(size[1], BLOCK), size[0])](
                        src,
                        out_storage,
                        size[0],
                        size[1],
                        target_offset,
                        src_off,
                        stride[0],
                        stride[1],
                        src.stride()[0],
                        src.stride()[1],
                        CONTIG_SRC=contig_src,
                        BLOCK=BLOCK,
                    )
                else:
                    size_t = torch.tensor(size, dtype=torch.int64, device=dev)
                    stride_t = torch.tensor(stride, dtype=torch.int64, device=dev)
                    src_stride_t = torch.tensor(
                        list(src.stride()), dtype=torch.int64, device=dev
                    )
                    _scatter_nd_kernel[(triton.cdiv(expected_numel, BLOCK),)](
                        src,
                        out_storage,
                        expected_numel,
                        target_offset,
                        src_off,
                        size_t,
                        stride_t,
                        src_stride_t,
                        NDIM=ndim,
                        CONTIG_SRC=contig_src,
                        BLOCK=BLOCK,
                    )
        else:
            _copy_storage_kernel[(triton.cdiv(storage_numel, BLOCK),)](
                storage_view,
                out_storage,
                storage_numel,
                BLOCK=BLOCK,
            )

    return torch.as_strided(
        out_storage, self.size(), self.stride(), self.storage_offset()
    )
