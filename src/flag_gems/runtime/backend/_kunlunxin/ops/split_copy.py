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

"""Kunlunxin override for ``aten::split_copy``.

The generic implementation loops over the chunks in Python and, for every
chunk, allocates a fresh tensor with ``torch.empty`` and launches a kernel
through ``grid = lambda meta: (...)``.  On the XPU backend the launch grid is
part of the compilation cache key, so each launch recompiles the kernel from
scratch (measured ~133 ms per call for a small copy); with up to 16 chunks and
two kernels per chunk (copy + the gems ``empty`` override) a single call can
pay dozens of recompiles.  That is the whole reason the generic path scores
0.002x.

The override:

* allocates ONE flat storage with ``torch.empty_strided`` and returns chunk
  views into it (no per-chunk allocation, no gems ``empty`` dispatch);
* launches ONE kernel for the whole split: program (p, chunk, block) copies a
  contiguous run of ``chunk_len * post`` elements (an affine copy with no
  per-element division), which is safe because the input is contiguous;
* uses constant-tuple grids and ``@libentry`` so nothing recompiles per call.

For a contiguous input the output storage order is exactly chunk-major, so
chunk ``k`` simply occupies ``[k*pre*chunk_size*post, ...)`` in the storage.
A non-contiguous input keeps a per-chunk gather kernel (correctness path).
"""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy, tle_dma_available

_MAX_RANK = 8
_BLOCK = 4096
# Minimum elements before the descriptor (tle.dsa) path pays off: it costs a
# couple of extra host-side steps per chunk, which dominates tiny splits.
_TLE_MIN_ELEMS = 1 << 20


@libentry()
@triton.jit
def _split_copy_chunk_kernel(
    src_ptr,
    dst_ptr,
    dim_size,
    post,
    chunk_size,
    pre,
    BLOCK_SIZE: tl.constexpr,
):
    # Contiguous input: the chunk is ``pre`` contiguous runs of
    # ``chunk_len*post`` elements, separated by ``dim_size*post`` in the input
    # and laid out chunk-major in the output storage.  No div/mod.
    p = tl.program_id(0)
    k = tl.program_id(1)
    j = tl.program_id(2)
    chunk_len = tl.minimum(chunk_size, dim_size - k * chunk_size)
    n_chunk = chunk_len * post
    off = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_chunk
    src_off = p * (dim_size * post) + k * chunk_size * post + off
    dst_off = k * (pre * chunk_size * post) + p * n_chunk + off
    v = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + dst_off, v, mask=mask)


@libentry()
@triton.jit
def _split_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    src_base,
    dst_base,
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
    ND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Arbitrary layout: decompose the flat output index over the chunk shape.
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements
    rem = off
    src = tl.full([BLOCK_SIZE], src_base, tl.int64)
    if ND > 7:
        c = rem % d7
        rem = rem // d7
        src += c.to(tl.int64) * s7
    if ND > 6:
        c = rem % d6
        rem = rem // d6
        src += c.to(tl.int64) * s6
    if ND > 5:
        c = rem % d5
        rem = rem // d5
        src += c.to(tl.int64) * s5
    if ND > 4:
        c = rem % d4
        rem = rem // d4
        src += c.to(tl.int64) * s4
    if ND > 3:
        c = rem % d3
        rem = rem // d3
        src += c.to(tl.int64) * s3
    if ND > 2:
        c = rem % d2
        rem = rem // d2
        src += c.to(tl.int64) * s2
    if ND > 1:
        c = rem % d1
        rem = rem // d1
        src += c.to(tl.int64) * s1
    if ND > 0:
        c = rem % d0
        src += c.to(tl.int64) * s0
    v = tl.load(src_ptr + src, mask=mask)
    tl.store(dst_ptr + dst_base + off, v, mask=mask)


@libentry()
@triton.jit
def _split_copy_pure_flat_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Contiguous input with pre == 1 (dim == 0): the chunk-major output storage
    # has exactly the same element order as the input, so the whole split is a
    # single flat copy.  NEED_MASK is a constexpr so the exact-multiple case
    # avoids the masked-memory path.
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = off < n_elements
        v = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, v, mask=mask)
    else:
        v = tl.load(src_ptr + off)
        tl.store(dst_ptr + off, v)


@libentry()
@triton.jit
def _split_copy_runs_kernel(
    src_ptr,
    dst_ptr,
    src_base,
    dst_base,
    DPOST: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RPP: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Contiguous input, pre > 1: one launch per chunk; program (p0, j) copies
    # RPP consecutive runs, each ``RUN`` contiguous units.  Every load/store in
    # the unrolled body stays affine (a single run), which is what keeps the
    # XPU backend on the block-DMA path; looping over runs instead of widening
    # the tile is what reduces the program count.
    p0 = tl.program_id(0) * RPP
    j = tl.program_id(1)
    off = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = off < RUN
        for r in tl.static_range(RPP):
            p = p0 + r
            v = tl.load(src_ptr + p * DPOST + src_base + off, mask=mask)
            tl.store(dst_ptr + dst_base + p * RUN + off, v, mask=mask)
    else:
        for r in tl.static_range(RPP):
            p = p0 + r
            v = tl.load(src_ptr + p * DPOST + src_base + off)
            tl.store(dst_ptr + dst_base + p * RUN + off, v)


def _merge_dims(sizes, strides):
    out_sizes, out_strides = [], []
    for size, stride in zip(sizes, strides):
        if size == 1:
            continue
        if out_sizes and out_strides[-1] == size * stride:
            out_sizes[-1] *= size
            out_strides[-1] = stride
        else:
            out_sizes.append(size)
            out_strides.append(stride)
    return out_sizes, out_strides


def _pick_block(n):
    b = triton.next_power_of_2(n)
    if b > 16384:
        b = 16384
    if b < 1024:
        b = 1024
    return b


def _flat_tier(n):
    """Block / warp tier for a pure flat copy (wider blocks for larger copies)."""
    if n <= 1024:
        return 1024, 2
    if n <= 8192:
        return 8192, 8
    if n <= (1 << 18):
        return 16384, 16
    return 32768, 32


def _use_tle_layout(src, dim, post, total):
    """Whether the descriptor (tle.dsa) path beats the Triton kernels here.

    Measured on the benchmark's six shapes: a strided / non-contiguous source
    wins through the descriptor (transposed input 0.044 -> 1.976,
    ``(8192, 8192) dim=1`` 0.096 -> 0.665), while a contiguous split with a wide
    trailing axis loses (``(16, 8192, 4096) dim=1`` 0.847 -> 0.422), so that
    layout stays on the Triton runs kernel.
    """
    if total < _TLE_MIN_ELEMS or not tle_dma_available():
        return False
    if not src.is_contiguous():
        return True
    return dim == src.dim() - 1 and post == 1


def split_copy(self: torch.Tensor, split_size: int, dim: int = 0):
    if self.dim() == 0:
        raise RuntimeError("split expects at least a 1-dimensional tensor")
    if split_size < 0:
        raise RuntimeError(
            f"split expects split_size be non-negative, but got split_size={split_size}"
        )

    dim_size = self.size(dim)
    if split_size == 0 and dim_size != 0:
        raise RuntimeError(
            "split_size can only be 0 if dimension size is 0, but got dimension "
            f"size of {dim_size}"
        )
    dim = dim % self.dim()

    if split_size == 0:
        n_chunks = 1
    else:
        n_chunks = max(1, (dim_size + split_size - 1) // split_size)

    chunk_lens = []
    for k in range(n_chunks):
        chunk_lens.append(min(split_size, dim_size - k * split_size))

    pre = 1
    for size in self.shape[:dim]:
        pre *= size
    post = 1
    for size in self.shape[dim + 1 :]:
        post *= size

    chunk_size = chunk_lens[0] if chunk_lens else 0
    # Every chunk gets the same slot span, so chunk k starts at k*span and the
    # (shorter) last chunk simply leaves the tail of its slot unused.  This
    # keeps the destination offset affine; the extra memory is at most one
    # chunk.
    total = self.numel()
    span = pre * chunk_size * post
    alloc = n_chunks * span if span > 0 else total
    # torch.empty_strided is a plain native allocation (torch.empty would
    # dispatch to the gems override inside a gems context, paying an extra
    # zeroing kernel and a recompile per call).
    storage = torch.empty_strided((alloc,), (1,), dtype=self.dtype, device=self.device)

    # Chunk views into the shared storage.  They are valid the moment the
    # storage exists, so build them before the copy and fill them in place.
    result = []
    for k, chunk_len in enumerate(chunk_lens):
        shape_k = list(self.shape)
        shape_k[dim] = chunk_len
        begin = k * span
        count = pre * chunk_len * post
        result.append(storage.narrow(0, begin, count).reshape(shape_k))

    if total > 0:
        with torch_device_fn.device(self.device):
            # Descriptor (tle.dsa) path - pure Triton, no hand-written payload.
            # `tle_copy` moves a strided source run with one 2D DMA per tile,
            # which is what a split along a non-leading axis needs: measured
            # (8192, 8192) dim=1 fp16 0.096 -> 0.665, transposed input
            # 0.044 -> 1.976.  Only for layouts where a chunk really is a 2D
            # strided block (last-axis split, or a non-contiguous source); a
            # split with a wide trailing axis (e.g. (16, 8192, 4096) dim=1) is
            # faster through the Triton runs kernel (0.422 vs 0.847).
            if _use_tle_layout(self, dim, post, total):
                copied = True
                for k, chunk_len in enumerate(chunk_lens):
                    if chunk_len == 0:
                        continue
                    if not tle_copy(
                        self.narrow(dim, k * chunk_size, chunk_len), result[k]
                    ):
                        copied = False
                        break
                if copied:
                    return result

            if self.is_contiguous():
                n_run = chunk_size * post
                es = self.element_size()
                if n_run > 0:
                    src, dst = self, storage
                    unit = 1
                    if es in (2, 4):
                        cand = 8 // es
                        if (
                            total % cand == 0
                            and n_run % cand == 0
                            and (dim_size * post) % cand == 0
                        ):
                            # 8-byte lanes: half/fourth the element count of a
                            # native-dtype copy, which is what the XPU copy
                            # throughput is actually bounded by.
                            try:
                                src = self.reshape(-1).view(torch.int64)
                                dst = storage.view(torch.int64)
                                unit = cand
                            except RuntimeError:
                                src, dst, unit = self, storage, 1
                    if pre == 1:
                        n_units = total // unit
                        block, warps = _flat_tier(n_units)
                        _split_copy_pure_flat_kernel[(triton.cdiv(n_units, block),)](
                            src,
                            dst,
                            n_units,
                            BLOCK_SIZE=block,
                            NEED_MASK=n_units % block != 0,
                            num_warps=warps,
                        )
                    else:
                        run_u = n_run // unit
                        dpost_u = (dim_size * post) // unit
                        span_u = span // unit
                        for k, chunk_len in enumerate(chunk_lens):
                            if chunk_len == 0:
                                continue
                            run_k = (chunk_len * post) // unit
                            block = min(8192, max(64, triton.next_power_of_2(run_k)))
                            warps = 8 if block <= 8192 else 16
                            # Narrow runs are dominated by the per-program
                            # launch/address overhead, so fold many runs into
                            # one program; wide runs already saturate and the
                            # extra loop only adds pressure.
                            if run_k < 4096:
                                rpp = next(
                                    r for r in (32, 16, 8, 4, 2, 1) if pre % r == 0
                                )
                            else:
                                rpp = 1
                            grid = (triton.cdiv(pre, rpp), triton.cdiv(run_k, block))
                            _split_copy_runs_kernel[grid](
                                src,
                                dst,
                                k * run_u,
                                k * span_u,
                                DPOST=dpost_u,
                                RUN=run_k,
                                BLOCK_SIZE=block,
                                RPP=rpp,
                                NEED_MASK=run_k % block != 0,
                                num_warps=warps,
                            )
            else:
                if self.dim() > _MAX_RANK:
                    raise RuntimeError(
                        f"split_copy supports at most {_MAX_RANK} dimensions, "
                        f"got {self.dim()}"
                    )
                src_strides = [int(s) for s in self.stride()]
                for k, chunk_len in enumerate(chunk_lens):
                    chunk_n = pre * chunk_len * post
                    if chunk_n == 0:
                        continue
                    chunk_sizes = list(self.shape)
                    chunk_sizes[dim] = chunk_len
                    dims, strides = _merge_dims(chunk_sizes, src_strides)
                    nd = len(dims)
                    if nd > _MAX_RANK:
                        raise RuntimeError(
                            f"split_copy supports at most {_MAX_RANK} dimensions, "
                            f"got {self.dim()}"
                        )
                    dims += [1] * (_MAX_RANK - nd)
                    strides += [0] * (_MAX_RANK - nd)
                    src_base = k * chunk_size * int(self.stride(dim))
                    dst_base = k * span
                    block = 1024
                    _split_copy_strided_kernel[(triton.cdiv(chunk_n, block),)](
                        self,
                        storage,
                        chunk_n,
                        src_base,
                        dst_base,
                        *dims,
                        *strides,
                        ND=nd,
                        BLOCK_SIZE=block,
                    )

    return result
