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

"""Kunlunxin (XPU3) override for ``aten::split_copy``.

Why this override exists (measured on P800 / KL3):

* The generic ``flag_gems.ops.split_copy`` builds its launch grid with a
  ``grid = lambda meta: ...`` *inside* the operator body.  On the xpubin
  backend the launch grid is baked into the compile-cache key, so a fresh
  lambda per call forces a full recompile on every split launch.  Fingerprint:
  a fixed ~135 ms per chunk independent of shape (``split_copy`` wall clock
  was 545-1255 ms, dtype-equal Gems Speedup ~0.002x).  All grids here are
  constant tuples.
* The generic per-element kernels recompute the source offset with an int64
  div/mod per element (contiguous path) or a full rank-len int64
  decomposition (strided path).  On XPU3 that lowers to scalar address math
  and moves the copies at ~30 GB/s.

Routing (validated in ``harness/perf_ir/probe_split_copy_tle.py`` and
``probe_split_copy_candidates3.py``):

* contiguous input, ``dim == 0``: the chunks concatenate to the whole tensor
  in flat order, so allocate one output buffer and move the whole tensor with
  the tle flat TMA copy (``tle_copy``), then hand out contiguous views.  One
  launch, ~0.95x of ``aten::split_copy``.
* contiguous input, ``dim > 0``: each chunk is ``outer`` contiguous runs of
  ``run_len`` elements.  Copy it with one flat 1-D launch whose address map
  ``(w // RW) * PW + base + (w % RW)`` is *piecewise affine* with ``RW``/``PW``
  as ``constexpr`` (constant div/mod lowers to mul-shift) and 8-byte
  (uint64) lanes.  ~0.84-1.0x of torch.
* non-contiguous input: hand each chunk view to ``tle_copy``; its
  ``tle.dsa`` row-copy / on-chip-transpose paths cover strided, transposed and
  broadcast layouts (transpose reached 1.7-2.2x).  Anything ``tle_copy``
  declines falls back to a local int64 rank-decomposition kernel that mirrors
  the generic algorithm (correct for every layout), so no correctness is lost.

Nothing here calls aten compute: the only torch calls are
``torch.empty_strided`` (allocation) and ``.narrow``/``.view``/``.reshape``
(metadata views).  Data movement is done by real XPU Triton / tle kernels.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

_MAX_RANK = 8
# 32768 uint64 words = 256 KB per program: the best measured flat-copy point
# on XPU3 (see the trajectory / wiki `technique-xpu3-uint64-flat-copy`).
_PACK_BLOCK = 32768
# Per-run element kernel (packing not expressible) tile.
_RUN_BLOCK = 4096
# Fallback strided decomposition tile.
_STRIDED_BLOCK = 1024
_INT32_SAFE = 2**31 - 1


@triton.jit
def _split_pack_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    src_base_w,
    RW: tl.constexpr,
    PW: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Copy one contiguous-input chunk as a piecewise-affine uint64 stream.

    ``RW`` words form one contiguous run (``chunk_len * dim_stride`` elements);
    runs repeat every ``PW`` words (``dim_size * dim_stride``); ``src_base_w``
    is the chunk's first run offset in words.  ``RW``/``PW`` are constexpr so
    ``w // RW`` / ``w % RW`` compile to constant multiply-shift, keeping the
    access affinely vectorizable.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    so = (off // RW) * PW + src_base_w + (off % RW)
    s = src_ptr.to(tl.pointer_type(tl.uint64))
    d = dst_ptr.to(tl.pointer_type(tl.uint64))
    if NEED_MASK:
        m = off < n_words
        tl.store(d + off, tl.load(s + so, mask=m), mask=m)
    else:
        tl.store(d + off, tl.load(s + so))


@triton.jit
def _split_run_kernel(
    src_ptr,
    dst_ptr,
    run_len,
    src_base,
    src_period,
    BLOCK: tl.constexpr,
):
    """Per-run copy for a contiguous-input chunk when uint64 packing is not
    expressible (odd run length / misaligned base).  Program ``(o, b)`` copies
    ``BLOCK`` elements of run ``o``; the run base is a scalar, no div/mod."""
    o = tl.program_id(0)
    b = tl.program_id(1)
    r = b * BLOCK + tl.arange(0, BLOCK)
    m = r < run_len
    r64 = r.to(tl.int64)
    so = o.to(tl.int64) * src_period + src_base + r64
    v = tl.load(src_ptr + so, mask=m)
    tl.store(dst_ptr + o.to(tl.int64) * run_len + r64, v, mask=m)


@triton.jit
def _split_strided_kernel(
    src_ptr,
    dst_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    n_elements,
    src_base,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """General strided gather for non-contiguous inputs the tle path declined.

    Mirrors the generic algorithm: decompose the output flat index over the
    chunk shape (innermost first) and scale each coordinate by the source
    stride; ``src_base`` folds the storage offset plus the chunk start.  int64
    offsets keep it correct for any address; it is the rare fallback path.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)

    src_strides = [s0, s1, s2, s3, s4, s5, s6, s7]
    chunk_dims = [d0, d1, d2, d3, d4, d5, d6, d7]

    rem = offsets
    src_offsets = tl.full([], src_base, tl.int64)
    for j in tl.static_range(RANK - 1, -1, -1):
        idx = rem % chunk_dims[j]
        rem = rem // chunk_dims[j]
        src_offsets += idx * src_strides[j]

    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def _contig_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _empty_contig(shape, dtype, device):
    """Contiguous allocation without going through ``torch.empty``.

    ``torch.empty`` may dispatch to a gems override (extra zeroing kernel + a
    per-call recompile in this environment); ``empty_strided`` with explicit
    contiguous strides is a pure native allocation.
    """
    shape = list(shape)
    return torch.empty_strided(
        tuple(shape), tuple(_contig_strides(shape)), dtype=dtype, device=device
    )


def _split_contig_dim0(self, split_size, dim_size, n_chunks, element_size):
    """contiguous ``dim == 0``: one flat copy of the whole tensor + views."""
    numel = self.numel()
    buffer = torch.empty_strided(
        (numel,), (1,), dtype=self.dtype, device=self.device
    )
    if numel > 0:
        handled = False
        # The whole tensor is contiguous; tle's flat TMA path is the fastest
        # measured move (~0.95-0.98x of aten). Fall back to a packed flat copy
        # when tle declines (alignment / dtype).
        flat_src = self.reshape(-1)
        if tle_copy(flat_src, buffer):
            handled = True
        elif (
            element_size in (1, 2, 4, 8)
            and self.data_ptr() % 8 == 0
            and buffer.data_ptr() % 8 == 0
            and (numel * element_size) % 8 == 0
            and numel * element_size // 8 <= _INT32_SAFE
        ):
            n_words = numel * element_size // 8
            grid = (triton.cdiv(n_words, _PACK_BLOCK),)
            _split_pack_kernel[grid](
                flat_src,
                buffer,
                n_words,
                0,
                RW=n_words,
                PW=1,
                BLOCK=_PACK_BLOCK,
                NEED_MASK=n_words % _PACK_BLOCK != 0,
            )
            handled = True
        if not handled:
            # Last resort: one per-run flat kernel (still a constant grid).
            grid = (1, triton.cdiv(numel, _RUN_BLOCK))
            _split_run_kernel[grid](flat_src, buffer, numel, 0, numel, BLOCK=_RUN_BLOCK)

    outer = self.numel() // dim_size if dim_size != 0 else 0
    results = []
    chunk_start = 0
    for _ in range(n_chunks):
        chunk_len = min(split_size, dim_size - chunk_start)
        out_shape = list(self.shape)
        out_shape[0] = chunk_len
        if buffer.numel() == 0:
            results.append(buffer.view(out_shape))
        else:
            flat_start = chunk_start * outer
            flat_end = (chunk_start + chunk_len) * outer
            results.append(buffer[flat_start:flat_end].view(out_shape))
        chunk_start += chunk_len
    return results


def _launch_contig_chunk(self, out, chunk_start, chunk_len, dim_stride, period, esize):
    """Copy one chunk of a contiguous input (dim may be anything)."""
    run_len = chunk_len * dim_stride
    numel = out.numel()
    outer = self.numel() // period if period != 0 else 1

    packable = (
        outer >= 1
        and esize in (1, 2, 4, 8)
        and self.data_ptr() % 8 == 0
        and out.data_ptr() % 8 == 0
        and (run_len * esize) % 8 == 0
        and (period * esize) % 8 == 0
        and (chunk_start * dim_stride * esize) % 8 == 0
        and (numel * esize // 8) <= _INT32_SAFE
    )
    if packable:
        rw = run_len * esize // 8
        pw = period * esize // 8
        base_w = chunk_start * dim_stride * esize // 8
        n_words = numel * esize // 8
        grid = (triton.cdiv(n_words, _PACK_BLOCK),)
        _split_pack_kernel[grid](
            self,
            out,
            n_words,
            base_w,
            RW=rw,
            PW=pw,
            BLOCK=_PACK_BLOCK,
            NEED_MASK=n_words % _PACK_BLOCK != 0,
        )
        return

    grid = (outer, triton.cdiv(run_len, _RUN_BLOCK))
    _split_run_kernel[grid](
        self,
        out,
        run_len,
        chunk_start * dim_stride,
        period,
        BLOCK=_RUN_BLOCK,
    )


def _launch_strided_chunk(self, out, chunk_start, chunk_len, dim, src_strides):
    """Generic rank-decomposition gather for non-contiguous chunks."""
    rank = self.dim()
    if rank > _MAX_RANK:
        raise RuntimeError(
            f"split_copy supports at most {_MAX_RANK} dimensions, got {rank}"
        )
    chunk_strides = list(src_strides) + [0] * (_MAX_RANK - rank)
    chunk_dims = list(self.shape)
    chunk_dims[dim] = chunk_len
    chunk_dims = chunk_dims + [1] * (_MAX_RANK - rank)
    n_elements = out.numel()
    src_base = chunk_start * src_strides[dim]
    grid = (triton.cdiv(n_elements, _STRIDED_BLOCK),)
    _split_strided_kernel[grid](
        self,
        out,
        *chunk_strides,
        *chunk_dims,
        n_elements,
        src_base,
        RANK=rank,
        BLOCK=_STRIDED_BLOCK,
    )


def split_copy(self: torch.Tensor, split_size: int, dim: int = 0):
    """Split ``self`` into chunks of ``split_size`` along ``dim`` and copy each.

    Mirrors ``aten::split_copy`` (see the generic implementation for the full
    contract): the last chunk is smaller on a non-divisible split, negative
    ``dim`` is normalized, every output is a freshly allocated contiguous
    tensor that never aliases the input, and the error messages match PyTorch.
    """
    logger.debug("GEMS_KUNLUNXIN SPLIT_COPY")

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
        # dim_size == 0; a single empty chunk.
        n_chunks = 1
    else:
        n_chunks = max(1, (dim_size + split_size - 1) // split_size)

    if self.is_contiguous():
        dim_stride = self.stride(dim)
        period = dim_size * dim_stride
        esize = self.element_size()

        # dim == 0: all chunks concatenate to the whole tensor -> single copy.
        if dim == 0:
            return _split_contig_dim0(self, split_size, dim_size, n_chunks, esize)

        result = []
        chunk_start = 0
        for _ in range(n_chunks):
            chunk_len = min(split_size, dim_size - chunk_start)
            out_shape = list(self.shape)
            out_shape[dim] = chunk_len
            out = _empty_contig(out_shape, self.dtype, self.device)
            if out.numel() > 0:
                _launch_contig_chunk(
                    self, out, chunk_start, chunk_len, dim_stride, period, esize
                )
            result.append(out)
            chunk_start += chunk_len
        return result

    # Non-contiguous input.
    rank = self.dim()
    if rank > _MAX_RANK:
        raise RuntimeError(
            f"split_copy supports at most {_MAX_RANK} dimensions, got {rank}"
        )
    src_strides = [int(s) for s in self.stride()]
    result = []
    chunk_start = 0
    for _ in range(n_chunks):
        chunk_len = min(split_size, dim_size - chunk_start)
        out_shape = list(self.shape)
        out_shape[dim] = chunk_len
        out = _empty_contig(out_shape, self.dtype, self.device)
        if out.numel() > 0:
            src_view = self.narrow(dim, chunk_start, chunk_len)
            handled = False
            with torch_device_fn.device(self.device):
                handled = tle_copy(src_view, out)
            if not handled:
                _launch_strided_chunk(self, out, chunk_start, chunk_len, dim, src_strides)
        result.append(out)
        chunk_start += chunk_len
    return result
