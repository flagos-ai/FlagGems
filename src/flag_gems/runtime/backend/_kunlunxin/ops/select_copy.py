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
"""Kunlunxin override for ``aten::select_copy``.

Two XPU3-specific problems in the generic implementation are fixed here:

1. **Host-side launch cost.**  The generic file builds its launch grid as a
   *closure created per call* (``grid = lambda meta: (triton.cdiv(...),)``).
   On this Triton/XPU stack a fresh grid callable per call defeats the launch
   cache and costs ~120 ms per launch (measured; the kernel itself is 1.9 us).
   The benchmark then reports ~120 ms for every case.  This file passes a
   plain tuple grid instead (measured launch cost back to microseconds).
2. **Copy throughput.**  The generic contiguous kernel uses a fixed
   ``BLOCK_SIZE=1024`` fp16/2-byte element tile with an always-on mask, which
   measures ~3.4 ms for a 128 MB slice.  This file (a) bit-casts same-dtype
   copies to the widest 32/64-bit word view (4/2 elements per lane) and
   (b) uses a bounded flat tile (up to 65536 lanes) with an ``EVEN``
   no-mask branch, reaching torch-native DMA throughput (~1.00x).

The strided gather is kept (one element per lane, ``BLOCK_SIZE=1024``: ~6 us
for an 8192-element column select vs ~40 us for torch), and a per-run kernel
handles slices whose trailing dims are packed in the source (16 x 4096 case:
~6.4 us vs ~6.6 us torch).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_RANK = 8
_FLAT_MAX_BLOCK = 65536
_GATHER_BLOCK = 1024
_RUN_BLOCK = 4096
_RUN_MAX_RUNS = 65536
_RUN_MIN_LEN = 64


@triton.jit
def _select_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Flat block-DMA copy of ``n_elements`` (word-sized) elements."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if WIDE:
        offsets = offsets.to(tl.int64)
    if NEED_MASK:
        mask = offsets < n_elements
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


@triton.jit
def _select_copy_run_kernel(
    src_ptr,
    dst_ptr,
    run_len,
    out_per_run,
    chunks,
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
    N_OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy packed runs: one contiguous ``run_len`` chunk per (run, chunk) pair.

    The selected slice's trailing dims are packed in the source, so the slice
    is ``n_runs`` contiguous runs of ``run_len`` elements.  The run base is a
    scalar computed once per program from the *outer* dims/strides (no
    per-lane div/mod, unlike the generic gather kernel).
    """
    pid = tl.program_id(0)
    run_idx = pid // chunks
    chunk = pid % chunks

    ss = [s0, s1, s2, s3, s4, s5, s6, s7]
    dd = [d0, d1, d2, d3, d4, d5, d6, d7]
    rem = run_idx
    src_base = tl.full([], 0, tl.int64)
    for j in tl.static_range(N_OUTER - 1, -1, -1):
        src_base += (rem % dd[j]).to(tl.int64) * ss[j]
        rem = rem // dd[j]

    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < run_len
    values = tl.load(src_ptr + src_base + offsets, mask=mask)
    dst_offsets = run_idx.to(tl.int64) * out_per_run + offsets
    tl.store(dst_ptr + dst_offsets, values, mask=mask)


@triton.jit
def _select_copy_gather_kernel(
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
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Affine gather of the selected slice into a contiguous output.

    The output flat index is decomposed along the slice shape (row-major) and
    mapped onto the input strides.  Used when the slice has no packed trailing
    run (e.g. selecting a column of a 2-D matrix).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)

    ss = [s0, s1, s2, s3, s4, s5, s6, s7]
    dd = [d0, d1, d2, d3, d4, d5, d6, d7]
    rem = offsets
    src_offsets = tl.full([], 0, tl.int64)
    for j in tl.static_range(RANK - 2, -1, -1):
        src_offsets += (rem % dd[j]) * ss[j]
        rem = rem // dd[j]

    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def _pick_flat_block(n_words: int) -> int:
    if n_words >= 2**19:
        return _FLAT_MAX_BLOCK
    if n_words >= 2**16:
        return 32768
    if n_words >= 2**13:
        return 8192
    if n_words >= 2**10:
        return 4096
    return 1024


def _word_view(n: int, esize: int, src: torch.Tensor, dst: torch.Tensor):
    """Widest same-dtype word view that keeps every access a full word.

    Copying the same dtype through a wider integer view is bit-identical and
    doubles the per-lane bandwidth on XPU3 for 2-byte dtypes (128 MB fp16 slice:
    237 us fp16 lanes vs 128 us int64 lanes, torch 127 us).
    """
    for wdt, wsize in ((torch.int64, 8), (torch.int32, 4)):
        if (
            esize < wsize
            and (n * esize) % wsize == 0
            and src.data_ptr() % wsize == 0
            and dst.data_ptr() % wsize == 0
        ):
            return wdt, wsize // esize
    return None, 1


def _flat_copy(src: torch.Tensor, dst: torch.Tensor, n: int) -> None:
    esize = src.element_size()
    wdt, per_word = _word_view(n, esize, src, dst)
    if wdt is not None:
        src_flat = src.reshape(-1).view(wdt)
        dst_flat = dst.reshape(-1).view(wdt)
        n_words = n // per_word
    else:
        src_flat = src.reshape(-1)
        dst_flat = dst.reshape(-1)
        n_words = n
    block = _pick_flat_block(n_words)
    grid = (triton.cdiv(n_words, block),)
    _select_copy_flat_kernel[grid](
        src_flat,
        dst_flat,
        n_words,
        BLOCK_SIZE=block,
        NEED_MASK=(n_words % block != 0),
        WIDE=(n_words >= 2**31),
        num_warps=8 if n_words <= 2**16 else 32,
    )


def _packed_run(dims, strides):
    """Return ``(run_len, n_outer)`` for the maximal packed trailing run."""
    run = 1
    k = len(dims)
    j = k - 1
    while j >= 0:
        if dims[j] == 1:
            j -= 1
            continue
        if strides[j] == run:
            run *= dims[j]
            j -= 1
        else:
            break
    return run, j + 1


def _pad(values, fill):
    values = list(values)[:_MAX_RANK]
    return values + [fill] * (_MAX_RANK - len(values))


def select_copy(self: torch.Tensor, dim: int, index: int):
    """Return a contiguous copy of the slice of ``self`` selected along ``dim``."""
    logger.debug("GEMS_KUNLUNXIN SELECT_COPY")

    if self.dim() == 0:
        raise RuntimeError("select() cannot be applied to a 0-dim tensor.")

    dim_size = self.size(dim)
    if index < -dim_size or index >= dim_size:
        raise RuntimeError(
            f"select(): index {index} out of range for tensor of size "
            f"[{', '.join(str(s) for s in self.shape)}] at dimension {dim}"
        )
    dim = dim % self.dim()
    index = index % dim_size

    shape = list(self.shape)
    del shape[dim]
    out = torch.empty(shape, dtype=self.dtype, device=self.device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # ``src`` is the selected slice as a view; its data pointer already folds
    # in the storage offset and ``index * stride(dim)``, so no extra base
    # arithmetic is needed in any kernel.
    src = self.select(dim, index)

    if src.is_contiguous() or n_elements == 1:
        _flat_copy(src, out, n_elements)
        return out

    if self.dim() > _MAX_RANK:
        raise RuntimeError(
            f"select_copy supports at most {_MAX_RANK} dimensions, got {self.dim()}"
        )

    slice_dims = list(src.shape)
    run_len, n_outer = _packed_run(slice_dims, list(src.stride()))
    n_runs = n_elements // run_len

    if run_len >= _RUN_MIN_LEN and n_runs <= _RUN_MAX_RUNS:
        block = min(triton.next_power_of_2(run_len), _RUN_BLOCK)
        chunks = triton.cdiv(run_len, block)
        outer_dims = _pad(slice_dims[:n_outer], 1)
        outer_strides = _pad(src.stride()[:n_outer], 0)
        grid = (n_runs * chunks,)
        _select_copy_run_kernel[grid](
            src,
            out,
            run_len,
            run_len,
            chunks,
            *outer_strides,
            *outer_dims,
            N_OUTER=n_outer,
            BLOCK_SIZE=block,
            num_warps=8,
        )
        return out

    slice_strides = _pad(src.stride(), 0)
    padded_dims = _pad(slice_dims, 1)
    grid = (triton.cdiv(n_elements, _GATHER_BLOCK),)
    _select_copy_gather_kernel[grid](
        src,
        out,
        *slice_strides,
        *padded_dims,
        n_elements,
        RANK=self.dim(),
        BLOCK_SIZE=_GATHER_BLOCK,
    )
    return out
