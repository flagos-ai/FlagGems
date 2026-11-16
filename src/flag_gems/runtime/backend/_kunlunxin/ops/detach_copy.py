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
"""Kunlunxin override for ``aten::detach_copy``.

The generic implementation copies with a fixed ``BLOCK_SIZE=1024`` tile of
native-dtype lanes and an always-on mask.  On XPU3 that is dominated by two
effects (see ``KERNEL_OPT_EXPERIENCE.md`` rows 39/70/71/75):

* 2-byte dtypes on 2-byte lanes pay a ~3x DMA penalty; a same-dtype copy can
  instead move the *widest integer word* per lane (bit-identical, 4/8 bytes),
* a large flat tile is what reaches block-DMA throughput, not a large grid,
  while the always-true mask forces the slow masked-memory lowering.

Measured on the benchmark matrix (2 MB / 4 MB copies), the generic path runs
at ~0.13x of native ``aten::detach_copy``; the word-view flat copy below runs
at ~0.94-1.05x (dtype-equal Gems Speedup ~1.00).

Non-contiguous inputs are materialized contiguously (the native contract) by
one of three device-side kernels, all driven by host-side shape/stride
arithmetic:

1. maximal packed trailing run >= 64: one program per (run, chunk) with the
   source base computed once per program from the outer dims/strides,
2. arbitrary layout, rank <= 8: per-element affine gather with the shape and
   stride of every dim passed as scalar kernel arguments,
3. rank > 8: the generic device-side metadata buffer variant (kept as the
   arbitrary-rank correctness path).

Complex inputs stay on the device path: lazy conjugate/negative bits are
resolved first (no-op views are untouched), the tensor is reinterpreted with
``torch.view_as_real`` and copied by the same kernels.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_RANK = 8
_FLAT_MAX_BLOCK = 32768
_RUN_BLOCK = 4096
_RUN_MIN_LEN = 64
_RUN_MAX_RUNS = 65536
_GATHER_BLOCK = 1024


@triton.jit
def _detach_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Flat block-DMA copy of ``n_words`` (word-sized) elements."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if WIDE:
        offsets = offsets.to(tl.int64)
    if NEED_MASK:
        mask = offsets < n_words
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


@triton.jit
def _detach_copy_run_kernel(
    src_ptr,
    dst_ptr,
    run_len,
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
    """Copy packed trailing runs: one contiguous ``run_len`` chunk per program.

    The source base is a scalar computed once per program from the *outer*
    dims/strides (no per-lane div/mod).  The mask is kept: on XPU3 a 1-D
    scalar-base access without a mask is silently miscompiled on
    non-contiguous runs (see ``view_as_real_copy``).
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
    dst_offsets = run_idx.to(tl.int64) * run_len + offsets
    tl.store(dst_ptr + dst_offsets, values, mask=mask)


@triton.jit
def _detach_copy_gather_kernel(
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
    """Affine gather of a strided tensor into a contiguous output.

    The output flat index is decomposed along the tensor's shape (row-major)
    and mapped onto the input strides.  ``src_ptr`` already folds in the
    storage offset, so a tensor whose base is a view needs no extra base
    arithmetic.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)

    ss = [s0, s1, s2, s3, s4, s5, s6, s7]
    dd = [d0, d1, d2, d3, d4, d5, d6, d7]
    rem = offsets
    src_offsets = tl.full([BLOCK_SIZE], 0, tl.int64)
    for j in tl.static_range(RANK - 1, -1, -1):
        src_offsets += (rem % dd[j]) * ss[j]
        rem = rem // dd[j]

    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _detach_copy_meta_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    meta_ptr,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Arbitrary-rank strided copy driven by a device metadata buffer.

    ``meta[2*i] = shape[i]``, ``meta[2*i+1] = stride[i]``.  Only used for
    non-contiguous tensors with more than ``_MAX_RANK`` dims.
    """
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx = offsets
    in_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for k in tl.static_range(ndim):
        i = ndim - 1 - k
        d = tl.load(meta_ptr + 2 * i).to(tl.int64)
        s = tl.load(meta_ptr + 2 * i + 1).to(tl.int64)
        c = idx % d
        idx = idx // d
        in_off += c * s
    values = tl.load(src_ptr + in_off, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def _pick_flat_block(n_words: int) -> int:
    """Block ladder: a large flat tile reaches block-DMA throughput on XPU3.

    Measured on the benchmark matrix, fewer/bigger programs win: 4 MB fp32 at
    32768 int32 words is ~1.00x of native, while a grid-fill tile (4096) is
    ~0.88x.  Small tensors shrink the tile to stay at the launch floor.
    """
    if n_words >= 2**19:
        return _FLAT_MAX_BLOCK
    if n_words >= 2**16:
        return 16384
    if n_words >= 2**13:
        return 8192
    if n_words >= 2**10:
        return 4096
    return 1024


def _word_view(n: int, esize: int, src: torch.Tensor, dst: torch.Tensor):
    """Widest same-dtype integer word view that keeps every access a full word.

    Copying the same dtype through an integer view is bit-identical.  2-byte
    lanes cost ~3x the DMA time of 4-byte lanes on XPU3 (knowledge row 75);
    int32 measured slightly ahead of int64 for the large benchmark cases.
    """
    for wdt, wsize in ((torch.int32, 4), (torch.int64, 8)):
        if (
            esize <= wsize
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
    _detach_copy_flat_kernel[grid](
        src_flat,
        dst_flat,
        n_words,
        BLOCK_SIZE=block,
        NEED_MASK=(n_words % block != 0),
        WIDE=(n_words >= 2**31),
        num_warps=16 if block >= 16384 else 8,
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


def _resolve_lazy_bits(input: torch.Tensor) -> torch.Tensor:
    """Materialize lazy conjugate / negative bits before any raw storage read.

    Both copy paths load the underlying storage directly, so a view carrying a
    lazy bit — e.g. ``torch._neg_view(x)`` or ``x.conj()`` — would be copied
    bitwise.  ``resolve_conj`` / ``resolve_neg`` return the input unchanged
    when no bit is set and otherwise materialize the logical values (a layout
    materialization, not part of the measured copy).  This mirrors the generic
    implementation and the native operator, whose freshly allocated result
    never carries a lazy bit.
    """
    if input.is_conj():
        input = input.resolve_conj()
    is_neg = getattr(input, "is_neg", None)
    if is_neg is not None and is_neg():
        input = input.resolve_neg()
    return input


def detach_copy(self: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``self`` detached from the autograd graph.

    Mirrors ``aten::detach_copy(Tensor self) -> Tensor``: the values are
    materialized into freshly allocated, dense contiguous memory, so the
    result never aliases the input.
    """
    logger.debug("GEMS_KUNLUNXIN DETACH_COPY")
    self = _resolve_lazy_bits(self)
    is_complex = self.dtype.is_complex
    if is_complex:
        self = torch.view_as_real(self)
    n = self.numel()
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if n == 0:
        # No elements to read: the freshly allocated (dense) output is already
        # the required result and a (0,) grid must not be launched.
        return torch.view_as_complex(out) if is_complex else out

    if self.is_contiguous():
        _flat_copy(self, out, n)
        return torch.view_as_complex(out) if is_complex else out

    shape = self.shape
    stride = self.stride()
    nd = len(shape)
    run_len, n_outer = _packed_run(shape, stride)
    if run_len >= _RUN_MIN_LEN and n_outer <= _MAX_RANK:
        n_runs = n // run_len
        if n_runs <= _RUN_MAX_RUNS:
            block = min(triton.next_power_of_2(run_len), _RUN_BLOCK)
            chunks = triton.cdiv(run_len, block)
            grid = (n_runs * chunks,)
            _detach_copy_run_kernel[grid](
                self,
                out,
                run_len,
                chunks,
                *_pad(stride[:n_outer], 0),
                *_pad(shape[:n_outer], 1),
                N_OUTER=n_outer,
                BLOCK_SIZE=block,
                num_warps=8,
            )
            return torch.view_as_complex(out) if is_complex else out

    if nd <= _MAX_RANK:
        grid = (triton.cdiv(n, _GATHER_BLOCK),)
        _detach_copy_gather_kernel[grid](
            self,
            out,
            *_pad(stride, 0),
            *_pad(shape, 1),
            n,
            RANK=nd,
            BLOCK_SIZE=_GATHER_BLOCK,
            num_warps=8,
        )
        return torch.view_as_complex(out) if is_complex else out

    # Arbitrary rank: hand the shape/stride metadata to the device.
    meta = torch.tensor(
        [v for s, st in zip(shape, stride) for v in (s, st)],
        dtype=torch.int64,
        device=self.device,
    )
    block = 256
    grid = (triton.cdiv(n, block),)
    _detach_copy_meta_kernel[grid](
        self, out, n, meta, ndim=nd, BLOCK_SIZE=block, num_warps=4
    )
    return torch.view_as_complex(out) if is_complex else out
