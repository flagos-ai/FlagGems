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

"""Kunlunxin (XPU3) override for ``aten::view_as_real_copy``.

The generic kernel expresses the (real, imag) pair of every complex element as
a trailing ``tl.arange(0, 2)`` lane and therefore emits a ``[BLOCK, 2]`` tile.
On TritonXPU that 2-D tile is (a) silently miscompiled on the strided path
(reads from wrong addresses) and (b) an interleaved, per-lane access on the
contiguous path -- 3-4 orders of magnitude off a streaming copy.

This override never builds a lane dimension:

* the copy is performed on a **32-bit integer alias** of the real view
  (``complex32 -> 1`` word, ``complex64 -> 2`` words, ``complex128 -> 4``
  words per element), which turns a pure bit move into the 4-byte DMA width
  this backend handles best;
* a fully contiguous source becomes one **1-D flat word copy**;
* a strided source is collapsed into contiguous **runs**: everything the
  innermost strides keep contiguous is one run, copied with a 1-D affine,
  unit-stride ``tl.load``/``tl.store``; the remaining dimensions become a
  *scalar* base offset per program, so no per-element gather is emitted.

All tensors passed to the kernels are real/word views of the caller's storage;
the wrapper allocates the output and preserves the generic semantics (dtype
map, ``shape + (2,)``, fresh non-aliasing tensor, empty early return, the
non-complex ``RuntimeError`` and the rank guard on the strided path).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Flat streaming copy: one program moves up to this many 32-bit words.  64K
# words = 256 KB per program; measured at parity with the native copy on the
# large benchmark tensors (32K words is ~8-12% behind on the biggest ones).
# The block is shrunk for small tensors so the grid still fills all 64 cores.
_FLAT_BLOCK = 65536
_MIN_FLAT_BLOCK = 1024
# Cores on the P800 (XPU3) part: keep at least this many programs.
_CORES = 64
# Per-run tile for the strided path; a run is copied in _RUN_BLOCK-word chunks.
# Larger chunks amortise the masked-access lowering cost (the only exact form
# for a strided run on this backend) and measured ~2x a 4K tile.
_RUN_BLOCK = 16384
# Leading (run) dimensions are decomposed with a fixed number of slots so the
# kernel always receives its full argument set; higher-rank inputs are
# rejected by the wrapper, exactly like the generic implementation.
_MAX_LEAD_RANK = 5

_REAL_DTYPE = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@libentry()
@triton.jit
def _varc_flat_kernel(src_ptr, dst_ptr, n_words, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    """1-D flat copy of a contiguous word range (no lane dimension)."""
    pid = ext.program_id(axis=0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        vals = tl.load(src_ptr + offs)
        tl.store(dst_ptr + offs, vals)
    else:
        mask = offs < n_words
        vals = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit
def _varc_run_kernel(
    src_ptr,
    dst_ptr,
    run_len,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one contiguous run per program, with all outer dims as scalar base.

    ``pid`` enumerates the runs in output (row-major) order, so the destination
    run starts at ``pid * run_len`` in the flat, contiguous output.  The source
    base is the digit decomposition of the same ``pid`` over the outer dims in
    logical order (innermost varies fastest), each coordinate scaled by the
    input's word stride.

    The access is always masked: on this backend an unmasked vector access
    whose base is a runtime scalar (``pid * stride``) is silently miscompiled
    for a non-contiguous run, while the masked form is exact at every block
    size measured.  A short run then pays a bounded probe cost, not wrong data.
    """
    pid = tl.program_id(axis=0)
    blk = tl.program_id(axis=1)
    rem = pid.to(tl.int64)
    src_base = 0
    if RANK > 0:
        idx = rem % d4
        rem = rem // d4
        src_base += idx * s4
    if RANK > 1:
        idx = rem % d3
        rem = rem // d3
        src_base += idx * s3
    if RANK > 2:
        idx = rem % d2
        rem = rem // d2
        src_base += idx * s2
    if RANK > 3:
        idx = rem % d1
        rem = rem // d1
        src_base += idx * s1
    if RANK > 4:
        idx = rem % d0
        src_base += idx * s0

    dst_base = pid.to(tl.int64) * run_len
    cols = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < run_len
    vals = tl.load(src_ptr + src_base + cols, mask=mask)
    tl.store(dst_ptr + dst_base + cols, vals, mask=mask)


def _word_view(t: torch.Tensor):
    """32-bit integer alias of ``t`` (metadata only), or None if impossible.

    ``view_as_real`` yields a trailing size-2 unit-stride dimension, so the
    ``fp16 -> int32`` merge, the ``fp32 -> int32`` identity and the
    ``fp64 -> int32`` split all succeed.  Any failure (e.g. an unexpected
    layout) simply keeps the real-dtype path.
    """
    try:
        return t.view(torch.int32)
    except RuntimeError:
        return None


def _collapse_runs(sizes, strides):
    """Split the layout into (trailing contiguous run, outer dims).

    Returns ``(run_len, outer_sizes, outer_strides)`` with the outer dims in
    logical (outermost-first) order and size-1 dims dropped.  A run is the
    whole tensor when every stride is contiguous; when the innermost stride is
    not 1 the run degenerates to a single element and every dim stays outer.
    """
    dims = [(int(n), int(s)) for n, s in zip(sizes, strides) if n != 1]
    if not dims:
        return 1, [], []

    # Walk the innermost dims as long as the strides keep them contiguous.
    i = len(dims) - 1
    run = dims[i][0]
    if dims[i][1] != 1:
        # No unit-stride innermost dim: per-element run, full decomposition.
        return 1, [n for n, _ in dims], [s for _, s in dims]
    while i - 1 >= 0 and dims[i - 1][1] == dims[i][1] * dims[i][0]:
        i -= 1
        run = 1
        for n, _ in dims[i:]:
            run *= n
    outer_sizes = [n for n, _ in dims[:i]]
    outer_strides = [s for _, s in dims[:i]]
    return run, outer_sizes, outer_strides


def view_as_real_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a complex tensor into a real tensor whose last dimension has size 2.

    Mirrors ``aten::view_as_real_copy`` on Kunlunxin: the copy is a 1-D affine
    word move (contiguous) or a run copy (strided), never a 2-D lane tile.
    """
    logger.debug("GEMS_KUNLUNXIN VIEW_AS_REAL_COPY")

    real_dtype = _REAL_DTYPE.get(self.dtype)
    if real_dtype is None:
        raise RuntimeError(
            f"view_as_real is only supported for complex tensors, not for "
            f"{str(self.dtype).replace('torch.', '')} tensors"
        )

    # ``view_as_real`` refuses tensors carrying a lazy conjugate or negative
    # bit; resolving them first is the same exact identity the generic
    # implementation uses and keeps the copy on the logical data.
    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    out = torch.empty(tuple(self.shape) + (2,), dtype=real_dtype, device=self.device)
    n_complex = self.numel()
    if n_complex == 0:
        return out

    # The real view shares the input's storage; the optional 32-bit alias turns
    # the (real, imag) pair into one/two/four words so the move rides the
    # 4-byte DMA width.  Neither is a compute op -- they only re-describe bits.
    src_real = torch.view_as_real(self)
    src_word = _word_view(src_real)
    dst_word = _word_view(out)

    with torch_device_fn.device(self.device):
        if src_word is not None and dst_word is not None:
            src_v, dst_v = src_word, dst_word
        else:
            src_v, dst_v = src_real, out

        if src_v.is_contiguous():
            n_words = dst_v.numel()
            block = _FLAT_BLOCK
            if n_words < _FLAT_BLOCK * _CORES:
                # Small tensor: keep one program per core instead of leaving
                # most of the grid empty.
                block = max(
                    _MIN_FLAT_BLOCK,
                    triton.next_power_of_2(triton.cdiv(n_words, _CORES)),
                )
            grid = (triton.cdiv(n_words, block),)
            _varc_flat_kernel[grid](
                src_v,
                dst_v,
                n_words,
                BLOCK=block,
                EVEN=(n_words % block == 0),
            )
            return out

        rank = self.dim()
        if rank > _MAX_LEAD_RANK:
            raise RuntimeError(
                f"view_as_real_copy supports at most {_MAX_LEAD_RANK} "
                f"dimensions, got {self.dim()}"
            )

        run_len, outer_sizes, outer_strides = _collapse_runs(
            src_v.shape, src_v.stride()
        )
        if len(outer_sizes) > _MAX_LEAD_RANK:
            raise RuntimeError(
                f"view_as_real_copy supports at most {_MAX_LEAD_RANK} "
                f"dimensions, got {self.dim()}"
            )
        pad = _MAX_LEAD_RANK - len(outer_sizes)
        # Pad at the front so the final `RANK` slots carry the outer dims in
        # logical order (outermost at slot 0, innermost at slot `RANK - 1`).
        # The kernel peels from the last slot, i.e. innermost varies fastest.
        dims = [1] * pad + outer_sizes
        strides = [0] * pad + outer_strides
        n_runs = 1
        for n in outer_sizes:
            n_runs *= n
        block = min(triton.next_power_of_2(run_len), _RUN_BLOCK)
        grid = (n_runs, triton.cdiv(run_len, block))
        _varc_run_kernel[grid](
            src_v,
            dst_v,
            run_len,
            *dims,
            *strides,
            RANK=len(outer_sizes),
            BLOCK=block,
        )
    return out
