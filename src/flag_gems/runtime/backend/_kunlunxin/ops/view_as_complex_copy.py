# Copyright 2026, The FlagOS Contributors.
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
"""Kunlunxin XPU3 override for ``aten::view_as_complex_copy``.

The generic implementation (``src/flag_gems/ops/view_as_complex_copy.py``)
addresses the (real, imag) pair as a trailing ``tl.arange(0, 2)`` lane and
builds a ``[BLOCK, 2]`` broadcast-masked 2-D tile.  On this backend that tile
compiles but silently miscompiles the strided path (wrong values) and, on the
contiguous path, degenerates to per-lane accesses (three to four orders of
magnitude slower than a streaming copy).  This is the "2-D tile / axis-1"
miscompile family documented in the local knowledge base.

The override keeps the host-side validation and semantics identical to the
generic version and replaces only the device-side copy, using strictly 1-D
tiles:

* contiguous input                 -> flat 1-D real copy (block DMA);
* non-contiguous, pair-word safe   -> the (real, imag) pair is reinterpreted as
  one wider integer word (``int32`` for fp16, ``int64`` for fp32), so the copy
  moves one word per complex.  When the innermost lead dimension is contiguous
  (stride 2) one program copies a whole row with a contiguous inner run; other
  layouts use a 1-D gather over the decomposed flat index;
* anything else (odd lead stride, unaligned storage, fp64) -> two 1-D masked
  loads / two 1-D masked stores over real elements.

No CPU/ATen numeric fallback and no aten copy-from shortcut: every byte is
moved by a Triton kernel on the device.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
# A (real, imag) pair reinterpreted as one wider integer word.  fp64 pairs are
# 16 B and have no scalar word type, so fp64 always takes the two-load path.
_WIDE_INT = {torch.float16: torch.int32, torch.float32: torch.int64}
_WIDE_TL = {torch.float16: tl.int32, torch.float32: tl.int64}

# Leading (batch) dimensions are decomposed with a fixed number of slots so the
# kernel always receives its full argument sets; higher-rank inputs are
# rejected by the wrapper.
_MAX_LEAD_RANK = 5

_FLAT_BLOCK = 32768  # reals per program on the contiguous path
_WORD_BLOCK = 8192  # pair-words per program on the gather path
_TWO_LOAD_BLOCK = 8192  # complexes per program on the odd-stride fallback
# A whole-row program keeps the row inside one tile; wide rows go to the gather
# instead so the per-program buffer cannot blow the local-memory budget.
_ROW_BLOCK_MAX = 32768


@triton.jit
def _flat_real_copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Stream ``n`` reals contiguously: a pure 1-D affine block copy."""
    pid = tl.program_id(axis=0).to(tl.int64)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=mask), mask=mask)


@triton.jit
def _row_pairword_copy_kernel(
    src_ptr,
    dst_ptr,
    D,
    od0,
    od1,
    od2,
    od3,
    od4,
    os0,
    os1,
    os2,
    os3,
    os4,
    OR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: scalar row base + contiguous inner word run.

    ``od``/``os`` are the outer lead dimensions and their pair-word strides in
    original order (outermost first), padded to five slots.  ``OR`` is the
    number of real outer dimensions.  The innermost lead dimension has stride
    2 (asserted on the host), so after the pair-word reinterpret it is a
    contiguous run of ``D`` words.
    """
    r = tl.program_id(axis=0).to(tl.int64)
    od = [od0, od1, od2, od3, od4]
    os_ = [os0, os1, os2, os3, os4]
    j = tl.arange(0, BLOCK)
    rem = tl.zeros([BLOCK], dtype=tl.int64) + r
    base = tl.zeros([BLOCK], dtype=tl.int64)
    for jj in tl.static_range(OR - 1, -1, -1):
        idx = rem % od[jj]
        rem = rem // od[jj]
        base += idx * os_[jj]
    mask = j < D
    tl.store(dst_ptr + r * D + j, tl.load(src_ptr + base + j, mask=mask), mask=mask)


@triton.jit
def _gather_pairword_copy_kernel(
    src_ptr,
    dst_ptr,
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
    n,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """1-D gather of pair-words over the flat complex index.

    The output is contiguous, so the flat complex index decomposes row-major
    over the leading dimensions; each recovered coordinate is scaled by the
    matching source stride expressed in pair-words.
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    lead_dims = [d0, d1, d2, d3, d4]
    lead_strides = [s0, s1, s2, s3, s4]
    rem = i
    base = tl.zeros([BLOCK], dtype=tl.int64)
    for jj in tl.static_range(RANK - 1, -1, -1):
        idx = rem % lead_dims[jj]
        rem = rem // lead_dims[jj]
        base += idx * lead_strides[jj]
    tl.store(dst_ptr + i, tl.load(src_ptr + base, mask=mask), mask=mask)


@triton.jit
def _two_load_copy_kernel(
    src_ptr,
    dst_ptr,
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
    n,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Real-granularity fallback: two 1-D masked loads + two 1-D masked stores.

    Used when the pair cannot be reinterpreted as one aligned word (odd lead
    stride, odd storage offset, or fp64).  The index decomposition is identical
    to the gather kernel but in real elements and one lane at a time.
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    lead_dims = [d0, d1, d2, d3, d4]
    lead_strides = [s0, s1, s2, s3, s4]
    rem = i
    base = tl.zeros([BLOCK], dtype=tl.int64)
    for jj in tl.static_range(RANK - 1, -1, -1):
        idx = rem % lead_dims[jj]
        rem = rem // lead_dims[jj]
        base += idx * lead_strides[jj]
    re = tl.load(src_ptr + base, mask=mask)
    im = tl.load(src_ptr + base + 1, mask=mask)
    tl.store(dst_ptr + i * 2, re, mask=mask)
    tl.store(dst_ptr + i * 2 + 1, im, mask=mask)


def _flat_storage(t: torch.Tensor) -> torch.Tensor:
    """1-D real view over the whole underlying storage (metadata only)."""
    n = t.untyped_storage().nbytes() // t.element_size()
    return torch.as_strided(t, (n,), (1,), 0)


def _pair_words(t: torch.Tensor) -> torch.Tensor:
    """Pair-word view of ``t`` starting at its own storage offset.

    Requires an even storage offset and an even storage size in real elements
    (both checked by :func:`_pairword_eligible`).
    """
    words = _flat_storage(t).view(_WIDE_INT[t.dtype])
    return words[t.storage_offset() // 2 :]


def _pairword_eligible(t: torch.Tensor) -> bool:
    if t.dtype not in _WIDE_INT:
        return False
    if t.storage_offset() % 2 != 0:
        return False
    if not all(s % 2 == 0 for s in t.stride()[:-1]):
        return False
    n_real = t.untyped_storage().nbytes() // t.element_size()
    return n_real % 2 == 0


def _lead_slots(t: torch.Tensor, rank: int, strides, div):
    dims = list(t.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
    strs = [s // div for s in strides] + [0] * (_MAX_LEAD_RANK - rank)
    return dims, strs


def view_as_complex_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a real tensor whose last dimension has size 2 into a complex tensor.

    Mirrors ``aten::view_as_complex_copy``: the out-of-place counterpart of
    ``torch.view_as_complex``. The input's last dimension must have size 2
    and stride 1 (batch dimensions may be non-contiguous); ``float32`` maps
    to ``complex64``, ``float64`` to ``complex128`` and ``float16`` to
    ``complex32``. The result is a contiguous, freshly allocated tensor that
    never aliases the input. The same errors PyTorch raises are raised: 0-dim
    inputs, a wrong last-dimension size or stride, and unsupported dtypes
    (bfloat16 and every non-float type).
    """
    logger.debug("GEMS_KUNLUNXIN VIEW_AS_COMPLEX_COPY")

    if self.dim() == 0:
        raise RuntimeError("Input tensor must have one or more dimensions")
    if self.size(-1) != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    complex_dtype = _COMPLEX_DTYPE.get(self.dtype)
    if complex_dtype is None:
        raise RuntimeError(
            "view_as_complex is only supported for half, float and double "
            "tensors, but got a tensor of scalar type: "
            f"{str(self.dtype).replace('torch.', '').capitalize()}"
        )
    if self.stride(-1) != 1:
        raise RuntimeError("Tensor must have a last dimension with stride 1")

    # ``view_as_real`` refuses tensors carrying a lazy conjugate or negative
    # bit; resolving them first is an exact identity for the reinterpreted
    # values and keeps the copy on the logical data.
    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    leading = tuple(self.shape[:-1])
    out = torch.empty(leading, dtype=complex_dtype, device=self.device)
    n_complex = out.numel()
    if n_complex == 0:
        return out

    out_real = torch.view_as_real(out)
    with torch_device_fn.device(self.device):
        if self.is_contiguous():
            n_real = 2 * n_complex
            _flat_real_copy_kernel[(triton.cdiv(n_real, _FLAT_BLOCK),)](
                self,
                out_real,
                n_real,
                BLOCK=_FLAT_BLOCK,
            )
            return out

        rank = self.dim() - 1
        if rank > _MAX_LEAD_RANK:
            raise RuntimeError(
                f"view_as_complex_copy supports at most {_MAX_LEAD_RANK + 1} "
                f"dimensions, got {self.dim()}"
            )

        if _pairword_eligible(self):
            src_words = _pair_words(self)
            dst_words = out_real.reshape(-1).view(_WIDE_INT[self.dtype])
            if self.stride(-2) == 2 and self.shape[-2] <= _ROW_BLOCK_MAX:
                # Contiguous row: one program copies the whole innermost run.
                d_inner = self.shape[-2]
                rows = n_complex // d_inner
                orank = rank - 1
                pad = _MAX_LEAD_RANK - orank
                outer_dims = list(self.shape[:-2]) + [1] * pad
                outer_strides = [s // 2 for s in self.stride()[:-2]] + [0] * pad
                block = triton.next_power_of_2(max(d_inner, 1))
                _row_pairword_copy_kernel[(rows,)](
                    src_words,
                    dst_words,
                    d_inner,
                    *outer_dims,
                    *outer_strides,
                    OR=orank,
                    BLOCK=block,
                )
            else:
                dims, strides = _lead_slots(self, rank, self.stride()[:-1], 2)
                _gather_pairword_copy_kernel[(triton.cdiv(n_complex, _WORD_BLOCK),)](
                    src_words,
                    dst_words,
                    *dims,
                    *strides,
                    n_complex,
                    RANK=rank,
                    BLOCK=_WORD_BLOCK,
                )
        else:
            dims, strides = _lead_slots(self, rank, self.stride()[:-1], 1)
            _two_load_copy_kernel[(triton.cdiv(n_complex, _TWO_LOAD_BLOCK),)](
                self,
                out_real,
                *dims,
                *strides,
                n_complex,
                RANK=rank,
                BLOCK=_TWO_LOAD_BLOCK,
            )
    return out
