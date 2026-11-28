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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 (Libra-class): 48 stream processors. A flat copy with one program per
# BLOCK pays a ~54ns/program dispatch tax (1024 programs @ 1M elements cost
# ~55us for ~4MB of traffic), so the grid is capped at one full wave and each
# program walks blocks with a grid-stride loop instead.
_GCU_NUM_SPC = 48
# Widest BLOCK: 64B/lane at num_warps=4 (256 lanes) for 1-byte dtypes and
# 64-128B/lane for wider ones, measured on GCU400.
_GCU_BLOCK_1B = 16384
_GCU_BLOCK_WIDE = 8192
# Below this element count the whole tensor fits in one narrow block, and a
# 8192-wide BLOCK would leave most lanes masked off. Measured on GCU400 with
# n=1024: BLOCK=1024 gives 3.07us vs 3.62us for the wide default (the masked
# lanes and their offset/compare arithmetic are pure overhead there).
# 1-byte dtypes are excluded: their load/store does not vectorize as well, and
# BLOCK=1024 measured *slower* there (4.89us vs 3.64us at n=1024, 6.86us vs
# 4.04us at n=65536), so they keep the wide block.
_GCU_SNUG_LIMIT = 2048
# Largest n that keeps tl.cdiv's internal (n + BLOCK - 1) inside int32 for
# every BLOCK in use (max 16384 = 2**14), so the fast kernel can rely on the
# i32 specialization of n_elements without an overflow-truncated block count.
_FLAT_I32_MAX = 2**31 - 2**15


def _flat_block_size(n: int, itemsize: int) -> int:
    if itemsize == 1:
        return _GCU_BLOCK_1B
    if n <= _GCU_SNUG_LIMIT:
        # Use a BLOCK sized to fit the data for small tensors, avoiding all lanes being masked out under a wide BLOCK.
        return max(256, triton.next_power_of_2(n))
    return _GCU_BLOCK_WIDE


def _interleaved_words(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a complex64 tensor's storage as interleaved fp32 words.

    The copy kernels read raw storage, so a lazy conjugate/negative bit has to
    be materialized before the copy. ``torch.view_as_real`` rejects an
    unresolved conjugated tensor, and ``resolve_conj`` cannot clear the bit on
    this backend (its registration hands back a raw clone that keeps it), so
    the bit is dropped by rebinding the storage instead -- pure metadata, the
    same trick as ``conj_physical._words_view`` -- and folded into the kernels
    as lane-negation flags. Word offsets are doubled complex-element offsets;
    every element stores its (real, imag) pair consecutively, so the copy
    output's parity identifies the lane.
    """
    words = torch.empty(0, dtype=torch.float32, device=t.device)
    words.set_(
        t.untyped_storage(),
        t.storage_offset() * 2,
        tuple(t.shape) + (2,),
        tuple(s * 2 for s in t.stride()) + (1,),
    )
    return words


@triton.jit
def _apply_lane_negation(
    value, offsets, NEGATE_REAL: tl.constexpr, NEGATE_IMAG: tl.constexpr
):
    # Interleaved (real, imag) word stream: even offsets carry the real part.
    # IEEE negation of an fp32 word is an exact sign-bit flip, so both the
    # raw-copy and the lazy-bit cases match aten bit-exactly.
    if NEGATE_REAL and NEGATE_IMAG:
        value = -value
    elif NEGATE_REAL:
        value = tl.where((offsets & 1) == 0, -value, value)
    elif NEGATE_IMAG:
        value = tl.where((offsets & 1) == 1, -value, value)
    return value


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _detach_copy_flat(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPC: tl.constexpr,
    NEGATE_REAL: tl.constexpr,
    NEGATE_IMAG: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in tl.range(pid, num_blocks, NUM_SPC):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, cache_modifier=".cg")
        val = _apply_lane_negation(val, offsets, NEGATE_REAL, NEGATE_IMAG)
        tl.store(out_ptr + offsets, val, mask=mask, cache_modifier=".cg")


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _detach_copy_flat_i64(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPC: tl.constexpr,
    NEGATE_REAL: tl.constexpr,
    NEGATE_IMAG: tl.constexpr,
):
    # Same walk, but the block count is derived in i64: n_elements may sit
    # within int32 range yet close enough to INT32_MAX that cdiv's
    # (n + BLOCK - 1) would overflow an i32 specialization.
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements.to(tl.int64), BLOCK_SIZE)
    for block_id in tl.range(pid, num_blocks, NUM_SPC):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, cache_modifier=".cg")
        val = _apply_lane_negation(val, offsets, NEGATE_REAL, NEGATE_IMAG)
        tl.store(out_ptr + offsets, val, mask=mask, cache_modifier=".cg")


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _detach_copy_strided(
    x_ptr,
    out_ptr,
    n_elements,
    meta_ptr,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEGATE_REAL: tl.constexpr,
    NEGATE_IMAG: tl.constexpr,
):
    # meta layout: for dim i in [0, ndim): meta[2*i] = shape[i], meta[2*i+1] = stride[i]
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
    val = tl.load(x_ptr + in_off, mask=mask)
    # Output offsets are linear, so their parity still identifies the lane.
    val = _apply_lane_negation(val, offsets, NEGATE_REAL, NEGATE_IMAG)
    tl.store(out_ptr + offsets, val, mask=mask)


def detach_copy(self: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``self`` detached from the autograd graph.

    Mirrors ``aten::detach_copy(Tensor self) -> Tensor``: the values are
    materialized into freshly allocated memory (equivalent to
    ``self.detach().clone()``), so the result never aliases the input.

    Contiguous inputs take a flat 1-D copy kernel; non-contiguous inputs take
    a strided kernel driven by a device-side shape/stride metadata buffer
    (``meta[2*i] = shape[i]``, ``meta[2*i+1] = stride[i]``). Either way the
    output is freshly allocated, dense and contiguous, matching the native
    operator, which also returns a contiguous tensor for a non-contiguous
    input.

    Complex inputs stay on the device path: complex64 is reinterpreted as its
    interleaved (real, imag) fp32 words, copied by the same kernels, and
    reassembled with ``torch.view_as_complex`` (Triton has no complex dtype
    mapping, so the kernels themselves stay real-valued). Lazy
    conjugate/negative bits fold into the kernels as lane-negation flags; the
    other complex dtypes go through ``torch.view_as_real``.
    """
    logger.debug("GEMS_ENFLAME DETACH_COPY")
    # Lazy conjugate/negative bits must be materialized before the kernels read
    # raw storage (no-op views are returned untouched); the kernels would
    # otherwise copy ``torch._neg_view(x)`` bitwise and return ``x`` instead of
    # ``-x``. Inlined rather than kept in a helper: at this launch scale an
    # extra Python frame is a measurable part of the total.
    is_complex = self.dtype.is_complex
    negate_real = False
    negate_imag = False
    if is_complex:
        is_neg_bit = bool(getattr(self, "is_neg", lambda: False)())
        conj_bit = bool(self.is_conj())
        if self.dtype == torch.complex64:
            # For logical value conj_A(neg_N(storage)) the real word is negated
            # iff N and the imaginary word iff N xor A.
            negate_real = is_neg_bit
            negate_imag = is_neg_bit != conj_bit
            self = _interleaved_words(self)
        else:
            # Other complex dtypes keep the aten path (unsupported on the
            # fp64-less devices this backend targets).
            if conj_bit:
                self = self.resolve_conj()
            if is_neg_bit:
                self = self.resolve_neg()
            self = torch.view_as_real(self)
    else:
        is_neg = getattr(self, "is_neg", None)
        if is_neg is not None and is_neg():
            self = self.resolve_neg()
    n = self.numel()
    # Fresh contiguous storage, exactly like native detach_copy. ``empty_like``
    # would preserve a non-contiguous input's strides, which both the kernels'
    # linear stores and the native layout contract rule out.
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if n == 0:
        # No elements to read: the freshly allocated (dense) output is already
        # the required result, and a grid of (0,) must not be launched.
        return torch.view_as_complex(out) if is_complex else out
    if self.is_contiguous():
        # Memory-op latency is amortized once each lane covers >=64B, i.e.
        # BLOCK 8192 for itemsize >= 2 and 16384 for 1-byte dtypes; small
        # tensors instead take a block snug to their size so the wide block's
        # masked-off lanes stop costing offset arithmetic. The grid is one full
        # wave at most; larger tensors loop grid-stride.
        block = _flat_block_size(n, self.dtype.itemsize)
        grid = (min((n + block - 1) // block, _GCU_NUM_SPC),)
        kernel = _detach_copy_flat if n <= _FLAT_I32_MAX else _detach_copy_flat_i64
        with torch_device_fn.device(self.device):
            kernel[grid](
                self,
                out,
                n,
                BLOCK_SIZE=block,
                NUM_SPC=_GCU_NUM_SPC,
                NEGATE_REAL=negate_real,
                NEGATE_IMAG=negate_imag,
                num_warps=4,
            )
    else:
        shape = self.shape
        stride = self.stride()
        nd = len(shape)
        # One host->device transfer for the shape/stride metadata; filling a
        # device tensor with a scalar ``meta[i] = v`` loop would launch
        # 2 * ndim separate device operations before the copy kernel.
        meta = torch.tensor(
            [v for s, st in zip(shape, stride) for v in (s, st)],
            dtype=torch.int64,
            device=self.device,
        )
        # Tuning measured on H20 (see the task report): the strided kernel's
        # per-element div/mod address computation prefers narrower blocks than
        # the flat path - BLOCK=256 beats the shared 1024 default by 10-58%
        # across the shapes swept, with num_warps=4 at the knee.
        BLOCK = 256
        grid = (triton.cdiv(n, BLOCK),)
        with torch_device_fn.device(self.device):
            _detach_copy_strided[grid](
                self,
                out,
                n,
                meta,
                ndim=nd,
                BLOCK_SIZE=BLOCK,
                NEGATE_REAL=negate_real,
                NEGATE_IMAG=negate_imag,
                num_warps=4,
            )
    return torch.view_as_complex(out) if is_complex else out
