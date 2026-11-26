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
"""Triton implementation of ``aten::_foreach_log2_`` (in-place, multi-tensor).

Semantics (mirrors ATen / the Gems reference):

* Every tensor of the list is transformed in place: ``T[i] = log2(T[i])``.
  Storage, strides, dtype, shape and the Python list object are untouched and
  the call returns ``None``.
* Floating dtypes only.  ``fp16``/``bf16`` are up-cast to ``fp32`` for the
  math and rounded back to the storage dtype (ATen's ``opmath_type``);
  ``fp32``/``fp64`` are computed natively.
* Integral / bool elements cannot be written back as float, so the call is
  rejected exactly like ATen (``result type Float can't be cast ...``).
* An empty list is rejected with ATen's "Tensor list must have at least one
  tensor." message; ``numel == 0`` elements are skipped (no launch).

Dispatch (measured on this target, rounds 1-5):

* Every workload is either launch-bound or bandwidth-bound, and the two want
  different launch shapes.  A group of equal-sized, 16-byte-aligned,
  flat-addressable tensors with ``numel <= _FUSE_MAX`` is processed by ONE
  launch (up to 16 tensors passed as individual pointer arguments, program
  ``pid`` belongs to tensor ``pid // bp``), which is 3-4x faster than one launch
  per tensor for 4096-element lists.  Larger tensors keep one masked 1-D launch
  per tensor, because a single launch spanning a whole group streams at about
  half the throughput of per-tensor launches on this backend.
* Per-call host cost is the binding constraint on short kernels, so the first
  call per specialization compiles and caches the CompiledKernel and later
  calls invoke its launcher directly instead of re-running ``JITFunction.run``
  binding, specialization, cache-key computation and MUSA alias analysis.
* Non-contiguous views always take a per-tensor shape/stride decomposition
  kernel; ``contiguous()`` is never called, so the in-place mutation contract
  and the original storage/strides are preserved.
* Duplicate list entries are launched sequentially so aliasing keeps ATen's
  apply-once-per-list-entry semantics.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_unary import _foreach_log2_ as default__foreach_log2_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _needs_fallback(seq):
    """True when the list cannot use the MUSA kernels (device or dtype)."""
    for t in seq:
        if not isinstance(t, torch.Tensor):
            return True
        if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
            return True
    return False


try:  # pre-bound launch path, see _launch_cached
    from triton import knobs as _knobs

    _FAST_LAUNCH = True
except Exception:  # pragma: no cover - any import drift falls back to the JIT call
    _FAST_LAUNCH = False

_BLOCK = 1024
_MAX_RANK = 8
# Per-tensor element count below which a single fused launch beats per-tensor
# launches.  At 1M elements one launch still costs more than the streaming it
# hides; at 16M the single-group launch loses ~2x throughput instead.
_FUSE_MAX = 1 << 20
# Pointer slots in _log2_multi_flat_kernel.
_SLOTS = 16

_ATEN_DTYPE_NAME = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
}


@triton.jit
def _log2_body(ptr, offs, mask, USE_F64: tl.constexpr):
    """In-place log2 of one tile (fp32 opmath for fp16/bf16)."""
    x = tl.load(ptr + offs, mask=mask)
    if USE_F64:
        y = tl.log2(x)
    else:
        y = tl.log2(x.to(tl.float32)).to(x.dtype)
    tl.store(ptr + offs, y, mask=mask)


@triton.jit
def _log2_flat_kernel(ptr, n_elements, USE_F64: tl.constexpr, BLOCK: tl.constexpr):
    """Contiguous (flat-addressable) in-place log2 for one tensor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    _log2_body(ptr, offs, mask, USE_F64)


@triton.jit
def _log2_multi_flat_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n_elements,
    bp,
    USE_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 equal-sized flat-addressable tensors.

    The pointers travel as kernel arguments -- no metadata table and no host to
    device copy -- so the addressing is identical to the single-tensor kernel.
    The grid is 1-D with ``bp = cdiv(numel, BLOCK)`` chunks per tensor, so
    program ``pid`` belongs to tensor ``pid // bp`` and the dispatch order
    matches per-tensor launches.  Unused slots repeat an existing pointer and
    are never selected because the grid only covers ``nt * bp`` programs.
    """
    pid = tl.program_id(0)
    t = pid // bp
    b = pid - t * bp
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    if t == 0:
        _log2_body(p0, offs, mask, USE_F64)
    if t == 1:
        _log2_body(p1, offs, mask, USE_F64)
    if t == 2:
        _log2_body(p2, offs, mask, USE_F64)
    if t == 3:
        _log2_body(p3, offs, mask, USE_F64)
    if t == 4:
        _log2_body(p4, offs, mask, USE_F64)
    if t == 5:
        _log2_body(p5, offs, mask, USE_F64)
    if t == 6:
        _log2_body(p6, offs, mask, USE_F64)
    if t == 7:
        _log2_body(p7, offs, mask, USE_F64)
    if t == 8:
        _log2_body(p8, offs, mask, USE_F64)
    if t == 9:
        _log2_body(p9, offs, mask, USE_F64)
    if t == 10:
        _log2_body(p10, offs, mask, USE_F64)
    if t == 11:
        _log2_body(p11, offs, mask, USE_F64)
    if t == 12:
        _log2_body(p12, offs, mask, USE_F64)
    if t == 13:
        _log2_body(p13, offs, mask, USE_F64)
    if t == 14:
        _log2_body(p14, offs, mask, USE_F64)
    if t == 15:
        _log2_body(p15, offs, mask, USE_F64)


@triton.jit
def _log2_strided_kernel(
    ptr,
    n_elements,
    RANK: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    S6: tl.constexpr,
    S7: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    T4: tl.constexpr,
    T5: tl.constexpr,
    T6: tl.constexpr,
    T7: tl.constexpr,
    USE_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generic strided in-place log2: flat logical index -> memory offset.

    The offsets are computed from the tensor's own shape/strides so the write
    lands on exactly the element the read came from; no staging buffer and no
    ``contiguous()`` copy is involved.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    rem = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        q = rem // S0
        off += (rem - q * S0) * T0
        rem = q
    if RANK > 1:
        q = rem // S1
        off += (rem - q * S1) * T1
        rem = q
    if RANK > 2:
        q = rem // S2
        off += (rem - q * S2) * T2
        rem = q
    if RANK > 3:
        q = rem // S3
        off += (rem - q * S3) * T3
        rem = q
    if RANK > 4:
        q = rem // S4
        off += (rem - q * S4) * T4
        rem = q
    if RANK > 5:
        q = rem // S5
        off += (rem - q * S5) * T5
        rem = q
    if RANK > 6:
        q = rem // S6
        off += (rem - q * S6) * T6
        rem = q
    if RANK > 7:
        q = rem // S7
        off += (rem - q * S7) * T7
        rem = q
    x = tl.load(ptr + off, mask=mask)
    if USE_F64:
        y = tl.log2(x)
    else:
        y = tl.log2(x.to(tl.float32)).to(x.dtype)
    tl.store(ptr + off, y, mask=mask)


_KCACHE = {}


def _launch_cached(jit_fn, key, args, nb):
    """Launch ``jit_fn`` through a cached CompiledKernel.

    ``JITFunction.run`` re-does argument binding, specialization refinement,
    cache-key computation, MUSA pointer-alias analysis and grid
    canonicalisation on every call; on short kernels that host path dominates
    (a 4096-element call measured ~0.041 ms against ATen's ~0.018 ms).  The
    compiled kernel is keyed by every property Triton specializes on -- dtype,
    device, and the ``==1`` / divisibility-16 facts of each runtime scalar and
    pointer -- so a cached entry is only reused for an identical
    specialization.  Returns False when the path is unusable, in which case the
    caller falls back to the ordinary ``kernel[grid](...)`` call.
    """
    k = _KCACHE.get(key)
    if k is None:
        try:
            k = jit_fn.warmup(*args, grid=(1,), num_warps=4)
            if k is not None:
                k._init_handles()
        except Exception:
            k = False
        _KCACHE[key] = k
    if not k:
        return False
    stream = _driver.active.get_current_stream(_driver.active.get_current_device())
    k.run(
        nb,
        1,
        1,
        stream,
        k.function,
        k.packed_metadata,
        k.launch_metadata((nb, 1, 1), stream, *args),
        _knobs.runtime.launch_enter_hook,
        _knobs.runtime.launch_exit_hook,
        *args,
    )
    return True


def _launch_flat(t, use_f64):
    """In-place log2 of one contiguous tensor."""
    n = t.numel()
    nb = (n + _BLOCK - 1) // _BLOCK
    if _FAST_LAUNCH:
        key = (
            _log2_flat_kernel,
            t.device,
            t.dtype,
            n == 1,
            (n & 15) == 0,
            (t.data_ptr() & 15) == 0,
        )
        if _launch_cached(_log2_flat_kernel, key, (t, n, use_f64, _BLOCK), nb):
            return
    _log2_flat_kernel[(nb,)](t, n, use_f64, _BLOCK, num_warps=4)


def _launch_multi(chunk, n0, bp, dtype, use_f64):
    """One launch for ``len(chunk)`` equal-sized contiguous tensors."""
    nt = len(chunk)
    pad = chunk[0]
    args = chunk + [pad] * (_SLOTS - nt) + [n0, bp, use_f64, _BLOCK]
    nb = bp * nt
    if _FAST_LAUNCH:
        key = (
            _log2_multi_flat_kernel,
            pad.device,
            dtype,
            n0 == 1,
            (n0 & 15) == 0,
            bp == 1,
            (bp & 15) == 0,
        )
        if _launch_cached(_log2_multi_flat_kernel, key, args, nb):
            return
    _log2_multi_flat_kernel[(nb,)](*args, num_warps=4)


def _launch_flat_group(tensors, dtype, use_f64):
    nt = len(tensors)
    if nt == 1:
        _launch_flat(tensors[0], use_f64)
        return
    n0 = tensors[0].numel()
    if n0 > _FUSE_MAX:
        # Bandwidth-bound: one launch per tensor streams ~2x faster here.
        for t in tensors:
            _launch_flat(t, use_f64)
        return
    for t in tensors:
        if t.numel() != n0 or (t.data_ptr() & 15):
            # Unequal sizes cannot share one chunk-per-tensor grid, and the
            # fused kernel makes no alignment claim of its own.
            for t in tensors:
                _launch_flat(t, use_f64)
            return
    bp = (n0 + _BLOCK - 1) // _BLOCK
    if nt <= _SLOTS:
        _launch_multi(tensors, n0, bp, dtype, use_f64)
        return
    for start in range(0, nt, _SLOTS):
        chunk = tensors[start : start + _SLOTS]
        if len(chunk) == 1:
            _launch_flat(chunk[0], use_f64)
        else:
            _launch_multi(chunk, n0, bp, dtype, use_f64)


def _launch_strided(t, use_f64):
    n = t.numel()
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    rank = len(dims)
    if rank == 0:
        _launch_flat(t, use_f64)
        return
    if rank > _MAX_RANK:
        raise NotImplementedError(
            "_foreach_log2_: tensor rank above 8 with non-contiguous layout"
        )
    sizes = [d[0] for d in dims] + [1] * (_MAX_RANK - rank)
    strides = [d[1] for d in dims] + [1] * (_MAX_RANK - rank)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _log2_strided_kernel[grid](
        t,
        n,
        rank,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        use_f64,
        _BLOCK,
        num_warps=4,
    )


def _foreach_log2_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_LOG2_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_log2_(self)
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
    # Validate every element before touching any of them.
    for t in self:
        dt = t.dtype
        if not dt.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + _ATEN_DTYPE_NAME.get(dt, str(dt).replace("torch.", "").capitalize())
            )

    groups = {}
    strided = []
    seen = set()
    aliased = False
    for t in self:
        n = t.numel()
        if n == 0:
            continue
        p = t.data_ptr()
        if p in seen:
            aliased = True
        else:
            seen.add(p)
        if n == 1 or t.is_contiguous():
            dt = t.dtype
            bucket = groups.get(dt)
            if bucket is None:
                groups[dt] = [t]
            else:
                bucket.append(t)
        else:
            strided.append(t)

    if aliased:
        # The same storage appears twice: ATen applies the op once per list
        # entry, so the writes must be ordered rather than fused.
        for t in self:
            if t.numel() == 0:
                continue
            use_f64 = t.dtype == torch.float64
            if t.numel() == 1 or t.is_contiguous():
                _launch_flat(t, use_f64)
            else:
                _launch_strided(t, use_f64)
        return None

    for t in strided:
        _launch_strided(t, t.dtype == torch.float64)
    for dt, bucket in groups.items():
        _launch_flat_group(bucket, dt, dt == torch.float64)
    return None
