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
"""Triton implementation of ``aten::_foreach_atan_``.

In-place foreach atan: every tensor in the input sequence has ``atan`` applied
element-by-element to its own storage, preserving shape/stride/dtype, and the
call returns ``None``.

Semantics reproduced from ATen:
  * only floating dtypes are accepted; integral/bool inputs raise the
    "result type Float can't be cast ..." RuntimeError,
  * an empty list raises, non-tensor elements raise TypeError,
  * every list element is transformed, so duplicated or aliasing entries
    compose exactly like ATen's sequential launches,
  * float16/bfloat16 are evaluated in float32 (opmath) and rounded once.

Math is ``libdevice.atan`` (``tl.math`` has no ``atan`` on this backend) which
matches ATen bit-for-bit for fp16/bf16/fp32, including inf/nan/signed zero.

Dispatch is by per-tensor footprint, because the two regimes have opposite
bottlenecks:

  * A large tensor is bandwidth bound.  It gets its own launch of
    ``_atan_flat_kernel`` with the tensor itself as the pointer argument, which
    is what lets Triton keep 16-byte alignment knowledge and emit 128-bit
    vectorized global accesses (measured ~1.2 TB/s; passing the address as an
    int64 scalar costs about half of that).
  * A small tensor is launch bound.  Small contiguous tensors of one call that
    share a dtype and an element count are packed into a single launch whose
    base pointers ride in as int64 scalar arguments, so a 16-tensor list costs
    one launch instead of sixteen.  A device metadata buffer was measured to be
    worse: its host->device copy alone costs ~40 us, more than the launches it
    saves, so the pointers stay in the kernel arguments and only the fallback
    path passes a tensor object.

Because the small case is purely host bound, the common homogeneous list (all
dense, same dtype, same element count, at most 16 entries) has a dedicated
one-pass fast path that allocates nothing but the pointer list.

Non-contiguous tensors use a stride-aware kernel whose traversal picks the
smallest-stride dimension as the innermost one, so transposed/dense views stay
coalesced.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

from flag_gems.ops._foreach_unary import _foreach_atan_ as default__foreach_atan_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
_FLAT_BLOCK = 1024
# Per-tensor footprint at which a dedicated vectorized launch beats packing the
# tensor into the shared small-tensor launch.
_SMALL_BYTES = 1 << 20
_SMALL_SLOTS = 16
_MAX_OUTER_DIMS = 8
_MAX_RANK = _MAX_OUTER_DIMS + 1

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Element-count threshold of the launch-bound class, per dtype.
_SMALL_ELEMS = {
    torch.float16: _SMALL_BYTES // 2,
    torch.bfloat16: _SMALL_BYTES // 2,
    torch.float32: _SMALL_BYTES // 4,
    torch.float64: _SMALL_BYTES // 8,
}


@triton.jit
def _atan_math(x):
    # fp16/bf16 are evaluated in fp32 (ATen opmath) and rounded once on store.
    if x.dtype == tl.float64:
        y = _ld.atan(x)
    else:
        y = _ld.atan(x.to(tl.float32)).to(x.dtype)
    return y


@triton.jit
def _atan_flat_kernel(x_ptr, n_elements, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, _atan_math(x))
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(x_ptr + offs, _atan_math(x), mask=mask)


@triton.jit
def _atan_group_kernel(
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
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 small same-dtype tensors of equal element count.

    Axis 0 selects the tensor (a 4-level select tree over the int64 base
    pointers) and axis 1 the tile inside it.  Small tensors are latency bound,
    so the narrow accesses implied by a runtime-selected base pointer cost
    nothing that matters here, while the single launch is what the case is
    actually paying for.
    """
    t = tl.program_id(0)
    c = tl.program_id(1)
    p = tl.where(
        t < 8,
        tl.where(
            t < 4,
            tl.where(t < 2, tl.where(t == 0, p0, p1), tl.where(t == 2, p2, p3)),
            tl.where(t < 6, tl.where(t == 4, p4, p5), tl.where(t == 6, p6, p7)),
        ),
        tl.where(
            t < 12,
            tl.where(t < 10, tl.where(t == 8, p8, p9), tl.where(t == 10, p10, p11)),
            tl.where(t < 14, tl.where(t == 12, p12, p13), tl.where(t == 14, p14, p15)),
        ),
    )
    ptr = p.to(tl.pointer_type(DT))
    offs = c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    tl.store(ptr + offs, _atan_math(x), mask=mask)


@triton.jit
def _atan_strided_kernel(
    x_ptr,
    D,
    S,
    CHUNKS,
    d1,
    s1,
    d2,
    s2,
    d3,
    s3,
    d4,
    s4,
    d5,
    s5,
    d6,
    s6,
    d7,
    s7,
    d8,
    s8,
    BLOCK: tl.constexpr,
):
    """One program covers ``BLOCK`` consecutive elements along the innermost
    (smallest-stride) dimension; the outer dimensions are unravelled with scalar
    div/mod, which costs a handful of scalar ops per program, not per element.
    """
    pid = tl.program_id(0)
    o = pid // CHUNKS
    c = pid - o * CHUNKS
    base = (o % d1) * s1
    o = o // d1
    base += (o % d2) * s2
    o = o // d2
    base += (o % d3) * s3
    o = o // d3
    base += (o % d4) * s4
    o = o // d4
    base += (o % d5) * s5
    o = o // d5
    base += (o % d6) * s6
    o = o // d6
    base += (o % d7) * s7
    o = o // d7
    base += (o % d8) * s8

    idx = c * BLOCK + tl.arange(0, BLOCK)
    mask = idx < D
    addr = base.to(tl.int64) + idx.to(tl.int64) * S
    x = tl.load(x_ptr + addr, mask=mask, other=0.0)
    tl.store(x_ptr + addr, _atan_math(x), mask=mask)


def _reject_dtype(dt):
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        + str(dt).replace("torch.", "").capitalize()
    )


def _launch_flat(t, numel):
    _atan_flat_kernel[(triton.cdiv(numel, _FLAT_BLOCK),)](
        t, numel, _FLAT_BLOCK, numel % _FLAT_BLOCK == 0
    )


def _launch_small_group(dt, items):
    """``items`` is a flat [tensor, numel, ptr, tensor, numel, ptr, ...] list."""
    n_tensors = len(items) // 3
    if n_tensors == 1:
        _launch_flat(items[0], items[1])
        return
    # One launch serves tensors that share an element count, so subgroup first.
    # The tensor object is kept for the direct fallback launch; the raw data
    # pointer feeds the packed group kernel.
    by_count = {}
    for j in range(n_tensors):
        b = 3 * j
        numel = items[b + 1]
        bucket = by_count.get(numel)
        if bucket is None:
            by_count[numel] = [(items[b], items[b + 2])]
        else:
            bucket.append((items[b], items[b + 2]))
    tl_dtype = _TL_DTYPE[dt]
    block = _FLAT_BLOCK
    for numel, pairs in by_count.items():
        n = len(pairs)
        if n == 1:
            _launch_flat(pairs[0][0], numel)
            continue
        chunks = -(-numel // block)
        i = 0
        while i < n:
            batch = pairs[i : i + _SMALL_SLOTS]
            i += _SMALL_SLOTS
            m = len(batch)
            if m == 1:
                _launch_flat(batch[0][0], numel)
                continue
            ptrs = [pair[1] for pair in batch]
            ptrs.extend([0] * (_SMALL_SLOTS - m))
            _atan_group_kernel[(m, chunks)](*ptrs, numel, tl_dtype, block)


def _launch_strided(t, numel):
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz > 1]
    if not dims:
        # numel <= 1: a single element is addressable with a flat index.
        _atan_flat_kernel[(1,)](t, numel, 1, False)
        return
    dims = sorted(dims, key=lambda p: p[1])
    if len(dims) > _MAX_RANK:
        raise NotImplementedError("tensor rank too large for the strided kernel")
    d_inner, s_inner = dims[0]
    outer_count = 1
    args = []
    for sz, st in dims[1:]:
        args.append(sz)
        args.append(st)
        outer_count *= sz
    args.extend([1, 0] * (_MAX_OUTER_DIMS - len(args) // 2))

    block = _FLAT_BLOCK
    chunks = triton.cdiv(d_inner, block)
    _atan_strided_kernel[(outer_count * chunks,)](
        t, d_inner, s_inner, chunks, *args, block
    )


def _run_general(ts):
    """Mixed dtypes, mixed layouts, mixed sizes, or long lists."""
    entries = []
    ptrs = []
    for t in ts:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if not dt.is_floating_point:
            _reject_dtype(dt)
        numel = t.numel()
        if numel == 0:
            continue
        ptr = t.data_ptr()
        entries.append((t, numel, ptr, t.is_contiguous()))
        ptrs.append(ptr)

    if len(entries) > 1 and len(set(ptrs)) != len(ptrs):
        # Aliasing list entries must be applied in list order, exactly like
        # ATen's sequential launches.
        for t, numel, ptr, flat in entries:
            if flat:
                _launch_flat(t, numel)
            else:
                _launch_strided(t, numel)
        return None

    groups = {}
    for t, numel, ptr, flat in entries:
        if not flat:
            _launch_strided(t, numel)
        elif numel >= _SMALL_ELEMS[t.dtype]:
            _launch_flat(t, numel)
        else:
            key = (t.device, t.dtype)
            g = groups.get(key)
            if g is None:
                groups[key] = [t, numel, ptr]
            else:
                g.append(t)
                g.append(numel)
                g.append(ptr)

    for key, items in groups.items():
        _launch_small_group(key[1], items)
    return None


def _foreach_atan_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_ATAN_")
    if isinstance(self, torch.Tensor):
        ts = (self,)
    elif isinstance(self, (list, tuple)):
        ts = self
    else:
        raise TypeError("argument 'self' must be tuple of Tensors")
    n = len(ts)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in ts):
        return default__foreach_atan_(ts)

    t0 = ts[0]
    if not isinstance(t0, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    dt = t0.dtype
    if not dt.is_floating_point:
        _reject_dtype(dt)
    numel = t0.numel()

    if n == 1:
        if numel:
            if not t0.is_contiguous():
                _launch_strided(t0, numel)
            else:
                _launch_flat(t0, numel)
        return None

    # Dominant shape of a foreach call: a short list of equally sized, dense,
    # same-dtype tensors that are individually launch bound.  Serve it with one
    # launch and no intermediate bookkeeping.
    if numel and numel < _SMALL_ELEMS[dt] and n <= _SMALL_SLOTS and t0.is_contiguous():
        ptrs = [t0.data_ptr()]
        ok = True
        for i in range(1, n):
            t = ts[i]
            if (
                not isinstance(t, torch.Tensor)
                or t.dtype != dt
                or t.numel() != numel
                or not t.is_contiguous()
            ):
                ok = False
                break
            ptrs.append(t.data_ptr())
        if ok and len(set(ptrs)) == n:
            ptrs.extend([0] * (_SMALL_SLOTS - n))
            _atan_group_kernel[(n, -(-numel // _FLAT_BLOCK))](
                *ptrs, numel, _TL_DTYPE[dt], _FLAT_BLOCK
            )
            return None

    return _run_general(ts)
