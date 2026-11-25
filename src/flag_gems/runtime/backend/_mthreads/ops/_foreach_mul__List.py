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
"""Triton implementation of aten::_foreach_mul_ (list overload).

For every pair (self[i], other[i]) the kernel performs an in-place element-wise
multiply that writes back into the storage of ``self[i]`` (no allocation and no
data movement of the mutated tensor).  ``other`` may be

* a sequence of tensors (any broadcastable shape, contiguous or not),
* a sequence of python numbers,
* a single python number applied to every element,
* a single tensor broadcast against every element.

The call returns ``None`` (the operator only mutates ``self``).

Products are formed in fp32 and rounded once onto the storage dtype, which is
exactly PyTorch's element-wise opmath for half/bfloat16 (the exact product of
two 16-bit values is representable in fp32, so a single rounding happens either
way) and a no-op for fp32.

Two launch shapes are used:

* a *uniform fused* kernel, when every list entry has the same dtype/shape and
  every operand is contiguous and small.  All entries are statically unrolled
  into one launch, which removes the per-entry launch overhead that dominates
  short tensor lists.  That launch reuses the already-compiled kernel handle
  (``kernel.warmup`` + the backend launcher) so per-call JIT argument binding is
  skipped; the handle is keyed by dtype, constexpr configuration, pointer
  alignment and size divisibility, i.e. exactly the specializations Triton bakes
  into the binary, so a different specialization falls back to the JIT path.
* otherwise one launch per pair, with contiguous, scalar and general
  strided/broadcast variants.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _triton_driver

from flag_gems.ops._foreach_binary import (
    _foreach_mul__List as default__foreach_mul__List,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_MAX_RANK = 8
_MAX_FUSE = 8
_CONTIG_BLOCK = 4096
_STRIDED_BLOCK = 1024
_INT32_MAX = 2147483647
# Fuse when one tensor of the list is small enough that per-launch cost, rather
# than steady-state bandwidth, still shows up in the call latency (<= 96 MB per
# tensor).  Larger lists keep one launch per pair.
_FUSE_BYTES_LIMIT = 96 << 20

# MODE: 0 -> other is a full tensor element
#       1 -> other is a python scalar
#       2 -> other is a 0-dim / 1-element tensor (one value broadcast)
MODE_TENSOR = 0
MODE_SCALAR = 1
MODE_SCALAR_PTR = 2


@triton.jit
def _mulc(a, b, OUT_TY: tl.constexpr):
    """Element-wise product computed in fp32, rounded once to ``OUT_TY``."""
    return (a.to(tl.float32) * b.to(tl.float32)).to(OUT_TY)


@triton.jit
def _k_contig(
    S,
    out_v,
    s,
    n,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    I64: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if EVEN:
        a = tl.load(S + offs)
        if MODE == 0:
            b = tl.load(out_v + offs)
        elif MODE == 2:
            b = tl.load(out_v)
        else:
            b = s
        tl.store(S + offs, _mulc(a, b, S.dtype.element_ty))
    else:
        a = tl.load(S + offs, mask=mask)
        if MODE == 0:
            b = tl.load(out_v + offs, mask=mask)
        elif MODE == 2:
            b = tl.load(out_v)
        else:
            b = s
        tl.store(S + offs, _mulc(a, b, S.dtype.element_ty), mask=mask)


@triton.jit
def _fused_step(S, out_v, offs, mask, EVEN: tl.constexpr):
    if EVEN:
        a = tl.load(S + offs)
        b = tl.load(out_v + offs)
        tl.store(S + offs, _mulc(a, b, S.dtype.element_ty))
    else:
        a = tl.load(S + offs, mask=mask)
        b = tl.load(out_v + offs, mask=mask)
        tl.store(S + offs, _mulc(a, b, S.dtype.element_ty), mask=mask)


@triton.jit
def _fused_step_s(S, s, offs, mask, EVEN: tl.constexpr):
    if EVEN:
        a = tl.load(S + offs)
        tl.store(S + offs, _mulc(a, s, S.dtype.element_ty))
    else:
        a = tl.load(S + offs, mask=mask)
        tl.store(S + offs, _mulc(a, s, S.dtype.element_ty), mask=mask)


@triton.jit
def _k_fused_t(
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    O0,
    O1,
    O2,
    O3,
    O4,
    O5,
    O6,
    O7,
    n,
    BLOCK: tl.constexpr,
    NT: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if NT > 0:
        _fused_step(S0, O0, offs, mask, EVEN)
    if NT > 1:
        _fused_step(S1, O1, offs, mask, EVEN)
    if NT > 2:
        _fused_step(S2, O2, offs, mask, EVEN)
    if NT > 3:
        _fused_step(S3, O3, offs, mask, EVEN)
    if NT > 4:
        _fused_step(S4, O4, offs, mask, EVEN)
    if NT > 5:
        _fused_step(S5, O5, offs, mask, EVEN)
    if NT > 6:
        _fused_step(S6, O6, offs, mask, EVEN)
    if NT > 7:
        _fused_step(S7, O7, offs, mask, EVEN)


@triton.jit
def _k_fused_s(
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    s,
    n,
    BLOCK: tl.constexpr,
    NT: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if NT > 0:
        _fused_step_s(S0, s, offs, mask, EVEN)
    if NT > 1:
        _fused_step_s(S1, s, offs, mask, EVEN)
    if NT > 2:
        _fused_step_s(S2, s, offs, mask, EVEN)
    if NT > 3:
        _fused_step_s(S3, s, offs, mask, EVEN)
    if NT > 4:
        _fused_step_s(S4, s, offs, mask, EVEN)
    if NT > 5:
        _fused_step_s(S5, s, offs, mask, EVEN)
    if NT > 6:
        _fused_step_s(S6, s, offs, mask, EVEN)
    if NT > 7:
        _fused_step_s(S7, s, offs, mask, EVEN)


@triton.jit
def _k_strided(
    S,
    out_v,
    s,
    n,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    s_off = tl.zeros((BLOCK,), dtype=tl.int64)
    o_off = tl.zeros((BLOCK,), dtype=tl.int64)
    if RANK >= 1:
        i0 = rem % n0
        rem = rem // n0
        s_off += i0 * a0
        o_off += i0 * b0
    if RANK >= 2:
        i1 = rem % n1
        rem = rem // n1
        s_off += i1 * a1
        o_off += i1 * b1
    if RANK >= 3:
        i2 = rem % n2
        rem = rem // n2
        s_off += i2 * a2
        o_off += i2 * b2
    if RANK >= 4:
        i3 = rem % n3
        rem = rem // n3
        s_off += i3 * a3
        o_off += i3 * b3
    if RANK >= 5:
        i4 = rem % n4
        rem = rem // n4
        s_off += i4 * a4
        o_off += i4 * b4
    if RANK >= 6:
        i5 = rem % n5
        rem = rem // n5
        s_off += i5 * a5
        o_off += i5 * b5
    if RANK >= 7:
        i6 = rem % n6
        rem = rem // n6
        s_off += i6 * a6
        o_off += i6 * b6
    if RANK >= 8:
        i7 = rem % n7
        rem = rem // n7
        s_off += i7 * a7
        o_off += i7 * b7
    a = tl.load(S + s_off, mask=mask)
    if MODE == 0:
        b = tl.load(out_v + o_off, mask=mask)
    elif MODE == 2:
        b = tl.load(out_v)
    else:
        b = s
    tl.store(S + s_off, _mulc(a, b, S.dtype.element_ty), mask=mask)


_FAST_CACHE = {}
_DRIVER = None
_HOOKS = None


def _driver():
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _triton_driver.active
    return _DRIVER


def _launch_hooks():
    global _HOOKS
    if _HOOKS is None:
        try:
            from triton import knobs

            _HOOKS = (
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
            )
        except Exception:
            _HOOKS = (None, None)
    return _HOOKS


def _fast_launch(kernel, key, grid, pos_args, rt_builder, kwargs):
    """Launch a cached compiled kernel, skipping per-call JIT argument binding.

    ``pos_args`` are the kernel arguments in declaration order (raw device
    pointers plus the constexpr values), which this backend's launcher forwards
    positionally.  ``rt_builder`` is only invoked while the kernel still has to
    be compiled.
    """
    entry = _FAST_CACHE.get(key)
    if entry is None:
        compiled = kernel.warmup(*rt_builder(), grid=(1,), **kwargs)
        entry = (compiled, compiled.run, compiled.function, compiled.packed_metadata)
        _FAST_CACHE[key] = entry
    drv = _driver()
    stream = drv.get_current_stream(drv.get_current_device())
    enter_hook, exit_hook = _launch_hooks()
    _, launcher, function, packed = entry
    launcher(
        grid,
        1,
        1,
        stream,
        function,
        packed,
        None,
        enter_hook,
        exit_hook,
        *pos_args,
    )


def _collapse(shape, sstride, ostride):
    """Merge dims into the smallest set of affine mixed-radix terms.

    Dims are ordered by increasing ``self`` stride so that consecutive flat
    indices hit consecutive addresses (coalescing) even for permuted views.
    """
    dims = sorted(
        ((sz, ss, os_) for sz, ss, os_ in zip(shape, sstride, ostride) if sz != 1),
        key=lambda d: d[1],
    )
    merged = []
    for sz, ss, os_ in dims:
        if merged:
            psz, pss, pos = merged[-1]
            if pss * psz == ss and (pos * psz == os_ or (pos == 0 and os_ == 0)):
                merged[-1] = (psz * sz, pss, pos)
                continue
        merged.append((sz, ss, os_))
    return merged


def _launch_contig(S, out_v, sval, n, mode):
    block = _CONTIG_BLOCK
    grid = ((n + block - 1) // block,)
    _k_contig[grid](
        S,
        out_v if isinstance(out_v, torch.Tensor) else S,
        sval,
        n,
        BLOCK=block,
        MODE=mode,
        I64=n > _INT32_MAX,
        EVEN=n % block == 0,
        num_warps=8,
    )


def _launch_strided(S, out_v, sval, mode):
    n = S.numel()
    if mode == MODE_TENSOR:
        ostride = out_v.expand(S.shape).stride()
    else:
        ostride = (0,) * S.dim()
    dims = _collapse(S.shape, S.stride(), ostride)
    if not dims:
        # single logical element
        _launch_contig(S, out_v, sval, n, mode)
        return
    if len(dims) == 1:
        sz, ss, os_ = dims[0]
        if ss == 1 and sz == n and (mode != MODE_TENSOR or os_ == 1):
            # both operands are linearly contiguous over the flat index
            _launch_contig(S, out_v, sval, n, mode)
            return
    rank = len(dims)
    ns = [1] * _MAX_RANK
    aa = [0] * _MAX_RANK
    bb = [0] * _MAX_RANK
    for i, (sz, ss, os_) in enumerate(dims):
        ns[i] = sz
        aa[i] = ss
        bb[i] = os_
    block = _STRIDED_BLOCK
    grid = ((n + block - 1) // block,)
    _k_strided[grid](
        S,
        out_v if isinstance(out_v, torch.Tensor) else S,
        sval,
        n,
        *ns,
        *aa,
        *bb,
        RANK=rank,
        BLOCK=block,
        MODE=mode,
        num_warps=4,
    )


def _run_pair(S, out_v):
    n = S.numel()
    if n == 0:
        return
    if isinstance(out_v, torch.Tensor):
        if out_v.numel() == 1:
            mode = MODE_SCALAR_PTR
        else:
            mode = MODE_TENSOR
        sval = 1.0
    else:
        mode = MODE_SCALAR
        sval = float(out_v)
    if S.is_contiguous():
        if mode == MODE_TENSOR and not (
            out_v.shape == S.shape and out_v.is_contiguous()
        ):
            _launch_strided(S, out_v, sval, mode)
            return
        _launch_contig(S, out_v, sval, n, mode)
        return
    _launch_strided(S, out_v, sval, mode)


def _launch_fused(tensors, others, start, count, n, scalar):
    """One launch covering ``count`` (<= _MAX_FUSE) uniform contiguous pairs."""
    s_tensors = tensors[start : start + count]
    block = 256 if n <= 65536 else 1024
    warps = 2 if n <= 65536 else 4
    grid = (n + block - 1) // block
    s_ptr = [t.data_ptr() for t in s_tensors]
    pad = _MAX_FUSE - count
    even = n % block == 0
    if scalar is None:
        o_tensors = others[start : start + count]
        o_ptr = [t.data_ptr() for t in o_tensors]
        kernel = _k_fused_t
        # Raw device pointers are forwarded as ints: the backend launcher then
        # skips the per-argument python ``data_ptr()`` call and driver query.
        pos_args = s_ptr + [s_ptr[0]] * pad + o_ptr + [o_ptr[0]] * pad
        pos_args += [n, block, count, even]
        aligned = all(p % 16 == 0 for p in s_ptr) and all(p % 16 == 0 for p in o_ptr)
        s_set, o_set = set(s_ptr), set(o_ptr)
        aliased = len(s_set) != count or len(o_set) != count or bool(s_set & o_set)

        def rt_builder(s_tensors=s_tensors, o_tensors=o_tensors, n=n, pad=pad):
            return (
                s_tensors
                + [s_tensors[0]] * pad
                + o_tensors
                + [o_tensors[0]] * pad
                + [n]
            )

    else:
        kernel = _k_fused_s
        pos_args = s_ptr + [s_ptr[0]] * pad + [scalar, n, block, count, even]
        aligned = all(p % 16 == 0 for p in s_ptr)
        aliased = len(set(s_ptr)) != count

        def rt_builder(s_tensors=s_tensors, n=n, scalar=scalar, pad=pad):
            return s_tensors + [s_tensors[0]] * pad + [scalar, n]

    kwargs = {"BLOCK": block, "NT": count, "EVEN": even, "num_warps": warps}
    if n > 1 and not aliased:
        key = (
            s_tensors[0].dtype,
            count,
            block,
            n % 16 == 0,
            even,
            aligned,
            scalar is None,
        )
        try:
            _fast_launch(kernel, key, grid, pos_args, rt_builder, kwargs)
            return
        except Exception:
            _FAST_CACHE.pop(key, None)
    kernel[(grid,)](*rt_builder(), **kwargs)


def _try_uniform(tensors, others, single, count):
    """Fuse the whole list into one launch when it is uniform and small."""
    first = tensors[0]
    if not first.is_contiguous():
        return False
    n = first.numel()
    if n == 0:
        return False
    dtype = first.dtype
    if n * dtype.itemsize > _FUSE_BYTES_LIMIT:
        return False
    shape = first.shape

    if others is None and not isinstance(single, torch.Tensor):
        scalar = float(single)
        for s in tensors:
            if s.dtype != dtype or s.shape != shape or not s.is_contiguous():
                return False
    else:
        scalar = None
        src = others if others is not None else [single] * count
        for i in range(count):
            s = tensors[i]
            o = src[i]
            if (
                s.dtype != dtype
                or s.shape != shape
                or not s.is_contiguous()
                or not isinstance(o, torch.Tensor)
                or o.shape != shape
                or not o.is_contiguous()
            ):
                return False

    start = 0
    while start < count:
        take = min(_MAX_FUSE, count - start)
        _launch_fused(tensors, others, start, take, n, scalar)
        start += take
    return True


def _foreach_mul__List(self, other):
    logger.debug("GEMS_MTHREADS _FOREACH_MUL__LIST")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_mul__List(self)
    """In-place foreach multiply; mutates every tensor of ``self``."""
    tensors = self if isinstance(self, (list, tuple)) else [self]
    count = len(tensors)
    if count == 0:
        return None
    if isinstance(other, (list, tuple)):
        if len(other) != count:
            raise RuntimeError(
                "Tensor list length mismatch: "
                f"self has {count} tensors but other has {len(other)}"
            )
        others = other
        single = None
    else:
        others = None
        single = other
    if count > 1 and _try_uniform(tensors, others, single, count):
        return None
    for i in range(count):
        _run_pair(tensors[i], others[i] if others is not None else single)
    return None
