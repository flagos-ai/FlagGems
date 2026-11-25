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
"""In-place foreach minimum against a scalar: aten::_foreach_minimum_.Scalar.

Every tensor of the input list is updated in place:

    x[i] <- minimum(x[i], scalar)      (NaN propagating, like torch.minimum)

Storage identity (data_ptr), shape and strides are preserved and the call
returns None, mirroring the ATen schema.

Two costs dominate this operator: the streaming read+modify+write of the data
and the per-tensor launch overhead (the pytest list has up to four tiny
tensors).  Therefore:

  * the whole list is fused into ONE kernel launch: grid = (blocks, arity) and
    ``program_id(1)`` selects the tensor, so a list of k tensors costs one
    launch instead of k;
  * a ``n % BLOCK == 0`` (EVEN) specialization drops the tail mask entirely;
  * dense and transposed-dense views collapse to a single unit-stride group
    (dims are sorted by stride and memory-adjacent neighbours merged), so no
    per-element index math is needed for them;
  * after the first call for a given kernel specialization the launch goes
    straight to the compiled kernel's launcher, skipping Triton's per-call
    argument binding / specialization / cache-key work and the launcher's
    global-scratch bookkeeping.  The launch always targets the *current*
    stream, and when Triton launch hooks are installed (profiling) or anything
    about that path is unexpected, the ordinary JIT launch is used instead.
"""

import torch
import triton
import triton.language as tl

try:
    from triton import knobs as _knobs
except Exception:  # pragma: no cover - older/newer Triton without knobs
    _knobs = None

import logging

from triton.runtime import driver as _driver_mod

from flag_gems.ops._foreach_binary import (
    _foreach_minimum__Scalar as default__foreach_minimum__Scalar,
)

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


_BLOCK = 1024
_NUM_WARPS = 4
_MAX_DIMS = 8
_MAX_FUSED = 4
_I32_SAFE = 2147479552
_DISABLED = object()
_FAST = {}
_RT = []
_HOOK_CALLS = None


@triton.jit
def _mn(x, s, IS_FLOAT: tl.constexpr):
    if IS_FLOAT:
        return tl.minimum(x, s, propagate_nan=tl.PropagateNan.ALL)
    return tl.minimum(x, s)


@triton.jit
def _flat_body(p, s, n, offs, IS_FLOAT: tl.constexpr, EVEN: tl.constexpr):
    # Single-use streaming lines: mark them evict-first so they leave L2 as
    # soon as they are consumed instead of displacing later traffic.
    if EVEN:
        x = tl.load(p + offs, eviction_policy="evict_first")
        tl.store(
            p + offs, _mn(x, s.to(x.dtype), IS_FLOAT), eviction_policy="evict_first"
        )
    else:
        m = offs < n
        x = tl.load(p + offs, mask=m, other=0.0, eviction_policy="evict_first")
        tl.store(
            p + offs,
            _mn(x, s.to(x.dtype), IS_FLOAT),
            mask=m,
            eviction_policy="evict_first",
        )


@triton.jit(do_not_specialize=["n"])
def _min1(p0, s, n, IS_FLOAT: tl.constexpr, EVEN: tl.constexpr, BLOCK: tl.constexpr):
    _flat_body(p0, s, n, tl.program_id(0) * BLOCK + tl.arange(0, BLOCK), IS_FLOAT, EVEN)


@triton.jit(do_not_specialize=["n"])
def _min2(
    p0, p1, s, n, IS_FLOAT: tl.constexpr, EVEN: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if tl.program_id(1) == 0:
        _flat_body(p0, s, n, offs, IS_FLOAT, EVEN)
    else:
        _flat_body(p1, s, n, offs, IS_FLOAT, EVEN)


@triton.jit(do_not_specialize=["n"])
def _min3(
    p0, p1, p2, s, n, IS_FLOAT: tl.constexpr, EVEN: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = tl.program_id(1)
    if t == 0:
        _flat_body(p0, s, n, offs, IS_FLOAT, EVEN)
    elif t == 1:
        _flat_body(p1, s, n, offs, IS_FLOAT, EVEN)
    else:
        _flat_body(p2, s, n, offs, IS_FLOAT, EVEN)


@triton.jit(do_not_specialize=["n"])
def _min4(
    p0,
    p1,
    p2,
    p3,
    s,
    n,
    IS_FLOAT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = tl.program_id(1)
    if t == 0:
        _flat_body(p0, s, n, offs, IS_FLOAT, EVEN)
    elif t == 1:
        _flat_body(p1, s, n, offs, IS_FLOAT, EVEN)
    elif t == 2:
        _flat_body(p2, s, n, offs, IS_FLOAT, EVEN)
    else:
        _flat_body(p3, s, n, offs, IS_FLOAT, EVEN)


@triton.jit
def _min_strided(
    ptr,
    s,
    n,
    e0,
    t0,
    e1,
    t1,
    e2,
    t2,
    e3,
    t3,
    e4,
    t4,
    e5,
    t5,
    e6,
    t6,
    e7,
    t7,
    ND: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rem = offs
    addr = tl.zeros((BLOCK,), dtype=tl.int64)
    if ND > 0:
        q = rem % e0
        rem = rem // e0
        addr += q.to(tl.int64) * t0
    if ND > 1:
        q = rem % e1
        rem = rem // e1
        addr += q.to(tl.int64) * t1
    if ND > 2:
        q = rem % e2
        rem = rem // e2
        addr += q.to(tl.int64) * t2
    if ND > 3:
        q = rem % e3
        rem = rem // e3
        addr += q.to(tl.int64) * t3
    if ND > 4:
        q = rem % e4
        rem = rem // e4
        addr += q.to(tl.int64) * t4
    if ND > 5:
        q = rem % e5
        rem = rem // e5
        addr += q.to(tl.int64) * t5
    if ND > 6:
        q = rem % e6
        rem = rem // e6
        addr += q.to(tl.int64) * t6
    if ND > 7:
        q = rem % e7
        rem = rem // e7
        addr += q.to(tl.int64) * t7
    x = tl.load(ptr + addr, mask=m, other=0.0)
    y = _mn(x, s.to(x.dtype), IS_FLOAT)
    tl.store(ptr + addr, y, mask=m)


_MIN_ARITY = (None, _min1, _min2, _min3, _min4)


def _canonical(t):
    """Memory-adjacent dim groups, innermost (smallest stride) first."""
    dims = [[sz, st] for sz, st in zip(t.shape, t.stride()) if sz > 1]
    if len(dims) <= 1:
        return dims
    dims = sorted(dims, key=lambda p: p[1])
    out = [dims[0]]
    for sz, st in dims[1:]:
        last = out[-1]
        if last[0] * last[1] == st:
            last[0] *= sz
        else:
            out.append([sz, st])
    return out


def _is_flat(t):
    if t.is_contiguous():
        return True
    groups = _canonical(t)
    return len(groups) == 1 and groups[0][1] == 1


def _launch_flat(ptrs, s, n, is_float, block):
    """One launch for 1..4 same-dtype, same-n, unit-stride tensors."""
    global _HOOK_CALLS
    arity = len(ptrs)
    nb = (n + block - 1) // block
    even = (n % block) == 0

    p0 = ptrs[0]
    dtype = p0.dtype
    aligned = (p0.data_ptr() & 15) == 0
    if aligned:
        for p in ptrs[1:]:
            if p.data_ptr() & 15:
                aligned = False
                break

    key = (arity, dtype, even, block, aligned)
    entry = _FAST.get(key)
    hc = _HOOK_CALLS
    if entry is not None and hc is not None and not (hc[0] or hc[1]):
        if not _RT:
            _RT.append(_driver_mod.active)
            _RT.append(_RT[0].get_current_device())
        launch, fn, meta = entry
        launch(
            nb,
            arity,
            1,
            _RT[0].get_current_stream(_RT[1]),
            fn,
            meta,
            None,
            None,
            None,
            *ptrs,
            s,
            n,
            is_float,
            even,
            block,
        )
        return

    kern = _MIN_ARITY[arity][(nb, arity)](
        *ptrs,
        s,
        n,
        IS_FLOAT=is_float,
        EVEN=even,
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )
    if entry is None:
        try:
            launcher = kern.run
            if _HOOK_CALLS is None and _knobs is not None:
                rt = _knobs.runtime
                _HOOK_CALLS = (rt.launch_enter_hook.calls, rt.launch_exit_hook.calls)
            # The raw launcher entry point skips the wrapper's per-call
            # global-scratch bookkeeping; only valid for scratch-free kernels.
            if getattr(launcher, "global_scratch_size", 1) == 0 and not getattr(
                launcher, "_debug_enabled", True
            ):
                _FAST[key] = (launcher.launch, kern.function, kern.packed_metadata)
            else:
                _FAST[key] = (launcher, kern.function, kern.packed_metadata)
        except Exception:
            _FAST[key] = _DISABLED


def _launch_strided(t, s, n, is_float, groups):
    args = []
    for sz, st in groups:
        args.append(sz)
        args.append(st)
    for _ in range(_MAX_DIMS - len(groups)):
        args.append(1)
        args.append(0)
    _min_strided[((n + _BLOCK - 1) // _BLOCK,)](
        t,
        s,
        n,
        *args,
        ND=len(groups),
        IS_FLOAT=is_float,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _apply_one(t, scalar):
    n = t.numel()
    if n == 0:
        return
    is_float = bool(t.dtype.is_floating_point)
    s = float(scalar) if is_float else int(scalar)
    if t.is_contiguous():
        _launch_flat((t,), s, n, is_float, _BLOCK)
        return
    groups = _canonical(t)
    if len(groups) == 1 and groups[0][1] == 1:
        _launch_flat((t,), s, n, is_float, _BLOCK)
        return
    if len(groups) > _MAX_DIMS:
        raise RuntimeError("foreach_minimum_: unsupported tensor layout")
    _launch_strided(t, s, n, is_float, groups)


def _apply_each(tensors, scalar):
    for t in tensors:
        _apply_one(t, scalar)


def _foreach_minimum__Scalar(self, scalar):
    logger.debug("GEMS_MTHREADS _FOREACH_MINIMUM__SCALAR")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_minimum__Scalar(self, scalar)
    if isinstance(self, torch.Tensor):
        _apply_one(self, scalar)
        return None
    nt = len(self)
    if nt == 0:
        return None
    t0 = self[0]
    dtype = t0.dtype
    if not dtype.is_floating_point:
        _apply_each(self, scalar)
        return None
    n = t0.numel()
    if nt == 1:
        if n == 0 or not t0.is_contiguous():
            _apply_one(t0, scalar)
            return None
    else:
        if n == 0:
            _apply_each(self, scalar)
            return None
        # Fusable only if every list element has the same dtype, the same
        # element count and unit-stride storage.
        for i in range(1, nt):
            t = self[i]
            if t.dtype is not dtype or t.numel() != n or not t.is_contiguous():
                _apply_each(self, scalar)
                return None
        if not t0.is_contiguous() and not _is_flat(t0):
            _apply_each(self, scalar)
            return None
    s = float(scalar)
    if nt <= _MAX_FUSED:
        _launch_flat(self, s, n, True, _BLOCK)
    else:
        i = 0
        while i < nt:
            _launch_flat(tuple(self[i : i + _MAX_FUSED]), s, n, True, _BLOCK)
            i += _MAX_FUSED
    return None
