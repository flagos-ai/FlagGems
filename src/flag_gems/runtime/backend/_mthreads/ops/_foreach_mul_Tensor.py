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
"""foreach_mul_tensor: out_i = self_i * other  (other is a 0-dim / scalar operand)."""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_mul_Tensor as default__foreach_mul_Tensor,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_BLOCK = 2048
_MAXDIM = 8


@triton.jit
def _tile(x_ptr, y_ptr, o, n, tile, BLOCK: tl.constexpr, UPCAST: tl.constexpr):
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy="evict_first")
    if UPCAST:
        r = x.to(tl.float32) * o.to(tl.float32)
    else:
        r = x * o
    tl.store(
        y_ptr + offs,
        r.to(y_ptr.dtype.element_ty),
        mask=mask,
        eviction_policy="evict_first",
    )


@triton.jit
def _tile_exact(x_ptr, y_ptr, o, tile, BLOCK: tl.constexpr, UPCAST: tl.constexpr):
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    if UPCAST:
        r = x.to(tl.float32) * o.to(tl.float32)
    else:
        r = x * o
    tl.store(y_ptr + offs, r.to(y_ptr.dtype.element_ty), eviction_policy="evict_first")


@triton.jit
def _mul4(
    x0,
    y0,
    n0,
    x1,
    y1,
    n1,
    x2,
    y2,
    n2,
    x3,
    y3,
    n3,
    o_ptr,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
):
    pid = tl.program_id(0)
    t0 = (n0 + BLOCK - 1) // BLOCK
    t1 = t0 + (n1 + BLOCK - 1) // BLOCK
    t2 = t1 + (n2 + BLOCK - 1) // BLOCK
    o = tl.load(o_ptr)
    if pid < t0:
        _tile(x0, y0, o, n0, pid, BLOCK, UPCAST)
    elif pid < t1:
        _tile(x1, y1, o, n1, pid - t0, BLOCK, UPCAST)
    elif pid < t2:
        _tile(x2, y2, o, n2, pid - t1, BLOCK, UPCAST)
    else:
        _tile(x3, y3, o, n3, pid - t2, BLOCK, UPCAST)


@triton.jit
def _mul4_exact(
    x0,
    y0,
    n0,
    x1,
    y1,
    n1,
    x2,
    y2,
    n2,
    x3,
    y3,
    n3,
    o_ptr,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
):
    pid = tl.program_id(0)
    t0 = (n0 + BLOCK - 1) // BLOCK
    t1 = t0 + (n1 + BLOCK - 1) // BLOCK
    t2 = t1 + (n2 + BLOCK - 1) // BLOCK
    o = tl.load(o_ptr)
    if pid < t0:
        _tile_exact(x0, y0, o, pid, BLOCK, UPCAST)
    elif pid < t1:
        _tile_exact(x1, y1, o, pid - t0, BLOCK, UPCAST)
    elif pid < t2:
        _tile_exact(x2, y2, o, pid - t1, BLOCK, UPCAST)
    else:
        _tile_exact(x3, y3, o, pid - t2, BLOCK, UPCAST)


@triton.jit
def _mul1(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, UPCAST: tl.constexpr):
    pid = tl.program_id(0)
    o = tl.load(o_ptr)
    _tile(x_ptr, y_ptr, o, n, pid, BLOCK, UPCAST)


@triton.jit
def _mul_strided(
    x_ptr,
    y_ptr,
    o_ptr,
    o_val,
    n,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    k0,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    k7,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    USE_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs.to(tl.int64)
    addr = tl.zeros([BLOCK], dtype=tl.int64)

    c = idx % r0
    idx = idx // r0
    addr += c * k0
    c = idx % r1
    idx = idx // r1
    addr += c * k1
    c = idx % r2
    idx = idx // r2
    addr += c * k2
    c = idx % r3
    idx = idx // r3
    addr += c * k3
    c = idx % r4
    idx = idx // r4
    addr += c * k4
    c = idx % r5
    idx = idx // r5
    addr += c * k5
    c = idx % r6
    idx = idx // r6
    addr += c * k6
    c = idx % r7
    addr += c * k7

    x = tl.load(x_ptr + addr, mask=mask)
    if USE_PTR:
        o = tl.load(o_ptr)
    else:
        o = o_val
    if UPCAST:
        r = x.to(tl.float32) * o.to(tl.float32)
    else:
        r = x * o
    tl.store(y_ptr + offs, r.to(y_ptr.dtype.element_ty), mask=mask)


def _tiles(numel):
    return (numel + _BLOCK - 1) // _BLOCK


def _upcast(dtype):
    return 1 if (dtype is torch.float16 or dtype is torch.bfloat16) else 0


def _launch_strided(t, out, other, o_val, numel, up, use_ptr):
    sizes = t.shape
    strides = t.stride()
    nd = len(sizes)
    rs = [1] * _MAXDIM
    ks = [0] * _MAXDIM
    for d in range(nd):
        rs[d] = sizes[nd - 1 - d]
        ks[d] = strides[nd - 1 - d]
    grid = (_tiles(numel),)
    _mul_strided[grid](
        t,
        out,
        other,
        o_val,
        numel,
        rs[0],
        rs[1],
        rs[2],
        rs[3],
        rs[4],
        rs[5],
        rs[6],
        rs[7],
        ks[0],
        ks[1],
        ks[2],
        ks[3],
        ks[4],
        ks[5],
        ks[6],
        ks[7],
        BLOCK=_BLOCK,
        UPCAST=up,
        USE_PTR=use_ptr,
    )


def _run_pairs(tensors, outs, other, up):
    for t, out in zip(tensors, outs):
        numel = t.numel()
        if numel == 0:
            continue
        if t.is_contiguous():
            _mul1[(_tiles(numel),)](t, out, other, numel, BLOCK=_BLOCK, UPCAST=up)
        else:
            _launch_strided(t, out, other, 0.0, numel, up, True)
    return outs


_FAST = {}


def _dispatch_mul(kernel, grid, args, up, key):
    """Launch a four-slot kernel, replaying the compiled variant once it exists.

    Triton's Python-level dispatch costs more than this tiny launch on the
    target, so the cached ``CompiledKernel`` is replayed directly, as the other
    MThreads foreach kernels do.  ``key`` carries everything the compile
    specialisation depends on (kernel, dtypes, per-slot numels and pointer
    alignment); a stale handle falls back to the ordinary launch path.
    """
    entry = _FAST.get(key)
    if entry is not None:
        compiled, stream = entry
        try:
            compiled.run(
                grid[0],
                1,
                1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                _BLOCK,
                up,
            )
            return
        except Exception:
            _FAST.pop(key, None)

    compiled = kernel[grid](*args, BLOCK=_BLOCK, UPCAST=up)
    try:
        drv = _driver.active
        _FAST[key] = (compiled, drv.get_current_stream(drv.get_current_device()))
    except Exception:
        _FAST.pop(key, None)


def _run_uniform(tensors, other):
    """Fast path: every tensor shares dtype, shape and contiguity."""
    n = len(tensors)
    first = tensors[0]
    up = _upcast(first.dtype)
    outs = [torch.empty_like(t) for t in tensors]
    i = 0
    while i < n:
        grp = list(tensors[i : i + 4])
        ogrp = list(outs[i : i + 4])
        nums = [t.numel() for t in grp]
        total = 0
        for q in nums:
            total += _tiles(q)
        if total:
            k = len(grp)
            if k < 4:
                grp = grp + [first] * (4 - k)
                ogrp = ogrp + [ogrp[0]] * (4 - k)
                nums = nums + [0] * (4 - k)
            exact = not (nums[0] & (_BLOCK - 1))
            kernel = _mul4_exact if exact else _mul4
            args = []
            for t, o, q in zip(grp, ogrp, nums):
                args += [t, o, q]
            args.append(other)
            key = (
                exact,
                first.dtype,
                other.dtype,
                up,
                tuple(nums),
                tuple(t.data_ptr() & 15 for t in grp),
                tuple(o.data_ptr() & 15 for o in ogrp),
                other.data_ptr() & 15,
            )
            _dispatch_mul(kernel, (total,), args, up, key)
        i += 4
    return outs


def _run_tensor(tensors, other):
    if type(tensors) is not list:
        tensors = list(tensors)
    n = len(tensors)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    first = tensors[0]
    dt = first.dtype
    s0 = first.shape
    for t in tensors:
        if t.dtype is not dt or t.shape != s0 or not t.is_contiguous():
            dev = first.device
            outs = [torch.empty(t.shape, dtype=t.dtype, device=dev) for t in tensors]
            return _run_pairs(tensors, outs, other, _upcast(dt))
    return _run_uniform(tensors, other)


def _run_scalar(tensors, other):
    tensors = list(tensors)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    dev = tensors[0].device
    outs = [torch.empty(t.shape, dtype=t.dtype, device=dev) for t in tensors]
    val = float(other) if tensors[0].dtype.is_floating_point else int(other)
    for t, out in zip(tensors, outs):
        numel = t.numel()
        if numel == 0:
            continue
        _launch_strided(t, out, t, val, numel, _upcast(t.dtype), False)
    return outs


def _foreach_mul_Tensor(self, other):
    logger.debug("GEMS_MTHREADS _FOREACH_MUL_TENSOR")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_mul_Tensor(self)
    tensors = self
    if isinstance(other, torch.Tensor):
        if other.dim() == 0:
            return _run_tensor(tensors, other)
        raise RuntimeError(
            "scalar tensor expected to be 0 dim but it has %d dimensions "
            "and %d elements." % (other.dim(), other.numel())
        )
    if isinstance(other, (list, tuple)):
        if len(other) != len(tensors):
            raise RuntimeError(
                "Tensor lists must have the same number of tensors, got "
                "%d and %d" % (len(tensors), len(other))
            )
        outs = []
        for t, o in zip(tensors, other):
            if isinstance(o, torch.Tensor):
                if o.dim() != 0:
                    raise NotImplementedError(
                        "only scalar operands are supported for the list overload"
                    )
                outs.append(_run_tensor([t], o)[0])
            else:
                outs.append(_run_scalar([t], o)[0])
        return outs
    return _run_scalar(tensors, other)
