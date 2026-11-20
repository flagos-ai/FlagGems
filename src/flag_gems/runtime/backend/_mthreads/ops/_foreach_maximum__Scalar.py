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
"""In-place foreach maximum with a scalar: aten::_foreach_maximum_.Scalar.

Semantics (verified against torch 2.9.1 on the target):
  for each tensor t in self:  t[i] = max_num(t[i], scalar_rounded_to_t.dtype)
  NaN propagates, ties keep the first (tensor) operand, storage is preserved,
  and the call returns None (Gems pytest in-place contract).

Architecture: one Triton launch per contiguous same-dtype group that shares a
single element count (2-D grid (blocks, T), up to 4 tensors per launch).  A
branch-light fast path covers the common list of 2..4 contiguous 16/32-bit
float tensors; non-contiguous tensors, float64, bool/int lists, unequal element
counts and longer lists fall back to per-tensor launches.  BLOCK/num_warps are
fixed at 1024/4: a bounded autotune search over BLOCK {1024,2048,4096} and
num_warps {2,4,8,16} left the bandwidth-bound workloads unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_maximum__Scalar as default__foreach_maximum__Scalar,
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
_MAX_RANK = 8
_FUSE = 4

_FP32 = torch.float32
_FP16 = torch.float16
_BF16 = torch.bfloat16


@triton.jit
def _fmax_body(
    p,
    s_ptr,
    n,
    offs,
    s,
    F64: tl.constexpr,
    UPCAST: tl.constexpr,
    SCALAR_NAN: tl.constexpr,
):
    mask = offs < n
    x = tl.load(p + offs, mask=mask)
    if SCALAR_NAN:
        res = x - x
    else:
        if F64:
            sx = tl.load(s_ptr)
            res = tl.where(x < sx, sx, x)
        elif UPCAST:
            xf = x.to(tl.float32)
            sf = s.to(tl.float32)
            res = tl.where(xf < sf, sf, xf).to(x.dtype)
        else:
            sx = s.to(x.dtype)
            res = tl.where(x < sx, sx, x)
    tl.store(p + offs, res, mask=mask)


@triton.jit
def _fmax_flat(
    x_ptr,
    s_ptr,
    n,
    s,
    F64: tl.constexpr,
    UPCAST: tl.constexpr,
    SCALAR_NAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _fmax_body(x_ptr, s_ptr, n, offs, s, F64, UPCAST, SCALAR_NAN)


@triton.jit
def _fmax_fused4(
    p0,
    p1,
    p2,
    p3,
    n,
    s,
    UPCAST: tl.constexpr,
    SCALAR_NAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    if pid1 == 0:
        _fmax_body(p0, p0, n, offs, s, False, UPCAST, SCALAR_NAN)
    elif pid1 == 1:
        _fmax_body(p1, p1, n, offs, s, False, UPCAST, SCALAR_NAN)
    elif pid1 == 2:
        _fmax_body(p2, p2, n, offs, s, False, UPCAST, SCALAR_NAN)
    else:
        _fmax_body(p3, p3, n, offs, s, False, UPCAST, SCALAR_NAN)


@triton.jit
def _fmax_strided(
    x_ptr,
    s_ptr,
    n,
    s,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    R: tl.constexpr,
    F64: tl.constexpr,
    UPCAST: tl.constexpr,
    SCALAR_NAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    dims = (d0, d1, d2, d3, d4, d5, d6, d7)
    strs = (t0, t1, t2, t3, t4, t5, t6, t7)
    for k in tl.static_range(R):
        dk = dims[k]
        i = rem % dk
        rem = rem // dk
        off += i * strs[k]
    x = tl.load(x_ptr + off, mask=mask)
    if SCALAR_NAN:
        res = x - x
    else:
        if F64:
            sx = tl.load(s_ptr)
            res = tl.where(x < sx, sx, x)
        elif UPCAST:
            xf = x.to(tl.float32)
            sf = s.to(tl.float32)
            res = tl.where(xf < sf, sf, xf).to(x.dtype)
        else:
            sx = s.to(x.dtype)
            res = tl.where(x < sx, sx, x)
    tl.store(x_ptr + off, res, mask=mask)


def _collapse_dims(t):
    """Innermost-first (stride, size) dims, adjacent dims merged when possible."""
    items = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        items.append((st, sz))
    items = sorted(items, key=lambda p: abs(p[0]))
    merged = []
    for st, sz in items:
        if merged and merged[-1][0] * merged[-1][1] == st:
            merged[-1][1] *= sz
        else:
            merged.append([st, sz])
    return merged


def _launch(t, dtype_scalar, s_nan, f64_holder):
    n = t.numel()
    if n == 0:
        return
    dt = t.dtype
    f64 = dt == torch.float64
    upcast = dt == _FP16 or dt == _BF16
    blk = _BLOCK
    if t.is_contiguous():
        grid = ((n + blk - 1) // blk,)
        _fmax_flat[grid](
            t,
            t if not f64 else f64_holder[0],
            n,
            dtype_scalar,
            F64=f64,
            UPCAST=upcast,
            SCALAR_NAN=s_nan,
            BLOCK=blk,
        )
        return
    dims = _collapse_dims(t)
    if len(dims) > _MAX_RANK:
        # Fall back to slicing off the outermost dimension (views only).
        for i in range(t.shape[0]):
            _launch(t[i], dtype_scalar, s_nan, f64_holder)
        return
    sizes = [1] * _MAX_RANK
    strides = [0] * _MAX_RANK
    for k, (st, sz) in enumerate(dims):
        strides[k] = st
        sizes[k] = sz
    grid = ((n + blk - 1) // blk,)
    _fmax_strided[grid](
        t,
        t,
        n,
        dtype_scalar,
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
        R=len(dims),
        F64=f64,
        UPCAST=upcast,
        SCALAR_NAN=s_nan,
        BLOCK=blk,
    )


def _launch_group(ts, dtype_scalar, s_nan, upcast):
    """One launch for a contiguous same-dtype group sharing one element count."""
    T = len(ts)
    n = ts[0].numel()
    p = ts + [ts[0]] * (_FUSE - T)
    grid = ((n + _BLOCK - 1) // _BLOCK, T)
    _fmax_fused4[grid](
        p[0],
        p[1],
        p[2],
        p[3],
        n,
        dtype_scalar,
        UPCAST=upcast,
        SCALAR_NAN=s_nan,
        BLOCK=_BLOCK,
    )


_INT_ERR = "result type Float can't be cast to the desired output type {}"


def _foreach_maximum__Scalar(self, scalar, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_MAXIMUM__SCALAR")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_maximum__Scalar(self, scalar, *args, **kwargs)
    if isinstance(self, torch.Tensor) or not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_maximum_() received an invalid combination of arguments - "
            "got (Tensor, Number), but expected (tuple of Tensors self, Number scalar)"
        )
    n_t = len(self)
    if n_t == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    s_val = scalar.item() if isinstance(scalar, torch.Tensor) else scalar
    s_nan = isinstance(s_val, float) and s_val != s_val

    # Branch-light fast path: 2..4 contiguous 16/32-bit float tensors, one numel.
    if 2 <= n_t <= 4 and not s_nan:
        t0 = self[0]
        dt = t0.dtype
        if (dt == _FP32 or dt == _FP16 or dt == _BF16) and t0.is_contiguous():
            n = t0.numel()
            if n:
                ok = True
                for i in range(1, n_t):
                    ti = self[i]
                    if ti.dtype != dt or not ti.is_contiguous() or ti.numel() != n:
                        ok = False
                        break
                if ok:
                    sc = float(s_val)
                    if n_t == 2:
                        _launch_group([t0, self[1]], sc, False, dt != _FP32)
                    elif n_t == 3:
                        _launch_group([t0, self[1], self[2]], sc, False, dt != _FP32)
                    else:
                        _launch_group(
                            [t0, self[1], self[2], self[3]], sc, False, dt != _FP32
                        )
                    return None

    f64_holder = [None]
    groups = {}
    order = []
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("expected Tensor as element of self")
        if t.numel() == 0:
            continue
        dt = t.dtype
        if dt == torch.bool:
            if not isinstance(s_val, bool):
                raise RuntimeError(
                    "result type Long can't be cast to the desired output type bool"
                )
            t = t.view(torch.int8)
            dt = torch.int8
            sc = 1 if s_val else 0
        elif dt.is_complex:
            raise NotImplementedError("complex foreach maximum")
        elif dt.is_floating_point:
            sc = float(s_val)
        elif isinstance(s_val, float):
            if s_val != s_val or s_val != int(s_val):
                raise RuntimeError(_INT_ERR.format(dt))
            sc = int(s_val)
        else:
            sc = int(s_val)

        if not t.is_contiguous() or dt == torch.float64:
            if dt == torch.float64 and f64_holder[0] is None:
                f64_holder[0] = torch.tensor(
                    [float(s_val)], dtype=torch.float64, device=t.device
                )
            _launch(t, sc, s_nan, f64_holder)
            continue
        key = (dt, sc)
        g = groups.get(key)
        if g is None:
            groups[key] = g = []
            order.append(key)
        g.append(t)

    for key in order:
        ts = groups[key]
        sc = key[1]
        upcast = key[0] == _FP16 or key[0] == _BF16
        nt = len(ts)
        if nt == 1:
            _launch(ts[0], sc, s_nan, f64_holder)
            continue
        n = ts[0].numel()
        uniform = True
        for t in ts:
            if t.numel() != n:
                uniform = False
                break
        if not uniform:
            for t in ts:
                _launch(t, sc, s_nan, f64_holder)
            continue
        for i in range(0, nt, _FUSE):
            chunk = ts[i : i + _FUSE]
            if len(chunk) == 1:
                _launch(chunk[0], sc, s_nan, f64_holder)
            else:
                _launch_group(chunk, sc, s_nan, upcast)

    return None
