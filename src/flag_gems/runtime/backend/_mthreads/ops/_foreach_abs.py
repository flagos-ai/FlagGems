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
"""Triton implementation of aten::_foreach_abs.

Applies |x| elementwise to every tensor of the input sequence, returning a
freshly allocated tensor per input (no aliasing with the input storage).

The whole list is processed by one fused Triton launch per group of up to
_KMAX tensors (cumulative block offsets + per-slot branch), which removes the
per-tensor Python-side launch cost that dominates short workloads. For the
common well-aligned case the already-compiled kernel is launched directly from
its CompiledKernel handle, skipping Triton's per-call binding and cache lookup.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_abs as default__foreach_abs

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
_BLOCK = 1024
_WARPS = 4
_KMAX = 16
_MAXD = 8

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))
_INT_DTYPES = frozenset(
    (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
)
_COMPLEX_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}
_PLAIN_DTYPES = _FLOAT_DTYPES | _INT_DTYPES

_FLAT_FAST = {}
_FUSED_FAST = {}
_STREAM_FN = None


@triton.jit
def _abs_block(x_ptr, out_ptr, n, pid, BLOCK: tl.constexpr, IS_FLOAT: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if IS_FLOAT:
        y = tl.abs(x)
    else:
        y = tl.where(x < 0, -x, x)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _abs_flat_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, IS_FLOAT: tl.constexpr):
    _abs_block(x_ptr, out_ptr, n, tl.program_id(0), BLOCK, IS_FLOAT)


@triton.jit
def _abs_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    # c1..c15 are cumulative block offsets: slot k starts at block c_k (c0 = 0)
    # and owns the programs p with c_k <= p < c_{k+1} (c16 = total).
    pid = tl.program_id(0)
    if pid < c1:
        _abs_block(x0, o0, n0, pid, BLOCK, IS_FLOAT)
    elif pid < c2:
        _abs_block(x1, o1, n1, pid - c1, BLOCK, IS_FLOAT)
    elif pid < c3:
        _abs_block(x2, o2, n2, pid - c2, BLOCK, IS_FLOAT)
    elif pid < c4:
        _abs_block(x3, o3, n3, pid - c3, BLOCK, IS_FLOAT)
    elif pid < c5:
        _abs_block(x4, o4, n4, pid - c4, BLOCK, IS_FLOAT)
    elif pid < c6:
        _abs_block(x5, o5, n5, pid - c5, BLOCK, IS_FLOAT)
    elif pid < c7:
        _abs_block(x6, o6, n6, pid - c6, BLOCK, IS_FLOAT)
    elif pid < c8:
        _abs_block(x7, o7, n7, pid - c7, BLOCK, IS_FLOAT)
    elif pid < c9:
        _abs_block(x8, o8, n8, pid - c8, BLOCK, IS_FLOAT)
    elif pid < c10:
        _abs_block(x9, o9, n9, pid - c9, BLOCK, IS_FLOAT)
    elif pid < c11:
        _abs_block(x10, o10, n10, pid - c10, BLOCK, IS_FLOAT)
    elif pid < c12:
        _abs_block(x11, o11, n11, pid - c11, BLOCK, IS_FLOAT)
    elif pid < c13:
        _abs_block(x12, o12, n12, pid - c12, BLOCK, IS_FLOAT)
    elif pid < c14:
        _abs_block(x13, o13, n13, pid - c13, BLOCK, IS_FLOAT)
    elif pid < c15:
        _abs_block(x14, o14, n14, pid - c14, BLOCK, IS_FLOAT)
    else:
        _abs_block(x15, o15, n15, pid - c15, BLOCK, IS_FLOAT)


@triton.jit
def _abs_strided_kernel(
    x_ptr,
    out_ptr,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    COMPLEX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    lin = offs.to(tl.int64)
    src = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM >= 8:
        src += (lin % sh7) * st7
        lin = lin // sh7
    if NDIM >= 7:
        src += (lin % sh6) * st6
        lin = lin // sh6
    if NDIM >= 6:
        src += (lin % sh5) * st5
        lin = lin // sh5
    if NDIM >= 5:
        src += (lin % sh4) * st4
        lin = lin // sh4
    if NDIM >= 4:
        src += (lin % sh3) * st3
        lin = lin // sh3
    if NDIM >= 3:
        src += (lin % sh2) * st2
        lin = lin // sh2
    if NDIM >= 2:
        src += (lin % sh1) * st1
        lin = lin // sh1
    if NDIM >= 1:
        src += lin * st0
    if COMPLEX:
        re = tl.load(x_ptr + 2 * src, mask=mask, other=0.0)
        im = tl.load(x_ptr + 2 * src + 1, mask=mask, other=0.0)
        y = tl.sqrt(re * re + im * im)
    else:
        x = tl.load(x_ptr + src, mask=mask, other=0)
        if IS_FLOAT:
            y = tl.abs(x)
        else:
            y = tl.where(x < 0, -x, x)
    tl.store(out_ptr + offs, y, mask=mask)


def _stream(dev):
    global _STREAM_FN
    fn = _STREAM_FN
    if fn is None:
        fn = _STREAM_FN = triton.runtime.driver.active.get_current_stream
    return fn(dev)


def _fast_ok(ck, nargs):
    """True when a CompiledKernel can be launched with the plain ABI."""
    try:
        runner = ck.run
    except Exception:  # noqa: BLE001
        return False
    if getattr(runner, "user_arg_count", None) != nargs:
        return False
    return not getattr(runner, "_needs_runtime_expansion", True)


def _launch_flat(x, out, n, is_float):
    gx = (n + _BLOCK - 1) // _BLOCK
    if (
        n >= 16
        and not (n & 15)
        and not (x.data_ptr() & 15)
        and not (out.data_ptr() & 15)
    ):
        dev = x.device.index or 0
        key = (dev, x.dtype, is_float)
        ck = _FLAT_FAST.get(key)
        if ck is None:
            try:
                ck = _abs_flat_kernel.warmup(
                    x,
                    out,
                    n,
                    BLOCK=_BLOCK,
                    IS_FLOAT=is_float,
                    num_warps=_WARPS,
                    grid=(gx,),
                )
                if not _fast_ok(ck, 5):
                    ck = False
            except Exception:  # noqa: BLE001
                ck = False
            _FLAT_FAST[key] = ck
        if ck is not False:
            ck.run(
                gx,
                1,
                1,
                _stream(dev),
                ck.function,
                ck.packed_metadata,
                None,
                None,
                None,
                x,
                out,
                n,
                _BLOCK,
                is_float,
            )
            return
    _abs_flat_kernel[(gx,)](
        x, out, n, BLOCK=_BLOCK, IS_FLOAT=is_float, num_warps=_WARPS
    )


def _compress_dims(shape, strides):
    """Merge adjacent dimensions that are contiguous in memory."""
    ns = []
    nst = []
    for s, st in zip(shape, strides):
        if s == 1:
            continue
        if ns and nst[-1] == st * s:
            ns[-1] *= s
            nst[-1] = st
        else:
            ns.append(s)
            nst.append(st)
    if not ns:
        ns = [1]
        nst = [0]
    return ns, nst


def _grid(n):
    return ((n + _BLOCK - 1) // _BLOCK,)


def _launch_strided(passed, out, n, shape, strides, is_float, is_complex):
    shape, strides = _compress_dims(shape, strides)
    nd = len(shape)
    if nd > _MAXD:
        raise RuntimeError("abs: too many dimensions")
    sc = shape + [1] * (_MAXD - nd)
    st = strides + [0] * (_MAXD - nd)
    _abs_strided_kernel[_grid(n)](
        passed,
        out,
        n,
        sc[0],
        sc[1],
        sc[2],
        sc[3],
        sc[4],
        sc[5],
        sc[6],
        sc[7],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        st[6],
        st[7],
        NDIM=nd,
        BLOCK=_BLOCK,
        IS_FLOAT=is_float,
        COMPLEX=is_complex,
        num_warps=_WARPS,
    )


def _abs_complex(x):
    real_dtype = _COMPLEX_REAL[x.dtype]
    n = x.numel()
    perm = None
    xp = None
    try:
        xr = x.view(real_dtype)
    except RuntimeError:
        strides = list(x.stride())
        perm = sorted(range(x.dim()), key=lambda d: strides[d])
        xp = x.permute(perm)
        xr = xp.view(real_dtype)
    if perm is None:
        out = torch.empty(x.shape, dtype=real_dtype, device=x.device)
        store_buf = out
        view_shape = list(x.shape)
        view_strides = list(x.stride())
    else:
        store_buf = torch.empty(list(xp.shape), dtype=real_dtype, device=x.device)
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        out = store_buf.permute(inv)
        view_shape = list(xp.shape)
        view_strides = list(xp.stride())
    if n == 0:
        return out
    _launch_strided(xr, store_buf, n, view_shape, view_strides, True, True)
    return out


def _abs_one(x):
    dtype = x.dtype
    if dtype in _COMPLEX_REAL:
        return _abs_complex(x)
    if dtype in _FLOAT_DTYPES:
        is_float = True
    elif dtype in _INT_DTYPES:
        is_float = False
    else:
        raise RuntimeError("\"abs\" not implemented for '%s'" % (dtype,))
    n = x.numel()
    if x.is_contiguous():
        out = torch.empty_like(x)
        if n == 0:
            return out
        _launch_flat(x, out, n, is_float)
        return out
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    if n == 0:
        return out
    _launch_strided(x, out, n, list(x.shape), list(x.stride()), is_float, False)
    return out


def _is_plain(x):
    return x.dtype in _PLAIN_DTYPES and x.numel() > 0 and x.is_contiguous()


def _launch_fused(chunk, outs, is_float):
    nslots = len(chunk)
    lens = [t.numel() for t in chunk]
    aligned = True
    for i in range(nslots):
        n = lens[i]
        if (
            n < 16
            or (n & 15)
            or (chunk[i].data_ptr() & 15)
            or (outs[i].data_ptr() & 15)
        ):
            aligned = False
            break
    blocks = [(v + _BLOCK - 1) // _BLOCK for v in lens]
    if aligned:
        # block offsets become multiples of 16, keeping Triton's per-argument
        # specialization identical for every call that shares a cache entry
        blocks = [(b + 15) & ~15 for b in blocks]
    cum = [0]
    for b in blocks:
        cum.append(cum[-1] + b)
    total = cum[-1]
    pad = _KMAX - nslots
    xs = list(chunk) + [chunk[0]] * pad
    os_ = list(outs) + [outs[0]] * pad
    ns = lens + [0] * pad
    cs = cum[1:_KMAX]
    cs = cs + [total] * (_KMAX - 1 - len(cs))
    args = tuple(xs) + tuple(os_) + tuple(ns) + tuple(cs) + (_BLOCK, is_float)
    if aligned:
        dev = chunk[0].device.index or 0
        key = (dev, chunk[0].dtype, is_float, nslots)
        ck = _FUSED_FAST.get(key)
        if ck is None:
            try:
                ck = _abs_fused_kernel.warmup(*args, grid=(total,), num_warps=_WARPS)
                if not _fast_ok(ck, 65):
                    ck = False
            except Exception:  # noqa: BLE001
                ck = False
            _FUSED_FAST[key] = ck
        if ck is not False:
            ck.run(
                total,
                1,
                1,
                _stream(dev),
                ck.function,
                ck.packed_metadata,
                None,
                None,
                None,
                *args,
            )
            return
    _abs_fused_kernel[(total,)](*args, num_warps=_WARPS)


def _foreach_abs(self):
    logger.debug("GEMS_MTHREADS _FOREACH_ABS")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor)
        or t.device.type != "musa"
        or t.dtype == torch.bool
        or t.dtype.is_complex
        for t in self
    ):
        return default__foreach_abs(self)
    items = list(self)
    if not items:
        raise RuntimeError("_foreach_abs(): expected a non-empty TensorList")
    n_items = len(items)
    outs = [None] * n_items
    i = 0
    while i < n_items:
        x = items[i]
        if x.dtype in _PLAIN_DTYPES and x.numel() > 0 and x.is_contiguous():
            dtype = x.dtype
            device = x.device
            limit = min(n_items, i + _KMAX)
            j = i + 1
            while j < limit:
                y = items[j]
                if (
                    y.dtype not in _PLAIN_DTYPES
                    or y.numel() == 0
                    or not y.is_contiguous()
                    or y.dtype != dtype
                    or y.device != device
                ):
                    break
                j += 1
            for k in range(i, j):
                outs[k] = torch.empty_like(items[k])
            is_float = dtype in _FLOAT_DTYPES
            if j == i + 1:
                _launch_flat(x, outs[i], x.numel(), is_float)
            else:
                _launch_fused(items[i:j], outs[i:j], is_float)
            i = j
        else:
            outs[i] = _abs_one(x)
            i += 1
    return outs
