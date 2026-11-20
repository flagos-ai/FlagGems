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

"""Triton implementation of aten::_foreach_sqrt (elementwise sqrt over a tensor list)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_MAX_DIMS = 8
_BLOCK = 1024
_MAX_T = 16

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)
_REAL_OF_COMPLEX = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@triton.jit
def _csqrt_pair(re, im):
    """ATen/thrust complex sqrt: hypot + the C99 division form (bit-exact)."""
    r = libdevice.hypot(re, im)
    pos = re > 0
    t = tl.sqrt((r + re) * 0.5)
    u = tl.sqrt((r - re) * 0.5)
    neg = im < 0
    io_n = tl.where(neg, -u, u)
    ro_n = tl.abs(im) / (2.0 * u)
    ro = tl.where(pos, t, ro_n)
    io = tl.where(pos, im / (2.0 * t), io_n)
    zero = (re == 0) & (im == 0)
    ro = tl.where(zero, re, ro)
    io = tl.where(zero, im, io)
    return ro, io


@triton.jit
def _sqrt_body(
    x_ptr,
    y_ptr,
    pid,
    n,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
    FP64: tl.constexpr,
    CPLX: tl.constexpr,
):
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < n
    if CPLX:
        re = tl.load(x_ptr + 2 * offs, mask=mask)
        im = tl.load(x_ptr + 2 * offs + 1, mask=mask)
        if not FP64:
            re = re.to(tl.float32)
            im = im.to(tl.float32)
        ro, io = _csqrt_pair(re, im)
        tl.store(y_ptr + 2 * offs, ro, mask=mask)
        tl.store(y_ptr + 2 * offs + 1, io, mask=mask)
    else:
        x = tl.load(x_ptr + offs, mask=mask)
        if FP64:
            y = tl.sqrt(x)
        else:
            y = tl.sqrt(x.to(tl.float32))
        tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _sqrt_flat_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
    FP64: tl.constexpr,
    CPLX: tl.constexpr,
):
    _sqrt_body(x_ptr, y_ptr, tl.program_id(0), n, BLOCK, LONG, FP64, CPLX)


@triton.jit
def _sqrt_multi_kernel(
    x0,
    y0,
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    x4,
    y4,
    x5,
    y5,
    x6,
    y6,
    x7,
    y7,
    x8,
    y8,
    x9,
    y9,
    x10,
    y10,
    x11,
    y11,
    x12,
    y12,
    x13,
    y13,
    x14,
    y14,
    x15,
    y15,
    n,
    BLOCK: tl.constexpr,
    LONG: tl.constexpr,
    FP64: tl.constexpr,
    CPLX: tl.constexpr,
):
    """One launch for up to 16 same-dtype, same-length flat tensors.

    grid = (cdiv(n, BLOCK), nt); program_id(1) selects the tensor.
    """
    tx = tl.program_id(1)
    pid = tl.program_id(0)
    if tx == 0:
        xp = x0
        yp = y0
    elif tx == 1:
        xp = x1
        yp = y1
    elif tx == 2:
        xp = x2
        yp = y2
    elif tx == 3:
        xp = x3
        yp = y3
    elif tx == 4:
        xp = x4
        yp = y4
    elif tx == 5:
        xp = x5
        yp = y5
    elif tx == 6:
        xp = x6
        yp = y6
    elif tx == 7:
        xp = x7
        yp = y7
    elif tx == 8:
        xp = x8
        yp = y8
    elif tx == 9:
        xp = x9
        yp = y9
    elif tx == 10:
        xp = x10
        yp = y10
    elif tx == 11:
        xp = x11
        yp = y11
    elif tx == 12:
        xp = x12
        yp = y12
    elif tx == 13:
        xp = x13
        yp = y13
    elif tx == 14:
        xp = x14
        yp = y14
    else:
        xp = x15
        yp = y15
    _sqrt_body(xp, yp, pid, n, BLOCK, LONG, FP64, CPLX)


@triton.jit
def _sqrt_strided_kernel(
    x_ptr,
    y_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
    CPLX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs.to(tl.int64)
    addr = tl.zeros([BLOCK], dtype=tl.int64)
    q = idx % s7
    idx = idx // s7
    addr += q * t7
    q = idx % s6
    idx = idx // s6
    addr += q * t6
    q = idx % s5
    idx = idx // s5
    addr += q * t5
    q = idx % s4
    idx = idx // s4
    addr += q * t4
    q = idx % s3
    idx = idx // s3
    addr += q * t3
    q = idx % s2
    idx = idx // s2
    addr += q * t2
    q = idx % s1
    idx = idx // s1
    addr += q * t1
    q = idx % s0
    addr += q * t0
    if CPLX:
        re = tl.load(x_ptr + 2 * addr, mask=mask)
        im = tl.load(x_ptr + 2 * addr + 1, mask=mask)
        if not FP64:
            re = re.to(tl.float32)
            im = im.to(tl.float32)
        ro, io = _csqrt_pair(re, im)
        tl.store(y_ptr + 2 * offs, ro, mask=mask)
        tl.store(y_ptr + 2 * offs + 1, io, mask=mask)
    else:
        x = tl.load(x_ptr + addr, mask=mask)
        if FP64:
            y = tl.sqrt(x)
        else:
            y = tl.sqrt(x.to(tl.float32))
        tl.store(y_ptr + offs, y, mask=mask)


def _out_dtype(dt):
    if dt in _FLOAT_DTYPES or dt in _COMPLEX_DTYPES:
        return dt
    return torch.float32


def _dense_order(t):
    """Permutation ``order`` such that ``t.permute(order)`` is contiguous, else None."""
    dims = t.dim()
    if dims == 0:
        return None
    sizes = t.shape
    strides = t.stride()
    order = sorted(range(dims), key=lambda i: (sizes[i] == 1, -strides[i]))
    expected = 1
    for i in reversed(order):
        if sizes[i] == 1:
            continue
        if strides[i] != expected:
            return None
        expected *= sizes[i]
    return order


def _inv_order(order):
    inv = [0] * len(order)
    for pos, d in enumerate(order):
        inv[d] = pos
    return inv


def _kernel_args(x, y):
    """Real (interleaved) pointer tensors plus dtype flags."""
    if x.is_complex():
        return torch.view_as_real(x), torch.view_as_real(y), True
    return x, y, False


def _pick_block(esize, n):
    """16 bytes per thread (256-thread CTAs) for every element size.

    16-bit dtypes get wider blocks only on large tensors: on small ones the
    extra bytes per thread would cost CTAs in a launch-bound regime.
    """
    if esize == 2:
        if n <= 8192:
            return 1024
        if n >= (1 << 20):
            return 4096
        return 2048
    if esize == 1:
        return 4096
    return 1024


def _launch_flat(x_ptr, y_ptr, n, cplx):
    block = _pick_block(x_ptr.element_size(), n)
    grid = (triton.cdiv(n, block),)
    _sqrt_flat_kernel[grid](
        x_ptr,
        y_ptr,
        n,
        BLOCK=block,
        LONG=n >= (1 << 31),
        FP64=(x_ptr.dtype == torch.float64),
        CPLX=cplx,
        num_warps=4,
    )


def _launch_multi(entries, n):
    """entries: list of (x_ptr, y_ptr, cplx) already in real-view form."""
    nt = len(entries)
    args = []
    for i in range(_MAX_T):
        entry = entries[i] if i < nt else entries[0]
        args.append(entry[0])
        args.append(entry[1])
    block = _pick_block(entries[0][0].element_size(), n)
    grid = (triton.cdiv(n, block), nt)
    _sqrt_multi_kernel[grid](
        *args,
        n,
        BLOCK=block,
        LONG=n >= (1 << 31),
        FP64=(entries[0][0].dtype == torch.float64),
        CPLX=entries[0][2],
        num_warps=4,
    )


def _launch_strided(t, out):
    n = t.numel()
    x_ptr, y_ptr, cplx = _kernel_args(t, out)
    dims = t.dim()
    sizes = [1] * _MAX_DIMS
    strides = [0] * _MAX_DIMS
    shape = t.shape
    stride = t.stride()
    for i in range(dims):
        sizes[_MAX_DIMS - dims + i] = shape[i]
        strides[_MAX_DIMS - dims + i] = stride[i]
    grid = (triton.cdiv(n, _BLOCK),)
    _sqrt_strided_kernel[grid](
        x_ptr,
        y_ptr,
        n,
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
        BLOCK=_BLOCK,
        FP64=(x_ptr.dtype == torch.float64),
        CPLX=cplx,
        num_warps=4,
    )


def _run_generic(out, t):
    """Per-tensor path for layouts that are not flat-addressable."""
    if t.dim() > _MAX_DIMS:
        for i in range(t.shape[0]):
            _run_generic(out.select(0, i), t.select(0, i))
        return
    _launch_strided(t, out)


def _prepare(t_raw):
    """Allocate the output and return (t, out, x_view, y_view) in flat order."""
    if t_raw.dtype == torch.bool:
        t = t_raw.view(torch.uint8)
    else:
        t = t_raw
    dt = _out_dtype(t_raw.dtype)
    order = None if t.is_contiguous() else _dense_order(t)
    if order is not None and t.numel() > 0:
        tp = t.permute(order)
        buf = torch.empty(t.numel(), dtype=dt, device=t.device)
        out = buf.view(tp.shape).permute(_inv_order(order))
        return t, out, tp, out.permute(order)
    out = torch.empty(t.shape, dtype=dt, device=t.device)
    return t, out, t, out


def foreach_sqrt(self, *args):
    tensors = [self, *args] if isinstance(self, torch.Tensor) else list(self)
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    outs = []
    groups = {}
    for t_raw in tensors:
        t, out, x_view, y_view = _prepare(t_raw)
        outs.append(out)
        n = t.numel()
        if n == 0:
            continue
        if x_view.is_contiguous() and y_view.is_contiguous():
            x_ptr, y_ptr, cplx = _kernel_args(x_view, y_view)
            key = (x_ptr.dtype, y_ptr.dtype, n, cplx)
            groups.setdefault(key, []).append((x_ptr, y_ptr, cplx))
        else:
            _run_generic(out, t)
    for key, entries in groups.items():
        n = key[2]
        if len(entries) == 1:
            _launch_flat(entries[0][0], entries[0][1], n, entries[0][2])
        else:
            for s in range(0, len(entries), _MAX_T):
                _launch_multi(entries[s : s + _MAX_T], n)
    return outs
