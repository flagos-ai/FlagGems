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

"""Triton implementation of aten::_foreach_sinh (elementwise sinh over a tensor list)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_MAX_RANK = 8
_MAX_FUSE = 16
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)

# Bounded, explicit search set for the streaming elementwise kernels (no timing in run()).
_FLAT_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=1),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
]


@triton.jit
def _sinh_value(x, IS_F64: tl.constexpr):
    """sinh with ATen-compatible opmath (fp32 for f16/bf16/f32, fp64 stays fp64)."""
    if IS_F64:
        y = libdevice.sinh(x)
    else:
        xf = x.to(tl.float32)
        y = 0.5 * (tl.exp(xf) - tl.exp(-xf))
    return y


@triton.jit
def _sinh_block(x_ptr, o_ptr, pid, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = _sinh_value(x, IS_F64)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_FLAT_CONFIGS, key=["n"])
@triton.jit
def _sinh_flat_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr):
    pid = tl.program_id(0)
    _sinh_block(x_ptr, o_ptr, pid, n, BLOCK, IS_F64)


@triton.autotune(configs=_FLAT_CONFIGS, key=["n0"])
@triton.jit
def _sinh_fused_kernel(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    x4,
    o4,
    x5,
    o5,
    x6,
    o6,
    x7,
    o7,
    x8,
    o8,
    x9,
    o9,
    x10,
    o10,
    x11,
    o11,
    x12,
    o12,
    x13,
    o13,
    x14,
    o14,
    x15,
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
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    """One launch for a whole list: grid axis 1 selects the tensor, axis 0 the block."""
    pid = tl.program_id(0)
    t = tl.program_id(1)
    if t == 0:
        _sinh_block(x0, o0, pid, n0, BLOCK, IS_F64)
    elif t == 1:
        _sinh_block(x1, o1, pid, n1, BLOCK, IS_F64)
    elif t == 2:
        _sinh_block(x2, o2, pid, n2, BLOCK, IS_F64)
    elif t == 3:
        _sinh_block(x3, o3, pid, n3, BLOCK, IS_F64)
    elif t == 4:
        _sinh_block(x4, o4, pid, n4, BLOCK, IS_F64)
    elif t == 5:
        _sinh_block(x5, o5, pid, n5, BLOCK, IS_F64)
    elif t == 6:
        _sinh_block(x6, o6, pid, n6, BLOCK, IS_F64)
    elif t == 7:
        _sinh_block(x7, o7, pid, n7, BLOCK, IS_F64)
    elif t == 8:
        _sinh_block(x8, o8, pid, n8, BLOCK, IS_F64)
    elif t == 9:
        _sinh_block(x9, o9, pid, n9, BLOCK, IS_F64)
    elif t == 10:
        _sinh_block(x10, o10, pid, n10, BLOCK, IS_F64)
    elif t == 11:
        _sinh_block(x11, o11, pid, n11, BLOCK, IS_F64)
    elif t == 12:
        _sinh_block(x12, o12, pid, n12, BLOCK, IS_F64)
    elif t == 13:
        _sinh_block(x13, o13, pid, n13, BLOCK, IS_F64)
    elif t == 14:
        _sinh_block(x14, o14, pid, n14, BLOCK, IS_F64)
    else:
        _sinh_block(x15, o15, pid, n15, BLOCK, IS_F64)


@triton.jit
def _sinh_strided_kernel(
    x_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    IS_F64: tl.constexpr,
    COMPLEX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    xo = tl.zeros((BLOCK,), dtype=tl.int64)
    oo = tl.zeros((BLOCK,), dtype=tl.int64)
    if RANK > 0:
        i0 = (offs // d0) % s0
        xo += i0 * a0
        oo += i0 * b0
    if RANK > 1:
        i1 = (offs // d1) % s1
        xo += i1 * a1
        oo += i1 * b1
    if RANK > 2:
        i2 = (offs // d2) % s2
        xo += i2 * a2
        oo += i2 * b2
    if RANK > 3:
        i3 = (offs // d3) % s3
        xo += i3 * a3
        oo += i3 * b3
    if RANK > 4:
        i4 = (offs // d4) % s4
        xo += i4 * a4
        oo += i4 * b4
    if RANK > 5:
        i5 = (offs // d5) % s5
        xo += i5 * a5
        oo += i5 * b5
    if RANK > 6:
        i6 = (offs // d6) % s6
        xo += i6 * a6
        oo += i6 * b6
    if RANK > 7:
        i7 = (offs // d7) % s7
        xo += i7 * a7
        oo += i7 * b7
    if COMPLEX:
        # x_ptr/o_ptr are real views; xo/oo address the real component
        re = tl.load(x_ptr + xo, mask=mask)
        im = tl.load(x_ptr + xo + 1, mask=mask)
        if IS_F64:
            a = re
            b = im
        else:
            a = re.to(tl.float32)
            b = im.to(tl.float32)
        ea = libdevice.exp(a)
        ena = libdevice.exp(-a)
        sh = 0.5 * (ea - ena)
        ch = 0.5 * (ea + ena)
        cb = libdevice.cos(b)
        sb = libdevice.sin(b)
        tl.store(o_ptr + oo, (sh * cb).to(o_ptr.dtype.element_ty), mask=mask)
        tl.store(o_ptr + oo + 1, (ch * sb).to(o_ptr.dtype.element_ty), mask=mask)
    else:
        x = tl.load(x_ptr + xo, mask=mask)
        y = _sinh_value(x, IS_F64)
        tl.store(o_ptr + oo, y.to(o_ptr.dtype.element_ty), mask=mask)


def _flat_config(n):
    if n <= 4096:
        return 256, 1
    if n <= 65536:
        return 1024, 4
    if n <= (1 << 20):
        return 2048, 4
    return 4096, 8


def _is_dense_permuted(x):
    """True when x's elements occupy exactly the offsets 0..numel-1 (a pure permutation)."""
    if x.layout != torch.strided:
        return False
    if x.numel() == 0:
        return True
    exp = 1
    for size, stride in sorted(zip(x.shape, x.stride()), key=lambda p: p[1]):
        if size == 1:
            continue
        if stride != exp:
            return False
        exp *= size
    return True


def _collapse(dims):
    """Merge adjacent (size, stride...) tuples when all layouts stay consistent."""
    out = []
    for entry in dims:
        size = entry[0]
        if size == 1:
            continue
        if out:
            prev = out[-1]
            if all(prev[1 + i] == entry[1 + i] * size for i in range(len(entry) - 1)):
                out[-1] = (prev[0] * size,) + prev[1:]
                continue
        out.append(entry)
    return out


def _launch_strided(x, out, n, xstrides, ostrides, shape, is_f64, is_complex):
    dims = _collapse(
        [(size, si, so) for size, si, so in zip(shape, xstrides, ostrides)]
    )
    rank = len(dims)
    sizes = [d[0] for d in dims] + [1] * (_MAX_RANK - rank)
    istr = [d[1] for d in dims] + [0] * (_MAX_RANK - rank)
    ostr = [d[2] for d in dims] + [0] * (_MAX_RANK - rank)
    divs = []
    for i in range(rank):
        acc = 1
        for j in range(i + 1, rank):
            acc *= sizes[j]
        divs.append(acc)
    divs = divs + [1] * (_MAX_RANK - rank)

    block, warps = _flat_config(n)
    _sinh_strided_kernel[(triton.cdiv(n, block),)](
        x,
        out,
        n,
        *sizes,
        *divs,
        *istr,
        *ostr,
        RANK=rank,
        BLOCK=block,
        IS_F64=is_f64,
        COMPLEX=is_complex,
        num_warps=warps,
    )


def _sinh_one(x):
    if x.dtype in _COMPLEX_DTYPES:
        out = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return out
        xr = torch.view_as_real(x)
        orr = torch.view_as_real(out)
        _launch_strided(
            xr,
            orr,
            n,
            xr.stride()[:-1],
            orr.stride()[:-1],
            x.shape,
            x.dtype == torch.complex128,
            True,
        )
        return out

    is_float = x.dtype in _FLOAT_DTYPES
    if is_float:
        out = torch.empty_like(x)
        is_f64 = x.dtype == torch.float64
    else:
        out = torch.empty(x.shape, dtype=torch.get_default_dtype(), device=x.device)
        is_f64 = False

    n = x.numel()
    if n == 0:
        return out

    if x.is_contiguous():
        _sinh_flat_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
            x, out, n, IS_F64=is_f64
        )
        return out

    if out.stride() == x.stride() and _is_dense_permuted(x):
        _sinh_flat_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
            x, out, n, IS_F64=is_f64
        )
        return out

    if not out.is_contiguous():
        out = torch.empty(x.shape, dtype=out.dtype, device=x.device)

    _launch_strided(x, out, n, x.stride(), out.stride(), x.shape, is_f64, False)
    return out


def _sinh_fused(tensors):
    """One launch for the whole list when every tensor is contiguous with one shared dtype."""
    dt = tensors[0].dtype
    if dt in _COMPLEX_DTYPES:
        return None
    dev = tensors[0].device
    for x in tensors:
        if x.dtype != dt or x.device != dev or not x.is_contiguous():
            return None

    is_float = dt in _FLOAT_DTYPES
    is_f64 = dt == torch.float64
    outs = []
    ns = []
    for x in tensors:
        ns.append(x.numel())
        if is_float:
            outs.append(torch.empty_like(x))
        else:
            outs.append(
                torch.empty(x.shape, dtype=torch.get_default_dtype(), device=dev)
            )

    pad_x = None
    pad_o = None
    for x, o, n in zip(tensors, outs, ns):
        if n:
            pad_x = x
            pad_o = o
            break
    if pad_x is None:
        return tuple(outs)

    xs = []
    os_ = []
    padded_ns = []
    for i, x in enumerate(tensors):
        if ns[i]:
            xs.append(x)
            os_.append(outs[i])
            padded_ns.append(ns[i])
        else:
            xs.append(pad_x)
            os_.append(pad_o)
            padded_ns.append(0)
    k = len(xs)
    while len(xs) < _MAX_FUSE:
        xs.append(pad_x)
        os_.append(pad_o)
        padded_ns.append(0)

    max_n = max(ns) if ns else 1
    args = []
    for x, o in zip(xs, os_):
        args.append(x)
        args.append(o)
    args.extend(padded_ns)
    _sinh_fused_kernel[lambda meta: (triton.cdiv(max_n, meta["BLOCK"]), k)](
        *args, IS_F64=is_f64
    )
    return tuple(outs)


def foreach_sinh(self):
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if 1 < len(self) <= _MAX_FUSE:
        fused = _sinh_fused(self)
        if fused is not None:
            return fused
    outs = []
    for x in self:
        outs.append(_sinh_one(x))
    return tuple(outs)
