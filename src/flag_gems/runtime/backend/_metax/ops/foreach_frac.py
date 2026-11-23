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

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8
_N_MAX = 16
_FUSE_MAX_BYTES = 256 * 1024 * 1024
_I64_LIMIT = 1 << 31
_TINY_ELEMS = 1 << 16
_TINY_BLOCK = 512

_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def _frac_chunk(
    x_ptr,
    y_ptr,
    n,
    start,
    pid,
    UPCAST: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if I64:
        offs = (pid - start).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = (pid - start) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if UPCAST:
        xf = x.to(tl.float32)
        t = tl.where(xf >= 0.0, tl.floor(xf), tl.ceil(xf))
        y = (xf - t).to(x.dtype)
    else:
        y = x - tl.where(x >= 0.0, tl.floor(x), tl.ceil(x))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _frac_contig_kernel(
    x_ptr, y_ptr, n, UPCAST: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if UPCAST:
        xf = x.to(tl.float32)
        t = tl.where(xf >= 0.0, tl.floor(xf), tl.ceil(xf))
        y = (xf - t).to(x.dtype)
    else:
        t = tl.where(x >= 0.0, tl.floor(x), tl.ceil(x))
        y = x - t
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _frac_fused16_kernel(
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
    x4,
    y4,
    n4,
    x5,
    y5,
    n5,
    x6,
    y6,
    n6,
    x7,
    y7,
    n7,
    x8,
    y8,
    n8,
    x9,
    y9,
    n9,
    x10,
    y10,
    n10,
    x11,
    y11,
    n11,
    x12,
    y12,
    n12,
    x13,
    y13,
    n13,
    x14,
    y14,
    n14,
    x15,
    y15,
    n15,
    UPCAST: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = (n0 + BLOCK - 1) // BLOCK
    if pid < b:
        _frac_chunk(x0, y0, n0, 0, pid, UPCAST, I64, BLOCK)
    else:
        b1 = b + (n1 + BLOCK - 1) // BLOCK
        if pid < b1:
            _frac_chunk(x1, y1, n1, b, pid, UPCAST, I64, BLOCK)
        else:
            b = b1 + (n2 + BLOCK - 1) // BLOCK
            if pid < b:
                _frac_chunk(x2, y2, n2, b1, pid, UPCAST, I64, BLOCK)
            else:
                b1 = b + (n3 + BLOCK - 1) // BLOCK
                if pid < b1:
                    _frac_chunk(x3, y3, n3, b, pid, UPCAST, I64, BLOCK)
                else:
                    b = b1 + (n4 + BLOCK - 1) // BLOCK
                    if pid < b:
                        _frac_chunk(x4, y4, n4, b1, pid, UPCAST, I64, BLOCK)
                    else:
                        b1 = b + (n5 + BLOCK - 1) // BLOCK
                        if pid < b1:
                            _frac_chunk(x5, y5, n5, b, pid, UPCAST, I64, BLOCK)
                        else:
                            b = b1 + (n6 + BLOCK - 1) // BLOCK
                            if pid < b:
                                _frac_chunk(x6, y6, n6, b1, pid, UPCAST, I64, BLOCK)
                            else:
                                b1 = b + (n7 + BLOCK - 1) // BLOCK
                                if pid < b1:
                                    _frac_chunk(x7, y7, n7, b, pid, UPCAST, I64, BLOCK)
                                else:
                                    b = b1 + (n8 + BLOCK - 1) // BLOCK
                                    if pid < b:
                                        _frac_chunk(
                                            x8, y8, n8, b1, pid, UPCAST, I64, BLOCK
                                        )
                                    else:
                                        b1 = b + (n9 + BLOCK - 1) // BLOCK
                                        if pid < b1:
                                            _frac_chunk(
                                                x9, y9, n9, b, pid, UPCAST, I64, BLOCK
                                            )
                                        else:
                                            b = b1 + (n10 + BLOCK - 1) // BLOCK
                                            if pid < b:
                                                _frac_chunk(
                                                    x10,
                                                    y10,
                                                    n10,
                                                    b1,
                                                    pid,
                                                    UPCAST,
                                                    I64,
                                                    BLOCK,
                                                )
                                            else:
                                                b1 = b + (n11 + BLOCK - 1) // BLOCK
                                                if pid < b1:
                                                    _frac_chunk(
                                                        x11,
                                                        y11,
                                                        n11,
                                                        b,
                                                        pid,
                                                        UPCAST,
                                                        I64,
                                                        BLOCK,
                                                    )
                                                else:
                                                    b = b1 + (n12 + BLOCK - 1) // BLOCK
                                                    if pid < b:
                                                        _frac_chunk(
                                                            x12,
                                                            y12,
                                                            n12,
                                                            b1,
                                                            pid,
                                                            UPCAST,
                                                            I64,
                                                            BLOCK,
                                                        )
                                                    else:
                                                        b1 = (
                                                            b
                                                            + (n13 + BLOCK - 1) // BLOCK
                                                        )
                                                        if pid < b1:
                                                            _frac_chunk(
                                                                x13,
                                                                y13,
                                                                n13,
                                                                b,
                                                                pid,
                                                                UPCAST,
                                                                I64,
                                                                BLOCK,
                                                            )
                                                        else:
                                                            b = (
                                                                b1
                                                                + (n14 + BLOCK - 1)
                                                                // BLOCK
                                                            )
                                                            if pid < b:
                                                                _frac_chunk(
                                                                    x14,
                                                                    y14,
                                                                    n14,
                                                                    b1,
                                                                    pid,
                                                                    UPCAST,
                                                                    I64,
                                                                    BLOCK,
                                                                )
                                                            else:
                                                                b1 = (
                                                                    b
                                                                    + (n15 + BLOCK - 1)
                                                                    // BLOCK
                                                                )
                                                                if pid < b1:
                                                                    _frac_chunk(
                                                                        x15,
                                                                        y15,
                                                                        n15,
                                                                        b,
                                                                        pid,
                                                                        UPCAST,
                                                                        I64,
                                                                        BLOCK,
                                                                    )


@triton.jit
def _frac_strided_kernel(
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
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    ys0,
    ys1,
    ys2,
    ys3,
    ys4,
    ys5,
    ys6,
    ys7,
    UPCAST: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    xoff = tl.zeros([BLOCK], dtype=tl.int64)
    yoff = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        r = rem % s0
        rem = rem // s0
        xoff += r * xs0
        yoff += r * ys0
    if RANK > 1:
        r = rem % s1
        rem = rem // s1
        xoff += r * xs1
        yoff += r * ys1
    if RANK > 2:
        r = rem % s2
        rem = rem // s2
        xoff += r * xs2
        yoff += r * ys2
    if RANK > 3:
        r = rem % s3
        rem = rem // s3
        xoff += r * xs3
        yoff += r * ys3
    if RANK > 4:
        r = rem % s4
        rem = rem // s4
        xoff += r * xs4
        yoff += r * ys4
    if RANK > 5:
        r = rem % s5
        rem = rem // s5
        xoff += r * xs5
        yoff += r * ys5
    if RANK > 6:
        r = rem % s6
        rem = rem // s6
        xoff += r * xs6
        yoff += r * ys6
    if RANK > 7:
        r = rem % s7
        rem = rem // s7
        xoff += r * xs7
        yoff += r * ys7
    x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
    if UPCAST:
        xf = x.to(tl.float32)
        t = tl.where(xf >= 0.0, tl.floor(xf), tl.ceil(xf))
        y = (xf - t).to(x.dtype)
    else:
        t = tl.where(x >= 0.0, tl.floor(x), tl.ceil(x))
        y = x - t
    tl.store(y_ptr + yoff, y, mask=mask)


def _upcast(dtype):
    return dtype in (torch.float16, torch.bfloat16)


def _block_for(dtype, total=0):
    if total <= _TINY_ELEMS:
        return _TINY_BLOCK
    return 2048 if dtype in (torch.float16, torch.bfloat16) else _BLOCK


def _launch_contig(t, out):
    n = t.numel()
    if n == 0:
        return
    block = _block_for(t.dtype, n)
    grid = (triton.cdiv(n, block),)
    _frac_contig_kernel[grid](
        t,
        out,
        n,
        UPCAST=_upcast(t.dtype),
        I64=n >= _I64_LIMIT,
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )


def _launch_fused(ts, os, dtype):
    ns = [t.numel() for t in ts]
    total_elems = 0
    for v in ns:
        total_elems += v
    block = _block_for(dtype, total_elems)
    total = 0
    i64 = False
    for v in ns:
        total += triton.cdiv(v, block)
        if v >= _I64_LIMIT:
            i64 = True
    if total == 0:
        return
    args = []
    for t, o, v in zip(ts, os, ns):
        args.append(t)
        args.append(o)
        args.append(v)
    pad = _N_MAX - len(ts)
    if pad > 0:
        args.extend([ts[0], os[0], 0] * pad)
    _frac_fused16_kernel[(total,)](
        *args,
        UPCAST=_upcast(dtype),
        I64=i64,
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )


def _launch_strided(t, out):
    shape = t.shape
    ist = t.stride()
    ost = out.stride()
    dims = sorted(
        (i for i in range(len(shape)) if shape[i] != 1),
        key=lambda i: abs(ist[i]),
    )
    rank = len(dims)
    sz = [1] * _MAX_RANK
    xs = [0] * _MAX_RANK
    ys = [0] * _MAX_RANK
    for k, i in enumerate(dims):
        sz[k] = shape[i]
        xs[k] = ist[i]
        ys[k] = ost[i]
    n = t.numel()
    block = _block_for(t.dtype, n)
    grid = (triton.cdiv(n, block),)
    _frac_strided_kernel[grid](
        t,
        out,
        n,
        sz[0],
        sz[1],
        sz[2],
        sz[3],
        sz[4],
        sz[5],
        sz[6],
        sz[7],
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        ys[0],
        ys[1],
        ys[2],
        ys[3],
        ys[4],
        ys[5],
        ys[6],
        ys[7],
        UPCAST=_upcast(t.dtype),
        RANK=rank,
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )


def foreach_frac(self):
    if isinstance(self, torch.Tensor):
        raise TypeError(
            "_foreach_frac(): argument 'self' (position 1) must be tuple of Tensors, not Tensor"
        )
    if isinstance(self, (list, tuple)):
        items = list(self)
    else:
        raise TypeError(
            "_foreach_frac(): argument 'self' (position 1) must be tuple of Tensors, "
            f"not {type(self).__name__}"
        )
    if len(items) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in items:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "_foreach_frac(): argument 'self' (position 1) must be tuple of Tensors, "
                f"not {type(t).__name__}"
            )
        if t.dtype not in _FLOAT_DTYPES:
            name = _DTYPE_NAMES.get(t.dtype, str(t.dtype))
            raise NotImplementedError(f"\"frac_cuda\" not implemented for '{name}'")

    outs = [torch.empty_like(t) for t in items]
    contig = [t.is_contiguous() and o.is_contiguous() for t, o in zip(items, outs)]

    n_items = len(items)
    i = 0
    while i < n_items:
        if not contig[i]:
            _launch_strided(items[i], outs[i])
            i += 1
            continue
        t0 = items[i]
        j = i + 1
        chunk_numel = t0.numel()
        while (
            j < n_items
            and j - i < _N_MAX
            and contig[j]
            and items[j].dtype == t0.dtype
            and items[j].device == t0.device
        ):
            chunk_numel += items[j].numel()
            j += 1
        if j - i == 1:
            if t0.numel() > 0:
                _launch_contig(t0, outs[i])
        elif chunk_numel * t0.element_size() <= _FUSE_MAX_BYTES:
            _launch_fused(items[i:j], outs[i:j], t0.dtype)
        else:
            for k in range(i, j):
                _launch_contig(items[k], outs[k])
        i = j
    return outs
