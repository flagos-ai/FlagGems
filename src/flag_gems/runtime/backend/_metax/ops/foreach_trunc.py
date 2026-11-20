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

"""Triton implementation of aten::_foreach_trunc.

Semantics (verified against the target's ATen reference):
  * floating dtypes (fp16/bf16/fp32/fp64): elementwise trunc, exact for -0.0, NaN, +-inf
  * integer dtypes: identity (fresh storage, value/dtype/shape preserved)
  * bool -> NotImplementedError('"trunc_cuda" not implemented for ...')
  * complex -> NotImplementedError('"foreach_unary_op_cuda" not implemented for ...')
  * empty tensor list -> RuntimeError("Tensor list must have at least one tensor.")
  * one freshly allocated output per input, output strides follow torch.empty_like

Launch strategy: a dtype group whose members are contiguous, share one shape and
fit one int32 index space is served by ONE launch. The 16 input pointers are
kernel ARGUMENTS selected per program with a static where chain (they keep their
compile-time alignment, unlike a device pointer table), and the outputs are
slices of a single freshly allocated buffer, so only one base pointer is needed.
Every other case falls back to one launch per tensor (flat kernel when
contiguous, rank-collapsed strided kernel otherwise).

The single-tensor kernels are split by value kind so that the hot launch carries
only three runtime arguments and no constexpr keywords: the per-call host cost
is now the dominant term on small and mid-size workloads.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024  # flat kernel tile (module constant, no launch kwarg)
_BLOCK2 = 4096  # 2-byte flat tile: 16B per lane instead of 8B at 1024
_MULTI_BLOCK = 4096  # fused tile: amortizes the pointer-selection chain
_MULTI_WARPS = 8
_MAX_RANK = 8
_I32_LIMIT = 2**31 - 1

# Triton kernels may only read globals that are tl.constexpr instances.
_TL_BLOCK = tl.constexpr(_BLOCK)
_TL_BLOCK2 = tl.constexpr(_BLOCK2)
_TL_MULTI_BLOCK = tl.constexpr(_MULTI_BLOCK)

_KIND_FLOAT = 0  # fp32: convert through fp32
_KIND_F64 = 1  # exact fp64 select
_KIND_INT = 2  # value identity
_KIND_F16 = 3  # fp16 / bf16: wider tile

_KIND = {
    torch.float16: _KIND_F16,
    torch.bfloat16: _KIND_F16,
    torch.float32: _KIND_FLOAT,
    torch.float64: _KIND_F64,
    torch.int8: _KIND_INT,
    torch.uint8: _KIND_INT,
    torch.int16: _KIND_INT,
    torch.int32: _KIND_INT,
    torch.int64: _KIND_INT,
}


@triton.jit
def _trunc_flat_kernel(x_ptr, o_ptr, n):
    pid = tl.program_id(0)
    offs = pid * _TL_BLOCK + tl.arange(0, _TL_BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # this backend's floor/ceil only accept fp32/fp64; the fp32 round trip is
    # exact for fp16/bf16 because every result is representable
    xf = x.to(tl.float32)
    y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(x.dtype)
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _trunc_flat2_kernel(x_ptr, o_ptr, n):
    pid = tl.program_id(0)
    offs = pid * _TL_BLOCK2 + tl.arange(0, _TL_BLOCK2)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(x.dtype)
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _trunc_flat_f64_kernel(x_ptr, o_ptr, n):
    pid = tl.program_id(0)
    offs = pid * _TL_BLOCK + tl.arange(0, _TL_BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _trunc_flat_int_kernel(x_ptr, o_ptr, n):
    pid = tl.program_id(0)
    offs = pid * _TL_BLOCK + tl.arange(0, _TL_BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x, mask=mask)


@triton.jit
def _trunc_flat_wide_kernel(x_ptr, o_ptr, n, IS_FLOAT: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * _TL_BLOCK + tl.arange(0, _TL_BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if IS_FLOAT:
        xf = x.to(tl.float32)
        y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(x.dtype)
    else:
        y = x
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _trunc_multi_kernel(
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
    ob,
    n,
    IS_FLOAT: tl.constexpr,
    EXACT64: tl.constexpr,
    NT: tl.constexpr,
):
    """One launch for NT contiguous tensors that all share one shape.

    Output of tensor t is slice t of the freshly allocated buffer `ob`
    (n elements apart), so only that buffer's base pointer is needed.
    """
    tid = tl.program_id(1)
    pid = tl.program_id(0)
    xp = x0
    if NT > 1:
        xp = tl.where(tid == 1, x1, xp)
    if NT > 2:
        xp = tl.where(tid == 2, x2, xp)
        xp = tl.where(tid == 3, x3, xp)
    if NT > 4:
        xp = tl.where(tid == 4, x4, xp)
        xp = tl.where(tid == 5, x5, xp)
        xp = tl.where(tid == 6, x6, xp)
        xp = tl.where(tid == 7, x7, xp)
    if NT > 8:
        xp = tl.where(tid == 8, x8, xp)
        xp = tl.where(tid == 9, x9, xp)
        xp = tl.where(tid == 10, x10, xp)
        xp = tl.where(tid == 11, x11, xp)
        xp = tl.where(tid == 12, x12, xp)
        xp = tl.where(tid == 13, x13, xp)
        xp = tl.where(tid == 14, x14, xp)
        xp = tl.where(tid == 15, x15, xp)
    offs = pid * _TL_MULTI_BLOCK + tl.arange(0, _TL_MULTI_BLOCK)
    mask = offs < n
    x = tl.load(xp + offs, mask=mask)
    if IS_FLOAT:
        if EXACT64:
            y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
        else:
            xf = x.to(tl.float32)
            y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(x.dtype)
    else:
        y = x
    tl.store(ob + tid * n + offs, y, mask=mask)


@triton.jit
def _ravel(
    rem,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    RANK: tl.constexpr,
):
    """Linear index -> storage offset for dims enumerated innermost first."""
    off = (rem % d0) * s0
    rem = rem // d0
    if RANK > 1:
        off += (rem % d1) * s1
        rem = rem // d1
    if RANK > 2:
        off += (rem % d2) * s2
        rem = rem // d2
    if RANK > 3:
        off += (rem % d3) * s3
        rem = rem // d3
    if RANK > 4:
        off += (rem % d4) * s4
        rem = rem // d4
    if RANK > 5:
        off += (rem % d5) * s5
        rem = rem // d5
    if RANK > 6:
        off += (rem % d6) * s6
        rem = rem // d6
    if RANK > 7:
        off += (rem % d7) * s7
        rem = rem // d7
    return off


@triton.jit(
    do_not_specialize=[
        "n",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "xs0",
        "xs1",
        "xs2",
        "xs3",
        "xs4",
        "xs5",
        "xs6",
        "xs7",
        "os0",
        "os1",
        "os2",
        "os3",
        "os4",
        "os5",
        "os6",
        "os7",
    ]
)
def _trunc_strided_kernel(
    x_ptr,
    o_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    RANK: tl.constexpr,
    SAME: tl.constexpr,
    EXACT64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xo = _ravel(
        offs,
        d0,
        d1,
        d2,
        d3,
        d4,
        d5,
        d6,
        d7,
        xs0,
        xs1,
        xs2,
        xs3,
        xs4,
        xs5,
        xs6,
        xs7,
        RANK,
    )
    if SAME:
        oo = xo
    else:
        oo = _ravel(
            offs,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            os0,
            os1,
            os2,
            os3,
            os4,
            os5,
            os6,
            os7,
            RANK,
        )
    x = tl.load(x_ptr + xo, mask=mask)
    if IS_FLOAT:
        if EXACT64:
            y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
        else:
            xf = x.to(tl.float32)
            y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(x.dtype)
    else:
        y = x
    tl.store(o_ptr + oo, y, mask=mask)


def _collapse_pair(shape, si, so):
    """Merge adjacent dims only when BOTH layouts are contiguous across them."""
    dims = []
    xi = []
    xo = []
    for s, a, b in zip(shape, si, so):
        if s == 1:
            continue
        if dims and xi[-1] == s * a and xo[-1] == s * b:
            dims[-1] *= s
            xi[-1] = a
            xo[-1] = b
        else:
            dims.append(s)
            xi.append(a)
            xo.append(b)
    return dims, xi, xo


_UNSUPPORTED_NAMES = {
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _raise_unsupported(dtype):
    if dtype is torch.bool:
        raise NotImplementedError("\"trunc_cuda\" not implemented for 'Bool'")
    raise NotImplementedError(
        "\"foreach_unary_op_cuda\" not implemented for '%s'"
        % _UNSUPPORTED_NAMES.get(dtype, str(dtype))
    )


def _launch_flat(t, o, n, kind):
    """Contiguous single-tensor launch: three runtime args, no constexpr kwargs."""
    if kind == _KIND_F16 and n <= _I32_LIMIT:
        _trunc_flat2_kernel[((n + _BLOCK2 - 1) // _BLOCK2,)](t, o, n, num_warps=8)
        return
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if n > _I32_LIMIT:
        _trunc_flat_wide_kernel[grid](
            t, o, n, IS_FLOAT=(kind != _KIND_INT), num_warps=4
        )
    elif kind == _KIND_INT:
        _trunc_flat_int_kernel[grid](t, o, n)
    elif kind == _KIND_F64:
        _trunc_flat_f64_kernel[grid](t, o, n)
    else:
        _trunc_flat_kernel[grid](t, o, n)


def _launch_single(t, o, n, kind):
    if t.is_contiguous():
        _launch_flat(t, o, n, kind)
        return
    is_float = kind != _KIND_INT
    exact64 = kind == _KIND_F64
    dims, xs, os_ = _collapse_pair(t.shape, t.stride(), o.stride())
    r = len(dims)
    if r > _MAX_RANK:
        raise RuntimeError("foreach_trunc: unsupported tensor rank %d" % r)
    pad = _MAX_RANK - r
    rd = dims[::-1] + [1] * pad
    rxs = xs[::-1] + [0] * pad
    ros = os_[::-1] + [0] * pad
    _trunc_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t,
        o,
        n,
        rd[0],
        rd[1],
        rd[2],
        rd[3],
        rd[4],
        rd[5],
        rd[6],
        rd[7],
        rxs[0],
        rxs[1],
        rxs[2],
        rxs[3],
        rxs[4],
        rxs[5],
        rxs[6],
        rxs[7],
        ros[0],
        ros[1],
        ros[2],
        ros[3],
        ros[4],
        ros[5],
        ros[6],
        ros[7],
        BLOCK=_BLOCK,
        IS_FLOAT=is_float,
        RANK=r,
        SAME=(xs == os_),
        EXACT64=exact64,
        num_warps=4,
    )


def foreach_trunc(self):
    tensors = self if isinstance(self, (list, tuple)) else [self]
    n_t = len(tensors)
    if n_t == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    if n_t == 1:
        t = tensors[0]
        kind = _KIND.get(t.dtype)
        if kind is None:
            _raise_unsupported(t.dtype)
        o = torch.empty_like(t)
        n = t.numel()
        if n != 0:
            _launch_single(t, o, n, kind)
        return (o,)

    groups = {}
    for i, t in enumerate(tensors):
        dtype = t.dtype
        if dtype in _KIND:
            idxs = groups.get(dtype)
            if idxs is None:
                groups[dtype] = [i]
            else:
                idxs.append(i)
        else:
            _raise_unsupported(dtype)

    outs = [None] * n_t
    for dtype, idxs in groups.items():
        kind = _KIND[dtype]
        if len(idxs) > 1:
            first = tensors[idxs[0]]
            n0 = first.numel()
            shape0 = first.shape
            if 0 < n0 <= (_I32_LIMIT - _MULTI_BLOCK) // len(idxs):
                fuse = True
                for i in idxs:
                    t = tensors[i]
                    if t.shape != shape0 or not t.is_contiguous():
                        fuse = False
                        break
                if fuse:
                    nt = len(idxs)
                    buf = torch.empty(
                        (nt,) + tuple(shape0), dtype=dtype, device=first.device
                    )
                    views = buf.unbind(0)
                    grp = [tensors[i] for i in idxs]
                    for k, i in enumerate(idxs):
                        outs[i] = views[k]
                    tail = grp[0]
                    _trunc_multi_kernel[((n0 + _MULTI_BLOCK - 1) // _MULTI_BLOCK, nt)](
                        grp[0],
                        grp[1] if nt > 1 else tail,
                        grp[2] if nt > 2 else tail,
                        grp[3] if nt > 3 else tail,
                        grp[4] if nt > 4 else tail,
                        grp[5] if nt > 5 else tail,
                        grp[6] if nt > 6 else tail,
                        grp[7] if nt > 7 else tail,
                        grp[8] if nt > 8 else tail,
                        grp[9] if nt > 9 else tail,
                        grp[10] if nt > 10 else tail,
                        grp[11] if nt > 11 else tail,
                        grp[12] if nt > 12 else tail,
                        grp[13] if nt > 13 else tail,
                        grp[14] if nt > 14 else tail,
                        grp[15] if nt > 15 else tail,
                        buf,
                        n0,
                        IS_FLOAT=(kind != _KIND_INT),
                        EXACT64=(kind == _KIND_F64),
                        NT=nt,
                        num_warps=_MULTI_WARPS,
                    )
                    continue
        for i in idxs:
            t = tensors[i]
            o = torch.empty_like(t)
            outs[i] = o
            n = t.numel()
            if n != 0:
                _launch_single(t, o, n, kind)
    return tuple(outs)
