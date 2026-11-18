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

"""Triton implementation of aten::_foreach_ceil (Tensor[] -> Tensor[]).

Reference semantics measured on the target (PyTorch 2.10.0+metax):
  * float16 / bfloat16 / float32 / float64 -> same dtype, elementwise ceil,
  * integral dtypes -> same dtype, identity (integers are already integral;
    no promotion happens in this build),
  * bool -> NotImplementedError, complex -> NotImplementedError,
  * empty list -> RuntimeError, bare tensor / non-sequence -> TypeError,
  * the returned container is always a tuple,
  * outputs are freshly allocated and, like empty_like, keep the input strides
    when the input is dense (transpose/permute) and are contiguous otherwise
    (stepped slices, expanded views).

Launch shape:
  * homogeneous list (same dtype/device, every tensor contiguous, >1 non-empty
    tensor): one launch serves up to _K tensors via work-conserving cumulative
    chunk boundaries, or exactly _K2 equal-numel tensors via a 2-D
    (chunk, tensor) grid.
  * otherwise one flat kernel per contiguous tensor, or one stride-decomposed
    kernel per noncontiguous tensor.

Host cost is part of the measured path, so the per-call work is one pass over
the list, a dict-based dtype classification and direct JITFunction.run launches
(no per-call grid-subscript closure, no cdiv call).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_BLOCK_SHIFT = 10  # log2(_BLOCK)
# 16-bit dtypes use twice the tile so every dtype moves 16 B per thread.
_BLOCK_WIDE = 2048
_BLOCK_WIDE_SHIFT = 11
_K = 8  # tensors per boundary-selected grouped launch
_K2 = 16  # tensors per equal-numel 2-D-grid grouped launch
_MAXDIM = 7  # real dims transmitted; the rest of the padded slot is size 1

_CPLX_NAMES = {torch.complex64: "ComplexFloat", torch.complex128: "ComplexDouble"}
_MODE_DIRECT = 1  # fp32 / fp64: libdevice ceil in the native type
_MODE_WIDE = 0  # fp16 / bf16 / fp8: exact via fp32
_MODE_IDENTITY = 2  # integral: already integral

_INFO_DIRECT = (_MODE_DIRECT, _BLOCK, _BLOCK_SHIFT)
_INFO_WIDE = (_MODE_WIDE, _BLOCK_WIDE, _BLOCK_WIDE_SHIFT)
_INFO_IDENTITY = (_MODE_IDENTITY, _BLOCK, _BLOCK_SHIFT)
_INFO = {
    torch.float16: _INFO_WIDE,
    torch.bfloat16: _INFO_WIDE,
    torch.float32: _INFO_DIRECT,
    torch.float64: _INFO_DIRECT,
}
for _fp8 in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
):
    _fp8_dtype = getattr(torch, _fp8, None)
    if _fp8_dtype is not None:
        _INFO[_fp8_dtype] = _INFO_WIDE


@triton.jit
def _apply_ceil(x, MODE: tl.constexpr):
    # Literal modes: Triton kernels may not read module-level globals.
    if MODE == 1:  # fp32 / fp64
        return tl.math.ceil(x)
    elif MODE == 0:  # fp16 / bf16 / fp8: exact through fp32
        return tl.math.ceil(x.to(tl.float32)).to(x.dtype)
    else:  # integral: identity
        return x


@triton.jit
def _ceil_flat_kernel(x_ptr, out_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, _apply_ceil(x, MODE), mask=mask)


@triton.jit
def _group_tile(
    x_ptr, out_ptr, n, local_pid, lane, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    offs = local_pid.to(tl.int64) * BLOCK + lane
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, _apply_ceil(x, MODE), mask=mask)


@triton.jit
def _ceil_group_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    # Number of exclusive chunk boundaries at or below pid == slot index.
    t = (
        (pid >= b1).to(tl.int32)
        + (pid >= b2).to(tl.int32)
        + (pid >= b3).to(tl.int32)
        + (pid >= b4).to(tl.int32)
        + (pid >= b5).to(tl.int32)
        + (pid >= b6).to(tl.int32)
        + (pid >= b7).to(tl.int32)
    )
    if t == 0:
        _group_tile(x0, o0, n0, pid, lane, MODE, BLOCK)
    elif t == 1:
        _group_tile(x1, o1, n1, pid - b1, lane, MODE, BLOCK)
    elif t == 2:
        _group_tile(x2, o2, n2, pid - b2, lane, MODE, BLOCK)
    elif t == 3:
        _group_tile(x3, o3, n3, pid - b3, lane, MODE, BLOCK)
    elif t == 4:
        _group_tile(x4, o4, n4, pid - b4, lane, MODE, BLOCK)
    elif t == 5:
        _group_tile(x5, o5, n5, pid - b5, lane, MODE, BLOCK)
    elif t == 6:
        _group_tile(x6, o6, n6, pid - b6, lane, MODE, BLOCK)
    else:
        _group_tile(x7, o7, n7, pid - b7, lane, MODE, BLOCK)


@triton.jit
def _ceil_group16_kernel(
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
    n,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Equal-numel lists: the tensor index is the second grid axis, so one
    # launch serves all 16 tensors with no boundaries and only one size.
    t = tl.program_id(1)
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    if t == 0:
        _group_tile(x0, o0, n, pid, lane, MODE, BLOCK)
    elif t == 1:
        _group_tile(x1, o1, n, pid, lane, MODE, BLOCK)
    elif t == 2:
        _group_tile(x2, o2, n, pid, lane, MODE, BLOCK)
    elif t == 3:
        _group_tile(x3, o3, n, pid, lane, MODE, BLOCK)
    elif t == 4:
        _group_tile(x4, o4, n, pid, lane, MODE, BLOCK)
    elif t == 5:
        _group_tile(x5, o5, n, pid, lane, MODE, BLOCK)
    elif t == 6:
        _group_tile(x6, o6, n, pid, lane, MODE, BLOCK)
    elif t == 7:
        _group_tile(x7, o7, n, pid, lane, MODE, BLOCK)
    elif t == 8:
        _group_tile(x8, o8, n, pid, lane, MODE, BLOCK)
    elif t == 9:
        _group_tile(x9, o9, n, pid, lane, MODE, BLOCK)
    elif t == 10:
        _group_tile(x10, o10, n, pid, lane, MODE, BLOCK)
    elif t == 11:
        _group_tile(x11, o11, n, pid, lane, MODE, BLOCK)
    elif t == 12:
        _group_tile(x12, o12, n, pid, lane, MODE, BLOCK)
    elif t == 13:
        _group_tile(x13, o13, n, pid, lane, MODE, BLOCK)
    elif t == 14:
        _group_tile(x14, o14, n, pid, lane, MODE, BLOCK)
    else:
        _group_tile(x15, o15, n, pid, lane, MODE, BLOCK)


@triton.jit
def _ceil_strided_kernel(
    x_ptr,
    out_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    MODE: tl.constexpr,
    SAME: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    i0 = offs % d0
    r = offs // d0
    i1 = r % d1
    r = r // d1
    i2 = r % d2
    r = r // d2
    i3 = r % d3
    r = r // d3
    i4 = r % d4
    r = r // d4
    i5 = r % d5
    r = r // d5
    i6 = r % d6

    off_in = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5 + i6 * s6
    if SAME:
        off_out = off_in
    else:
        off_out = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5 + i6 * t6
    x = tl.load(x_ptr + off_in, mask=mask)
    tl.store(out_ptr + off_out, _apply_ceil(x, MODE), mask=mask)


def _merge_dims(t, o):
    """(size, in_stride, out_stride) with adjacent dims merged when legal.

    Innermost (smallest nonzero |stride|) first so the fastest varying flat
    index walks the most compact dimension; zero-stride dims go outermost.
    """
    merged = []
    for d in range(t.dim()):
        size = t.shape[d]
        sin = t.stride(d)
        sout = o.stride(d)
        if merged:
            psize, psin, psout = merged[-1]
            if psin == size * sin and psout == size * sout:
                merged[-1] = (psize * size, sin, sout)
                continue
        merged.append((size, sin, sout))
    return [e for e in sorted(merged, key=lambda e: (e[1] == 0, abs(e[1])))]


def _launch_one(t, o, n, mode, blk, shift):
    if t.is_contiguous():
        _ceil_flat_kernel[((n + blk - 1) >> shift,)](t, o, n, MODE=mode, BLOCK=blk)
        return
    dims = _merge_dims(t, o)
    if len(dims) > _MAXDIM:
        raise RuntimeError("_foreach_ceil: too many dimensions")
    shapes = [1] * _MAXDIM
    sin = [0] * _MAXDIM
    sout = [0] * _MAXDIM
    same = True
    for k, (size, a, b) in enumerate(dims):
        shapes[k] = size
        sin[k] = a
        sout[k] = b
        if a != b:
            same = False
    _ceil_strided_kernel[((n + _BLOCK - 1) >> _BLOCK_SHIFT,)](
        t,
        o,
        n,
        *shapes,
        *sin,
        *sout,
        MODE=mode,
        SAME=same,
        BLOCK=_BLOCK,
    )


def _launch_group(xs, ys, ns, start, cnt, mode, blk, shift):
    stop = start + _K
    if stop > cnt:
        stop = cnt
    xp = xs[start:stop]
    yp = ys[start:stop]
    nv = ns[start:stop]
    k = stop - start
    if k < _K:
        xp += [xp[0]] * (_K - k)
        yp += [yp[0]] * (_K - k)
        nv += [0] * (_K - k)
    bounds = []
    c = 0
    for i in range(_K - 1):
        c += (nv[i] + blk - 1) >> shift
        bounds.append(c)
    total = c + ((nv[_K - 1] + blk - 1) >> shift)
    _ceil_group_kernel[(total,)](*xp, *yp, *nv, *bounds, MODE=mode, BLOCK=blk)


def foreach_ceil(self):
    tensors = self
    if not isinstance(tensors, (list, tuple)) or isinstance(tensors, torch.Tensor):
        raise TypeError(
            "_foreach_ceil(): argument 'self' (position 1) must be tuple of "
            "Tensors, not Tensor"
        )
    nitems = len(tensors)
    if nitems == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    infos = []
    numels = []
    dt0 = None
    dev0 = None
    homogeneous = nitems > 1
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "_foreach_ceil(): argument 'self' (position 1) must be tuple of "
                "Tensors, not %s" % type(t).__name__
            )
        dt = t.dtype
        info = _INFO.get(dt)
        if info is None:
            name = _CPLX_NAMES.get(dt)
            if name is not None:
                raise NotImplementedError(
                    "\"foreach_unary_op_cuda\" not implemented for '%s'" % name
                )
            if dt is torch.bool:
                raise NotImplementedError("\"ceil_cuda\" not implemented for 'Bool'")
            info = _INFO_WIDE if dt.is_floating_point else _INFO_IDENTITY
        infos.append(info)
        numels.append(t.numel())
        if dt0 is None:
            dt0 = dt
            dev0 = t.device
        elif homogeneous and (
            dt is not dt0 or t.device != dev0 or not t.is_contiguous()
        ):
            homogeneous = False

    if nitems == 1:
        out = torch.empty_like(tensors[0])
        n = numels[0]
        if n:
            mode, blk, shift = infos[0]
            _launch_one(tensors[0], out, n, mode, blk, shift)
        return (out,)

    outs = [torch.empty_like(t) for t in tensors]

    if homogeneous:
        xs = []
        ys = []
        ns = []
        for i in range(nitems):
            n = numels[i]
            if n:
                xs.append(tensors[i])
                ys.append(outs[i])
                ns.append(n)
        cnt = len(xs)
        if cnt > 1:
            mode, blk, shift = infos[0]
            if cnt == _K2 and min(ns) == max(ns):
                n = ns[0]
                _ceil_group16_kernel[((n + blk - 1) >> shift, _K2)](
                    *xs, *ys, n, MODE=mode, BLOCK=blk
                )
                return tuple(outs)
            for start in range(0, cnt, _K):
                _launch_group(xs, ys, ns, start, cnt, mode, blk, shift)
            return tuple(outs)

    for i in range(nitems):
        n = numels[i]
        if n:
            mode, blk, shift = infos[i]
            _launch_one(tensors[i], outs[i], n, mode, blk, shift)
    return tuple(outs)
