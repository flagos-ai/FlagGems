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

"""foreach_mul with a scalar list (out-of-place), Triton implementation.

out_i = self_i * scalars_i, element-wise, in fp32 opmath for fp16/bf16 (matching
ATen's opmath_symmetric_gpu_kernel_with_scalars).

Tensors of one call that share (device, dtype, memory-path) are fused into a
single 2D-grid launch: program_id(1) selects the tensor slot, program_id(0) the
element block.  Anything non-uniform falls back to one launch per tensor.
"""

import struct

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NW = 4
_MAX_SLOTS = 8

_M0 = 0  # fp16 / bf16 / fp32 -> fp32 opmath, one rounding on store
_M1 = 1  # fp64, scalar kept at double precision via hi/lo split
_M2 = 2  # integral dtypes, arithmetic in int64 then truncating store


@triton.jit
def _in_off(
    offs,
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
    RANK: tl.constexpr,
):
    """Row-major logical index -> physical element offset for a strided view."""
    rem = offs
    io = tl.zeros_like(offs).to(tl.int64)
    if RANK > 7:
        d = rem % s7
        rem = rem // s7
        io += d.to(tl.int64) * t7
    if RANK > 6:
        d = rem % s6
        rem = rem // s6
        io += d.to(tl.int64) * t6
    if RANK > 5:
        d = rem % s5
        rem = rem // s5
        io += d.to(tl.int64) * t5
    if RANK > 4:
        d = rem % s4
        rem = rem // s4
        io += d.to(tl.int64) * t4
    if RANK > 3:
        d = rem % s3
        rem = rem // s3
        io += d.to(tl.int64) * t3
    if RANK > 2:
        d = rem % s2
        rem = rem // s2
        io += d.to(tl.int64) * t2
    if RANK > 1:
        d = rem % s1
        rem = rem // s1
        io += d.to(tl.int64) * t1
    if RANK > 0:
        io += rem.to(tl.int64) * t0
    return io


@triton.jit
def _flat_kernel(
    x_ptr, o_ptr, n, s_f, s_hi, s_lo, s_i, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=0)
    if MODE == 0:
        y = v.to(tl.float32) * s_f.to(tl.float32)
    elif MODE == 1:
        y = v.to(tl.float64) * (s_hi.to(tl.float64) + s_lo.to(tl.float64))
    else:
        y = v.to(tl.int64) * s_i
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _fused4(
    x0, x1, x2, x3, o0, o1, o2, o3, n0, n1, n2, n3, s0, s1, s2, s3, BLOCK: tl.constexpr
):
    """Four flat fp16/bf16/fp32 multiplies in one launch (lean signature)."""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if j == 0:
        m = offs < n0
        y = tl.load(x0 + offs, mask=m, other=0).to(tl.float32) * s0.to(tl.float32)
        tl.store(o0 + offs, y.to(o0.dtype.element_ty), mask=m)
    if j == 1:
        m = offs < n1
        y = tl.load(x1 + offs, mask=m, other=0).to(tl.float32) * s1.to(tl.float32)
        tl.store(o1 + offs, y.to(o1.dtype.element_ty), mask=m)
    if j == 2:
        m = offs < n2
        y = tl.load(x2 + offs, mask=m, other=0).to(tl.float32) * s2.to(tl.float32)
        tl.store(o2 + offs, y.to(o2.dtype.element_ty), mask=m)
    if j == 3:
        m = offs < n3
        y = tl.load(x3 + offs, mask=m, other=0).to(tl.float32) * s3.to(tl.float32)
        tl.store(o3 + offs, y.to(o3.dtype.element_ty), mask=m)


@triton.jit
def _flat_fused(
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
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Up to 8 independent flat fp16/bf16/fp32 multiplies in one launch."""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if j == 0:
        m = offs < n0
        y = tl.load(x0 + offs, mask=m, other=0).to(tl.float32) * s0.to(tl.float32)
        tl.store(o0 + offs, y.to(o0.dtype.element_ty), mask=m)
    if G > 1:
        if j == 1:
            m = offs < n1
            y = tl.load(x1 + offs, mask=m, other=0).to(tl.float32) * s1.to(tl.float32)
            tl.store(o1 + offs, y.to(o1.dtype.element_ty), mask=m)
    if G > 2:
        if j == 2:
            m = offs < n2
            y = tl.load(x2 + offs, mask=m, other=0).to(tl.float32) * s2.to(tl.float32)
            tl.store(o2 + offs, y.to(o2.dtype.element_ty), mask=m)
    if G > 3:
        if j == 3:
            m = offs < n3
            y = tl.load(x3 + offs, mask=m, other=0).to(tl.float32) * s3.to(tl.float32)
            tl.store(o3 + offs, y.to(o3.dtype.element_ty), mask=m)
    if G > 4:
        if j == 4:
            m = offs < n4
            y = tl.load(x4 + offs, mask=m, other=0).to(tl.float32) * s4.to(tl.float32)
            tl.store(o4 + offs, y.to(o4.dtype.element_ty), mask=m)
    if G > 5:
        if j == 5:
            m = offs < n5
            y = tl.load(x5 + offs, mask=m, other=0).to(tl.float32) * s5.to(tl.float32)
            tl.store(o5 + offs, y.to(o5.dtype.element_ty), mask=m)
    if G > 6:
        if j == 6:
            m = offs < n6
            y = tl.load(x6 + offs, mask=m, other=0).to(tl.float32) * s6.to(tl.float32)
            tl.store(o6 + offs, y.to(o6.dtype.element_ty), mask=m)
    if G > 7:
        if j == 7:
            m = offs < n7
            y = tl.load(x7 + offs, mask=m, other=0).to(tl.float32) * s7.to(tl.float32)
            tl.store(o7 + offs, y.to(o7.dtype.element_ty), mask=m)


@triton.jit
def _strided_kernel(
    x_ptr,
    o_ptr,
    n,
    s_f,
    s_hi,
    s_lo,
    s_i,
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
    MODE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    io = _in_off(
        offs, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7, RANK
    )
    v = tl.load(x_ptr + io, mask=m, other=0)
    if MODE == 0:
        y = v.to(tl.float32) * s_f.to(tl.float32)
    elif MODE == 1:
        y = v.to(tl.float64) * (s_hi.to(tl.float64) + s_lo.to(tl.float64))
    else:
        y = v.to(tl.int64) * s_i
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


_MODE_BY_DTYPE = {
    torch.float16: _M0,
    torch.bfloat16: _M0,
    torch.float32: _M0,
    torch.float64: _M1,
    torch.int8: _M2,
    torch.int16: _M2,
    torch.int32: _M2,
    torch.int64: _M2,
    torch.uint8: _M2,
}


def _mode_of(dtype):
    try:
        return _MODE_BY_DTYPE[dtype]
    except KeyError:
        return -1


def _is_dense(t):
    """True when the covered physical offsets are exactly [0, numel)."""
    if t.numel() == 0:
        return True
    dims = [(st, sz) for st, sz in zip(t.stride(), t.shape) if sz != 1]
    if len(dims) <= 1:
        return True
    dims.sort(key=lambda p: p[0])
    exp = 1
    for st, sz in dims:
        if st != exp:
            return False
        exp *= sz
    return True


def _collapse(t):
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for i in range(len(dims) - 1, -1, -1):
        sz, st = dims[i]
        if out and st == out[-1][1] * out[-1][0]:
            out[-1] = (out[-1][0] * sz, st)
        else:
            out.append((sz, st))
    out.reverse()
    return out


def _scalar_args(s, mode):
    if mode == _M1:
        d = float(s)
        hi = struct.unpack("<f", struct.pack("<f", d))[0]
        return None, hi, d - hi, None
    if mode == _M2:
        return None, None, None, int(s)
    return float(s), None, None, None


def _launch_fused_generic(ts, outs, ns, ss, maxn):
    """Pad a homogeneous flat group to 8 slots and launch once."""
    g = len(ts)
    xs = list(ts)
    yo = list(outs)
    nn = list(ns)
    ssf = list(ss)
    px = xs[0]
    po = yo[0]
    while len(xs) < _MAX_SLOTS:
        xs.append(px)
        yo.append(po)
        nn.append(0)
        ssf.append(0.0)
    _flat_fused[((maxn + _BLOCK - 1) // _BLOCK, g)](
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        yo[0],
        yo[1],
        yo[2],
        yo[3],
        yo[4],
        yo[5],
        yo[6],
        yo[7],
        nn[0],
        nn[1],
        nn[2],
        nn[3],
        nn[4],
        nn[5],
        nn[6],
        nn[7],
        ssf[0],
        ssf[1],
        ssf[2],
        ssf[3],
        ssf[4],
        ssf[5],
        ssf[6],
        ssf[7],
        G=g,
        BLOCK=_BLOCK,
        num_warps=_NW,
    )


def _launch_one(t, o, n, mode, flat, s_f, s_hi, s_lo, s_i):
    grid = (triton.cdiv(n, _BLOCK),)
    if flat:
        _flat_kernel[grid](
            t, o, n, s_f, s_hi, s_lo, s_i, MODE=mode, BLOCK=_BLOCK, num_warps=_NW
        )
        return
    dims = _collapse(t)
    if len(dims) > 8:
        raise NotImplementedError("rank too large")
    shapes = [1] * 8
    strides = [0] * 8
    for k, (sz, st) in enumerate(dims):
        shapes[k] = sz
        strides[k] = st
    _strided_kernel[grid](
        t,
        o,
        n,
        s_f,
        s_hi,
        s_lo,
        s_i,
        shapes[0],
        shapes[1],
        shapes[2],
        shapes[3],
        shapes[4],
        shapes[5],
        shapes[6],
        shapes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        MODE=mode,
        RANK=len(dims),
        BLOCK=_BLOCK,
        num_warps=_NW,
    )


def foreach_mul_scalar_list(self, scalars=None, *args, **kwargs):
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    if isinstance(scalars, (list, tuple)):
        if len(scalars) != len(tensors):
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
        seq = scalars
    else:
        seq = [scalars] * len(tensors)

    L = len(tensors)

    # Fast path: every tensor is contiguous with one shared fp32-opmath dtype
    # and device, so the whole list becomes a single 2D-grid launch.
    if 2 <= L <= _MAX_SLOTS:
        t0 = tensors[0]
        dt = t0.dtype
        if _MODE_BY_DTYPE.get(dt, -1) == _M0 and t0.is_contiguous():
            dev = t0.get_device()
            ns = []
            ss = []
            ok = True
            for i in range(L):
                t = tensors[i]
                if t.dtype is not dt or t.get_device() != dev or not t.is_contiguous():
                    ok = False
                    break
                ns.append(t.numel())
                ss.append(float(seq[i]))
            if ok:
                outs = [torch.empty_like(t) for t in tensors]
                maxn = max(ns)
                if maxn:
                    if L == 4:
                        _fused4[((maxn + _BLOCK - 1) // _BLOCK, 4)](
                            tensors[0],
                            tensors[1],
                            tensors[2],
                            tensors[3],
                            outs[0],
                            outs[1],
                            outs[2],
                            outs[3],
                            ns[0],
                            ns[1],
                            ns[2],
                            ns[3],
                            ss[0],
                            ss[1],
                            ss[2],
                            ss[3],
                            BLOCK=_BLOCK,
                            num_warps=_NW,
                        )
                    else:
                        _launch_fused_generic(tensors, outs, ns, ss, maxn)
                return outs

    recs = []
    for i in range(L):
        t = tensors[i]
        mode = _mode_of(t.dtype)
        if mode < 0:
            raise NotImplementedError("unsupported dtype: %s" % (t.dtype,))
        n = t.numel()
        if mode == _M0:
            s_f = float(seq[i])
            s_hi = s_lo = s_i = None
        else:
            s_f, s_hi, s_lo, s_i = _scalar_args(seq[i], mode)
        if n == 0:
            o = torch.empty_like(t)
            flat = True
        elif t.is_contiguous():
            o = torch.empty_like(t)
            flat = True
        elif _is_dense(t):
            o = torch.empty_like(t)
            flat = o.stride() == t.stride()
            if not flat:
                o = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        else:
            o = torch.empty(t.shape, dtype=t.dtype, device=t.device)
            flat = False
        recs.append((t, o, n, mode, flat, s_f, s_hi, s_lo, s_i))

    if L:
        r0 = recs[0]
        uniform = r0[3] == _M0 and r0[4]
        if uniform:
            t0 = r0[0]
            for j in range(1, L):
                r = recs[j]
                if (
                    (not r[4])
                    or r[3] != _M0
                    or r[0].dtype != t0.dtype
                    or r[0].device != t0.device
                ):
                    uniform = False
                    break
        if uniform and L > 1:
            maxn = 0
            for r in recs:
                if r[2] > maxn:
                    maxn = r[2]
            if maxn:
                if L <= _MAX_SLOTS:
                    _launch_fused_generic(
                        [r[0] for r in recs],
                        [r[1] for r in recs],
                        [r[2] for r in recs],
                        [r[5] for r in recs],
                        maxn,
                    )
                else:
                    for start in range(0, L, _MAX_SLOTS):
                        part = recs[start : start + _MAX_SLOTS]
                        m = 0
                        for r in part:
                            if r[2] > m:
                                m = r[2]
                        if m:
                            _launch_fused_generic(
                                [r[0] for r in part],
                                [r[1] for r in part],
                                [r[2] for r in part],
                                [r[5] for r in part],
                                m,
                            )
            return [r[1] for r in recs]

    for r in recs:
        if r[2] != 0:
            _launch_one(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
    return [r[1] for r in recs]
