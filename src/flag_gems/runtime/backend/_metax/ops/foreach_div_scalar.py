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

"""Triton implementation of ``aten::_foreach_div.Scalar``.

Semantics: for every tensor ``t`` in the input list and scalar ``c``,
``out_i[j] = t_i[j] / c`` computed in fp32 opmath and stored back in the
tensor's own dtype.  Outputs are freshly allocated (never aliasing inputs).

Fast path: the list is grouped into runs of same-dtype, *dense* tensors
(contiguous, or a dense permutation such as a transposed view).  For a dense
tensor the element storage is a contiguous run of ``numel`` elements, so when
the output is allocated with the identical stride layout the whole operation
becomes a flat streaming copy-with-divide.  Up to eight such tensors are
handled by a single fused launch (one program range per slot), which keeps the
per-call launch count at ``ceil(N/8)`` instead of ``N``.

Slow path: non-dense views (gaps in storage) fall back to a gather kernel with
in-kernel stride decomposition after TensorIterator-style dimension collapse.
"""

import torch
import triton
import triton.language as tl

_MAX_SLOTS = 8
_BLOCK = 1024
_NUM_WARPS = 4
_STRIDED_BLOCK = 1024


@triton.jit
def _div_tile(src, dst, base, n, s, BLOCK: tl.constexpr):
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=0.0)
    y = x.to(tl.float32) / s
    tl.store(dst + offs, y, mask=mask)


@triton.jit
def _foreach_div_scalar_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    s,
    BLOCK: tl.constexpr,
):
    """Each program handles one BLOCK-wide tile of exactly one slot."""
    pid = tl.program_id(0)
    if pid < q1:
        _div_tile(x0, y0, pid * BLOCK, n0, s, BLOCK)
    elif pid < q2:
        _div_tile(x1, y1, (pid - q1) * BLOCK, n1, s, BLOCK)
    elif pid < q3:
        _div_tile(x2, y2, (pid - q2) * BLOCK, n2, s, BLOCK)
    elif pid < q4:
        _div_tile(x3, y3, (pid - q3) * BLOCK, n3, s, BLOCK)
    elif pid < q5:
        _div_tile(x4, y4, (pid - q4) * BLOCK, n4, s, BLOCK)
    elif pid < q6:
        _div_tile(x5, y5, (pid - q5) * BLOCK, n5, s, BLOCK)
    elif pid < q7:
        _div_tile(x6, y6, (pid - q6) * BLOCK, n6, s, BLOCK)
    else:
        _div_tile(x7, y7, (pid - q7) * BLOCK, n7, s, BLOCK)


@triton.jit
def _strided_div_kernel(
    x,
    y,
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
    sp,
    s,
    USE_PTR: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather with in-kernel stride decomposition; contiguous store."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM >= 1:
        off += (rem % sh0) * st0
        rem = rem // sh0
    if NDIM >= 2:
        off += (rem % sh1) * st1
        rem = rem // sh1
    if NDIM >= 3:
        off += (rem % sh2) * st2
        rem = rem // sh2
    if NDIM >= 4:
        off += (rem % sh3) * st3
        rem = rem // sh3
    if NDIM >= 5:
        off += (rem % sh4) * st4
        rem = rem // sh4
    if NDIM >= 6:
        off += (rem % sh5) * st5
        rem = rem // sh5
    if NDIM >= 7:
        off += (rem % sh6) * st6
        rem = rem // sh6
    if NDIM >= 8:
        off += (rem % sh7) * st7
        rem = rem // sh7
    if USE_PTR:
        val = tl.load(sp).to(tl.float32)
    else:
        val = s
    xv = tl.load(x + off, mask=mask, other=0.0)
    tl.store(y + offs, xv.to(tl.float32) / val, mask=mask)


def _is_dense(t):
    """True when the view covers a contiguous run of storage (no gaps)."""
    shape = t.shape
    stride = t.stride()
    pairs = sorted(
        (int(stride[i]), int(shape[i])) for i in range(len(shape)) if shape[i] != 1
    )
    expect = 1
    for st, sz in pairs:
        if st != expect:
            return False
        expect *= sz
    return True


def _collapse_dims(t):
    """Collapse dims (innermost first) so the kernel can decompose offsets."""
    shape = t.shape
    stride = t.stride()
    sizes = []
    strides = []
    for i in range(len(shape) - 1, -1, -1):
        sz = int(shape[i])
        st = int(stride[i])
        if sz == 1:
            continue
        if sizes and st == strides[-1] * sizes[-1]:
            sizes[-1] = sizes[-1] * sz
        else:
            sizes.append(sz)
            strides.append(st)
    return sizes, strides


def _out_dtype(t):
    if t.is_floating_point() or t.is_complex():
        return t.dtype
    return torch.get_default_dtype()


def _alloc_out(t, dense):
    dtype = _out_dtype(t)
    if dtype == t.dtype and t.is_contiguous():
        return torch.empty_like(t)
    if dense:
        return torch.empty_strided(t.shape, t.stride(), dtype=dtype, device=t.device)
    return torch.empty(t.shape, dtype=dtype, device=t.device)


def _launch_flat(ins, outs, ns, value):
    k = len(ins)
    if k == 0:
        return
    ins = list(ins)
    outs = list(outs)
    ns = list(ns)
    if k < _MAX_SLOTS:
        pad = _MAX_SLOTS - k
        ins += [ins[0]] * pad
        outs += [outs[0]] * pad
        ns += [0] * pad
    counts = [(nn + _BLOCK - 1) // _BLOCK if nn > 0 else 0 for nn in ns]
    q = []
    acc = 0
    for c in counts[:7]:
        acc += c
        q.append(acc)
    total = sum(counts)
    if total == 0:
        return
    _foreach_div_scalar_kernel[(total,)](
        ins[0],
        ins[1],
        ins[2],
        ins[3],
        ins[4],
        ins[5],
        ins[6],
        ins[7],
        outs[0],
        outs[1],
        outs[2],
        outs[3],
        outs[4],
        outs[5],
        outs[6],
        outs[7],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        q[0],
        q[1],
        q[2],
        q[3],
        q[4],
        q[5],
        q[6],
        value,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_strided(t, out, value=None, scalar_ptr=None):
    sizes, strides = _collapse_dims(t)
    if not sizes:
        sizes, strides = [1], [1]
    ndim = len(sizes)
    if ndim > 8:
        raise NotImplementedError(
            "_foreach_div: views with more than 8 collapsed dims are unsupported"
        )
    sh = sizes + [1] * (8 - ndim)
    st = strides + [0] * (8 - ndim)
    n = int(t.numel())
    grid = ((n + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK,)
    use_ptr = scalar_ptr is not None
    _strided_div_kernel[grid](
        t,
        out,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        st[6],
        st[7],
        scalar_ptr if use_ptr else t,
        float(value) if value is not None else 0.0,
        USE_PTR=use_ptr,
        NDIM=ndim,
        BLOCK=_STRIDED_BLOCK,
        num_warps=4,
    )


def _launch_div_ptr(t, out, divisor):
    """Gather kernel (contiguous store) with the divisor read from memory."""
    n = int(t.numel())
    if _is_dense(t):
        ndim = 1
        sh = [n, 1, 1, 1, 1, 1, 1, 1]
        st = [1, 0, 0, 0, 0, 0, 0, 0]
    else:
        sizes, strides = _collapse_dims(t)
        ndim = len(sizes)
        if ndim > 8:
            raise NotImplementedError(
                "_foreach_div: views with more than 8 collapsed dims are unsupported"
            )
        sh = sizes + [1] * (8 - ndim)
        st = strides + [0] * (8 - ndim)
    grid = ((n + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK,)
    _strided_div_kernel[grid](
        t,
        out,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        st[0],
        st[1],
        st[2],
        st[3],
        st[4],
        st[5],
        st[6],
        st[7],
        divisor,
        0.0,
        USE_PTR=True,
        NDIM=ndim,
        BLOCK=_STRIDED_BLOCK,
        num_warps=4,
    )


def _run_number(tensors, value):
    outs = [None] * len(tensors)
    # (device, dtype) -> list of flat-capable (tensor, out) pairs in order
    groups = {}
    order = []
    for i, t in enumerate(tensors):
        if t.numel() == 0:
            outs[i] = _alloc_out(t, True)
            continue
        dense = _is_dense(t)
        out = _alloc_out(t, dense)
        outs[i] = out
        if dense:
            key = (t.device, t.dtype)
            bucket = groups.get(key)
            if bucket is None:
                bucket = []
                groups[key] = bucket
                order.append(key)
            bucket.append((t, out))
        else:
            _launch_strided(t, out, value=value)
    for key in order:
        bucket = groups[key]
        for start in range(0, len(bucket), _MAX_SLOTS):
            chunk = bucket[start : start + _MAX_SLOTS]
            _launch_flat(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                [int(c[0].numel()) for c in chunk],
                value,
            )
    return outs


def _run_scalar_list(tensors, scalars):
    if len(scalars) != len(tensors):
        raise ValueError(
            "Tensor list length %d does not match scalar list length %d"
            % (len(tensors), len(scalars))
        )
    outs = [None] * len(tensors)
    for i, (t, sc) in enumerate(zip(tensors, scalars)):
        if isinstance(sc, torch.Tensor):
            out = _alloc_out(t, False) if not t.is_contiguous() else _alloc_out(t, True)
            outs[i] = out
            if t.numel() != 0:
                _launch_div_ptr(t, out, sc)
            continue
        if t.numel() == 0:
            outs[i] = _alloc_out(t, True)
            continue
        dense = _is_dense(t)
        outs[i] = _alloc_out(t, dense)
        if dense:
            _launch_flat([t], [outs[i]], [int(t.numel())], float(sc))
        else:
            _launch_strided(t, outs[i], value=float(sc))
    return outs


def _run_tensor_divisor(tensors, divisor):
    if divisor.numel() != 1:
        raise NotImplementedError(
            "_foreach_div: Tensor-list divisor (elementwise) is unsupported"
        )
    outs = [None] * len(tensors)
    for i, t in enumerate(tensors):
        if t.numel() == 0:
            outs[i] = _alloc_out(t, True)
            continue
        out = torch.empty(t.shape, dtype=_out_dtype(t), device=t.device)
        outs[i] = out
        _launch_div_ptr(t, out, divisor)
    return outs


def foreach_div_scalar(self, scalar=None, *args, **kwargs):
    """Apply ``self / scalar`` out-of-place to every tensor in ``self``."""
    other = scalar
    if other is None:
        if "scalar" in kwargs:
            other = kwargs.pop("scalar")
        elif "other" in kwargs:
            other = kwargs.pop("other")
    is_single = isinstance(self, torch.Tensor)
    tensors = [self] if is_single else list(self)
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if isinstance(other, torch.Tensor):
        outs = _run_tensor_divisor(tensors, other)
    elif isinstance(other, (list, tuple)):
        outs = _run_scalar_list(tensors, list(other))
    else:
        outs = _run_number(tensors, float(other))
    return outs[0] if is_single else outs
