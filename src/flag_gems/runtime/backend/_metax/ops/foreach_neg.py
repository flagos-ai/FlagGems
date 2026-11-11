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

"""foreach_neg: out_i = -self_i for every tensor in the input sequence.

Triton implementation.  Fresh contiguous output per input tensor; the input
may have any dense strided layout.  Elementwise sign flip in the native dtype
(exact for every dtype).  Empty tensor lists and bool tensors are rejected
exactly like ATen.

Dispatch:
  * single tensor            -> flat (contiguous) or collapsed-strides kernel
  * several small tensors of
    one dtype, all contiguous -> one fused launch for up to 16 tensors
  * otherwise                -> one launch per tensor

The timed path of this operator is dominated by per-call overhead, so the
launch configuration is baked into the kernels (no BLOCK/num_warps kwargs)
and the host path keeps the Python work minimal.
"""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_BLOCK = 1024
_FUSED_K = 16
_FUSE_MAX_ELEMS = 1 << 20

_BOOL_MSG = (
    "Boolean negative, the `-` operator, is not supported, "
    "use the `logical_not` operator instead."
)
_EMPTY_MSG = "Tensor list must have at least one tensor."

_empty_like = torch.empty_like
_bool = torch.bool
_CONTIG = torch.contiguous_format
_view_as_real = torch.view_as_real


@triton.jit
def _neg_flat(x_ptr, y_ptr, n):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n
    tl.store(y_ptr + offs, -tl.load(x_ptr + offs, mask=mask), mask=mask)


@triton.jit
def _neg_strided(
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
    NDIM: tl.constexpr,
):
    """Read x through its (collapsed, innermost-first) strides, write flat."""
    pid = tl.program_id(0)
    offs = (pid * 1024 + tl.arange(0, 1024)).to(tl.int64)
    mask = offs < n
    rem = offs
    src = tl.zeros([1024], dtype=tl.int64)
    if NDIM > 0:
        src += (rem % s0) * t0
        rem = rem // s0
    if NDIM > 1:
        src += (rem % s1) * t1
        rem = rem // s1
    if NDIM > 2:
        src += (rem % s2) * t2
        rem = rem // s2
    if NDIM > 3:
        src += (rem % s3) * t3
        rem = rem // s3
    if NDIM > 4:
        src += (rem % s4) * t4
        rem = rem // s4
    if NDIM > 5:
        src += (rem % s5) * t5
        rem = rem // s5
    if NDIM > 6:
        src += (rem % s6) * t6
        rem = rem // s6
    if NDIM > 7:
        src += (rem % s7) * t7
        rem = rem // s7
    tl.store(y_ptr + offs, -tl.load(x_ptr + src, mask=mask), mask=mask)


@triton.jit
def _neg_fused(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
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
):
    """Negate up to 16 contiguous tensors with one launch."""
    pid = tl.program_id(0)
    local = tl.arange(0, 1024)
    b0 = tl.cdiv(n0, 1024)
    b1 = b0 + tl.cdiv(n1, 1024)
    b2 = b1 + tl.cdiv(n2, 1024)
    b3 = b2 + tl.cdiv(n3, 1024)
    b4 = b3 + tl.cdiv(n4, 1024)
    b5 = b4 + tl.cdiv(n5, 1024)
    b6 = b5 + tl.cdiv(n6, 1024)
    b7 = b6 + tl.cdiv(n7, 1024)
    b8 = b7 + tl.cdiv(n8, 1024)
    b9 = b8 + tl.cdiv(n9, 1024)
    b10 = b9 + tl.cdiv(n10, 1024)
    b11 = b10 + tl.cdiv(n11, 1024)
    b12 = b11 + tl.cdiv(n12, 1024)
    b13 = b12 + tl.cdiv(n13, 1024)
    b14 = b13 + tl.cdiv(n14, 1024)
    b15 = b14 + tl.cdiv(n15, 1024)
    if pid < b0:
        offs = pid * 1024 + local
        m = offs < n0
        tl.store(o0 + offs, -tl.load(p0 + offs, mask=m), mask=m)
    elif pid < b1:
        offs = (pid - b0) * 1024 + local
        m = offs < n1
        tl.store(o1 + offs, -tl.load(p1 + offs, mask=m), mask=m)
    elif pid < b2:
        offs = (pid - b1) * 1024 + local
        m = offs < n2
        tl.store(o2 + offs, -tl.load(p2 + offs, mask=m), mask=m)
    elif pid < b3:
        offs = (pid - b2) * 1024 + local
        m = offs < n3
        tl.store(o3 + offs, -tl.load(p3 + offs, mask=m), mask=m)
    elif pid < b4:
        offs = (pid - b3) * 1024 + local
        m = offs < n4
        tl.store(o4 + offs, -tl.load(p4 + offs, mask=m), mask=m)
    elif pid < b5:
        offs = (pid - b4) * 1024 + local
        m = offs < n5
        tl.store(o5 + offs, -tl.load(p5 + offs, mask=m), mask=m)
    elif pid < b6:
        offs = (pid - b5) * 1024 + local
        m = offs < n6
        tl.store(o6 + offs, -tl.load(p6 + offs, mask=m), mask=m)
    elif pid < b7:
        offs = (pid - b6) * 1024 + local
        m = offs < n7
        tl.store(o7 + offs, -tl.load(p7 + offs, mask=m), mask=m)
    elif pid < b8:
        offs = (pid - b7) * 1024 + local
        m = offs < n8
        tl.store(o8 + offs, -tl.load(p8 + offs, mask=m), mask=m)
    elif pid < b9:
        offs = (pid - b8) * 1024 + local
        m = offs < n9
        tl.store(o9 + offs, -tl.load(p9 + offs, mask=m), mask=m)
    elif pid < b10:
        offs = (pid - b9) * 1024 + local
        m = offs < n10
        tl.store(o10 + offs, -tl.load(p10 + offs, mask=m), mask=m)
    elif pid < b11:
        offs = (pid - b10) * 1024 + local
        m = offs < n11
        tl.store(o11 + offs, -tl.load(p11 + offs, mask=m), mask=m)
    elif pid < b12:
        offs = (pid - b11) * 1024 + local
        m = offs < n12
        tl.store(o12 + offs, -tl.load(p12 + offs, mask=m), mask=m)
    elif pid < b13:
        offs = (pid - b12) * 1024 + local
        m = offs < n13
        tl.store(o13 + offs, -tl.load(p13 + offs, mask=m), mask=m)
    elif pid < b14:
        offs = (pid - b13) * 1024 + local
        m = offs < n14
        tl.store(o14 + offs, -tl.load(p14 + offs, mask=m), mask=m)
    elif pid < b15:
        offs = (pid - b14) * 1024 + local
        m = offs < n15
        tl.store(o15 + offs, -tl.load(p15 + offs, mask=m), mask=m)


def _collapse(x):
    """Collapse a strided layout to <= _MAX_DIMS dims (outermost first)."""
    shape = x.shape
    stride = x.stride()
    merged_s = []
    merged_t = []
    for i in range(len(shape)):
        s = shape[i]
        t = stride[i]
        if s == 1:
            continue
        if merged_s and merged_t[-1] == t * s:
            merged_s[-1] *= s
        else:
            merged_s.append(s)
            merged_t.append(t)
    return merged_s, merged_t


def _launch_strided(x, out, n):
    ms, mt = _collapse(x)
    nd = len(ms)
    if nd == 0:
        _neg_flat[((n + 1023) >> 10,)](x, out, n)
        return
    if nd > _MAX_DIMS:
        raise RuntimeError("foreach_neg: unsupported tensor rank")
    pad = _MAX_DIMS - nd
    # the kernel decomposes the flat logical index innermost dim first
    ms = ms[::-1] + [1] * pad
    mt = mt[::-1] + [0] * pad
    _neg_strided[((n + 1023) >> 10,)](x, out, n, *ms, *mt, NDIM=nd)


def _launch_fused(srcs, outs, ns):
    blocks = 0
    for v in ns:
        blocks += (v + 1023) >> 10
    if blocks == 0:
        return
    pad = _FUSED_K - len(srcs)
    if pad:
        srcs = srcs + [srcs[0]] * pad
        outs = outs + [outs[0]] * pad
        ns = ns + [0] * pad
    _neg_fused[(blocks,)](*srcs, *outs, *ns)


def foreach_neg(self):
    count = len(self)
    if count == 1:
        x = self[0]
        dt = x.dtype
        if dt == _bool:
            raise RuntimeError(_BOOL_MSG)
        cx = dt.is_complex
        cont = x.is_contiguous()
        if cont and not cx:
            out = _empty_like(x)
            nel = x.numel()
            if nel:
                _neg_flat[((nel + 1023) >> 10,)](x, out, nel)
            return [out]
        out = _empty_like(x, memory_format=_CONTIG)
        src = _view_as_real(x) if cx else x
        dst = _view_as_real(out) if cx else out
        nel = src.numel()
        if nel:
            if cont:
                _neg_flat[((nel + 1023) >> 10,)](src, dst, nel)
            else:
                _launch_strided(src, dst, nel)
        return [out]

    if count == 0:
        raise RuntimeError(_EMPTY_MSG)

    # multi-tensor fast path: one dtype, all contiguous, small total
    dt = self[0].dtype
    if dt != _bool and not dt.is_complex:
        ns = []
        total = 0
        for t in self:
            if t.dtype is not dt or not t.is_contiguous():
                break
            v = t.numel()
            ns.append(v)
            total += v
        else:
            if total <= _FUSE_MAX_ELEMS:
                outs = [_empty_like(t) for t in self]
                if count <= _FUSED_K:
                    _launch_fused(self, outs, ns)
                else:
                    for i in range(0, count, _FUSED_K):
                        j = i + _FUSED_K
                        _launch_fused(self[i:j], outs[i:j], ns[i:j])
                return outs

    outs = []
    append = outs.append
    for x in self:
        dt = x.dtype
        if dt == _bool:
            raise RuntimeError(_BOOL_MSG)
        cx = dt.is_complex
        cont = x.is_contiguous()
        if cont and not cx:
            out = _empty_like(x)
            nel = x.numel()
            if nel:
                _neg_flat[((nel + 1023) >> 10,)](x, out, nel)
        else:
            out = _empty_like(x, memory_format=_CONTIG)
            src = _view_as_real(x) if cx else x
            dst = _view_as_real(out) if cx else out
            nel = src.numel()
            if nel:
                if cont:
                    _neg_flat[((nel + 1023) >> 10,)](src, dst, nel)
                else:
                    _launch_strided(src, dst, nel)
        append(out)
    return outs
