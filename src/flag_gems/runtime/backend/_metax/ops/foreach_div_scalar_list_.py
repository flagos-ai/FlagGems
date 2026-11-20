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

"""In-place foreach division by a scalar list (aten::_foreach_div_.ScalarList).

Design: all contiguous tensors of the same dtype in one call are packed into a
single Triton launch of up to 8 slots. Each program owns one BLOCK-sized chunk
of one slot; the slot (pointer, numel, scalar) is chosen by a uniform scalar
branch chain built from host-computed chunk prefixes. Numels stay specialized
so Triton keeps the divisibility property of the mask bound and emits
vectorized (v4) predicated loads/stores; only the chunk prefixes, which feed
scalar compares, are left unspecialized. Non-contiguous tensors keep a
per-tensor launch with a stride-decoding kernel whose dimensions are collapsed
and sorted by ascending stride so the storage-contiguous dimension is
lane-fastest. 16-bit dtypes are divided in fp32 opmath with a single rounding
on store; 32-bit dtypes divide in fp32. run() mutates in place and returns None.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
MAX_DIMS = 8
GROUP = 8


@triton.jit
def _div_flat_kernel(ptr, n, s, IS16: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask)
    if IS16:
        y = (x.to(tl.float32) / s).to(x.dtype)
    else:
        y = x / s
    tl.store(ptr + offs, y, mask=mask)


@triton.jit
def _div_uniform4_kernel(
    p0, p1, p2, p3, n, chunks, s0, s1, s2, s3, IS16: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    slot = pid // chunks
    idx = pid - slot * chunks
    if slot == 0:
        p = p0
        s = s0
    elif slot == 1:
        p = p1
        s = s1
    elif slot == 2:
        p = p2
        s = s2
    else:
        p = p3
        s = s3
    offs = idx.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask)
    if IS16:
        y = (x.to(tl.float32) / s).to(x.dtype)
    else:
        y = x / s
    tl.store(p + offs, y, mask=mask)


@triton.jit
def _div_uniform8_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n,
    chunks,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    IS16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = pid // chunks
    idx = pid - slot * chunks
    if slot == 0:
        p = p0
        s = s0
    elif slot == 1:
        p = p1
        s = s1
    elif slot == 2:
        p = p2
        s = s2
    elif slot == 3:
        p = p3
        s = s3
    elif slot == 4:
        p = p4
        s = s4
    elif slot == 5:
        p = p5
        s = s5
    elif slot == 6:
        p = p6
        s = s6
    else:
        p = p7
        s = s7
    offs = idx.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask)
    if IS16:
        y = (x.to(tl.float32) / s).to(x.dtype)
    else:
        y = x / s
    tl.store(p + offs, y, mask=mask)


@triton.jit(do_not_specialize=["c0", "c1", "c2", "c3", "c4", "c5", "c6"])
def _div_fused_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    IS16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < c0:
        p = p0
        n = n0
        s = s0
        idx = pid
    elif pid < c1:
        p = p1
        n = n1
        s = s1
        idx = pid - c0
    elif pid < c2:
        p = p2
        n = n2
        s = s2
        idx = pid - c1
    elif pid < c3:
        p = p3
        n = n3
        s = s3
        idx = pid - c2
    elif pid < c4:
        p = p4
        n = n4
        s = s4
        idx = pid - c3
    elif pid < c5:
        p = p5
        n = n5
        s = s5
        idx = pid - c4
    elif pid < c6:
        p = p6
        n = n6
        s = s6
        idx = pid - c5
    else:
        p = p7
        n = n7
        s = s7
        idx = pid - c6
    offs = idx.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask)
    if IS16:
        y = (x.to(tl.float32) / s).to(x.dtype)
    else:
        y = x / s
    tl.store(p + offs, y, mask=mask)


@triton.jit
def _div_strided_kernel(
    ptr,
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
    NDIM: tl.constexpr,
    IS16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    i0 = rem % d0
    rem = rem // d0
    off += i0 * t0
    if NDIM >= 2:
        i1 = rem % d1
        rem = rem // d1
        off += i1 * t1
    if NDIM >= 3:
        i2 = rem % d2
        rem = rem // d2
        off += i2 * t2
    if NDIM >= 4:
        i3 = rem % d3
        rem = rem // d3
        off += i3 * t3
    if NDIM >= 5:
        i4 = rem % d4
        rem = rem // d4
        off += i4 * t4
    if NDIM >= 6:
        i5 = rem % d5
        rem = rem // d5
        off += i5 * t5
    if NDIM >= 7:
        i6 = rem % d6
        rem = rem // d6
        off += i6 * t6
    if NDIM >= 8:
        i7 = rem % d7
        rem = rem // d7
        off += i7 * t7
    x = tl.load(ptr + off, mask=mask)
    if IS16:
        y = (x.to(tl.float32) / s).to(x.dtype)
    else:
        y = x / s
    tl.store(ptr + off, y, mask=mask)


def _collapse(t):
    """Collapse a view's shape/stride to canonical (dim, stride) pairs."""
    shape = t.shape
    stride = t.stride()
    dims = []
    strides = []
    for d in range(len(shape) - 1, -1, -1):
        sz = shape[d]
        if sz == 1:
            continue
        st = stride[d]
        if dims and st == strides[-1] * dims[-1]:
            dims[-1] *= sz
        else:
            dims.append(sz)
            strides.append(st)
    pairs = sorted(zip(strides, dims))
    return [p[1] for p in pairs], [p[0] for p in pairs]


def foreach_div_scalar_list_(self, scalars):
    if isinstance(self, torch.Tensor):
        tensors = [self]
    else:
        tensors = list(self)
    nt = len(tensors)
    if nt == 0:
        return None

    if isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            vals = [float(scalars)] * nt
        else:
            vals = [float(v) for v in scalars]
    elif isinstance(scalars, (int, float)):
        vals = [float(scalars)] * nt
    else:
        vals = [v if isinstance(v, float) else float(v) for v in scalars]

    if len(vals) != nt:
        raise RuntimeError(
            "Tensor list must have same number of elements as scalar list."
        )

    groups = {}
    for i in range(nt):
        t = tensors[i]
        n = t.numel()
        if n == 0:
            continue
        s = vals[i]
        if t.is_contiguous():
            g = groups.get(t.dtype)
            if g is None:
                groups[t.dtype] = [t, s]
            else:
                g.append(t)
                g.append(s)
        else:
            dims, strides = _collapse(t)
            nd = len(dims)
            while len(dims) < MAX_DIMS:
                dims.append(1)
                strides.append(0)
            grid = ((n + BLOCK - 1) // BLOCK,)
            _div_strided_kernel[grid](
                t,
                n,
                s,
                dims[0],
                dims[1],
                dims[2],
                dims[3],
                dims[4],
                dims[5],
                dims[6],
                dims[7],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                strides[4],
                strides[5],
                strides[6],
                strides[7],
                NDIM=nd,
                IS16=(t.dtype == torch.float16 or t.dtype == torch.bfloat16),
                BLOCK=BLOCK,
                num_warps=NUM_WARPS,
            )

    for dtype, g in groups.items():
        is16 = dtype == torch.float16 or dtype == torch.bfloat16
        m = len(g) >> 1
        j = 0
        while j < m:
            k = GROUP if m - j > GROUP else m - j
            if k == 1:
                t = g[2 * j]
                n = t.numel()
                _div_flat_kernel[((n + BLOCK - 1) // BLOCK,)](
                    t,
                    n,
                    g[2 * j + 1],
                    IS16=is16,
                    BLOCK=BLOCK,
                    num_warps=NUM_WARPS,
                )
            else:
                ps = []
                ns = []
                ss = []
                for q in range(k):
                    t = g[2 * (j + q)]
                    ps.append(t)
                    ns.append(t.numel())
                    ss.append(g[2 * (j + q) + 1])
                n0 = ns[0]
                uniform = True
                for q in range(1, k):
                    if ns[q] != n0:
                        uniform = False
                        break
                while len(ps) < GROUP:
                    ps.append(ps[0])
                    ns.append(0)
                    ss.append(0.0)
                if uniform:
                    chunks = (n0 + BLOCK - 1) // BLOCK
                    grid = (chunks * k,)
                    if k <= 4:
                        _div_uniform4_kernel[grid](
                            ps[0],
                            ps[1],
                            ps[2],
                            ps[3],
                            n0,
                            chunks,
                            ss[0],
                            ss[1],
                            ss[2],
                            ss[3],
                            IS16=is16,
                            BLOCK=BLOCK,
                            num_warps=NUM_WARPS,
                        )
                    else:
                        _div_uniform8_kernel[grid](
                            ps[0],
                            ps[1],
                            ps[2],
                            ps[3],
                            ps[4],
                            ps[5],
                            ps[6],
                            ps[7],
                            n0,
                            chunks,
                            ss[0],
                            ss[1],
                            ss[2],
                            ss[3],
                            ss[4],
                            ss[5],
                            ss[6],
                            ss[7],
                            IS16=is16,
                            BLOCK=BLOCK,
                            num_warps=NUM_WARPS,
                        )
                else:
                    cs = []
                    total = 0
                    for q in range(k):
                        total += (ns[q] + BLOCK - 1) // BLOCK
                        cs.append(total)
                    for q in range(GROUP - k):
                        cs.append(total)
                    _div_fused_kernel[(total,)](
                        ps[0],
                        ps[1],
                        ps[2],
                        ps[3],
                        ps[4],
                        ps[5],
                        ps[6],
                        ps[7],
                        ns[0],
                        ns[1],
                        ns[2],
                        ns[3],
                        ns[4],
                        ns[5],
                        ns[6],
                        ns[7],
                        cs[0],
                        cs[1],
                        cs[2],
                        cs[3],
                        cs[4],
                        cs[5],
                        cs[6],
                        ss[0],
                        ss[1],
                        ss[2],
                        ss[3],
                        ss[4],
                        ss[5],
                        ss[6],
                        ss[7],
                        IS16=is16,
                        BLOCK=BLOCK,
                        num_warps=NUM_WARPS,
                    )
            j += k
    return None
