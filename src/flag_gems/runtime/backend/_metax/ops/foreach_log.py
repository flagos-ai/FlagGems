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

"""Self-contained Triton implementation of aten::_foreach_log."""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_FLAT_BLOCK = 2048
_FLAT_WARPS = 4
_SMALL_BLOCK = 1024
_SMALL_WARPS = 4
_SMALL_ELEMS = 65536
_BATCH = 4
_BATCH_SMALL = 16


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@triton.jit
def _log_flat_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if x_ptr.dtype.element_ty == tl.float64:
        y = tl.log(v)
    else:
        y = tl.log(v.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log_flat_batch4(
    x0,
    x1,
    x2,
    x3,
    o0,
    o1,
    o2,
    o3,
    n0,
    n1,
    n2,
    n3,
    BLOCK: tl.constexpr,
):
    """Applies log to up to four same-dtype dense tensors in one launch.

    Program ids are mapped to (tensor, block) by comparing against the running
    sum of per-tensor block counts, so no device-side metadata is needed.
    """
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = b0 + tl.cdiv(n1, BLOCK)
    b2 = b1 + tl.cdiv(n2, BLOCK)

    if pid < b0:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n0
        v = tl.load(x0 + offs, mask=m, other=1.0)
        if x0.dtype.element_ty == tl.float64:
            y = tl.log(v)
        else:
            y = tl.log(v.to(tl.float32))
        tl.store(o0 + offs, y.to(o0.dtype.element_ty), mask=m)
    elif pid < b1:
        offs = (pid - b0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n1
        v = tl.load(x1 + offs, mask=m, other=1.0)
        if x1.dtype.element_ty == tl.float64:
            y = tl.log(v)
        else:
            y = tl.log(v.to(tl.float32))
        tl.store(o1 + offs, y.to(o1.dtype.element_ty), mask=m)
    elif pid < b2:
        offs = (pid - b1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n2
        v = tl.load(x2 + offs, mask=m, other=1.0)
        if x2.dtype.element_ty == tl.float64:
            y = tl.log(v)
        else:
            y = tl.log(v.to(tl.float32))
        tl.store(o2 + offs, y.to(o2.dtype.element_ty), mask=m)
    else:
        offs = (pid - b2) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n3
        v = tl.load(x3 + offs, mask=m, other=1.0)
        if x3.dtype.element_ty == tl.float64:
            y = tl.log(v)
        else:
            y = tl.log(v.to(tl.float32))
        tl.store(o3 + offs, y.to(o3.dtype.element_ty), mask=m)


@triton.jit
def _log_flat_batch16(
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
    BLOCK: tl.constexpr,
):
    """Applies log to up to sixteen same-dtype dense tensors in one launch.

    A program resolves its (tensor, block) pair by comparing its id against the
    running sum of per-tensor block counts, derived inline from the scalar size
    arguments. Used only when the whole group is launch-bound; wide packing
    loses bandwidth on large streams, which stay on the four-slot kernel.
    """
    pid = tl.program_id(0)
    b1 = tl.cdiv(n0, BLOCK)
    b2 = b1 + tl.cdiv(n1, BLOCK)
    b3 = b2 + tl.cdiv(n2, BLOCK)
    b4 = b3 + tl.cdiv(n3, BLOCK)
    b5 = b4 + tl.cdiv(n4, BLOCK)
    b6 = b5 + tl.cdiv(n5, BLOCK)
    b7 = b6 + tl.cdiv(n6, BLOCK)
    b8 = b7 + tl.cdiv(n7, BLOCK)
    b9 = b8 + tl.cdiv(n8, BLOCK)
    b10 = b9 + tl.cdiv(n9, BLOCK)
    b11 = b10 + tl.cdiv(n10, BLOCK)
    b12 = b11 + tl.cdiv(n11, BLOCK)
    b13 = b12 + tl.cdiv(n12, BLOCK)
    b14 = b13 + tl.cdiv(n13, BLOCK)
    b15 = b14 + tl.cdiv(n14, BLOCK)

    src = x0
    dst = o0
    base = pid
    cnt = n0
    if pid >= b1:
        src = x1
        dst = o1
        base = pid - b1
        cnt = n1
    if pid >= b2:
        src = x2
        dst = o2
        base = pid - b2
        cnt = n2
    if pid >= b3:
        src = x3
        dst = o3
        base = pid - b3
        cnt = n3
    if pid >= b4:
        src = x4
        dst = o4
        base = pid - b4
        cnt = n4
    if pid >= b5:
        src = x5
        dst = o5
        base = pid - b5
        cnt = n5
    if pid >= b6:
        src = x6
        dst = o6
        base = pid - b6
        cnt = n6
    if pid >= b7:
        src = x7
        dst = o7
        base = pid - b7
        cnt = n7
    if pid >= b8:
        src = x8
        dst = o8
        base = pid - b8
        cnt = n8
    if pid >= b9:
        src = x9
        dst = o9
        base = pid - b9
        cnt = n9
    if pid >= b10:
        src = x10
        dst = o10
        base = pid - b10
        cnt = n10
    if pid >= b11:
        src = x11
        dst = o11
        base = pid - b11
        cnt = n11
    if pid >= b12:
        src = x12
        dst = o12
        base = pid - b12
        cnt = n12
    if pid >= b13:
        src = x13
        dst = o13
        base = pid - b13
        cnt = n13
    if pid >= b14:
        src = x14
        dst = o14
        base = pid - b14
        cnt = n14
    if pid >= b15:
        src = x15
        dst = o15
        base = pid - b15
        cnt = n15

    offs = base * BLOCK + tl.arange(0, BLOCK)
    m = offs < cnt
    v = tl.load(src + offs, mask=m, other=1.0)
    if x0.dtype.element_ty == tl.float64:
        y = tl.log(v)
    else:
        y = tl.log(v.to(tl.float32))
    tl.store(dst + offs, y.to(o0.dtype.element_ty), mask=m)


@triton.jit
def _log_strided_kernel(
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
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs.to(tl.int64)
    io = tl.zeros([BLOCK], dtype=tl.int64)

    q = rem % d0
    io += q * s0
    rem = rem // d0
    q = rem % d1
    io += q * s1
    rem = rem // d1
    q = rem % d2
    io += q * s2
    rem = rem // d2
    q = rem % d3
    io += q * s3
    rem = rem // d3
    q = rem % d4
    io += q * s4
    rem = rem // d4
    q = rem % d5
    io += q * s5
    rem = rem // d5
    q = rem % d6
    io += q * s6
    rem = rem // d6
    q = rem % d7
    io += q * s7

    v = tl.load(x_ptr + io, mask=mask, other=1.0)
    if x_ptr.dtype.element_ty == tl.float64:
        y = tl.log(v)
    else:
        y = tl.log(v.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log_complex_flat_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    re = tl.load(x_ptr + 2 * offs, mask=mask, other=1.0)
    im = tl.load(x_ptr + 2 * offs + 1, mask=mask, other=0.0)
    if x_ptr.dtype.element_ty == tl.float64:
        mag = 0.5 * tl.log(re * re + im * im)
        ang = tl.extra.cuda.libdevice.atan2(im, re)
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
        mag = 0.5 * tl.log(ref * ref + imf * imf)
        ang = tl.extra.cuda.libdevice.atan2(imf, ref)
    ot = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, mag.to(ot), mask=mask)
    tl.store(o_ptr + 2 * offs + 1, ang.to(ot), mask=mask)


@triton.jit
def _log_complex_strided_kernel(
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
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs.to(tl.int64)
    io = tl.zeros([BLOCK], dtype=tl.int64)

    q = rem % d0
    io += q * s0
    rem = rem // d0
    q = rem % d1
    io += q * s1
    rem = rem // d1
    q = rem % d2
    io += q * s2
    rem = rem // d2
    q = rem % d3
    io += q * s3
    rem = rem // d3
    q = rem % d4
    io += q * s4
    rem = rem // d4
    q = rem % d5
    io += q * s5
    rem = rem // d5
    q = rem % d6
    io += q * s6
    rem = rem // d6
    q = rem % d7
    io += q * s7

    re = tl.load(x_ptr + 2 * io, mask=mask, other=1.0)
    im = tl.load(x_ptr + 2 * io + 1, mask=mask, other=0.0)
    if x_ptr.dtype.element_ty == tl.float64:
        mag = 0.5 * tl.log(re * re + im * im)
        ang = tl.extra.cuda.libdevice.atan2(im, re)
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
        mag = 0.5 * tl.log(ref * ref + imf * imf)
        ang = tl.extra.cuda.libdevice.atan2(imf, ref)
    ot = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, mag.to(ot), mask=mask)
    tl.store(o_ptr + 2 * offs + 1, ang.to(ot), mask=mask)


# --------------------------------------------------------------------------
# Host helpers
# --------------------------------------------------------------------------
def _collapse_dims(x):
    """Rank-reduced (dim, stride) pairs; size-1 dims are dropped and
    adjacent dims are merged when they are contiguous."""
    dims = []
    strides = []
    shape = x.shape
    stride = x.stride()
    for i in range(len(shape)):
        d = int(shape[i])
        s = int(stride[i])
        if d == 1:
            continue
        if dims and s == strides[-1] * dims[-1]:
            dims[-1] = dims[-1] * d
        else:
            dims.append(d)
            strides.append(s)
    pad = _MAX_DIMS - len(dims)
    if pad < 0:
        raise NotImplementedError(
            "foreach_log: input rank too large for the strided path"
        )
    return [1] * pad + dims, [0] * pad + strides


def _launch_strided(kernel, dims_src, xp, op, n, grid):
    dims, strides = _collapse_dims(dims_src)
    kernel[grid](
        xp,
        op,
        n,
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
        BLOCK=_FLAT_BLOCK,
        num_warps=_FLAT_WARPS,
    )


def foreach_log(self, *args, **kwargs):
    if isinstance(self, torch.Tensor) or args or kwargs:
        raise TypeError("_foreach_log() expects a single sequence of tensors")
    tensors = list(self)
    count = len(tensors)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = []
    groups = {}
    for x in tensors:
        if x.is_floating_point() or x.is_complex():
            out = torch.empty_like(x)
        else:
            out = torch.empty_like(x, dtype=torch.get_default_dtype())
        outs.append(out)
        n = x.numel()
        if n == 0:
            continue
        if n <= _SMALL_ELEMS:
            block = _SMALL_BLOCK
            warps = _SMALL_WARPS
        else:
            block = _FLAT_BLOCK
            warps = _FLAT_WARPS
        grid = ((n + block - 1) // block,)
        if x.is_complex():
            xf = torch.view_as_real(x)
            of = torch.view_as_real(out)
            if out.stride() == x.stride():
                _log_complex_flat_kernel[grid](xf, of, n, BLOCK=block, num_warps=warps)
            else:
                _launch_strided(_log_complex_strided_kernel, x, xf, of, n, grid)
            continue
        if out.stride() != x.stride():
            _launch_strided(_log_strided_kernel, x, x, out, n, grid)
            continue
        if count == 1:
            _log_flat_kernel[grid](x, out, n, BLOCK=block, num_warps=warps)
            continue
        key = (x.device, x.dtype)
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [(x, out, n)]
        else:
            bucket.append((x, out, n))

    if not groups:
        return tuple(outs)

    for items in groups.values():
        total_items = len(items)
        total_elems = 0
        for it in items:
            total_elems += it[2]
        if total_elems <= _SMALL_ELEMS:
            block = _SMALL_BLOCK
            warps = _SMALL_WARPS
            width = _BATCH_SMALL
            kernel = _log_flat_batch16
        else:
            block = _FLAT_BLOCK
            warps = _FLAT_WARPS
            width = _BATCH
            kernel = _log_flat_batch4

        if total_items == 1:
            x, out, n = items[0]
            _log_flat_kernel[((n + block - 1) // block,)](
                x, out, n, BLOCK=block, num_warps=warps
            )
            continue

        for start in range(0, total_items, width):
            chunk = items[start : start + width]
            if len(chunk) == 1:
                x, out, n = chunk[0]
                _log_flat_kernel[((n + block - 1) // block,)](
                    x, out, n, BLOCK=block, num_warps=warps
                )
                continue
            xs = []
            os_ = []
            ns = []
            total = 0
            for it in chunk:
                xs.append(it[0])
                os_.append(it[1])
                ns.append(it[2])
                total += (it[2] + block - 1) // block
            last_x = xs[-1]
            last_o = os_[-1]
            while len(xs) < width:
                xs.append(last_x)
                os_.append(last_o)
                ns.append(0)
            kernel[(total,)](*xs, *os_, *ns, BLOCK=block, num_warps=warps)

    return tuple(outs)
