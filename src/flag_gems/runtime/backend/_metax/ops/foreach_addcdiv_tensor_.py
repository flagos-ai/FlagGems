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

"""Triton implementation of aten::_foreach_addcdiv_.Tensor (in-place).

self_i += scalars_i * (tensor1_i / tensor2_i)

Launch-cost architecture (the tiny-list cases are dominated by host-side kernel
binding, not by GPU work):
  * fast path  : homogeneous tensor list (same dtype/device/contiguity/numel,
                 n <= 16) -> ONE launch, 2-D grid (tiles, n) where program
                 axis 1 selects the list element.  Element base addresses are
                 handed over as plain int64 scalars (one data_ptr each) and
                 cast to pointers in-kernel, so the launch carries only two
                 tensor arguments (a dtype anchor and the scalars tensor)
                 instead of 48.
  * general    : lists that mix dtypes/devices/shapes/contiguity fall back to a
                 per-(dtype, device, contiguity) fused launch with runtime
                 prefix-bound arm dispatch, plus a strided kernel for
                 non-contiguous elements.
fp16/bf16/fp32 use fp32 opmath (ATen opmath_t == float); fp64 uses fp64.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_BLOCK_HALF = 2048
_ARM = 16

_META = []
for _i in range(_ARM):
    _META.append("x%d" % _i)
for _i in range(_ARM):
    _META.append("n%d" % _i)

_PNAMES = []
for _i in range(_ARM):
    _PNAMES.append("a%d" % _i)
for _i in range(_ARM):
    _PNAMES.append("b%d" % _i)
for _i in range(_ARM):
    _PNAMES.append("c%d" % _i)
_PNAMES.append("numel")

_ZEROS = [0.0] * _ARM


@triton.jit
def _tile(
    a_ptr,
    b_ptr,
    c_ptr,
    n,
    t,
    s,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = t * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        c = tl.load(c_ptr + offs)
        if F64:
            r = a + s * (b / c)
        else:
            r = a.to(tl.float32) + s * (b.to(tl.float32) / c.to(tl.float32))
        tl.store(a_ptr + offs, r)
    else:
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        c = tl.load(c_ptr + offs, mask=mask, other=0.0)
        if F64:
            r = a + s * (b / c)
        else:
            r = a.to(tl.float32) + s * (b.to(tl.float32) / c.to(tl.float32))
        tl.store(a_ptr + offs, r, mask=mask)


@triton.jit(do_not_specialize=_PNAMES)
def _fused_ptrs(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    anchor,
    s_ptr,
    numel,
    s0: tl.constexpr,
    s1: tl.constexpr,
    s2: tl.constexpr,
    s3: tl.constexpr,
    s4: tl.constexpr,
    s5: tl.constexpr,
    s6: tl.constexpr,
    s7: tl.constexpr,
    s8: tl.constexpr,
    s9: tl.constexpr,
    s10: tl.constexpr,
    s11: tl.constexpr,
    s12: tl.constexpr,
    s13: tl.constexpr,
    s14: tl.constexpr,
    s15: tl.constexpr,
    USE_S_PTR: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    e = tl.program_id(1)
    pty = tl.pointer_type(anchor.dtype.element_ty)
    if e == 0:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 0)
        else:
            sv = s0
        _tile(
            tl.cast(a0, pty),
            tl.cast(b0, pty),
            tl.cast(c0, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 1:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 1)
        else:
            sv = s1
        _tile(
            tl.cast(a1, pty),
            tl.cast(b1, pty),
            tl.cast(c1, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 2:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 2)
        else:
            sv = s2
        _tile(
            tl.cast(a2, pty),
            tl.cast(b2, pty),
            tl.cast(c2, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 3:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 3)
        else:
            sv = s3
        _tile(
            tl.cast(a3, pty),
            tl.cast(b3, pty),
            tl.cast(c3, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 4:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 4)
        else:
            sv = s4
        _tile(
            tl.cast(a4, pty),
            tl.cast(b4, pty),
            tl.cast(c4, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 5:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 5)
        else:
            sv = s5
        _tile(
            tl.cast(a5, pty),
            tl.cast(b5, pty),
            tl.cast(c5, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 6:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 6)
        else:
            sv = s6
        _tile(
            tl.cast(a6, pty),
            tl.cast(b6, pty),
            tl.cast(c6, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 7:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 7)
        else:
            sv = s7
        _tile(
            tl.cast(a7, pty),
            tl.cast(b7, pty),
            tl.cast(c7, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 8:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 8)
        else:
            sv = s8
        _tile(
            tl.cast(a8, pty),
            tl.cast(b8, pty),
            tl.cast(c8, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 9:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 9)
        else:
            sv = s9
        _tile(
            tl.cast(a9, pty),
            tl.cast(b9, pty),
            tl.cast(c9, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 10:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 10)
        else:
            sv = s10
        _tile(
            tl.cast(a10, pty),
            tl.cast(b10, pty),
            tl.cast(c10, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 11:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 11)
        else:
            sv = s11
        _tile(
            tl.cast(a11, pty),
            tl.cast(b11, pty),
            tl.cast(c11, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 12:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 12)
        else:
            sv = s12
        _tile(
            tl.cast(a12, pty),
            tl.cast(b12, pty),
            tl.cast(c12, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 13:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 13)
        else:
            sv = s13
        _tile(
            tl.cast(a13, pty),
            tl.cast(b13, pty),
            tl.cast(c13, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    elif e == 14:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 14)
        else:
            sv = s14
        _tile(
            tl.cast(a14, pty),
            tl.cast(b14, pty),
            tl.cast(c14, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )
    else:
        if USE_S_PTR:
            sv = tl.load(s_ptr + 15)
        else:
            sv = s15
        _tile(
            tl.cast(a15, pty),
            tl.cast(b15, pty),
            tl.cast(c15, pty),
            numel,
            t,
            sv,
            EVEN,
            F64,
            BLOCK,
        )


@triton.jit(do_not_specialize=_META)
def _fused16(
    a0,
    b0,
    c0,
    a1,
    b1,
    c1,
    a2,
    b2,
    c2,
    a3,
    b3,
    c3,
    a4,
    b4,
    c4,
    a5,
    b5,
    c5,
    a6,
    b6,
    c6,
    a7,
    b7,
    c7,
    a8,
    b8,
    c8,
    a9,
    b9,
    c9,
    a10,
    b10,
    c10,
    a11,
    b11,
    c11,
    a12,
    b12,
    c12,
    a13,
    b13,
    c13,
    a14,
    b14,
    c14,
    a15,
    b15,
    c15,
    s_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
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
    USE_S_PTR: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < x0:
        if USE_S_PTR:
            s = tl.load(s_ptr + 0)
        else:
            s = s0
        _tile(a0, b0, c0, n0, pid, s, EVEN, F64, BLOCK)
    elif pid < x1:
        if USE_S_PTR:
            s = tl.load(s_ptr + 1)
        else:
            s = s1
        _tile(a1, b1, c1, n1, pid - x0, s, EVEN, F64, BLOCK)
    elif pid < x2:
        if USE_S_PTR:
            s = tl.load(s_ptr + 2)
        else:
            s = s2
        _tile(a2, b2, c2, n2, pid - x1, s, EVEN, F64, BLOCK)
    elif pid < x3:
        if USE_S_PTR:
            s = tl.load(s_ptr + 3)
        else:
            s = s3
        _tile(a3, b3, c3, n3, pid - x2, s, EVEN, F64, BLOCK)
    elif pid < x4:
        if USE_S_PTR:
            s = tl.load(s_ptr + 4)
        else:
            s = s4
        _tile(a4, b4, c4, n4, pid - x3, s, EVEN, F64, BLOCK)
    elif pid < x5:
        if USE_S_PTR:
            s = tl.load(s_ptr + 5)
        else:
            s = s5
        _tile(a5, b5, c5, n5, pid - x4, s, EVEN, F64, BLOCK)
    elif pid < x6:
        if USE_S_PTR:
            s = tl.load(s_ptr + 6)
        else:
            s = s6
        _tile(a6, b6, c6, n6, pid - x5, s, EVEN, F64, BLOCK)
    elif pid < x7:
        if USE_S_PTR:
            s = tl.load(s_ptr + 7)
        else:
            s = s7
        _tile(a7, b7, c7, n7, pid - x6, s, EVEN, F64, BLOCK)
    elif pid < x8:
        if USE_S_PTR:
            s = tl.load(s_ptr + 8)
        else:
            s = s8
        _tile(a8, b8, c8, n8, pid - x7, s, EVEN, F64, BLOCK)
    elif pid < x9:
        if USE_S_PTR:
            s = tl.load(s_ptr + 9)
        else:
            s = s9
        _tile(a9, b9, c9, n9, pid - x8, s, EVEN, F64, BLOCK)
    elif pid < x10:
        if USE_S_PTR:
            s = tl.load(s_ptr + 10)
        else:
            s = s10
        _tile(a10, b10, c10, n10, pid - x9, s, EVEN, F64, BLOCK)
    elif pid < x11:
        if USE_S_PTR:
            s = tl.load(s_ptr + 11)
        else:
            s = s11
        _tile(a11, b11, c11, n11, pid - x10, s, EVEN, F64, BLOCK)
    elif pid < x12:
        if USE_S_PTR:
            s = tl.load(s_ptr + 12)
        else:
            s = s12
        _tile(a12, b12, c12, n12, pid - x11, s, EVEN, F64, BLOCK)
    elif pid < x13:
        if USE_S_PTR:
            s = tl.load(s_ptr + 13)
        else:
            s = s13
        _tile(a13, b13, c13, n13, pid - x12, s, EVEN, F64, BLOCK)
    elif pid < x14:
        if USE_S_PTR:
            s = tl.load(s_ptr + 14)
        else:
            s = s14
        _tile(a14, b14, c14, n14, pid - x13, s, EVEN, F64, BLOCK)
    else:
        if USE_S_PTR:
            s = tl.load(s_ptr + 15)
        else:
            s = s15
        _tile(a15, b15, c15, n15, pid - x14, s, EVEN, F64, BLOCK)


@triton.jit(
    do_not_specialize=["n", "s_idx", "d0", "d1", "d2", "d3", "t0", "t1", "t2", "t3"]
)
def _addcdiv_strided_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    s_ptr,
    s_val,
    n,
    s_idx,
    d0,
    d1,
    d2,
    d3,
    t0,
    t1,
    t2,
    t3,
    USE_S_PTR: tl.constexpr,
    F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = (rem % d0) * t0
    rem = rem // d0
    off += (rem % d1) * t1
    rem = rem // d1
    off += (rem % d2) * t2
    rem = rem // d2
    off += (rem % d3) * t3
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    c = tl.load(c_ptr + off, mask=mask, other=0.0)
    if USE_S_PTR:
        s = tl.load(s_ptr + s_idx)
    else:
        s = s_val
    if F64:
        r = a + s.to(tl.float64) * (b / c)
    else:
        r = a.to(tl.float32) + s.to(tl.float32) * (b.to(tl.float32) / c.to(tl.float32))
    tl.store(a_ptr + off, r, mask=mask)


def _split_dims(t):
    sizes = list(t.shape)
    strides = list(t.stride())
    while len(sizes) < 4:
        sizes.append(1)
        strides.append(0)
    return (
        int(sizes[0]),
        int(sizes[1]),
        int(sizes[2]),
        int(sizes[3]),
        int(strides[0]),
        int(strides[1]),
        int(strides[2]),
        int(strides[3]),
    )


def _launch_fused(idxs, self, t1, t2, f64, even, use_s_ptr, s_view, s_vals, blk):
    total = 0
    ptrs = []
    xs = []
    ns = []
    for i in idxs:
        numel = self[i].numel()
        total += (numel + blk - 1) // blk
        xs.append(total)
        ns.append(numel)
        ptrs.append(self[i])
        ptrs.append(t1[i])
        ptrs.append(t2[i])
    # pad unused arms with arm-0 pointers / terminal bounds
    while len(ptrs) < 3 * _ARM:
        ptrs.append(ptrs[0])
        ptrs.append(ptrs[1])
        ptrs.append(ptrs[2])
        xs.append(total)
        ns.append(0)
    args = ptrs
    if use_s_ptr:
        args.append(s_view)
        args.extend([0.0] * _ARM)
    else:
        args.append(ptrs[0])
        args.extend(s_vals)
    args.extend(xs)
    args.extend(ns)
    _fused16[(total,)](
        *args,
        USE_S_PTR=use_s_ptr,
        EVEN=even,
        F64=f64,
        BLOCK=blk,
        num_warps=4,
    )


def foreach_addcdiv_tensor_(self, tensor1, tensor2, scalars):
    n = len(self)
    if n == 0:
        return None

    if isinstance(scalars, torch.Tensor):
        if scalars.dim() == 1 and scalars.numel() == n and scalars.device.type != "cpu":
            s_ptr = scalars
            s_vals = None
        else:
            if scalars.dim() > 0:
                s_vals = scalars.tolist()
                if len(s_vals) < n:
                    s_vals = list(s_vals) * n
            else:
                s_vals = [scalars.item()] * n
            s_ptr = None
    else:
        s_vals = [float(scalars)] * n
        s_ptr = None

    # ---- fast path: homogeneous list -> one launch, two tensor arguments --
    a0 = self[0]
    dtype = a0.dtype
    dev = a0.device
    numel0 = a0.numel()
    homog = n <= _ARM and numel0 > 0 and len(tensor1) >= n and len(tensor2) >= n
    if homog:
        pa = []
        pb = []
        pc = []
        ea = pa.append
        eb = pb.append
        ec = pc.append
        for a, b, c in zip(self, tensor1, tensor2):
            if (
                a.numel() != numel0
                or a.dtype is not dtype
                or a.device != dev
                or not a.is_contiguous()
                or not b.is_contiguous()
                or not c.is_contiguous()
            ):
                homog = False
                break
            ea(a.data_ptr())
            eb(b.data_ptr())
            ec(c.data_ptr())

    if homog:
        pad = _ARM - n
        if pad:
            pa += [pa[0]] * pad
            pb += [pb[0]] * pad
            pc += [pc[0]] * pad
        use_s_ptr = s_ptr is not None
        blk = _BLOCK_HALF if dtype.itemsize <= 2 else _BLOCK
        args = pa + pb + pc
        args.append(a0)
        args.append(s_ptr if use_s_ptr else a0)
        args.append(numel0)
        if use_s_ptr:
            args.extend(_ZEROS)
        else:
            sv = s_vals[:n]
            if pad:
                sv.extend(_ZEROS[:pad])
            args.extend(sv)
        _fused_ptrs[((numel0 + blk - 1) // blk, n)](
            *args,
            USE_S_PTR=use_s_ptr,
            EVEN=(numel0 % blk == 0),
            F64=(dtype is torch.float64),
            BLOCK=blk,
            num_warps=4,
        )
        return None

    # ---- general path: group by (dtype, device, contiguity) ---------------
    buckets = {}
    for i in range(n):
        a = self[i]
        if a.numel() == 0:
            continue
        contig = (
            a.is_contiguous()
            and tensor1[i].is_contiguous()
            and tensor2[i].is_contiguous()
        )
        key = (a.dtype, a.device, contig)
        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = [i]
        else:
            bucket.append(i)

    for (dtype, device, contig), idxs in buckets.items():
        f64 = dtype is torch.float64
        blk = _BLOCK_HALF if dtype.itemsize <= 2 else _BLOCK
        use_s_ptr = s_ptr is not None
        if not contig:
            for i in idxs:
                a = self[i]
                numel = a.numel()
                if use_s_ptr:
                    sp = s_ptr
                    s_val = 0.0
                    s_idx = i
                else:
                    sp = a
                    s_val = float(s_vals[i])
                    s_idx = 0
                d0, d1, d2, d3, t0, t1, t2, t3 = _split_dims(a)
                _addcdiv_strided_kernel[((numel + blk - 1) // blk,)](
                    a,
                    tensor1[i],
                    tensor2[i],
                    sp,
                    s_val,
                    numel,
                    s_idx,
                    d0,
                    d1,
                    d2,
                    d3,
                    t0,
                    t1,
                    t2,
                    t3,
                    USE_S_PTR=use_s_ptr,
                    F64=f64,
                    BLOCK=blk,
                    num_warps=4,
                )
            continue
        even = True
        for i in idxs:
            if self[i].numel() % blk:
                even = False
                break
        for c in range(0, len(idxs), _ARM):
            chunk = idxs[c : c + _ARM]
            if use_s_ptr:
                _launch_fused(
                    chunk,
                    self,
                    tensor1,
                    tensor2,
                    f64,
                    even,
                    True,
                    s_ptr.narrow(0, c, len(chunk)),
                    None,
                    blk,
                )
            else:
                sv = [float(s_vals[i]) for i in chunk]
                if len(sv) < _ARM:
                    sv.extend([0.0] * (_ARM - len(sv)))
                _launch_fused(
                    chunk,
                    self,
                    tensor1,
                    tensor2,
                    f64,
                    even,
                    False,
                    None,
                    sv,
                    blk,
                )
    return None
