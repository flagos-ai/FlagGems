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

"""Triton implementation of aten::_foreach_sub_.List (in-place, list of tensors).

self[i] <- self[i] - alpha * other[i]   (elementwise, in place, no output allocation)

Design:
  * homogeneous lists (<= _MAXK contiguous tensors of one shape/dtype/device) are
    updated by a single launch of a 2-D grid kernel, grid = (blocks, nlist), so the
    kernel needs only 10 runtime arguments (8 pointers + n + alpha).
  * heterogeneous / non-contiguous members fall back to a bound-split flat kernel
    and a rank-specialized strided kernel (zero strides supported, i.e. expanded
    ``other``).
  * math dtype follows ATen opmath: fp32 for fp16/bf16, native for fp32/fp64/ints.
  * the per-call host cost dominates tiny-list workloads, so launches replay a
    cached compiled-kernel handle with the runtime driver stream and no
    launch-metadata/hooks (skipping Triton's arg binder, specialization-key
    computation and per-call LazyDict), and the device pointers are handed to the
    launcher as plain integers (its PyLong fast path).  The cache key repeats
    everything Triton specializes on: pointer dtype/device, the 16B alignment
    residue of every pointer, integer scalar values and constexpr options.  A miss
    or any exception falls back to the ordinary JIT launch.
  * the tail mask is compiled out when numel is a multiple of BLOCK.
"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _tr_driver

_MAXK = 4
_NDMAX = 5

_KCACHE = {}
_KCACHE_MAX = 512
_DIRECT_OK = True
_DRIVER = None


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _fsub_compute(a, b, alpha, IS_F64: tl.constexpr, IS_INT: tl.constexpr):
    if IS_INT:
        v = a - alpha * b
    else:
        if IS_F64:
            v = a - alpha * b
        else:
            v = (a.to(tl.float32) - alpha * b.to(tl.float32)).to(a.dtype)
    return v


@triton.jit
def _fsub_line(
    s_ptr,
    o_ptr,
    base,
    n,
    alpha,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_INT: tl.constexpr,
):
    offs = tl.max_contiguous(tl.multiple_of(base + tl.arange(0, BLOCK), BLOCK), BLOCK)
    if NEED_MASK:
        mask = offs < n
        a = tl.load(s_ptr + offs, mask=mask)
        b = tl.load(o_ptr + offs, mask=mask)
        tl.store(s_ptr + offs, _fsub_compute(a, b, alpha, IS_F64, IS_INT), mask=mask)
    else:
        a = tl.load(s_ptr + offs)
        b = tl.load(o_ptr + offs)
        tl.store(s_ptr + offs, _fsub_compute(a, b, alpha, IS_F64, IS_INT))


@triton.jit
def _fsub_flat1(
    s_ptr,
    o_ptr,
    n,
    alpha,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_INT: tl.constexpr,
):
    pid = tl.program_id(0)
    _fsub_line(s_ptr, o_ptr, pid * BLOCK, n, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT)


@triton.jit
def _fsub_flat_s(
    s0,
    o0,
    s1,
    o1,
    s2,
    o2,
    s3,
    o3,
    n,
    alpha,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_INT: tl.constexpr,
):
    """Homogeneous chunk: grid = (blocks, nlist); one tensor per program_id(1)."""
    base = tl.program_id(0) * BLOCK
    j = tl.program_id(1)
    if j == 0:
        _fsub_line(s0, o0, base, n, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT)
    else:
        if j == 1:
            _fsub_line(s1, o1, base, n, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT)
        else:
            if j == 2:
                _fsub_line(s2, o2, base, n, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT)
            else:
                _fsub_line(s3, o3, base, n, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT)


@triton.jit
def _fsub_flat(
    s0,
    o0,
    s1,
    o1,
    s2,
    o2,
    s3,
    o3,
    n0,
    n1,
    n2,
    n3,
    b0,
    b1,
    b2,
    b3,
    alpha,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_INT: tl.constexpr,
):
    """Heterogeneous chunk: one 1-D grid split by host-computed block bounds."""
    pid = tl.program_id(0)
    if pid < b0:
        _fsub_line(s0, o0, pid * BLOCK, n0, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT)
    else:
        if pid < b1:
            _fsub_line(
                s1, o1, (pid - b0) * BLOCK, n1, alpha, BLOCK, NEED_MASK, IS_F64, IS_INT
            )
        else:
            if pid < b2:
                _fsub_line(
                    s2,
                    o2,
                    (pid - b1) * BLOCK,
                    n2,
                    alpha,
                    BLOCK,
                    NEED_MASK,
                    IS_F64,
                    IS_INT,
                )
            else:
                _fsub_line(
                    s3,
                    o3,
                    (pid - b2) * BLOCK,
                    n3,
                    alpha,
                    BLOCK,
                    NEED_MASK,
                    IS_F64,
                    IS_INT,
                )


@triton.jit
def _fsub_nd(
    s_ptr,
    o_ptr,
    n,
    alpha,
    d0,
    d1,
    d2,
    d3,
    d4,
    ss0,
    ss1,
    ss2,
    ss3,
    ss4,
    os0,
    os1,
    os2,
    os3,
    os4,
    RANK: tl.constexpr,
    OFLAT: tl.constexpr,
    SAME: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    IS_INT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    rem = offs
    so = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        i = rem % d0
        rem = rem // d0
        so += i * ss0
    if RANK > 1:
        i = rem % d1
        rem = rem // d1
        so += i * ss1
    if RANK > 2:
        i = rem % d2
        rem = rem // d2
        so += i * ss2
    if RANK > 3:
        i = rem % d3
        rem = rem // d3
        so += i * ss3
    if RANK > 4:
        i = rem % d4
        so += i * ss4

    if SAME:
        oo = so
    else:
        if OFLAT:
            oo = offs
        else:
            rem2 = offs
            oo = tl.zeros([BLOCK], dtype=tl.int64)
            if RANK > 0:
                i = rem2 % d0
                rem2 = rem2 // d0
                oo += i * os0
            if RANK > 1:
                i = rem2 % d1
                rem2 = rem2 // d1
                oo += i * os1
            if RANK > 2:
                i = rem2 % d2
                rem2 = rem2 // d2
                oo += i * os2
            if RANK > 3:
                i = rem2 % d3
                rem2 = rem2 // d3
                oo += i * os3
            if RANK > 4:
                i = rem2 % d4
                oo += i * os4

    a = tl.load(s_ptr + so, mask=mask)
    b = tl.load(o_ptr + oo, mask=mask)
    tl.store(s_ptr + so, _fsub_compute(a, b, alpha, IS_F64, IS_INT), mask=mask)


# --------------------------------------------------------------------------- #
# host helpers
# --------------------------------------------------------------------------- #
def _np2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _cfg(numel):
    """Fixed (BLOCK, num_warps) heuristic; autotune is unsafe for in-place work."""
    if numel <= 256:
        return max(32, _np2(numel)), 1
    if numel <= 4096:
        return 256, 2
    if numel <= 262144:
        return 512, 4
    return 1024, 4


def _raw_launch(comp, gx, gy, args, block, need_mask, is_f64, is_int):
    """Replay a compiled kernel: driver stream, no launch metadata and no hooks."""
    global _DRIVER
    d = _DRIVER
    if d is None:
        d = _DRIVER = _tr_driver.active
    dev = d.get_current_device()
    stream = d.get_current_stream(dev)
    comp.run(
        gx,
        gy,
        1,
        stream,
        comp.function,
        comp.packed_metadata,
        None,
        None,
        None,
        *args,
        block,
        need_mask,
        is_f64,
        is_int,
    )


def _cached_launch(
    kern_fn, key, gx, gy, targs, dargs, block, need_mask, is_f64, is_int, nw
):
    """Launch through the cached compiled handle, recompiling via JIT on a miss.

    ``targs`` holds tensor arguments for the JIT fallback, ``dargs`` the same
    values (integer pointers) for the precompiled replay.
    """
    global _DIRECT_OK
    if _DIRECT_OK:
        comp = _KCACHE.get(key)
        if comp is not None:
            try:
                _raw_launch(comp, gx, gy, dargs, block, need_mask, is_f64, is_int)
                return comp
            except Exception:
                _KCACHE.pop(key, None)
    comp = kern_fn[(gx, gy)](
        *targs,
        BLOCK=block,
        NEED_MASK=need_mask,
        IS_F64=is_f64,
        IS_INT=is_int,
        num_warps=nw,
    )
    if _DIRECT_OK:
        if hasattr(comp, "run") and hasattr(comp, "packed_metadata"):
            if len(_KCACHE) >= _KCACHE_MAX:
                _KCACHE.clear()
            _KCACHE[key] = comp
        else:
            _DIRECT_OK = False
    return comp


def _nd_dims(shape, sst, ost):
    """Innermost-first (size, self_stride, other_stride) with size-1 dropped and
    mergeable adjacent dims collapsed."""
    sz = []
    ss = []
    os_ = []
    for d in range(len(shape) - 1, -1, -1):
        sd = shape[d]
        if sd == 1:
            continue
        sz.append(sd)
        ss.append(sst[d])
        os_.append(ost[d])
    i = 0
    while i + 1 < len(sz):
        if ss[i] * sz[i] == ss[i + 1] and os_[i] * sz[i] == os_[i + 1]:
            sz[i + 1] = sz[i + 1] * sz[i]
            del sz[i]
            del ss[i]
            del os_[i]
        else:
            i += 1
    return sz, ss, os_


def _launch_nd(s, o, nelem, alpha_v, is_f64, is_int):
    sz, ss, os_ = _nd_dims(s.shape, s.stride(), o.stride())
    if len(sz) > _NDMAX:
        n0 = s.shape[0]
        for i in range(n0):
            si = s.select(0, i)
            oi = o.select(0, i)
            _launch_nd(si, oi, si.numel(), alpha_v, is_f64, is_int)
        return
    rank = len(sz)
    while len(sz) < _NDMAX:
        sz.append(1)
        ss.append(0)
        os_.append(0)
    BLOCK, nw = _cfg(nelem)
    same = tuple(s.stride()) == tuple(o.stride())
    _fsub_nd[((nelem + BLOCK - 1) // BLOCK,)](
        s,
        o,
        nelem,
        alpha_v,
        sz[0],
        sz[1],
        sz[2],
        sz[3],
        sz[4],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        os_[0],
        os_[1],
        os_[2],
        os_[3],
        os_[4],
        RANK=rank,
        OFLAT=o.is_contiguous(),
        SAME=same,
        BLOCK=BLOCK,
        IS_F64=is_f64,
        IS_INT=is_int,
        num_warps=nw,
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def foreach_sub_list_(self, other, alpha=1):
    if isinstance(self, (list, tuple)):
        sel = self
        nlist = len(sel)
    else:
        sel = (self,)
        nlist = 1

    if isinstance(other, (list, tuple)):
        if len(other) != nlist:
            raise RuntimeError("Tensor list must match in size")
        oth = other
    else:
        oth = (other,) * nlist

    if nlist == 0:
        return None

    if type(alpha) is not float and type(alpha) is not int:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        else:
            alpha = float(alpha)

    # ---- fast path: <= _MAXK contiguous tensors of one shape/dtype/device --- #
    s0 = sel[0]
    n0 = s0.numel()
    if 0 < nlist <= _MAXK and n0 > 0:
        o0 = oth[0]
        dtype = s0.dtype
        dev = s0.device
        ok = o0.shape == s0.shape and s0.is_contiguous() and o0.is_contiguous()
        tens = [s0, o0]
        if ok:
            for i in range(1, _MAXK):
                if i < nlist:
                    s = sel[i]
                    o = oth[i]
                    if (
                        s.dtype is not dtype
                        or s.device != dev
                        or s.shape != o.shape
                        or not s.is_contiguous()
                        or not o.is_contiguous()
                        or s.numel() != n0
                    ):
                        ok = False
                        break
                    tens.append(s)
                    tens.append(o)
                else:
                    tens.append(s0)
                    tens.append(o0)
        if ok:
            is_f64 = dtype == torch.float64
            is_int = not dtype.is_floating_point
            av = int(alpha) if is_int else float(alpha)
            block, nw = _cfg(n0)
            need_mask = (n0 % block) != 0
            p0 = tens[0].data_ptr()
            p1 = tens[1].data_ptr()
            p2 = tens[2].data_ptr()
            p3 = tens[3].data_ptr()
            p4 = tens[4].data_ptr()
            p5 = tens[5].data_ptr()
            p6 = tens[6].data_ptr()
            p7 = tens[7].data_ptr()
            _cached_launch(
                _fsub_flat_s,
                (
                    dtype,
                    dev,
                    block,
                    nw,
                    nlist,
                    need_mask,
                    is_f64,
                    is_int,
                    n0,
                    p0 & 15,
                    p1 & 15,
                    p2 & 15,
                    p3 & 15,
                    p4 & 15,
                    p5 & 15,
                    p6 & 15,
                    p7 & 15,
                ),
                (n0 + block - 1) // block,
                nlist,
                tens + [n0, av],
                [
                    p0,
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    p7,
                    n0,
                    av,
                ],
                block,
                need_mask,
                is_f64,
                is_int,
                nw,
            )
            return None

    # ---- general path: group by (dtype, device), chunk, fuse flat chunks ---- #
    groups = None
    for i in range(nlist):
        s = sel[i]
        o = oth[i]
        if o.shape != s.shape:
            o = o.expand(s.shape)
        nelem = s.numel()
        if nelem == 0:
            continue
        key = (s.dtype, s.device)
        if groups is None:
            groups = {}
        g = groups.get(key)
        if g is None:
            groups[key] = g = []
        g.append((s, o, nelem, s.is_contiguous(), o.is_contiguous()))

    if groups is None:
        return None

    for (dtype, device), items in groups.items():
        is_f64 = dtype == torch.float64
        is_int = not dtype.is_floating_point
        alpha_v = int(alpha) if is_int else float(alpha)

        cnt = len(items)
        start = 0
        while start < cnt:
            chunk = items[start : start + _MAXK]
            start += _MAXK
            flat_ok = True
            for it in chunk:
                if not (it[3] and it[4]):
                    flat_ok = False
                    break
            if flat_ok:
                nmax = 0
                for it in chunk:
                    if it[2] > nmax:
                        nmax = it[2]
                BLOCK, nw = _cfg(nmax)
                pts = []
                nas = []
                bas = []
                acc = 0
                for it in chunk:
                    acc += (it[2] + BLOCK - 1) // BLOCK
                    pts.append(it[0])
                    pts.append(it[1])
                    nas.append(it[2])
                    bas.append(acc)
                total = acc
                s_first, o_first = pts[0], pts[1]
                while len(nas) < _MAXK:
                    pts.append(s_first)
                    pts.append(o_first)
                    nas.append(0)
                    bas.append(total)
                ipts = [t.data_ptr() for t in pts]
                pts.extend(nas)
                pts.extend(bas)
                pts.append(alpha_v)
                ipts.extend(nas)
                ipts.extend(bas)
                ipts.append(alpha_v)
                _cached_launch(
                    _fsub_flat,
                    (
                        dtype,
                        device,
                        BLOCK,
                        nw,
                        is_f64,
                        is_int,
                        nas[0],
                        nas[1],
                        nas[2],
                        nas[3],
                        bas[0],
                        bas[1],
                        bas[2],
                        bas[3],
                        ipts[0] & 15,
                        ipts[1] & 15,
                        ipts[2] & 15,
                        ipts[3] & 15,
                        ipts[4] & 15,
                        ipts[5] & 15,
                        ipts[6] & 15,
                        ipts[7] & 15,
                    ),
                    total,
                    1,
                    pts,
                    ipts,
                    BLOCK,
                    True,
                    is_f64,
                    is_int,
                    nw,
                )
            else:
                for it in chunk:
                    s, o, nelem, sc, oc = it
                    if sc and oc:
                        BLOCK, nw = _cfg(nelem)
                        _fsub_flat1[((nelem + BLOCK - 1) // BLOCK,)](
                            s,
                            o,
                            nelem,
                            alpha_v,
                            BLOCK=BLOCK,
                            NEED_MASK=(nelem % BLOCK) != 0,
                            IS_F64=is_f64,
                            IS_INT=is_int,
                            num_warps=nw,
                        )
                    else:
                        _launch_nd(s, o, nelem, alpha_v, is_f64, is_int)

    return None
