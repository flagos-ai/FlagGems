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

"""Triton implementation of aten::_foreach_maximum_.ScalarList.

In-place elementwise maximum between each tensor of a tensor list and the
paired scalar (or a single broadcast scalar).  Storage of every input tensor is
preserved; only values change.  Returns ``None``.

Contiguous tensors that share dtype and element count are fused into a single
kernel launch (up to 4 per launch).  The fused launch re-uses the compiled
kernel handle directly so a repeated call does not pay Triton's per-call
specialization/cache-key dispatch, and the most common fused case (four
same-shape/same-dtype contiguous non-float64 tensors) uses a lean kernel with
the smallest possible argument list.  Everything else falls back to one launch
per tensor.
"""

import logging
import numbers
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_PACK_D = struct.Struct("<d")
_UNPACK_Q = struct.Struct("<q")

_F64 = torch.float64
_MAX_I32 = 2147483647

# (device, dtype, blocks, tensors-per-launch, block, large) -> callable runner
# bound to the already-compiled kernel of _fmax_multi / _fmax4.
_RUNNERS = {}
_RUNNERS4 = {}


@triton.jit
def _fmax_flat(
    x_ptr,
    n,
    s_f32,
    s_bits,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_F64: tl.constexpr,
    LARGE: tl.constexpr,
):
    pid = tl.program_id(0)
    if LARGE:
        pid = pid.to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if IS_F64:
        s = tl.cast(s_bits.to(tl.int64), tl.float64, bitcast=True)
        x = tl.maximum(x, s, propagate_nan=tl.PropagateNan.ALL)
    elif IS_FLOAT:
        x = tl.maximum(x, s_f32.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    else:
        x = tl.maximum(x, s_f32.to(x.dtype))
    tl.store(x_ptr + offs, x, mask=mask)


@triton.jit
def _fmax4(
    p0, p1, p2, p3, v0, v1, v2, v3, n, BLOCK: tl.constexpr, IS_FLOAT: tl.constexpr
):
    ib = tl.program_id(0)
    ti = tl.program_id(1)
    offs = ib * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    p = p0
    sv = v0
    if ti == 1:
        p = p1
        sv = v1
    if ti == 2:
        p = p2
        sv = v2
    if ti == 3:
        p = p3
        sv = v3

    x = tl.load(p + offs, mask=mask, eviction_policy="evict_first")
    if IS_FLOAT:
        x = tl.maximum(x, sv.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    else:
        x = tl.maximum(x, sv.to(x.dtype))
    tl.store(p + offs, x, mask=mask, eviction_policy="evict_first")


@triton.jit
def _fmax_multi(
    p0,
    p1,
    p2,
    p3,
    v0,
    v1,
    v2,
    v3,
    b0,
    b1,
    b2,
    b3,
    n,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_F64: tl.constexpr,
    LARGE: tl.constexpr,
):
    ib = tl.program_id(0)
    ti = tl.program_id(1)
    if LARGE:
        ib = ib.to(tl.int64)
    offs = ib * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    p = p0
    sv = v0
    sb = b0
    if NT > 1:
        if ti == 1:
            p = p1
            sv = v1
            sb = b1
    if NT > 2:
        if ti == 2:
            p = p2
            sv = v2
            sb = b2
    if NT > 3:
        if ti == 3:
            p = p3
            sv = v3
            sb = b3

    x = tl.load(p + offs, mask=mask)
    if IS_F64:
        s = tl.cast(sb.to(tl.int64), tl.float64, bitcast=True)
        x = tl.maximum(x, s, propagate_nan=tl.PropagateNan.ALL)
    elif IS_FLOAT:
        x = tl.maximum(x, sv.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    else:
        x = tl.maximum(x, sv.to(x.dtype))
    tl.store(p + offs, x, mask=mask)


@triton.jit
def _fmax_strided(
    x_ptr,
    inner_size,
    inner_stride,
    inner_blocks,
    s_f32,
    s_bits,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_F64: tl.constexpr,
):
    gid = tl.program_id(0)
    ib = gid % inner_blocks
    ob = gid // inner_blocks

    sizes = (o0, o1, o2, o3, o4, o5, o6, o7)
    steps = (t0, t1, t2, t3, t4, t5, t6, t7)
    rem = ob
    base = 0
    for k in tl.static_range(NDIM):
        base += (rem % sizes[k]) * steps[k]
        rem = rem // sizes[k]

    idx = ib * BLOCK + tl.arange(0, BLOCK)
    mask = idx < inner_size
    off = base + idx * inner_stride
    x = tl.load(x_ptr + off, mask=mask)
    if IS_F64:
        s = tl.cast(s_bits.to(tl.int64), tl.float64, bitcast=True)
        x = tl.maximum(x, s, propagate_nan=tl.PropagateNan.ALL)
    elif IS_FLOAT:
        x = tl.maximum(x, s_f32.to(x.dtype), propagate_nan=tl.PropagateNan.ALL)
    else:
        x = tl.maximum(x, s_f32.to(x.dtype))
    tl.store(x_ptr + off, x, mask=mask)


def _bits(v):
    return _UNPACK_Q.unpack(_PACK_D.pack(float(v)))[0]


def _plan_strided(t):
    """Collapse (shape, stride) metadata into (inner_size, inner_stride, outer)."""
    dims = [(int(sz), int(st)) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return 1, 1, ()
    dims.sort(key=lambda d: d[1])
    merged = []
    for sz, st in dims:
        if merged:
            msz, mst = merged[-1]
            if mst * msz == st:
                merged[-1] = (msz * sz, mst)
                continue
        merged.append((sz, st))
    inner_size, inner_stride = merged[0]
    return inner_size, inner_stride, tuple(merged[1:])


def _apply(tensors, vals, dev):
    total = len(tensors)
    i = 0
    while i < total:
        t = tensors[i]
        numel = t.numel()
        if numel == 0:
            i += 1
            continue
        dt = t.dtype
        is_float = dt.is_floating_point
        is_f64 = dt == _F64

        if not t.is_contiguous():
            inner_size, inner_stride, outer = _plan_strided(t)
            osz = [1, 1, 1, 1, 1, 1, 1, 1]
            ost = [0, 0, 0, 0, 0, 0, 0, 0]
            outer_size = 1
            for k, (sz, st) in enumerate(outer):
                osz[k] = sz
                ost[k] = st
                outer_size *= sz
            block = 1024 if inner_size > 1024 else 256
            inner_blocks = (inner_size + block - 1) // block
            _fmax_strided[(inner_blocks * outer_size,)](
                t,
                inner_size,
                inner_stride,
                inner_blocks,
                float(vals[i]),
                _bits(vals[i]),
                osz[0],
                osz[1],
                osz[2],
                osz[3],
                osz[4],
                osz[5],
                osz[6],
                osz[7],
                ost[0],
                ost[1],
                ost[2],
                ost[3],
                ost[4],
                ost[5],
                ost[6],
                ost[7],
                NDIM=len(outer),
                BLOCK=block,
                IS_FLOAT=is_float,
                IS_F64=is_f64,
                num_warps=4,
            )
            i += 1
            continue

        # Extend over following contiguous same-dtype / same-numel tensors so a
        # homogeneous list costs one launch instead of len(list) launches.
        # Aliasing storage is never fused (that would race).
        dptr = t.data_ptr()
        j = i + 1
        lim = i + 4
        if lim > total:
            lim = total
        while j < lim:
            u = tensors[j]
            if (
                u.dtype is not dt
                or u.numel() != numel
                or not u.is_contiguous()
                or u.data_ptr() == dptr
            ):
                break
            j += 1

        if j == i + 1:
            block = 1024
            nb = (numel + block - 1) // block
            _fmax_flat[(nb,)](
                t,
                numel,
                float(vals[i]),
                _bits(vals[i]),
                BLOCK=block,
                IS_FLOAT=is_float,
                IS_F64=is_f64,
                LARGE=numel > _MAX_I32,
                num_warps=4,
            )
            i = j
            continue

        nt = j - i
        # 2048 elements over 4 warps (256 lanes) gives 8 elements per thread,
        # i.e. 16-byte vector accesses for 16-bit dtypes too.
        block = 2048
        nb = (numel + block - 1) // block
        u = tensors[i + 1]
        v0 = float(vals[i])
        v1 = float(vals[i + 1])

        if nt == 4 and not is_f64 and numel <= _MAX_I32:
            # Lean path: 11 kernel slots instead of 18.
            y = tensors[i + 2]
            z = tensors[i + 3]
            v2 = float(vals[i + 2])
            v3 = float(vals[i + 3])
            key = (dev, dt, nb)
            runner = _RUNNERS4.get(key)
            if runner is None:
                ck = _fmax4[(nb, 4, 1)](
                    t,
                    u,
                    y,
                    z,
                    v0,
                    v1,
                    v2,
                    v3,
                    numel,
                    BLOCK=block,
                    IS_FLOAT=is_float,
                    num_warps=4,
                )
                if len(_RUNNERS4) > 256:
                    _RUNNERS4.clear()
                _RUNNERS4[key] = ck[(nb, 4, 1)]
            else:
                runner(t, u, y, z, v0, v1, v2, v3, numel, block, is_float)
            i = j
            continue

        if nt == 2:
            rargs = (t, u, t, t, v0, v1, 0.0, 0.0, 0, 0, 0, 0, numel)
        elif nt == 3:
            rargs = (
                t,
                u,
                tensors[i + 2],
                t,
                v0,
                v1,
                float(vals[i + 2]),
                0.0,
                0,
                0,
                0,
                0,
                numel,
            )
        else:
            rargs = (
                t,
                u,
                tensors[i + 2],
                tensors[i + 3],
                v0,
                v1,
                float(vals[i + 2]),
                float(vals[i + 3]),
                0,
                0,
                0,
                0,
                numel,
            )
        if is_f64:
            # The IEEE-754 bit pattern is only consumed by the float64 element
            # type; every other dtype uses the fp32 argument.
            b2 = _bits(vals[i + 2]) if nt > 2 else 0
            b3 = _bits(vals[i + 3]) if nt > 3 else 0
            rargs = rargs[:8] + (_bits(vals[i]), _bits(vals[i + 1]), b2, b3, numel)

        key = (dev, dt, nb, nt, block, numel > _MAX_I32)
        runner = _RUNNERS.get(key)
        if runner is None:
            ck = _fmax_multi[(nb, nt, 1)](
                *rargs,
                NT=nt,
                BLOCK=block,
                IS_FLOAT=is_float,
                IS_F64=is_f64,
                LARGE=numel > _MAX_I32,
                num_warps=4,
            )
            if len(_RUNNERS) > 512:
                _RUNNERS.clear()
            _RUNNERS[key] = ck[(nb, nt, 1)]
        else:
            # A compiled-kernel launch takes one slot per declared parameter,
            # constexprs included, in signature order.
            runner(*rargs, nt, block, is_float, is_f64, numel > _MAX_I32)
        i = j


def _foreach_maximum__ScalarList(self, scalars):
    logger.debug("GEMS_HYGON _FOREACH_MAXIMUM__SCALARLIST")
    tensors = self if isinstance(self, (list, tuple)) else [self]
    count = len(tensors)

    if type(scalars) is list:
        vals = scalars
        if len(vals) != count:
            raise RuntimeError("Tensor list and scalar list must have the same length.")
    elif isinstance(scalars, numbers.Number):
        vals = (scalars,) * count
    else:
        vals = list(scalars)
        if len(vals) != count:
            raise RuntimeError("Tensor list and scalar list must have the same length.")

    if count == 0:
        return None

    dev = tensors[0].get_device()
    if dev >= 0 and dev != torch.cuda.current_device():
        with torch.cuda.device(dev):
            _apply(tensors, vals, dev)
    else:
        _apply(tensors, vals, dev)
    return None
