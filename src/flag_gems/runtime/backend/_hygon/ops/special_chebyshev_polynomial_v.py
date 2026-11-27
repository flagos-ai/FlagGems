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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim as libdevice

logger = logging.getLogger(__name__)
_BIG_NUMEL = 4 << 20
_BIG = (1024, 4)
_SMALL = (256, 4)


@triton.jit
def _chev_v_value(xv, nv):
    degree = nv.to(tl.int64)
    high = (degree > 8) & (tl.abs(xv) < 1.0)
    p = tl.full(xv.shape, 1.0, xv.dtype)
    q = xv + xv - 1.0
    active = (degree >= 2) & ~high & (tl.abs(xv) != 1.0)
    k = 2
    while tl.sum(active.to(tl.int32), 0) > 0:
        r = (xv + xv) * q - p
        p = tl.where(active, q, p)
        q = tl.where(active, r, q)
        k += 1
        active = (degree >= k) & ~high & (tl.abs(xv) != 1.0) & (q == q)
    result = tl.where(degree == 0, 1.0, q)
    if tl.sum(high.to(tl.int32), 0) > 0:
        th = libdevice.acos(xv)
        closed = libdevice.cos((degree.to(xv.dtype) + 0.5) * th) / libdevice.cos(
            0.5 * th
        )
        odd_value = tl.where(
            degree % 2 == 0, degree + degree + 1, -(degree + degree + 1)
        )
        closed = tl.where(libdevice.sin(th * 0.5) == 1.0, odd_value, closed)
        result = tl.where(high, closed, result)
    endpoint = tl.where(
        xv > 0,
        1.0,
        tl.where(degree % 2 == 0, degree + degree + 1, -(degree + degree + 1)),
    )
    result = tl.where(tl.abs(xv) == 1.0, endpoint, result)
    return tl.where(degree < 0, 0.0, result)


@triton.jit
def _chev_v_flat(
    x_ptr, n_ptr, out_ptr, numel, IS_F64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
    nv = tl.load(n_ptr + offs, mask=mask, other=0)
    if IS_F64:
        r = _chev_v_value(xv, nv)
    else:
        r = _chev_v_value(xv.to(tl.float32), nv)
    tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _chev_v_bcast(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    o0,
    o1,
    o2,
    o3,
    xs0,
    xs1,
    xs2,
    xs3,
    ns0,
    ns1,
    ns2,
    ns3,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    t = offs
    i3 = t % o3
    t = t // o3
    i2 = t % o2
    t = t // o2
    i1 = t % o1
    i0 = t // o1
    x_off = i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3
    n_off = i0 * ns0 + i1 * ns1 + i2 * ns2 + i3 * ns3
    xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    nv = tl.load(n_ptr + n_off, mask=mask, other=0)
    if IS_F64:
        r = _chev_v_value(xv, nv)
    else:
        r = _chev_v_value(xv.to(tl.float32), nv)
    tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty), mask=mask)


def _pad4(shape, strides, out_shape):
    """Right-align an operand's shape/strides to the 4-d output space."""
    nd = len(shape)
    pads = 4 - nd
    shp = (1,) * pads + tuple(shape)
    st = [0] * 4
    for i in range(nd):
        st[pads + i] = strides[i]
    for i in range(4):
        if shp[i] == 1 and out_shape[i] != 1:
            st[i] = 0
    return (shp, tuple(st))


def special_chebyshev_polynomial_v(x, n):
    logger.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_V")
    x_shape = x.shape
    n_shape = n.shape
    shape = x_shape if x_shape == n_shape else torch.broadcast_shapes(x_shape, n_shape)
    dt = x.dtype
    ndt = n.dtype
    if ndt != dt and n.is_floating_point():
        dt = torch.promote_types(dt, ndt)
    out = torch.empty(shape, dtype=dt, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out
    is_f64 = dt == torch.float64 and x.dtype == torch.float64
    block, warps = _BIG if numel >= _BIG_NUMEL else _SMALL
    if (
        x_shape == shape
        and n_shape == shape
        and x.is_contiguous()
        and n.is_contiguous()
    ):
        grid = ((numel + block - 1) // block,)
        _chev_v_flat[grid](
            x, n, out, numel, IS_F64=is_f64, BLOCK=block, num_warps=warps
        )
        return out
    oshp = tuple(out.shape)
    out_strides = (
        (oshp[1] * oshp[2] * oshp[3], oshp[2] * oshp[3], oshp[3], 1)
        if len(oshp) == 4
        else None
    )
    if out_strides is None:
        st = []
        acc = 1
        for s in reversed(oshp):
            st.append(acc)
            acc *= s
        out_strides = tuple(reversed(st))
        out_strides = (1,) * (4 - len(oshp)) + out_strides
    o4 = (1,) * (4 - len(oshp)) + oshp
    _, xs = _pad4(x.shape, x.stride(), o4)
    _, ns = _pad4(n.shape, n.stride(), o4)
    grid = ((numel + block - 1) // block,)
    _chev_v_bcast[grid](
        x,
        n,
        out,
        numel,
        o4[0],
        o4[1],
        o4[2],
        o4[3],
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        IS_F64=is_f64,
        BLOCK=block,
        num_warps=warps,
    )
    return out
