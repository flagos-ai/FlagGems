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

"""_foreach_add.ScalarList implemented with Triton kernels.

out[i] = self[i] + scalars[i]   (scalar lifted to self[i].dtype, one rounding)
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_add_ScalarList as default_foreach_add_ScalarList,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ----------------------------------------------------------------- kernels


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=8),
    ],
    key=["n_elem"],
)
@triton.jit
def _add_scalar_kernel(
    x_ptr,
    o_ptr,
    n_elem,
    s,
    CAST_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    v = tl.load(x_ptr + offs, mask=mask)
    if CAST_S:
        r = v + s.to(v.dtype)
    else:
        r = v + s
    tl.store(o_ptr + offs, r, mask=mask)


@triton.jit
def _fused_tile(
    x_ptr,
    o_ptr,
    n_elem,
    s,
    base,
    lo,
    CAST_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Handle one tensor's tile range [lo, lo + padded(n_elem))."""
    hi = lo + ((n_elem + (BLOCK - 1)) // BLOCK) * BLOCK
    if (base >= lo) & (base < hi):
        offs = base - lo + tl.arange(0, BLOCK)
        mask = offs < n_elem
        v = tl.load(x_ptr + offs, mask=mask)
        if CAST_S:
            r = v + s.to(v.dtype)
        else:
            r = v + s
        tl.store(o_ptr + offs, r, mask=mask)
    return hi


@triton.jit
def _add_scalar_fused_kernel(
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
    CAST_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0) * BLOCK
    lo = base * 0
    lo = _fused_tile(x0, o0, n0, s0, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x1, o1, n1, s1, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x2, o2, n2, s2, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x3, o3, n3, s3, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x4, o4, n4, s4, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x5, o5, n5, s5, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x6, o6, n6, s6, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x7, o7, n7, s7, base, lo, CAST_S, BLOCK)


@triton.jit
def _add_scalar_same4_kernel(
    x0,
    x1,
    x2,
    x3,
    o0,
    o1,
    o2,
    o3,
    n_elem,
    s0,
    s1,
    s2,
    s3,
    CAST_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Four equal-length contiguous tensors in one launch (shared numel)."""
    base = tl.program_id(0) * BLOCK
    lo = base * 0
    lo = _fused_tile(x0, o0, n_elem, s0, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x1, o1, n_elem, s1, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x2, o2, n_elem, s2, base, lo, CAST_S, BLOCK)
    lo = _fused_tile(x3, o3, n_elem, s3, base, lo, CAST_S, BLOCK)


@triton.jit
def _add_scalar_strided_kernel(
    x_ptr,
    o_ptr,
    n_elem,
    s,
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
    CAST_S: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n_elem
    rem = idx
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    if NDIM >= 8:
        c = rem % sh7
        rem = rem // sh7
        off = off + c * st7
    if NDIM >= 7:
        c = rem % sh6
        rem = rem // sh6
        off = off + c * st6
    if NDIM >= 6:
        c = rem % sh5
        rem = rem // sh5
        off = off + c * st5
    if NDIM >= 5:
        c = rem % sh4
        rem = rem // sh4
        off = off + c * st4
    if NDIM >= 4:
        c = rem % sh3
        rem = rem // sh3
        off = off + c * st3
    if NDIM >= 3:
        c = rem % sh2
        rem = rem // sh2
        off = off + c * st2
    if NDIM >= 2:
        c = rem % sh1
        rem = rem // sh1
        off = off + c * st1
    if NDIM >= 1:
        c = rem % sh0
        rem = rem // sh0
        off = off + c * st0
    v = tl.load(x_ptr + off, mask=mask)
    if CAST_S:
        r = v + s.to(v.dtype)
    else:
        r = v + s
    tl.store(o_ptr + idx, r, mask=mask)


# ----------------------------------------------------------------- helpers

_MAX_FUSE = 8
_INT32_MAX = 2147483647


def _cfg(n_elem):
    """BLOCK / num_warps heuristic for an elementwise pass over n_elem items."""
    if n_elem >= 1024:
        return 1024, 4
    if n_elem > 512:
        return 512, 2
    if n_elem > 128:
        return 256, 1
    return 128, 1


def _padded_total(ns, block):
    total = 0
    for n in ns:
        total += ((n + block - 1) // block) * block
    return total


def _launch_one(x, o, n_elem, s, cast_s):
    if n_elem == 0:
        return
    grid = lambda meta: ((n_elem + meta["BLOCK"] - 1) // meta["BLOCK"],)
    _add_scalar_kernel[grid](x, o, n_elem, s, CAST_S=cast_s)


def _launch_fused(xs, outs, ns, ss, cast_s):
    """One launch covers the whole padded homogeneous contiguous group."""
    m = len(xs)
    if m == 1:
        _launch_one(xs[0], outs[0], ns[0], ss[0], cast_s)
        return
    block, warps = _cfg(max(ns))
    total = _padded_total(ns, block)
    if total == 0:
        return
    pad = _MAX_FUSE - m
    xs = xs + [xs[0]] * pad
    outs = outs + [outs[0]] * pad
    ns = ns + [0] * pad
    ss = ss + [0.0] * pad
    grid = ((total + block - 1) // block,)
    _add_scalar_fused_kernel[grid](
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
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
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        CAST_S=cast_s,
        BLOCK=block,
        num_warps=warps,
    )


def _collapse(t):
    """Drop size-1 dims and merge contiguous neighbours (metadata only)."""
    dims = []
    for sz, st in zip(t.shape, t.stride()):
        sz = int(sz)
        if sz != 1:
            dims.append((sz, int(st)))
    merged = []
    for sz, st in reversed(dims):
        if merged:
            psz, pst = merged[-1]
            if st == psz * pst:
                merged[-1] = (sz * psz, pst)
                continue
        merged.append((sz, st))
    merged.reverse()
    return merged


def _launch_strided(x, o, n_elem, s, dims, cast_s):
    if n_elem == 0:
        return
    nd = len(dims)
    if nd > 8:
        raise NotImplementedError("foreach_add: too many dimensions")
    sh = [1] * 8
    st = [0] * 8
    for i, (sz, stride) in enumerate(dims):
        sh[i] = sz
        st[i] = stride
    block, warps = _cfg(n_elem)
    grid = ((n_elem + block - 1) // block,)
    _add_scalar_strided_kernel[grid](
        x,
        o,
        n_elem,
        s,
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
        CAST_S=cast_s,
        NDIM=nd,
        BLOCK=block,
        num_warps=warps,
    )


def _out_dtype(t):
    dt = t.dtype
    if dt.is_floating_point:
        return dt
    return torch.result_type(t, 1.0)


# ----------------------------------------------------------------- entry


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_add_ScalarList(self, scalars, alpha=None):
    logger.debug("GEMS_MTHREADS _FOREACH_ADD_SCALARLIST")

    if not _list_supported(self):
        return default_foreach_add_ScalarList(self, scalars, alpha)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_add_ScalarList(self, scalars, alpha)


def _launch_foreach_add_ScalarList(self, scalars, alpha=None):
    tensors = self
    n = len(tensors)
    if n == 0:
        return []

    if isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            s_list = [scalars] * n
        else:
            s_list = list(scalars)
    elif isinstance(scalars, (list, tuple)):
        s_list = list(scalars)
    else:
        s_list = [scalars] * n

    if len(s_list) != n:
        raise RuntimeError(
            "_foreach_add_: expected the number of scalars to match the "
            "number of tensors"
        )
    if alpha is not None and alpha != 1:
        s_list = [v * alpha for v in s_list]
    for i, v in enumerate(s_list):
        if isinstance(v, torch.Tensor):
            s_list[i] = v.item()

    t0 = tensors[0]
    dev = t0.device
    dtype0 = t0.dtype
    homog = True
    contig = True
    for t in tensors:
        if t.dtype is not dtype0 or t.device != dev:
            homog = False
            break
        if not t.is_contiguous():
            contig = False

    if homog and contig and dtype0.is_floating_point:
        n0 = t0.numel()
        same = True
        for t in tensors:
            if t.numel() != n0:
                same = False
                break
        block, warps = _cfg(n0)
        if same:
            total = n * (((n0 + block - 1) // block) * block)
        else:
            total = _padded_total([t.numel() for t in tensors], block)
        if same and n == 4 and total < _INT32_MAX:
            # one allocation, four zero-copy views, one launch
            buf = torch.empty((4,) + t0.shape, dtype=dtype0, device=dev)
            outs = list(buf.unbind(0))
            if total:
                grid = ((total + block - 1) // block,)
                _add_scalar_same4_kernel[grid](
                    tensors[0],
                    tensors[1],
                    tensors[2],
                    tensors[3],
                    outs[0],
                    outs[1],
                    outs[2],
                    outs[3],
                    n0,
                    s_list[0],
                    s_list[1],
                    s_list[2],
                    s_list[3],
                    CAST_S=True,
                    BLOCK=block,
                    num_warps=warps,
                )
            return outs
        outs = [torch.empty(t.shape, dtype=dtype0, device=dev) for t in tensors]
        if n > 1 and total < _INT32_MAX:
            if same:
                _launch_fused(tensors, outs, [n0] * n, s_list, True)
            else:
                _launch_fused(tensors, outs, [t.numel() for t in tensors], s_list, True)
        else:
            for i in range(n):
                _launch_one(tensors[i], outs[i], tensors[i].numel(), s_list[i], True)
        return outs

    outs = [None] * n
    groups = {}
    order = []
    for i, t in enumerate(tensors):
        key = (t.device, t.dtype, t.is_contiguous())
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
            order.append(key)
        bucket.append(i)

    for key in order:
        idxs = groups[key]
        gdt = key[1]
        gdev = key[0]
        gcontig = key[2]
        odt = _out_dtype(tensors[idxs[0]])
        cast_s = (odt is gdt) and gdt.is_floating_point
        if gcontig:
            xs = []
            os_ = []
            ns = []
            ss = []
            for i in idxs:
                t = tensors[i]
                o = torch.empty(t.shape, dtype=odt, device=gdev)
                outs[i] = o
                xs.append(t)
                os_.append(o)
                ns.append(t.numel())
                ss.append(s_list[i])
            block, _warps = _cfg(max(ns))
            if len(xs) > 1 and _padded_total(ns, block) < _INT32_MAX:
                _launch_fused(xs, os_, ns, ss, cast_s)
                continue
            for j in range(len(xs)):
                _launch_one(xs[j], os_[j], ns[j], ss[j], cast_s)
        else:
            for i in idxs:
                t = tensors[i]
                o = torch.empty(t.shape, dtype=odt, device=gdev)
                outs[i] = o
                _launch_strided(t, o, t.numel(), s_list[i], _collapse(t), cast_s)
    return outs
