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
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_add__ScalarList as default_foreach_add__ScalarList,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _add_contig_kernel(x_ptr, n_elem, s, BLOCK: tl.constexpr, LONG: tl.constexpr):
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x + s, mask=mask)


@triton.jit
def _add_multi4_kernel(
    x0,
    n0,
    s0,
    x1,
    n1,
    s1,
    x2,
    n2,
    s2,
    x3,
    n3,
    s3,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    if g == 0:
        m = offs < n0
        v = tl.load(x0 + offs, mask=m)
        tl.store(x0 + offs, v + s0, mask=m)
    elif g == 1:
        m = offs < n1
        v = tl.load(x1 + offs, mask=m)
        tl.store(x1 + offs, v + s1, mask=m)
    elif g == 2:
        m = offs < n2
        v = tl.load(x2 + offs, mask=m)
        tl.store(x2 + offs, v + s2, mask=m)
    else:
        m = offs < n3
        v = tl.load(x3 + offs, mask=m)
        tl.store(x3 + offs, v + s3, mask=m)


@triton.jit
def _add_strided_kernel(
    x_ptr,
    n_elem,
    s,
    sz0,
    sz1,
    sz2,
    sz3,
    st0,
    st1,
    st2,
    st3,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    if RANK >= 1:
        i0 = rem % sz0
        rem = rem // sz0
        off = off + i0 * st0
    if RANK >= 2:
        i1 = rem % sz1
        rem = rem // sz1
        off = off + i1 * st1
    if RANK >= 3:
        i2 = rem % sz2
        rem = rem // sz2
        off = off + i2 * st2
    if RANK >= 4:
        i3 = rem % sz3
        rem = rem // sz3
        off = off + i3 * st3
    x = tl.load(x_ptr + off, mask=mask)
    tl.store(x_ptr + off, x + s, mask=mask)


@triton.jit
def _add_table_kernel(x_ptr, n_elem, s, sz_ptr, st_ptr, RANK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    for k in range(RANK):
        sz = tl.load(sz_ptr + k)
        st = tl.load(st_ptr + k)
        i = rem % sz
        rem = rem // sz
        off = off + i * st
    x = tl.load(x_ptr + off, mask=mask)
    tl.store(x_ptr + off, x + s, mask=mask)


# Cached (launcher, function handle, packed metadata) per (dtype, BLOCK, num_warps).
# Populated by warmup(), which produces exactly the binary a normal launch with
# 16-byte-aligned pointers and 16-element-multiple sizes would compile.
_FAST = {}
_LAST = None
_GET_STREAM = None
_FAST_DEV = -1


def _collapse(shape, stride):
    """Fold dims (fastest first) whose strides make them contiguous, drop size-1."""
    dims = []
    for sz, st in zip(reversed(shape), reversed(stride)):
        if sz == 1:
            continue
        if dims and st == dims[-1][0] * dims[-1][1]:
            dims[-1][0] *= sz
        else:
            dims.append([sz, st])
    return dims


def _bucket(n, esize):
    if n <= 1024:
        blk = 32
        while blk < n:
            blk <<= 1
        if blk <= 128:
            nw = 1
        elif blk <= 512:
            nw = 2
        else:
            nw = 4
        return blk, nw
    if esize >= 4 and n > 4096:
        # ~2 KB of streaming span per CTA is what measured best for 4-byte types.
        return 512, 2
    return 1024, 4


def _fast_launch4(dt, blk, nw, nblk, a, n0, s0, b, n1, s1, c, n2, s2, d, n3, s3):
    global _LAST, _GET_STREAM, _FAST_DEV
    last = _LAST
    if last is not None and last[0] is dt and last[1] == blk and last[2] == nw:
        launcher = last[3]
        func = last[4]
        packed = last[5]
    else:
        entry = _FAST.get((dt, blk, nw))
        if entry is None:
            compiled = _add_multi4_kernel.warmup(
                a,
                n0,
                s0,
                b,
                n1,
                s1,
                c,
                n2,
                s2,
                d,
                n3,
                s3,
                BLOCK=blk,
                num_warps=nw,
                grid=(1, 4, 1),
            )
            entry = (compiled.run, compiled.function, compiled.packed_metadata)
            _FAST[(dt, blk, nw)] = entry
            if _GET_STREAM is None:
                _GET_STREAM = _driver.active.get_current_stream
                _FAST_DEV = _driver.active.get_current_device()
        launcher, func, packed = entry
        _LAST = (dt, blk, nw, launcher, func, packed)
    launcher(
        nblk,
        4,
        1,
        _GET_STREAM(_FAST_DEV),
        func,
        packed,
        None,
        None,
        None,
        a,
        n0,
        s0,
        b,
        n1,
        s1,
        c,
        n2,
        s2,
        d,
        n3,
        s3,
        blk,
    )


def _fast_launch(dt, blk, nw, nblk, nsel, args):
    global _GET_STREAM, _FAST_DEV
    entry = _FAST.get((dt, blk, nw))
    if entry is None:
        compiled = _add_multi4_kernel.warmup(
            *args, BLOCK=blk, num_warps=nw, grid=(1, 4, 1)
        )
        entry = (compiled.run, compiled.function, compiled.packed_metadata)
        _FAST[(dt, blk, nw)] = entry
        if _GET_STREAM is None:
            _GET_STREAM = _driver.active.get_current_stream
            _FAST_DEV = _driver.active.get_current_device()
    launcher, func, packed = entry
    launcher(
        nblk,
        nsel,
        1,
        _GET_STREAM(_FAST_DEV),
        func,
        packed,
        None,
        None,
        None,
        *args,
        blk,
    )


def _launch_single(t, n, s):
    """One tensor per launch (non-contiguous, or oversize element counts)."""
    blk, nw = _bucket(n, t.dtype.itemsize)
    long_ = n >= 2147483648
    grid = (1,) if n <= 1024 else ((n + blk - 1) // blk,)
    if t.is_contiguous():
        _add_contig_kernel[grid](t, n, s, BLOCK=blk, LONG=long_, num_warps=nw)
        return

    dims = _collapse(t.shape, t.stride())
    rank = len(dims)
    if rank <= 4:
        sz = [1, 1, 1, 1]
        st = [0, 0, 0, 0]
        for i, d in enumerate(dims):
            sz[i] = int(d[0])
            st[i] = int(d[1])
        _add_strided_kernel[grid](
            t,
            n,
            s,
            sz[0],
            sz[1],
            sz[2],
            sz[3],
            st[0],
            st[1],
            st[2],
            st[3],
            RANK=rank,
            BLOCK=blk,
            num_warps=nw,
        )
        return

    sizes = torch.tensor([int(d[0]) for d in dims], dtype=torch.int64, device=t.device)
    strides = torch.tensor(
        [int(d[1]) for d in dims], dtype=torch.int64, device=t.device
    )
    _add_table_kernel[grid](t, n, s, sizes, strides, rank, BLOCK=blk, num_warps=nw)


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


def _foreach_add__ScalarList(self, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_ADD__SCALARLIST")

    if not _list_supported(self):
        return default_foreach_add__ScalarList(self, scalars)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_add__ScalarList(self, scalars)


def _launch_foreach_add__ScalarList(self, scalars):
    tensors = self if isinstance(self, list) else list(self)
    if not isinstance(scalars, (list, tuple)):
        scalars = list(scalars)
    nt = len(tensors)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if len(scalars) != nt:
        raise RuntimeError(
            "Tensor list must have same number of elements as scalar list."
        )

    # Fast path: one to four contiguous, same floating dtype, 16-byte aligned
    # tensors whose element counts are multiples of 16 -> one cached launch.
    if nt == 4:
        a = tensors[0]
        dt = a.dtype
        if dt.is_floating_point:
            b = tensors[1]
            c = tensors[2]
            d = tensors[3]
            n0 = a.numel()
            n1 = b.numel()
            n2 = c.numel()
            n3 = d.numel()
            if (
                (n0 & 15) == 0
                and n0
                and (n1 & 15) == 0
                and (n2 & 15) == 0
                and (n3 & 15) == 0
            ):
                if (
                    b.dtype is dt
                    and c.dtype is dt
                    and d.dtype is dt
                    and a.is_contiguous()
                    and b.is_contiguous()
                    and c.is_contiguous()
                    and d.is_contiguous()
                    and (a.data_ptr() & 15) == 0
                    and (b.data_ptr() & 15) == 0
                    and (c.data_ptr() & 15) == 0
                    and (d.data_ptr() & 15) == 0
                ):
                    m = n0
                    if n1 > m:
                        m = n1
                    if n2 > m:
                        m = n2
                    if n3 > m:
                        m = n3
                    blk, nw = _bucket(m, dt.itemsize)
                    s0 = scalars[0]
                    s1 = scalars[1]
                    s2 = scalars[2]
                    s3 = scalars[3]
                    if type(s0) is not float:
                        s0 = float(s0)
                    if type(s1) is not float:
                        s1 = float(s1)
                    if type(s2) is not float:
                        s2 = float(s2)
                    if type(s3) is not float:
                        s3 = float(s3)
                    _fast_launch4(
                        dt,
                        blk,
                        nw,
                        (m + blk - 1) // blk,
                        a,
                        n0,
                        s0,
                        b,
                        n1,
                        s1,
                        c,
                        n2,
                        s2,
                        d,
                        n3,
                        s3,
                    )
                    return None
    elif nt < 4:
        first = tensors[0]
        dt = first.dtype
        if dt.is_floating_point:
            ns = []
            ss = []
            ok = True
            for i in range(nt):
                t = tensors[i]
                n = t.numel()
                if (
                    (n & 15)
                    or t.dtype is not dt
                    or (t.data_ptr() & 15)
                    or not t.is_contiguous()
                ):
                    ok = False
                    break
                ns.append(n)
                sv = scalars[i]
                ss.append(sv if type(sv) is float else float(sv))
            if ok and ns[0]:
                maxn = ns[0]
                for n in ns:
                    if n > maxn:
                        maxn = n
                blk, nw = _bucket(maxn, dt.itemsize)
                args = []
                for i in range(4):
                    if i < nt:
                        args.append(tensors[i])
                        args.append(ns[i])
                        args.append(ss[i])
                    else:
                        args.append(first)
                        args.append(0)
                        args.append(0.0)
                _fast_launch(dt, blk, nw, (maxn + blk - 1) // blk, nt, args)
                return None

    groups = {}
    for i in range(nt):
        t = tensors[i]
        n = t.numel()
        if n == 0:
            continue
        s = scalars[i]
        if type(s) is not float and type(s) is not int:
            if isinstance(s, torch.Tensor):
                s = s.item()
        if t.dtype.is_floating_point:
            s = float(s)
        elif isinstance(s, float):
            raise RuntimeError(
                "result type Float can't be cast to the desired output type"
            )
        else:
            s = int(s)

        if n < 2147483648 and t.is_contiguous():
            key = t.dtype
            g = groups.get(key)
            if g is None:
                groups[key] = [(t, n, s)]
            else:
                g.append((t, n, s))
        else:
            _launch_single(t, n, s)

    for items in groups.values():
        n_items = len(items)
        if n_items == 1:
            _launch_single(items[0][0], items[0][1], items[0][2])
        else:
            for base in range(0, n_items, 4):
                chunk = items[base : base + 4]
                maxn = 0
                for it in chunk:
                    if it[1] > maxn:
                        maxn = it[1]
                blk, nw = _bucket(maxn, chunk[0][0].dtype.itemsize)
                nblk = (maxn + blk - 1) // blk
                args = []
                first = chunk[0]
                for k in range(4):
                    if k < len(chunk):
                        t, n, s = chunk[k]
                        args.append(t)
                        args.append(n)
                        args.append(s)
                    else:
                        args.append(first[0])
                        args.append(0)
                        args.append(first[2])
                _add_multi4_kernel[(nblk, len(chunk))](*args, BLOCK=blk, num_warps=nw)

    return None
