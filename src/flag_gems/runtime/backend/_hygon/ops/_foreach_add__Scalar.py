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
from triton import knobs as _knobs

logger = logging.getLogger(__name__)

_BLOCK = 1024
_BLOCK_HALF = 2048
_NUM_WARPS = 4
_FUSABLE = (torch.float16, torch.float32, torch.bfloat16)
# Fuse a whole contiguous group into one launch only when the group is small
# enough that launch count dominates; above this the branchy fused kernel
# measurably loses bandwidth against the straight-line per-tensor kernel.
_FUSE_MAX = 1 << 20
_INT32_MAX = 0x7FFFFFFF

_RT = _knobs.runtime
try:
    _STREAM = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - fallback for builds without it
    _STREAM = None

# key -> (launcher, function, packed_metadata) of an already compiled kernel
_FAST = {}


@triton.jit
def _add_flat_kernel(ptr, n, scalar, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask)
    tl.store(ptr + offs, x.to(tl.float32) + scalar, mask=mask)


@triton.jit
def _add_exact_kernel(ptr, scalar, BLOCK: tl.constexpr):
    """In-place add for a contiguous tensor whose numel is a multiple of BLOCK."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + offs)
    tl.store(ptr + offs, x.to(tl.float32) + scalar)


@triton.jit
def _add_flat_f64_kernel(ptr, n, sptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(sptr)
    x = tl.load(ptr + offs, mask=mask)
    tl.store(ptr + offs, x + s, mask=mask)


@triton.jit
def _add_flat_i64_kernel(ptr, n, scalar, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask)
    tl.store(ptr + offs, x + scalar, mask=mask)


@triton.jit
def _add_strided_kernel(
    ptr,
    n,
    scalar,
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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    r7 = idx % s7
    idx = idx // s7
    off += r7 * t7
    r6 = idx % s6
    idx = idx // s6
    off += r6 * t6
    r5 = idx % s5
    idx = idx // s5
    off += r5 * t5
    r4 = idx % s4
    idx = idx // s4
    off += r4 * t4
    r3 = idx % s3
    idx = idx // s3
    off += r3 * t3
    r2 = idx % s2
    idx = idx // s2
    off += r2 * t2
    r1 = idx % s1
    idx = idx // s1
    off += r1 * t1
    r0 = idx % s0
    off += r0 * t0
    x = tl.load(ptr + off, mask=mask)
    tl.store(ptr + off, x.to(tl.float32) + scalar, mask=mask)


@triton.jit
def _fused_add_kernel(
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
    scalar,
    BLOCK: tl.constexpr,
):
    """One launch adds a scalar in place to up to 8 contiguous same-dtype tensors.

    Slots occupy consecutive program-id block ranges; a slot with n == 0 owns no
    block, so its guard is never true and its pointer is never dereferenced.
    """
    pid = tl.program_id(0)
    b = tl.arange(0, BLOCK)
    c0 = tl.cdiv(n0, BLOCK)
    c1 = c0 + tl.cdiv(n1, BLOCK)
    c2 = c1 + tl.cdiv(n2, BLOCK)
    c3 = c2 + tl.cdiv(n3, BLOCK)
    c4 = c3 + tl.cdiv(n4, BLOCK)
    c5 = c4 + tl.cdiv(n5, BLOCK)
    c6 = c5 + tl.cdiv(n6, BLOCK)
    c7 = c6 + tl.cdiv(n7, BLOCK)
    if pid < c0:
        offs = pid * BLOCK + b
        m = offs < n0
        x = tl.load(p0 + offs, mask=m)
        tl.store(p0 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c0) & (pid < c1):
        offs = (pid - c0) * BLOCK + b
        m = offs < n1
        x = tl.load(p1 + offs, mask=m)
        tl.store(p1 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c1) & (pid < c2):
        offs = (pid - c1) * BLOCK + b
        m = offs < n2
        x = tl.load(p2 + offs, mask=m)
        tl.store(p2 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c2) & (pid < c3):
        offs = (pid - c2) * BLOCK + b
        m = offs < n3
        x = tl.load(p3 + offs, mask=m)
        tl.store(p3 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c3) & (pid < c4):
        offs = (pid - c3) * BLOCK + b
        m = offs < n4
        x = tl.load(p4 + offs, mask=m)
        tl.store(p4 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c4) & (pid < c5):
        offs = (pid - c4) * BLOCK + b
        m = offs < n5
        x = tl.load(p5 + offs, mask=m)
        tl.store(p5 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c5) & (pid < c6):
        offs = (pid - c5) * BLOCK + b
        m = offs < n6
        x = tl.load(p6 + offs, mask=m)
        tl.store(p6 + offs, x.to(tl.float32) + scalar, mask=m)
    if (pid >= c6) & (pid < c7):
        offs = (pid - c6) * BLOCK + b
        m = offs < n7
        x = tl.load(p7 + offs, mask=m)
        tl.store(p7 + offs, x.to(tl.float32) + scalar, mask=m)


def _block_for(dt):
    """2-byte element types need twice the elements per thread for a wide access."""
    if dt == torch.float16 or dt == torch.bfloat16:
        return _BLOCK_HALF
    return _BLOCK


def _no_hooks():
    """True when no Triton launch hook needs the launch metadata dict."""
    enter = _RT.launch_enter_hook
    exit_ = _RT.launch_exit_hook
    return (enter is None or not enter.calls) and (exit_ is None or not exit_.calls)


def _cur_stream(dev):
    if _STREAM is not None and dev is not None:
        return _STREAM(dev)
    return torch.cuda.current_stream().cuda_stream


def _collapse(shape, strides):
    """Collapse adjacent dims that are contiguous in memory (pure metadata)."""
    dims = []
    for sz, st in zip(shape, strides):
        if sz == 1:
            continue
        if dims and dims[-1][1] == st * sz:
            dims[-1] = (dims[-1][0] * sz, st)
        else:
            dims.append((sz, st))
    if not dims:
        dims = [(1, 0)]
    return dims


def _as_scalar(scalar):
    if hasattr(scalar, "item"):
        return scalar.item()
    return scalar


def _launch_fused(chunk, s):
    n = len(chunk)
    dt = chunk[0].dtype
    blk = _block_for(dt)
    ints = []
    counts = []
    total = 0
    aligned = True
    edge = False
    for t in chunk:
        p = t.data_ptr()
        if p & 15:
            aligned = False
        c = t.numel()
        if c == 1 or c > _INT32_MAX:
            edge = True
        total += (c + blk - 1) // blk
        ints.append(p)
        counts.append(c)
    fast = aligned and total > 0 and not edge
    pad = 8 - n
    counts8 = counts if pad == 0 else counts + [0] * pad
    if fast:
        ints8 = ints if pad == 0 else ints + [ints[0]] * pad
        dev = chunk[0].get_device()
        key = ("fused", dt, dev)
        ent = _FAST.get(key)
        if ent is not None and _no_hooks():
            launcher, function, packed = ent
            launcher(
                total,
                1,
                1,
                _cur_stream(dev),
                function,
                packed,
                None,
                _RT.launch_enter_hook,
                _RT.launch_exit_hook,
                ints8[0],
                ints8[1],
                ints8[2],
                ints8[3],
                ints8[4],
                ints8[5],
                ints8[6],
                ints8[7],
                counts8[0],
                counts8[1],
                counts8[2],
                counts8[3],
                counts8[4],
                counts8[5],
                counts8[6],
                counts8[7],
                s,
                blk,
            )
            return
    ptrs = list(chunk)
    if pad:
        ptrs.extend([chunk[0]] * pad)
    ck = _fused_add_kernel[(total,)](
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        ptrs[5],
        ptrs[6],
        ptrs[7],
        counts8[0],
        counts8[1],
        counts8[2],
        counts8[3],
        counts8[4],
        counts8[5],
        counts8[6],
        counts8[7],
        s,
        BLOCK=blk,
        num_warps=_NUM_WARPS,
    )
    if fast and _FAST.get(key) is None and _no_hooks():
        _FAST[key] = (ck.run, ck.function, ck.packed_metadata)


def _launch_one(t, s):
    n = t.numel()
    if n == 0:
        return
    if t.dim() > 8:
        # rare: recurse over the leading index using zero-copy views
        for i in range(t.shape[0]):
            _launch_one(t[i], s)
        return
    dt = t.dtype
    blk = _block_for(dt)
    if t.is_contiguous():
        nb = (n + blk - 1) // blk
        if dt in _FUSABLE and 1 < n <= _INT32_MAX:
            p = t.data_ptr()
            if not (p & 15):
                dev = t.get_device()
                fs = float(s)
                if n % blk == 0:
                    key = ("exact", dt, dev)
                    ent = _FAST.get(key)
                    if ent is not None and _no_hooks():
                        launcher, function, packed = ent
                        launcher(
                            nb,
                            1,
                            1,
                            _cur_stream(dev),
                            function,
                            packed,
                            None,
                            _RT.launch_enter_hook,
                            _RT.launch_exit_hook,
                            p,
                            fs,
                            blk,
                        )
                        return
                    ck = _add_exact_kernel[(nb,)](
                        t, fs, BLOCK=blk, num_warps=_NUM_WARPS
                    )
                    if ent is None and _no_hooks():
                        _FAST[key] = (ck.run, ck.function, ck.packed_metadata)
                    return
                key = ("flat", dt, dev)
                ent = _FAST.get(key)
                if ent is not None and _no_hooks():
                    launcher, function, packed = ent
                    launcher(
                        nb,
                        1,
                        1,
                        _cur_stream(dev),
                        function,
                        packed,
                        None,
                        _RT.launch_enter_hook,
                        _RT.launch_exit_hook,
                        p,
                        n,
                        fs,
                        blk,
                    )
                    return
                ck = _add_flat_kernel[(nb,)](t, n, fs, BLOCK=blk, num_warps=_NUM_WARPS)
                if ent is None and _no_hooks():
                    _FAST[key] = (ck.run, ck.function, ck.packed_metadata)
                return
        if dt == torch.float64:
            sptr = torch.tensor(float(s), dtype=torch.float64, device=t.device)
            _add_flat_f64_kernel[(nb,)](t, n, sptr, BLOCK=blk, num_warps=_NUM_WARPS)
        elif dt.is_floating_point or dt.is_complex:
            _add_flat_kernel[(nb,)](t, n, float(s), BLOCK=blk, num_warps=_NUM_WARPS)
        else:
            _add_flat_i64_kernel[(nb,)](t, n, int(s), BLOCK=blk, num_warps=_NUM_WARPS)
        return

    dims = _collapse(list(t.shape), list(t.stride()))
    if len(dims) == 1 and dims[0][1] == 1:
        nb = (n + blk - 1) // blk
        if dt == torch.float64:
            sptr = torch.tensor(float(s), dtype=torch.float64, device=t.device)
            _add_flat_f64_kernel[(nb,)](t, n, sptr, BLOCK=blk, num_warps=_NUM_WARPS)
        else:
            _add_flat_kernel[(nb,)](t, n, float(s), BLOCK=blk, num_warps=_NUM_WARPS)
        return

    shp = [1] * 8
    strd = [0] * 8
    base = 8 - len(dims)
    for i, (sz, st) in enumerate(dims):
        shp[base + i] = sz
        strd[base + i] = st
    nb = (n + _BLOCK - 1) // _BLOCK
    _add_strided_kernel[(nb,)](
        t,
        n,
        float(s),
        shp[0],
        shp[1],
        shp[2],
        shp[3],
        shp[4],
        shp[5],
        shp[6],
        shp[7],
        strd[0],
        strd[1],
        strd[2],
        strd[3],
        strd[4],
        strd[5],
        strd[6],
        strd[7],
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _foreach_add__Scalar(self, scalar):
    logger.debug("GEMS_HYGON _FOREACH_ADD__SCALAR")
    if type(self) is torch.Tensor:
        tensors = (self,)
    else:
        tensors = self
    s = scalar.item() if hasattr(scalar, "item") else scalar
    n = len(tensors)
    # Fast prelude: a short list of contiguous tensors sharing one dtype and
    # device is the common case and needs no grouping bookkeeping.
    if 1 < n <= 8:
        dt = tensors[0].dtype
        if dt in _FUSABLE:
            dev = tensors[0].get_device()
            total_el = 0
            ok = True
            for t in tensors:
                num = t.numel()
                if (
                    not num
                    or t.dtype is not dt
                    or t.get_device() != dev
                    or not t.is_contiguous()
                ):
                    ok = False
                    break
                total_el += num
            if ok:
                if total_el <= _FUSE_MAX:
                    _launch_fused(tensors, float(s))
                else:
                    for t in tensors:
                        _launch_one(t, s)
                return None
    groups = None
    rest = None
    for t in tensors:
        if not t.numel():
            continue
        if t.dtype in _FUSABLE and t.is_contiguous():
            if groups is None:
                groups = {}
                groups[(t.get_device(), t.dtype)] = [t]
            else:
                key = (t.get_device(), t.dtype)
                g = groups.get(key)
                if g is None:
                    groups[key] = [t]
                else:
                    g.append(t)
        else:
            if rest is None:
                rest = []
            rest.append(t)
    if groups is not None:
        for g in groups.values():
            ng = len(g)
            if ng == 1:
                _launch_one(g[0], s)
                continue
            total_el = 0
            for t in g:
                total_el += t.numel()
            if total_el <= _FUSE_MAX:
                # small lists: one launch for the whole group beats N launches
                fs = float(s)
                for i in range(0, ng, 8):
                    chunk = g[i : i + 8]
                    if len(chunk) == 1:
                        _launch_one(chunk[0], s)
                    else:
                        _launch_fused(chunk, fs)
            else:
                # large lists: the branchy fused kernel costs bandwidth, so keep
                # the straight-line per-tensor kernel
                for t in g:
                    _launch_one(t, s)
    if rest is not None:
        for t in rest:
            _launch_one(t, s)
    return None
