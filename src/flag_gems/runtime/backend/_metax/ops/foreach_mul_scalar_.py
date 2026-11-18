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

"""In-place foreach scalar multiply (aten::_foreach_mul_.Scalar) in Triton.

Architecture
  * contiguous tensors of the same dtype are fused into ONE launch covering up to
    8 tensor slots (grid = (blocks, slots)); the hot benchmark shape (4 tensors)
    therefore costs a single launch instead of four.
  * non-contiguous tensors use a generic strided kernel (collapsed rank is a
    constexpr, addresses are decomposed with int64 math), one launch each.
  * the scalar is passed as an exact fp64 kernel argument, so fp16/bf16/fp32 use
    fp32 "opmath" (single rounding on store) and fp64 stays exact.
  * run() returns None (the Gem/pytest contract) while mutating every tensor in
    place through its own storage.
  * host side: the per-call Triton binder dominates small workloads (measured
    ~17us for a 3-argument launch, ~34us for a 9-argument launch), so after the
    first launch of a given exact signature the compiled kernel is cached and
    replayed straight through the driver launcher.  The launcher resolves each
    pointer argument with data_ptr() + mcPointerGetAttribute, so the fast path
    passes raw data_ptr integers instead of tensors.  The cache key holds slot
    count, BLOCK, num_warps, torch dtype, scalar and every data_ptr/numel, so a
    hit can only occur for a signature Triton would specialize identically;
    anything else falls back to the normal JIT launch.
"""

import triton
import triton.knobs as _tk
import triton.language as tl
from triton.runtime.driver import driver as _drvcfg

_MAX_RANK = 8
_MAX_SLOTS = 8

_DRV = _drvcfg.active
_LC = {}
_STREAM = None
_ENTER = None
_EXIT = None
_NOHOOK = False


@triton.jit
def _mulv(v, s):
    # s is an exact fp64 kernel argument (see the ``s: tl.float64`` annotations).
    if v.dtype == tl.float64:
        r = v * s
    else:
        r = (v.to(tl.float32) * s.to(tl.float32)).to(v.dtype)
    return r


@triton.jit
def _slot(p, n, s, bid, BLOCK: tl.constexpr):
    offs = bid * BLOCK + tl.arange(0, BLOCK)
    if (bid + 1) * BLOCK <= n:
        # full block: no mask, so the compiler can emit wide vector accesses
        v = tl.load(p + offs)
        tl.store(p + offs, _mulv(v, s))
    else:
        m = offs < n
        v = tl.load(p + offs, mask=m)
        tl.store(p + offs, _mulv(v, s), mask=m)


@triton.jit
def _mul1_kernel(p0, n0, s: tl.float64, BLOCK: tl.constexpr):
    _slot(p0, n0, s, tl.program_id(0), BLOCK)


@triton.jit
def _mul2_kernel(p0, p1, n0, n1, s: tl.float64, BLOCK: tl.constexpr):
    bid = tl.program_id(0)
    sl = tl.program_id(1)
    if sl == 0:
        _slot(p0, n0, s, bid, BLOCK)
    else:
        _slot(p1, n1, s, bid, BLOCK)


@triton.jit
def _mul4_kernel(p0, p1, p2, p3, n0, n1, n2, n3, s: tl.float64, BLOCK: tl.constexpr):
    bid = tl.program_id(0)
    sl = tl.program_id(1)
    if sl == 0:
        _slot(p0, n0, s, bid, BLOCK)
    elif sl == 1:
        _slot(p1, n1, s, bid, BLOCK)
    elif sl == 2:
        _slot(p2, n2, s, bid, BLOCK)
    else:
        _slot(p3, n3, s, bid, BLOCK)


@triton.jit
def _mul8_kernel(
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
    s: tl.float64,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    sl = tl.program_id(1)
    if sl == 0:
        _slot(p0, n0, s, bid, BLOCK)
    elif sl == 1:
        _slot(p1, n1, s, bid, BLOCK)
    elif sl == 2:
        _slot(p2, n2, s, bid, BLOCK)
    elif sl == 3:
        _slot(p3, n3, s, bid, BLOCK)
    elif sl == 4:
        _slot(p4, n4, s, bid, BLOCK)
    elif sl == 5:
        _slot(p5, n5, s, bid, BLOCK)
    elif sl == 6:
        _slot(p6, n6, s, bid, BLOCK)
    else:
        _slot(p7, n7, s, bid, BLOCK)


@triton.jit
def _mul_strided_kernel(
    x_ptr,
    n,
    s: tl.float64,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = offs < n
    sizes = [a0, a1, a2, a3, a4, a5, a6, a7]
    strides = [b0, b1, b2, b3, b4, b5, b6, b7]
    addr = tl.zeros((BLOCK,), dtype=tl.int64)
    rem = offs
    for k in tl.static_range(RANK):
        sz = sizes[k].to(tl.int64)
        sz = tl.where(sz < 1, 1, sz)
        addr += (rem % sz) * strides[k].to(tl.int64)
        rem = rem // sz
    v = tl.load(x_ptr + addr, mask=m)
    tl.store(x_ptr + addr, _mulv(v, s), mask=m)


def _cfg(n, esz):
    """BLOCK / num_warps for the fused contiguous kernel.

    Small tensors: one CTA per slot (BLOCK = next_pow2(numel), capped at 8192) with
    up to 8 warps, so a 4-slot launch is 4 CTAs instead of a flock of tiny ones.
    Large tensors: 16 KB per CTA at 32 B per thread (warp width is 64 on this
    target), i.e. half as many CTAs as the 16 B/thread layout.
    """
    ept = 16 // esz
    if ept < 1:
        ept = 1
    if n >= 65536:
        return 1024 * ept, 8
    b = 64
    while b < n:
        b <<= 1
    if b > 8192:
        b = 8192
    nw = b // (64 * ept)
    if nw < 1:
        nw = 1
    elif nw > 8:
        nw = 8
    return b, nw


def _cfg_strided(n, esz):
    """BLOCK / num_warps for the gather/scatter kernel (16B per thread)."""
    ept = 16 // esz
    if ept < 1:
        ept = 1
    block = 256 * ept
    if n >= block:
        return block, 4
    b = 64
    while b < n:
        b <<= 1
    nw = b // (64 * ept)
    if nw < 1:
        nw = 1
    elif nw > 4:
        nw = 4
    return b, nw


def _flush(batch, s, dt):
    """One launch for all tensors of ``batch`` (list of (tensor, numel, data_ptr))."""
    k = len(batch)
    maxn = 0
    for _, n, _ in batch:
        if n > maxn:
            maxn = n
    block, nw = _cfg(maxn, dt.itemsize)
    nb = (maxn + block - 1) // block
    tp = tuple([p for _, _, p in batch])
    tn = tuple([n for _, n, _ in batch])
    key = (k, block, nw, dt, s, tp, tn)

    ent = _LC.get(key)
    if ent is None:
        if k == 1:
            kern = _mul1_kernel
        elif k == 2:
            kern = _mul2_kernel
        elif k == 4:
            kern = _mul4_kernel
        else:
            kern = _mul8_kernel
        ck = kern[(nb, k)](
            *(tuple([t for t, _, _ in batch]) + tn + (s,)), BLOCK=block, num_warps=nw
        )
        if len(_LC) > 256:
            _LC.clear()
        _LC[key] = (ck.run, ck.function, ck.packed_metadata, ck)
        return

    rt = tp + tn + (s, block)
    grid = (nb, k, 1)
    lm = None if _NOHOOK else ent[3].launch_metadata(grid, _STREAM, *rt)
    ent[0](nb, k, 1, _STREAM, ent[1], ent[2], lm, _ENTER, _EXIT, *rt)


def _collapse(shape, stride):
    """Drop size-1 dims and merge adjacent dims that stay dense."""
    sizes = []
    sts = []
    for sz, st in zip(shape, stride):
        if sz == 1:
            continue
        if sizes and st == sizes[-1] * sts[-1]:
            sizes[-1] *= sz
        else:
            sizes.append(sz)
            sts.append(st)
    return sizes, sts


def _launch_strided(t, s):
    sizes, sts = _collapse(t.shape, t.stride())
    if not sizes:
        sizes, sts = [1], [0]
    if len(sizes) > _MAX_RANK:
        # extremely rare: peel the outermost dim and recurse over the views
        for i in range(sizes[0]):
            _launch_strided(t.select(0, i), s)
        return
    n = t.numel()
    block, nw = _cfg_strided(n, t.dtype.itemsize)
    pad_sz = sizes + [1] * (_MAX_RANK - len(sizes))
    pad_st = sts + [0] * (_MAX_RANK - len(sts))
    _mul_strided_kernel[((n + block - 1) // block,)](
        t,
        n,
        s,
        pad_sz[0],
        pad_sz[1],
        pad_sz[2],
        pad_sz[3],
        pad_sz[4],
        pad_sz[5],
        pad_sz[6],
        pad_sz[7],
        pad_st[0],
        pad_st[1],
        pad_st[2],
        pad_st[3],
        pad_st[4],
        pad_st[5],
        pad_st[6],
        pad_st[7],
        RANK=len(sizes),
        BLOCK=block,
        num_warps=nw,
    )


def foreach_mul_scalar_(self, scalar):
    global _STREAM, _ENTER, _EXIT, _NOHOOK
    s = scalar if type(scalar) is float else float(scalar)
    tensors = self if type(self) is list else list(self)
    if not tensors:
        return None
    enter = _tk.runtime.launch_enter_hook
    exit_ = _tk.runtime.launch_exit_hook
    _ENTER = enter
    _EXIT = exit_
    _NOHOOK = not getattr(enter, "hooks", 1) and not getattr(exit_, "hooks", 1)
    _STREAM = _DRV.get_current_stream(_DRV.get_current_device())

    groups = {}
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            dt = t.dtype
            item = (t, n, t.data_ptr())
            g = groups.get(dt)
            if g is None:
                groups[dt] = [item]
            else:
                g.append(item)
        else:
            _launch_strided(t, s)

    for dt, items in groups.items():
        if len(items) == 1:
            _flush(items, s, dt)
            continue
        esz = dt.itemsize
        batch = []
        spans = []
        for it in items:
            p = it[2]
            e = p + it[1] * esz
            clash = False
            for qs, qe in spans:
                if p < qe and qs < e:
                    clash = True
                    break
            if clash:
                _flush(batch, s, dt)
                batch = []
                spans = []
            batch.append(it)
            spans.append((p, e))
            if len(batch) == _MAX_SLOTS:
                _flush(batch, s, dt)
                batch = []
                spans = []
        if batch:
            _flush(batch, s, dt)
    return None
