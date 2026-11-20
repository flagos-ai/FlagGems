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

"""In-place foreach pow with a per-tensor scalar exponent.

Implements ``aten::_foreach_pow_.ScalarList``: for every i,
``self[i] = pow(self[i], exponent[i])`` in place, returning ``None``.

Design:
  * a contiguous, duplicate-free list is fused into ONE launch with a 2D grid
    ``(blocks, k)``; ``program_id(1)`` selects the tensor, so a block never
    straddles two tensors;
  * when every tensor has the same numel and the same exponent (the common
    foreach case) a compact kernel with one shared exponent and one shared
    numel is used, its mask is dropped when ``numel % BLOCK == 0``, and the
    common exponents 0.5/1/2/3/-0.5/-1/-2 are evaluated with a multiply/sqrt
    form instead of a generic transcendental;
  * duplicated / non-contiguous tensors fall back to ordered per-tensor
    launches (flat contiguous kernel or a collapsed-dim strided kernel);
  * pow is evaluated with libdevice ``powf`` in float32 (ATen opmath) and cast
    back to the storage dtype on store.
"""

import importlib

import triton
import triton.language as tl


def _resolve_pow():
    for name in (
        "triton.language.extra.libdevice",
        "triton.language.extra.cuda.libdevice",
        "triton.language.extra.metax.libdevice",
        "triton.language.extra.hip.libdevice",
    ):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        fn = getattr(mod, "pow", None)
        if fn is not None:
            return fn
    return None


_POW_FN = _resolve_pow()

if _POW_FN is not None:

    @triton.jit
    def _powf(x, e):
        return _POW_FN(x, e)

else:

    @triton.jit
    def _powf(x, e):
        s = tl.where(x < 0.0, -1.0, 1.0)
        return s * tl.exp2(e * tl.log2(tl.abs(x)))


@triton.jit
def _pow_blk(p, offs, n, e, BLOCK: tl.constexpr):
    mask = offs < n
    x = tl.load(p + offs, mask=mask, other=1.0).to(tl.float32)
    y = _powf(x, e)
    tl.store(p + offs, y, mask=mask)


@triton.jit
def _pow_math(x, e, SPECIAL: tl.constexpr):
    # Exact-in-value fast paths for the common foreach exponents; equal to
    # powf within tolerance and far cheaper than a generic transcendental.
    if SPECIAL == 1:
        y = tl.sqrt(x)
        # IEEE pow(-inf, 0.5) is +inf while sqrt(-inf) is NaN
        return tl.where(x == -float("inf"), float("inf"), y)
    elif SPECIAL == 2:
        return x * x
    elif SPECIAL == 3:
        return x * x * x
    elif SPECIAL == 4:
        return x
    elif SPECIAL == 6:
        return 1.0 / x
    elif SPECIAL == 7:
        return 1.0 / (x * x)
    else:
        return _powf(x, e)


@triton.jit
def _pow_blk_shared(
    p,
    offs,
    n,
    e,
    SPECIAL: tl.constexpr,
    EVEN: tl.constexpr,
    STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if EVEN:
        if STREAM:
            # in-place stream with no reuse: bypass L1 and mark the lines
            # evict-first instead of thrashing the 8 MB L2
            x = tl.load(p + offs, cache_modifier=".cg").to(tl.float32)
            tl.store(p + offs, _pow_math(x, e, SPECIAL), cache_modifier=".cs")
        else:
            x = tl.load(p + offs).to(tl.float32)
            tl.store(p + offs, _pow_math(x, e, SPECIAL))
    else:
        mask = offs < n
        x = tl.load(p + offs, mask=mask, other=1.0).to(tl.float32)
        tl.store(p + offs, _pow_math(x, e, SPECIAL), mask=mask)


@triton.jit
def _pow1(p, e, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _pow_blk(p, offs, n, e, BLOCK)


@triton.jit
def _pow_blk_even(p, offs, SPECIAL: tl.constexpr, STREAM: tl.constexpr):
    if STREAM:
        x = tl.load(p + offs, cache_modifier=".cg").to(tl.float32)
        tl.store(p + offs, _pow_math(x, 0.0, SPECIAL), cache_modifier=".cs")
    else:
        x = tl.load(p + offs).to(tl.float32)
        tl.store(p + offs, _pow_math(x, 0.0, SPECIAL))


@triton.jit
def _pow16s(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    SPECIAL: tl.constexpr,
    STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Minimal-signature kernel: 16 pointers only, no mask, no scalars."""
    tid = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _pow_blk_even(p0, offs, SPECIAL, STREAM)
    if tid == 1:
        _pow_blk_even(p1, offs, SPECIAL, STREAM)
    if tid == 2:
        _pow_blk_even(p2, offs, SPECIAL, STREAM)
    if tid == 3:
        _pow_blk_even(p3, offs, SPECIAL, STREAM)
    if tid == 4:
        _pow_blk_even(p4, offs, SPECIAL, STREAM)
    if tid == 5:
        _pow_blk_even(p5, offs, SPECIAL, STREAM)
    if tid == 6:
        _pow_blk_even(p6, offs, SPECIAL, STREAM)
    if tid == 7:
        _pow_blk_even(p7, offs, SPECIAL, STREAM)
    if tid == 8:
        _pow_blk_even(p8, offs, SPECIAL, STREAM)
    if tid == 9:
        _pow_blk_even(p9, offs, SPECIAL, STREAM)
    if tid == 10:
        _pow_blk_even(p10, offs, SPECIAL, STREAM)
    if tid == 11:
        _pow_blk_even(p11, offs, SPECIAL, STREAM)
    if tid == 12:
        _pow_blk_even(p12, offs, SPECIAL, STREAM)
    if tid == 13:
        _pow_blk_even(p13, offs, SPECIAL, STREAM)
    if tid == 14:
        _pow_blk_even(p14, offs, SPECIAL, STREAM)
    if tid == 15:
        _pow_blk_even(p15, offs, SPECIAL, STREAM)


@triton.jit
def _pow16u(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    e,
    n,
    SPECIAL: tl.constexpr,
    EVEN: tl.constexpr,
    STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tid = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _pow_blk_shared(p0, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 1:
        _pow_blk_shared(p1, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 2:
        _pow_blk_shared(p2, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 3:
        _pow_blk_shared(p3, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 4:
        _pow_blk_shared(p4, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 5:
        _pow_blk_shared(p5, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 6:
        _pow_blk_shared(p6, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 7:
        _pow_blk_shared(p7, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 8:
        _pow_blk_shared(p8, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 9:
        _pow_blk_shared(p9, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 10:
        _pow_blk_shared(p10, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 11:
        _pow_blk_shared(p11, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 12:
        _pow_blk_shared(p12, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 13:
        _pow_blk_shared(p13, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 14:
        _pow_blk_shared(p14, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)
    if tid == 15:
        _pow_blk_shared(p15, offs, n, e, SPECIAL, EVEN, STREAM, BLOCK)


@triton.jit
def _pow16(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    e9,
    e10,
    e11,
    e12,
    e13,
    e14,
    e15,
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
    tid = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _pow_blk(p0, offs, n0, e0, BLOCK)
    if tid == 1:
        _pow_blk(p1, offs, n1, e1, BLOCK)
    if tid == 2:
        _pow_blk(p2, offs, n2, e2, BLOCK)
    if tid == 3:
        _pow_blk(p3, offs, n3, e3, BLOCK)
    if tid == 4:
        _pow_blk(p4, offs, n4, e4, BLOCK)
    if tid == 5:
        _pow_blk(p5, offs, n5, e5, BLOCK)
    if tid == 6:
        _pow_blk(p6, offs, n6, e6, BLOCK)
    if tid == 7:
        _pow_blk(p7, offs, n7, e7, BLOCK)
    if tid == 8:
        _pow_blk(p8, offs, n8, e8, BLOCK)
    if tid == 9:
        _pow_blk(p9, offs, n9, e9, BLOCK)
    if tid == 10:
        _pow_blk(p10, offs, n10, e10, BLOCK)
    if tid == 11:
        _pow_blk(p11, offs, n11, e11, BLOCK)
    if tid == 12:
        _pow_blk(p12, offs, n12, e12, BLOCK)
    if tid == 13:
        _pow_blk(p13, offs, n13, e13, BLOCK)
    if tid == 14:
        _pow_blk(p14, offs, n14, e14, BLOCK)
    if tid == 15:
        _pow_blk(p15, offs, n15, e15, BLOCK)


@triton.jit
def _pow_strided(
    p,
    e,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i = offs.to(tl.int64)
    o = (i % d0) * s0
    i = i // d0
    o += (i % d1) * s1
    i = i // d1
    o += (i % d2) * s2
    i = i // d2
    o += (i % d3) * s3
    i = i // d3
    o += (i % d4) * s4
    i = i // d4
    o += (i % d5) * s5
    i = i // d5
    o += (i % d6) * s6
    i = i // d6
    o += (i % d7) * s7
    x = tl.load(p + o, mask=mask, other=1.0).to(tl.float32)
    y = _powf(x, e)
    tl.store(p + o, y, mask=mask)


_SLOTS = 16
_SPECIAL = {
    0.5: 1,
    2.0: 2,
    3.0: 3,
    1.0: 4,
    -1.0: 6,
    -2.0: 7,
}


def _block_for(n):
    b = 64
    while b < n and b < 4096:
        b <<= 1
    return b


def _warps_for(block):
    if block <= 128:
        return 1
    if block <= 512:
        return 2
    if block <= 1024:
        return 4
    return 8


def _collapse(t):
    """Collapse a strided view to a minimal (sizes, strides) description."""
    sizes = []
    strides = []
    for s, st in zip(t.shape, t.stride()):
        if s == 1:
            continue
        if sizes and strides[-1] == st * s:
            sizes[-1] *= s
        else:
            sizes.append(s)
            strides.append(st)
    return sizes, strides


def _launch_one(t, e):
    n = t.numel()
    if n == 0:
        return
    e = float(e)
    block = _block_for(n)
    nw = _warps_for(block)
    grid = ((n + block - 1) // block,)
    if t.is_contiguous():
        _pow1[grid](t, e, n, BLOCK=block, num_warps=nw, num_stages=1)
        return
    ds, ss = _collapse(t)
    if len(ds) > 8:
        raise RuntimeError("_foreach_pow_: unsupported tensor rank")
    while len(ds) < 8:
        ds.append(1)
        ss.append(0)
    _pow_strided[grid](
        t,
        e,
        n,
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        BLOCK=block,
        num_warps=nw,
        num_stages=1,
    )


def _launch_uniform(ts, e, numel, special):
    k = len(ts)
    ps = list(ts)
    if k < _SLOTS:
        ps.extend([ps[0]] * (_SLOTS - k))
    block = _block_for(numel)
    grid = ((numel + block - 1) // block, k)
    stream = numel >= 1048576 and ps[0].element_size() == 2
    even = (numel % block) == 0
    nw = _warps_for(block)
    if special and even:
        # The shared exponent and even the element count are compile-time
        # constants here, so the launch carries only the 16 data pointers.
        _pow16s[grid](
            ps[0],
            ps[1],
            ps[2],
            ps[3],
            ps[4],
            ps[5],
            ps[6],
            ps[7],
            ps[8],
            ps[9],
            ps[10],
            ps[11],
            ps[12],
            ps[13],
            ps[14],
            ps[15],
            SPECIAL=special,
            STREAM=stream,
            BLOCK=block,
            num_warps=nw,
            num_stages=1,
        )
        return
    _pow16u[grid](
        ps[0],
        ps[1],
        ps[2],
        ps[3],
        ps[4],
        ps[5],
        ps[6],
        ps[7],
        ps[8],
        ps[9],
        ps[10],
        ps[11],
        ps[12],
        ps[13],
        ps[14],
        ps[15],
        e,
        numel,
        SPECIAL=special,
        EVEN=even,
        STREAM=stream,
        BLOCK=block,
        num_warps=nw,
        num_stages=1,
    )


def _launch_multi(ts, es):
    k = len(ts)
    anchor = None
    maxn = 0
    ps = []
    ns = []
    ee = []
    for t, e in zip(ts, es):
        nn = t.numel()
        if nn:
            if anchor is None:
                anchor = t
            if nn > maxn:
                maxn = nn
            ps.append(t)
            ns.append(nn)
            ee.append(e)
        else:
            ps.append(None)
            ns.append(0)
            ee.append(0.0)
    if maxn == 0:
        return
    while len(ps) < _SLOTS:
        ps.append(None)
        ns.append(0)
        ee.append(0.0)
    ps = [anchor if p is None else p for p in ps]
    ee = [float(v) for v in ee]
    block = _block_for(maxn)
    grid = ((maxn + block - 1) // block, k)
    _pow16[grid](
        ps[0],
        ps[1],
        ps[2],
        ps[3],
        ps[4],
        ps[5],
        ps[6],
        ps[7],
        ps[8],
        ps[9],
        ps[10],
        ps[11],
        ps[12],
        ps[13],
        ps[14],
        ps[15],
        ee[0],
        ee[1],
        ee[2],
        ee[3],
        ee[4],
        ee[5],
        ee[6],
        ee[7],
        ee[8],
        ee[9],
        ee[10],
        ee[11],
        ee[12],
        ee[13],
        ee[14],
        ee[15],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        ns[8],
        ns[9],
        ns[10],
        ns[11],
        ns[12],
        ns[13],
        ns[14],
        ns[15],
        BLOCK=block,
        num_warps=_warps_for(block),
        num_stages=1,
    )


def foreach_pow_scalar_list_(self, exponent=None, *args, **kwargs):
    tensors = self
    if exponent is None:
        exponent = kwargs.get("exponents", kwargs.get("exponent"))
    if exponent is None and args:
        exponent = args[0]

    n = len(tensors)
    if n != len(exponent):
        raise RuntimeError(
            "_foreach_pow_: expected the same number of tensors and scalars, "
            "but got %d tensors and %d scalars" % (n, len(exponent))
        )
    if n == 0:
        return None

    e0 = exponent[0]
    if type(e0) is float:
        exps = exponent if type(exponent) is list else list(exponent)
    else:
        exps = list(map(float, exponent))
        e0 = exps[0]

    t0 = tensors[0]
    if n == 1:
        _launch_one(t0, e0)
        return None

    nn0 = t0.numel()
    uniform = True
    fused = True
    seen = set()
    for i in range(n):
        t = tensors[i]
        if t.numel() != nn0 or exps[i] != e0:
            uniform = False
        if not t.is_contiguous():
            fused = False
            break
        it = id(t)
        if it in seen:
            fused = False
            break
        seen.add(it)

    if not fused:
        for t, e in zip(tensors, exps):
            _launch_one(t, e)
        return None

    if uniform:
        if nn0 == 0:
            return None
        special = _SPECIAL.get(e0, 0)
        if n <= _SLOTS:
            _launch_uniform(tensors, e0, nn0, special)
        else:
            for s in range(0, n, _SLOTS):
                _launch_uniform(tensors[s : s + _SLOTS], e0, nn0, special)
        return None

    if n <= _SLOTS:
        _launch_multi(tensors, exps)
    else:
        for s in range(0, n, _SLOTS):
            _launch_multi(tensors[s : s + _SLOTS], exps[s : s + _SLOTS])
    return None
