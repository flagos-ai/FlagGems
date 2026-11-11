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

"""In-place foreach addcmul (ScalarList overload) implemented with Triton.

self[i] = self[i] + scalars[i] * tensor1[i] * tensor2[i]   (in place, returns None)

Architecture: hybrid host dispatch over the (contiguous, shape-matched) fast
path.  Launch-bound sizes pack the whole list into one launch:
grid = (sum_i cdiv(n_i, BLOCK),), each program resolving its list member from
runtime cumulative block counts computed in-kernel from the per-slot numels, so
no extra host work or device metadata tensor is needed (slots a call does not
use carry n=0 and contribute zero blocks).  That launch is issued through the
compiled kernel's launcher directly (the JIT signature binding and cache-key
work is skipped, only the compiled code is cached), which removes most of the
per-call host cost of the wide argument list.  Streaming sizes instead issue one
low-register single-tensor launch per member, where host launch cost is hidden
behind the DRAM work.  Non-contiguous / broadcast members fall back to
per-tensor strided launches.
"""

import operator

import torch
import triton
import triton.language as tl

try:  # launcher internals (used only for the direct-launch fast path)
    import triton.runtime.jit as _triton_jit
    from triton.runtime import driver as _driver

    _DIRECT_AVAILABLE = hasattr(_triton_jit, "JITFunction")
except Exception:  # pragma: no cover - fall back to the normal launch path
    _triton_jit = None
    _driver = None
    _DIRECT_AVAILABLE = False

_BLOCK = 1024
_NUM_WARPS = 4
_NDIM_MAX = 8
_MAXS = 16
# Total packed grid (in blocks) below which one fused launch beats a launch per
# member.  Above it the call is DRAM-bound, host launch cost is hidden behind the
# GPU work, and the low-register single-tensor kernel has better streaming
# throughput.  Derived from measured dispatch evidence, not from SM counts.
_FUSE_BLOCKS = 2048

_CK_CACHE = {}
_DIRECT_DISABLED = not _DIRECT_AVAILABLE

# C-level metadata accessors: mapping them over a tensor list keeps the per-call
# validation out of the Python bytecode loop.
_NUMEL = torch.Tensor.numel
_CONTIG = torch.Tensor.is_contiguous
_DTYPE_OF = operator.attrgetter("dtype")


@triton.jit
def _elem(s_ptr, a_ptr, b_ptr, n, alpha, base, BLOCK: tl.constexpr):
    off = base + tl.arange(0, BLOCK)
    m = off < n
    s = tl.load(s_ptr + off, mask=m, other=0.0)
    a = tl.load(a_ptr + off, mask=m, other=0.0)
    b = tl.load(b_ptr + off, mask=m, other=0.0)
    acc = s.to(tl.float32) + alpha * a.to(tl.float32) * b.to(tl.float32)
    tl.store(s_ptr + off, acc, mask=m)


@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
    ]
)
def _fused16(
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
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
    v14,
    v15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    c0 = (n0 + (BLOCK - 1)) // BLOCK
    c1 = c0 + (n1 + (BLOCK - 1)) // BLOCK
    c2 = c1 + (n2 + (BLOCK - 1)) // BLOCK
    c3 = c2 + (n3 + (BLOCK - 1)) // BLOCK
    c4 = c3 + (n4 + (BLOCK - 1)) // BLOCK
    c5 = c4 + (n5 + (BLOCK - 1)) // BLOCK
    c6 = c5 + (n6 + (BLOCK - 1)) // BLOCK
    c7 = c6 + (n7 + (BLOCK - 1)) // BLOCK
    c8 = c7 + (n8 + (BLOCK - 1)) // BLOCK
    c9 = c8 + (n9 + (BLOCK - 1)) // BLOCK
    c10 = c9 + (n10 + (BLOCK - 1)) // BLOCK
    c11 = c10 + (n11 + (BLOCK - 1)) // BLOCK
    c12 = c11 + (n12 + (BLOCK - 1)) // BLOCK
    c13 = c12 + (n13 + (BLOCK - 1)) // BLOCK
    c14 = c13 + (n14 + (BLOCK - 1)) // BLOCK

    t = 0
    t = tl.where(pid >= c0, 1, t)
    t = tl.where(pid >= c1, 2, t)
    t = tl.where(pid >= c2, 3, t)
    t = tl.where(pid >= c3, 4, t)
    t = tl.where(pid >= c4, 5, t)
    t = tl.where(pid >= c5, 6, t)
    t = tl.where(pid >= c6, 7, t)
    t = tl.where(pid >= c7, 8, t)
    t = tl.where(pid >= c8, 9, t)
    t = tl.where(pid >= c9, 10, t)
    t = tl.where(pid >= c10, 11, t)
    t = tl.where(pid >= c11, 12, t)
    t = tl.where(pid >= c12, 13, t)
    t = tl.where(pid >= c13, 14, t)
    t = tl.where(pid >= c14, 15, t)

    if t == 0:
        _elem(s0, a0, b0, n0, v0, pid * BLOCK, BLOCK)
    elif t == 1:
        _elem(s1, a1, b1, n1, v1, (pid - c0) * BLOCK, BLOCK)
    elif t == 2:
        _elem(s2, a2, b2, n2, v2, (pid - c1) * BLOCK, BLOCK)
    elif t == 3:
        _elem(s3, a3, b3, n3, v3, (pid - c2) * BLOCK, BLOCK)
    elif t == 4:
        _elem(s4, a4, b4, n4, v4, (pid - c3) * BLOCK, BLOCK)
    elif t == 5:
        _elem(s5, a5, b5, n5, v5, (pid - c4) * BLOCK, BLOCK)
    elif t == 6:
        _elem(s6, a6, b6, n6, v6, (pid - c5) * BLOCK, BLOCK)
    elif t == 7:
        _elem(s7, a7, b7, n7, v7, (pid - c6) * BLOCK, BLOCK)
    elif t == 8:
        _elem(s8, a8, b8, n8, v8, (pid - c7) * BLOCK, BLOCK)
    elif t == 9:
        _elem(s9, a9, b9, n9, v9, (pid - c8) * BLOCK, BLOCK)
    elif t == 10:
        _elem(s10, a10, b10, n10, v10, (pid - c9) * BLOCK, BLOCK)
    elif t == 11:
        _elem(s11, a11, b11, n11, v11, (pid - c10) * BLOCK, BLOCK)
    elif t == 12:
        _elem(s12, a12, b12, n12, v12, (pid - c11) * BLOCK, BLOCK)
    elif t == 13:
        _elem(s13, a13, b13, n13, v13, (pid - c12) * BLOCK, BLOCK)
    elif t == 14:
        _elem(s14, a14, b14, n14, v14, (pid - c13) * BLOCK, BLOCK)
    else:
        _elem(s15, a15, b15, n15, v15, (pid - c14) * BLOCK, BLOCK)


@triton.jit(do_not_specialize=["n"])
def _fused16_h(
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
    n,
    v,
    BLOCK: tl.constexpr,
):
    """Fused addcmul for a list whose members share one numel and one alpha.

    grid = k * cdiv(n, BLOCK), so the member index is simply pid // cb and each
    member's element base is (pid - t * cb) * BLOCK.  Only 50 runtime arguments
    reach the launcher instead of 81 for the fully general form.
    """
    pid = tl.program_id(0)
    cb = (n + (BLOCK - 1)) // BLOCK
    t = pid // cb
    base = (pid - t * cb) * BLOCK
    if t == 0:
        _elem(s0, a0, b0, n, v, base, BLOCK)
    elif t == 1:
        _elem(s1, a1, b1, n, v, base, BLOCK)
    elif t == 2:
        _elem(s2, a2, b2, n, v, base, BLOCK)
    elif t == 3:
        _elem(s3, a3, b3, n, v, base, BLOCK)
    elif t == 4:
        _elem(s4, a4, b4, n, v, base, BLOCK)
    elif t == 5:
        _elem(s5, a5, b5, n, v, base, BLOCK)
    elif t == 6:
        _elem(s6, a6, b6, n, v, base, BLOCK)
    elif t == 7:
        _elem(s7, a7, b7, n, v, base, BLOCK)
    elif t == 8:
        _elem(s8, a8, b8, n, v, base, BLOCK)
    elif t == 9:
        _elem(s9, a9, b9, n, v, base, BLOCK)
    elif t == 10:
        _elem(s10, a10, b10, n, v, base, BLOCK)
    elif t == 11:
        _elem(s11, a11, b11, n, v, base, BLOCK)
    elif t == 12:
        _elem(s12, a12, b12, n, v, base, BLOCK)
    elif t == 13:
        _elem(s13, a13, b13, n, v, base, BLOCK)
    elif t == 14:
        _elem(s14, a14, b14, n, v, base, BLOCK)
    else:
        _elem(s15, a15, b15, n, v, base, BLOCK)


@triton.jit
def _addcmul_flat(
    s_ptr, a_ptr, b_ptr, n, alpha, BLOCK: tl.constexpr, F64: tl.constexpr
):
    if F64:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        s = tl.load(s_ptr + off, mask=m, other=0.0)
        a = tl.load(a_ptr + off, mask=m, other=0.0)
        b = tl.load(b_ptr + off, mask=m, other=0.0)
        tl.store(s_ptr + off, s + alpha.to(tl.float64) * a * b, mask=m)
    else:
        _elem(s_ptr, a_ptr, b_ptr, n, alpha, tl.program_id(0) * BLOCK, BLOCK)


@triton.jit
def _addcmul_strided(
    s_ptr,
    a_ptr,
    b_ptr,
    alpha,
    total,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    ps0,
    ps1,
    ps2,
    ps3,
    ps4,
    ps5,
    ps6,
    ps7,
    pa0,
    pa1,
    pa2,
    pa3,
    pa4,
    pa5,
    pa6,
    pa7,
    pb0,
    pb1,
    pb2,
    pb3,
    pb4,
    pb5,
    pb6,
    pb7,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < total
    so = tl.zeros([BLOCK], tl.int32)
    ao = tl.zeros([BLOCK], tl.int32)
    bo = tl.zeros([BLOCK], tl.int32)
    r = idx
    i = r % d7
    r = r // d7
    so += i * ps7
    ao += i * pa7
    bo += i * pb7
    i = r % d6
    r = r // d6
    so += i * ps6
    ao += i * pa6
    bo += i * pb6
    i = r % d5
    r = r // d5
    so += i * ps5
    ao += i * pa5
    bo += i * pb5
    i = r % d4
    r = r // d4
    so += i * ps4
    ao += i * pa4
    bo += i * pb4
    i = r % d3
    r = r // d3
    so += i * ps3
    ao += i * pa3
    bo += i * pb3
    i = r % d2
    r = r // d2
    so += i * ps2
    ao += i * pa2
    bo += i * pb2
    i = r % d1
    r = r // d1
    so += i * ps1
    ao += i * pa1
    bo += i * pb1
    i = r % d0
    so += i * ps0
    ao += i * pa0
    bo += i * pb0
    s = tl.load(s_ptr + so, mask=m, other=0.0)
    a = tl.load(a_ptr + ao, mask=m, other=0.0)
    b = tl.load(b_ptr + bo, mask=m, other=0.0)
    acc = s.to(tl.float32) + alpha * a.to(tl.float32) * b.to(tl.float32)
    tl.store(s_ptr + so, acc, mask=m)


def _build_direct(kernel, dtype, device, extra):
    """Compile one fused kernel once for this pointer dtype and cache its launcher.

    Warm-up uses deliberately misaligned views so the compiled code carries no
    alignment assumption, and runtime scalars that are neither 1 nor a multiple
    of the specialization divisibility, so no runtime value is baked in.  Only
    compiled code is cached, never tensor data.
    """
    dummies = []
    for _ in range(_MAXS * 3):
        dummies.append(torch.empty(65, dtype=dtype, device=device)[1:])
    ck = kernel.warmup(*dummies, *extra, BLOCK=_BLOCK, num_warps=_NUM_WARPS, grid=(1,))
    if hasattr(ck, "result"):
        ck = ck.result()
    dist = ()
    ctx_cls = getattr(_triton_jit, "DistributedRtContext", None)
    if ctx_cls is not None:
        ctx = ctx_cls()
        if getattr(ctx, "is_lite_mode", False):
            dist = (ctx.comm_ptr, ctx.mem_ptr)
    return (ck, dist)


def _direct(kernel, tag, extra, values, total):
    """Launch a cached compiled kernel without Triton's per-call JIT binding."""
    global _DIRECT_DISABLED
    first = values[0]
    key = (tag, first.dtype, first.device)
    ent = _CK_CACHE.get(key)
    if ent is None:
        try:
            ent = _build_direct(kernel, first.dtype, first.device, extra)
        except Exception:
            _DIRECT_DISABLED = True
            return False
        _CK_CACHE[key] = ent
    ck, dist = ent
    try:
        stream = _driver.active.get_current_stream(_driver.active.get_current_device())
        ck.run(
            total,
            1,
            1,
            stream,
            ck.function,
            ck.packed_metadata,
            None,
            None,
            None,
            *dist,
            *values,
        )
    except Exception:
        _DIRECT_DISABLED = True
        return False
    return True


_GEN_EXTRA = [4095] * _MAXS + [0.0] * _MAXS
_HOM_EXTRA = [4095, 0.0]


def _pad(ss, as_, bs):
    if len(ss) < _MAXS:
        # only copy when the argument lists must be padded to the slot count
        ss = list(ss)
        as_ = list(as_)
        bs = list(bs)
        ps, pa, pb = ss[0], as_[0], bs[0]
        while len(ss) < _MAXS:
            ss.append(ps)
            as_.append(pa)
            bs.append(pb)
    return ss, as_, bs


def _launch_fused(ss, as_, bs, ns, vs, total):
    if total == 0:
        return
    ss, as_, bs = _pad(ss, as_, bs)
    if len(ns) < _MAXS:
        ns = list(ns)
        vs = list(vs)
        ns += [0] * (_MAXS - len(ns))
        vs += [0.0] * (_MAXS - len(vs))
    if not _DIRECT_DISABLED and _direct(
        _fused16, 0, _GEN_EXTRA, (*ss, *as_, *bs, *ns, *vs, _BLOCK), total
    ):
        return
    _fused16[(total,)](
        *ss,
        *as_,
        *bs,
        *ns,
        *vs,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_fused_h(ss, as_, bs, n, v, total):
    if total == 0:
        return
    ss, as_, bs = _pad(ss, as_, bs)
    if not _DIRECT_DISABLED and _direct(
        _fused16_h, 1, _HOM_EXTRA, (*ss, *as_, *bs, n, v, _BLOCK), total
    ):
        return
    _fused16_h[(total,)](*ss, *as_, *bs, n, v, BLOCK=_BLOCK, num_warps=_NUM_WARPS)


def _launch_flat(s, a, b, n, alpha, blk):
    _addcmul_flat[((n + blk - 1) // blk,)](
        s, a, b, n, alpha, BLOCK=blk, F64=False, num_warps=_NUM_WARPS
    )


def _launch_one(s, a, b, alpha):
    n = s.numel()
    if n == 0:
        return
    f64 = s.dtype == torch.float64
    if (
        a.shape == s.shape
        and b.shape == s.shape
        and s.is_contiguous()
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _addcmul_flat[grid](
            s, a, b, n, alpha, BLOCK=_BLOCK, F64=f64, num_warps=_NUM_WARPS
        )
        return
    bshape = torch.broadcast_shapes(s.shape, a.shape, b.shape)
    if tuple(bshape) != tuple(s.shape):
        raise RuntimeError(
            "addcmul_: output with shape {} doesn't match the broadcast shape {}".format(
                tuple(s.shape), tuple(bshape)
            )
        )
    a = a.expand(s.shape)
    b = b.expand(s.shape)
    nd = s.dim()
    if nd > _NDIM_MAX:
        raise NotImplementedError("tensor rank above 8 is not supported")
    pad = _NDIM_MAX - nd
    shape = [1] * pad + list(s.shape)
    sstr = [0] * pad + list(s.stride())
    astr = [0] * pad + list(a.stride())
    bstr = [0] * pad + list(b.stride())
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _addcmul_strided[grid](
        s,
        a,
        b,
        alpha,
        n,
        *shape,
        *sstr,
        *astr,
        *bstr,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _fallback(ss, as_, bs, alphas, k):
    for i in range(k):
        _launch_one(ss[i], as_[i], bs[i], alphas[i])


def foreach_addcmul_scalar_list_(self, tensor1, tensor2, scalars):
    if isinstance(self, torch.Tensor):
        self = [self]
        if isinstance(tensor1, torch.Tensor):
            tensor1 = [tensor1]
        if isinstance(tensor2, torch.Tensor):
            tensor2 = [tensor2]
    k = len(self)
    if k == 0:
        return None
    if len(tensor1) != k or len(tensor2) != k:
        raise ValueError("tensor lists must all have the same length")
    if isinstance(scalars, (list, tuple)):
        alphas = [x if type(x) is float else float(x) for x in scalars]
        la = len(alphas)
        if la != k:
            if la == 1:
                alphas = alphas * k
            else:
                raise ValueError(
                    "scalars must be a list with one entry per tensor list member"
                )
    else:
        alphas = [float(scalars)] * k

    ss = self
    as_ = tensor1
    bs = tensor2
    d0 = ss[0].dtype
    if d0 == torch.float64:
        _fallback(ss, as_, bs, alphas, k)
        return None
    # The flat fast path is valid when every member triple is contiguous and
    # flat-index equivalent: equal numels plus ATen broadcast validity imply
    # equal shapes, so numel equality replaces the shape comparison.
    ns = list(map(_NUMEL, ss))
    if (
        list(map(_NUMEL, as_)) != ns
        or list(map(_NUMEL, bs)) != ns
        or not all(map(_CONTIG, ss))
        or not all(map(_CONTIG, as_))
        or not all(map(_CONTIG, bs))
    ):
        _fallback(ss, as_, bs, alphas, k)
        return None
    dts = list(map(_DTYPE_OF, ss))
    dts += map(_DTYPE_OF, as_)
    dts += map(_DTYPE_OF, bs)
    if dts.count(d0) != len(dts):
        _fallback(ss, as_, bs, alphas, k)
        return None
    n0 = ns[0]
    hom_n = ns.count(n0) == k
    if hom_n:
        blocks = k * ((n0 + 1023) >> 10)
    else:
        blocks = 0
        for n in ns:
            blocks += (n + 1023) >> 10

    if blocks <= _FUSE_BLOCKS:
        # launch-bound regime: pack the whole list into as few launches as possible
        if k > _MAXS:
            for j in range(0, k, _MAXS):
                hi = j + _MAXS
                if hi > k:
                    hi = k
                total = 0
                for i in range(j, hi):
                    total += (ns[i] + 1023) >> 10
                _launch_fused(
                    ss[j:hi], as_[j:hi], bs[j:hi], ns[j:hi], alphas[j:hi], total
                )
        elif hom_n and alphas.count(alphas[0]) == k:
            # every member shares one numel and one alpha: use the narrow kernel
            _launch_fused_h(ss, as_, bs, n0, alphas[0], blocks)
        else:
            _launch_fused(ss, as_, bs, ns, alphas, blocks)
    else:
        # streaming regime: one low-overhead kernel per member.  2-byte dtypes
        # move half the bytes per element, so they need twice the elements per
        # thread to stay request-bound rather than byte-bound.
        blk = _BLOCK if d0.itemsize >= 4 else 2 * _BLOCK
        for i in range(k):
            if ns[i]:
                _launch_flat(ss[i], as_[i], bs[i], ns[i], alphas[i], blk)
    return None
