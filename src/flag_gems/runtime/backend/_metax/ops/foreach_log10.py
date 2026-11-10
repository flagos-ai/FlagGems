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

"""Triton implementation of ``aten::_foreach_log10``.

Reference-semantics facts measured on the target MetaX C550 (torch 2.10):
  * floating dtypes are preserved; integral / bool inputs promote to the default
    floating dtype (float32); complex inputs stay complex,
  * an empty tensor list raises RuntimeError("Tensor list must have at least one tensor."),
  * outputs are freshly allocated, never alias the input storage,
  * special values follow IEEE (log10(0) = -inf, negative -> NaN, +inf -> +inf),
  * complex log10 is real = log10(hypot(a, b)), imag = atan2(b, a) * log10(e)
    (bit-exact versus ATen for complex128, 1-2 ulp for complex64).

Scheduling:
  * same-dtype contiguous tensors are fused into a single launch (up to 16 tensors per
    launch) with a per-program pointer/geometry selection, which removes the fixed
    per-launch cost that dominates short lists of small tensors;
  * a single streaming tensor uses a flat kernel whose per-thread working set is
    searched once per (dtype, size bucket) by a bounded autotune list; launch-overhead
    dominated small tensors keep a fixed geometry;
  * non-contiguous and complex tensors fall back to per-tensor launches.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024
_NUM_WARPS = 4
_INT32_MAX = 1 << 31
_SMALL_N = 1 << 18
_MID_N = 1 << 24
_MAX_K = 16

_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=4),
    triton.Config({"BLOCK": 16384}, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_warps=4),
]

_DT_KEY = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


def _dt_key(dt):
    key = _DT_KEY.get(dt)
    if key is None:
        key = 4 + int(getattr(dt, "itemsize", 0))
    return key


def _bucket(n):
    if n < _SMALL_N:
        return 0
    if n < _MID_N:
        return 1
    return 2


def _match_dtype(dt):
    if dt.is_floating_point or dt.is_complex:
        return dt
    return torch.get_default_dtype()


def _collapse(shape, stride):
    """Metadata-only dim collapse: drop size-1 dims and merge adjacent ones.

    Returned in logical (outermost-first) order.
    """
    dims = [(s, st) for s, st in zip(shape, stride) if s != 1]
    merged = []
    for s, st in reversed(dims):
        if merged:
            ps, pst = merged[-1]
            if st == pst * ps:
                merged[-1] = (ps * s, pst)
                continue
        merged.append((s, st))
    merged.reverse()
    return merged


def _decomp(dims):
    """Shape/stride tuples ordered for linear-index decomposition (last dim first)."""
    if not dims:
        return (1,), (0,)
    shapes = tuple(int(s) for s, _ in reversed(dims))
    strides = tuple(int(st) for _, st in reversed(dims))
    return shapes, strides


@triton.jit
def _flat_body(
    in_ptr, out_ptr, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr, IDX64: tl.constexpr
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(in_ptr + offs, mask=m, eviction_policy="evict_first")
    if IS_F64:
        y = libdevice.log10(x)
    else:
        y = libdevice.fast_log2f(x.to(tl.float32)) * 0.3010299956639812
    tl.store(
        out_ptr + offs,
        y.to(out_ptr.dtype.element_ty),
        mask=m,
        eviction_policy="evict_first",
    )


@triton.jit
def _log10_flat_small(
    in_ptr, out_ptr, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr, IDX64: tl.constexpr
):
    _flat_body(in_ptr, out_ptr, n, BLOCK, IS_F64, IDX64)


@triton.autotune(configs=_CONFIGS, key=["BUCKET", "DT_KEY"])
@triton.jit
def _log10_flat_tuned(
    in_ptr,
    out_ptr,
    n,
    BUCKET: tl.constexpr,
    DT_KEY: tl.constexpr,
    IS_F64: tl.constexpr,
    IDX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _flat_body(in_ptr, out_ptr, n, BLOCK, IS_F64, IDX64)


@triton.jit
def _fused_body(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    NS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    ins = (i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15)
    outs = (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)
    ip = i0
    op = o0
    n = NS[0]
    base = pid
    off = 0
    for k in tl.static_range(1, K):
        off = off + (NS[k - 1] + BLOCK - 1) // BLOCK
        sel = pid >= off
        ip = tl.where(sel, ins[k], ip)
        op = tl.where(sel, outs[k], op)
        n = tl.where(sel, NS[k], n)
        base = tl.where(sel, pid - off, base)
    offs = base * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(ip + offs, mask=m, eviction_policy="evict_first")
    if IS_F64:
        y = libdevice.log10(x)
    else:
        y = libdevice.fast_log2f(x.to(tl.float32)) * 0.3010299956639812
    tl.store(
        op + offs, y.to(op.dtype.element_ty), mask=m, eviction_policy="evict_first"
    )


@triton.jit
def _log10_fused_small(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    NS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    _fused_body(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        i12,
        i13,
        i14,
        i15,
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
        NS,
        K,
        BLOCK,
        IS_F64,
    )


@triton.autotune(configs=_CONFIGS, key=["BUCKET", "DT_KEY", "K"])
@triton.jit
def _log10_fused_tuned(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    NS: tl.constexpr,
    K: tl.constexpr,
    BUCKET: tl.constexpr,
    DT_KEY: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    _fused_body(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        i12,
        i13,
        i14,
        i15,
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
        NS,
        K,
        BLOCK,
        IS_F64,
    )


@triton.jit
def _log10_strided_kernel(
    in_ptr,
    out_ptr,
    n,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    rem = offs
    src = tl.zeros_like(offs)
    for d in tl.static_range(RANK):
        c = rem % SHAPES[d]
        rem = rem // SHAPES[d]
        src += c * STRIDES[d]
    x = tl.load(in_ptr + src, mask=m, other=0)
    if IS_F64:
        y = libdevice.log10(x)
    else:
        y = libdevice.log10(x.to(tl.float32))
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _log10_complex_kernel(
    re_ptr,
    out_re_ptr,
    n,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    rem = offs
    src = tl.zeros_like(offs)
    for d in tl.static_range(RANK):
        c = rem % SHAPES[d]
        rem = rem // SHAPES[d]
        src += c * STRIDES[d]
    a = tl.load(re_ptr + 2 * src, mask=m, other=0)
    b = tl.load(re_ptr + 2 * src + 1, mask=m, other=0)
    if IS_F64:
        real = libdevice.log10(libdevice.hypot(a, b))
        imag = libdevice.atan2(b, a) * 0.4342944819032518
    else:
        af = a.to(tl.float32)
        bf = b.to(tl.float32)
        real = libdevice.log10(libdevice.hypot(af, bf))
        imag = libdevice.atan2(bf, af) * 0.4342944819032518
    tl.store(out_re_ptr + 2 * offs, real, mask=m)
    tl.store(out_re_ptr + 2 * offs + 1, imag, mask=m)


def _launch_one(t, out, n):
    if n == 0:
        return
    if t.dtype.is_complex:
        rv = torch.view_as_real(t)
        ro = torch.view_as_real(out)
        shapes, strides = _decomp(_collapse(t.shape, t.stride()))
        grid = (triton.cdiv(n, _BLOCK),)
        _log10_complex_kernel[grid](
            rv,
            ro,
            n,
            SHAPES=shapes,
            STRIDES=strides,
            RANK=len(shapes),
            BLOCK=_BLOCK,
            IS_F64=(t.dtype == torch.complex128),
            num_warps=_NUM_WARPS,
        )
        return
    is_f64 = t.dtype == torch.float64
    if t.is_contiguous():
        if n < _SMALL_N:
            _log10_flat_small[(triton.cdiv(n, _BLOCK),)](
                t,
                out,
                n,
                BLOCK=_BLOCK,
                IS_F64=is_f64,
                IDX64=(n >= _INT32_MAX),
                num_warps=_NUM_WARPS,
            )
        else:
            _log10_flat_tuned[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
                t,
                out,
                n,
                BUCKET=_bucket(n),
                DT_KEY=_dt_key(t.dtype),
                IS_F64=is_f64,
                IDX64=(n >= _INT32_MAX),
            )
    else:
        shapes, strides = _decomp(_collapse(t.shape, t.stride()))
        grid = (triton.cdiv(n, _BLOCK),)
        _log10_strided_kernel[grid](
            t,
            out,
            n,
            SHAPES=shapes,
            STRIDES=strides,
            RANK=len(shapes),
            BLOCK=_BLOCK,
            IS_F64=is_f64,
            num_warps=_NUM_WARPS,
        )


def _launch_fused(ts, os):
    k = len(ts)
    ns = tuple(int(t.numel()) for t in ts)
    ia = list(ts) + [ts[0]] * (_MAX_K - k)
    oa = list(os) + [os[0]] * (_MAX_K - k)
    is_f64 = ts[0].dtype == torch.float64
    if max(ns) < _SMALL_N:
        blk = _BLOCK
        _log10_fused_small[(sum((v + blk - 1) // blk for v in ns),)](
            *(ia + oa), NS=ns, K=k, BLOCK=blk, IS_F64=is_f64, num_warps=_NUM_WARPS
        )
        return

    def grid(meta):
        blk = meta["BLOCK"]
        return (sum((v + blk - 1) // blk for v in ns),)

    _log10_fused_tuned[grid](
        *(ia + oa),
        NS=ns,
        K=k,
        BUCKET=_bucket(max(ns)),
        DT_KEY=_dt_key(ts[0].dtype),
        IS_F64=is_f64,
    )


def _flat_ok(t):
    return (not t.dtype.is_complex) and t.is_contiguous() and 0 < t.numel() < _INT32_MAX


def foreach_log10(self):
    tensors = self if isinstance(self, (list, tuple)) else list(self)
    nt = len(tensors)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = []
    for t in tensors:
        outs.append(torch.empty(t.shape, dtype=_match_dtype(t.dtype), device=t.device))

    if nt == 1:
        _launch_one(tensors[0], outs[0], tensors[0].numel())
        return outs

    groups = {}
    singles = []
    for i in range(nt):
        t = tensors[i]
        if _flat_ok(t):
            groups.setdefault(t.dtype, []).append(i)
        else:
            singles.append(i)

    for idxs in groups.values():
        for c in range(0, len(idxs), _MAX_K):
            chunk = idxs[c : c + _MAX_K]
            if len(chunk) == 1:
                j = chunk[0]
                _launch_one(tensors[j], outs[j], tensors[j].numel())
            else:
                _launch_fused([tensors[j] for j in chunk], [outs[j] for j in chunk])

    for i in singles:
        _launch_one(tensors[i], outs[i], tensors[i].numel())
    return outs
