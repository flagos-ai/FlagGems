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

"""In-place foreach clamp_min against a scalar (Triton).

Semantics (ATen `_foreach_clamp_min_.Scalar`): for each tensor T and element x,
    x <- (isnan(x) ? x : max_num(x, s))
where max_num is the non-NaN-propagating IEEE maximum.  ATen narrows the scalar
to the tensor's opmath type; rounding it to the storage dtype is exact for
fp16/bf16 (monotone, idempotent), so the comparison is done directly in the
storage dtype.

Layouts: a tensor whose element addresses exactly cover a flat range
(contiguous, permuted/transposed, outer-dim slices) is processed as a flat 1-D
stream (coalesced, vectorizable); anything gapped uses a strided ND kernel.

Launch count: dense tensors sharing a device and dtype are fused into a single
launch (up to FUSE_MAX tensors).  The grid covers the sum of their per-tensor
block counts and each program resolves (tensor, local block) from its program id.

Host cost: per-call latency for short lists of small tensors is dominated by
host dispatch, so the common case (all tensors contiguous, same device and
dtype, equal element count) takes a minimal-argument, minimal-Python path: one
kernel with (p0..p3, n, blocks_per_tensor, s) and t = pid // bpt.  Every other
shape falls back to the general boundary-chain kernel or the strided ND kernel.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)

_BLOCK_BY_ITEM = {1: 4096, 2: 2048, 4: 1024, 8: 512}
FUSE_MAX = 4


@triton.jit
def _apply(x, s):
    # NaN-preserving IEEE maxNum: a NaN element stays NaN, a NaN scalar is a no-op.
    return tl.where(x != x, x, tl.maximum(x, s))


@triton.jit
def _clamp_flat_s(ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(ptr + offs, mask=m, other=0)
    tl.store(ptr + offs, _apply(x, s), mask=m)


@triton.jit
def _clamp_flat_ts(ptr, n, sp, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(ptr + offs, mask=m, other=0)
    s = tl.load(sp)
    tl.store(ptr + offs, _apply(x, s), mask=m)


@triton.jit
def _clamp_eq(p0, p1, p2, p3, n, bpt, s, BLOCK: tl.constexpr, K: tl.constexpr):
    """Fused in-place clamp for K<=4 dense tensors with identical element count."""
    pid = tl.program_id(0)
    t = pid // bpt
    o = pid - t * bpt
    p = p0
    if K > 1:
        if t == 1:
            p = p1
    if K > 2:
        if t == 2:
            p = p2
    if K > 3:
        if t == 3:
            p = p3
    offs = o * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(p + offs, mask=m, other=0)
    tl.store(p + offs, _apply(x, s), mask=m)


@triton.jit
def _clamp_flat_fused(
    p0, p1, p2, p3, n0, n1, n2, n3, s1, s2, s3, s, BLOCK: tl.constexpr, K: tl.constexpr
):
    """General fused variant: per-tensor element counts via cumulative starts."""
    pid = tl.program_id(0)
    t = 0
    o = pid
    if K > 1:
        if pid >= s1:
            t = 1
            o = pid - s1
    if K > 2:
        if pid >= s2:
            t = 2
            o = pid - s2
    if K > 3:
        if pid >= s3:
            t = 3
            o = pid - s3
    p = p0
    n = n0
    if t == 1:
        p = p1
        n = n1
    if t == 2:
        p = p2
        n = n2
    if t == 3:
        p = p3
        n = n3
    offs = o * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(p + offs, mask=m, other=0)
    tl.store(p + offs, _apply(x, s), mask=m)


@triton.jit
def _nd_addr(offs, shp0, shp1, shp2, shp3, str0, str1, str2, str3, NDIM: tl.constexpr):
    idx = offs
    addr = offs * 0
    if NDIM == 4:
        c = idx % shp3
        idx = idx // shp3
        addr += c * str3
    if NDIM >= 3:
        c = idx % shp2
        idx = idx // shp2
        addr += c * str2
    if NDIM >= 2:
        c = idx % shp1
        idx = idx // shp1
        addr += c * str1
    addr += idx * str0
    return addr


@triton.jit
def _clamp_nd_s(
    ptr,
    n,
    s,
    shp0,
    shp1,
    shp2,
    shp3,
    str0,
    str1,
    str2,
    str3,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    addr = _nd_addr(offs, shp0, shp1, shp2, shp3, str0, str1, str2, str3, NDIM)
    x = tl.load(ptr + addr, mask=m, other=0)
    tl.store(ptr + addr, _apply(x, s), mask=m)


@triton.jit
def _clamp_nd_ts(
    ptr,
    n,
    sp,
    shp0,
    shp1,
    shp2,
    shp3,
    str0,
    str1,
    str2,
    str3,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    addr = _nd_addr(offs, shp0, shp1, shp2, shp3, str0, str1, str2, str3, NDIM)
    x = tl.load(ptr + addr, mask=m, other=0)
    s = tl.load(sp)
    tl.store(ptr + addr, _apply(x, s), mask=m)


def _is_dense(t):
    """True when the element addresses cover exactly [0, numel) (flat-safe)."""
    if t.is_contiguous():
        return True
    pairs = [(int(sz), int(st)) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not pairs:
        return True
    pairs = sorted(pairs, key=lambda p: p[1])
    if pairs[0][1] != 1:
        return False
    for i in range(1, len(pairs)):
        if pairs[i][1] != pairs[i - 1][1] * pairs[i - 1][0]:
            return False
    return True


def _collapse(shape, strides):
    dims = [[int(sz), int(st)] for sz, st in zip(shape, strides) if sz != 1]
    changed = True
    while changed:
        changed = False
        out = []
        for sz, st in dims:
            if out and out[-1][1] * out[-1][0] == st:
                out[-1][0] *= sz
                changed = True
            else:
                out.append([sz, st])
        dims = out
    return [d[0] for d in dims], [d[1] for d in dims]


_CK_CACHE = {}


def _scalar_kind(s):
    """Triton's inferred dtype class of the runtime scalar argument."""
    if isinstance(s, float):
        return 0
    if -2147483648 <= s <= 2147483647:
        return 1
    return 2


def _launch_eq(chunk, n, bpt, block, s, dev_idx):
    """One fused launch for 1..FUSE_MAX dense tensors with identical numel.

    The first call for a (dtype, BLOCK, K, device, element count, block count,
    pointer-alignment) key compiles through the normal JIT path and keeps the
    returned CompiledKernel; later calls launch it directly, which skips
    Triton's Python binder, argument specialisation and key construction.  The
    cache key covers every property Triton specialises on (element type, the
    constexprs, pointer alignment mod 16, integer divisibility/equal-to-1 via
    the raw values, and the scalar's inferred dtype class), so a reused kernel
    is always the one the JIT would have selected.
    """
    k = len(chunk)
    p0 = chunk[0]
    if k == 1:
        p1 = p2 = p3 = p0
    elif k == 2:
        p1 = chunk[1]
        p2 = p3 = p0
    elif k == 3:
        p1 = chunk[1]
        p2 = chunk[2]
        p3 = p0
    else:
        p1 = chunk[1]
        p2 = chunk[2]
        p3 = chunk[3]
    total = bpt * k
    key = (
        p0.dtype,
        block,
        k,
        dev_idx,
        n,
        bpt,
        _scalar_kind(s),
        p0.data_ptr() & 15,
        p1.data_ptr() & 15,
        p2.data_ptr() & 15,
        p3.data_ptr() & 15,
    )
    ck = _CK_CACHE.get(key)
    if ck is None:
        if len(_CK_CACHE) > 512:
            _CK_CACHE.clear()
        _CK_CACHE[key] = _clamp_eq[(total,)](
            p0, p1, p2, p3, n, bpt, s, BLOCK=block, K=k, num_warps=4
        )
    else:
        ck[(total, 1, 1)](p0, p1, p2, p3, n, bpt, s, block, k)


def _launch_group(chunk, block, scalar):
    """One fused launch for 1..FUSE_MAX dense tensors with differing numels."""
    ns = [t.numel() for t in chunk]
    k = len(chunk)
    if k == 1:
        _clamp_flat_s[(triton.cdiv(ns[0], block),)](
            chunk[0], ns[0], scalar, BLOCK=block, num_warps=4
        )
        return
    b0 = triton.cdiv(ns[0], block)
    b1 = triton.cdiv(ns[1], block)
    s1 = b0
    s2 = s1 + b1
    total = s2
    if k > 2:
        b2 = triton.cdiv(ns[2], block)
        s3 = s2 + b2
        total = s3
    else:
        s3 = s2
    if k > 3:
        total = s3 + triton.cdiv(ns[3], block)
    p0 = chunk[0]
    p1 = chunk[1]
    p2 = chunk[2] if k > 2 else p0
    p3 = chunk[3] if k > 3 else p0
    n2 = ns[2] if k > 2 else 0
    n3 = ns[3] if k > 3 else 0
    _clamp_flat_fused[(total,)](
        p0,
        p1,
        p2,
        p3,
        ns[0],
        ns[1],
        n2,
        n3,
        s1,
        s2,
        s3,
        scalar,
        BLOCK=block,
        K=k,
        num_warps=4,
    )


def _foreach_clamp_min__Scalar(self, scalar):
    logger.debug("GEMS_HYGON _FOREACH_CLAMP_MIN__SCALAR")
    tensors = self if isinstance(self, list) else list(self)
    nt = len(tensors)
    if not nt:
        return None
    is_ts = isinstance(scalar, torch.Tensor)

    # ---- common case: contiguous, one device+dtype, identical element count
    if not is_ts:
        t0 = tensors[0]
        n = t0.numel()
        if n and t0.is_contiguous():
            dt = t0.dtype
            dev = t0.device
            ok = True
            for i in range(1, nt):
                t = tensors[i]
                if (
                    t.numel() != n
                    or t.dtype != dt
                    or t.device != dev
                    or not t.is_contiguous()
                ):
                    ok = False
                    break
            if ok:
                block = _BLOCK_BY_ITEM.get(t0.element_size(), 1024)
                di = dev.index
                cur = torch.cuda.current_device()
                if di is not None and di != cur:
                    torch.cuda.set_device(di)
                    cur = di
                bpt = -(-n // block)
                s = int(scalar) if dt in _INT_DTYPES else float(scalar)
                if nt <= FUSE_MAX:
                    _launch_eq(tensors, n, bpt, block, s, cur)
                else:
                    for i in range(0, nt, FUSE_MAX):
                        _launch_eq(tensors[i : i + FUSE_MAX], n, bpt, block, s, cur)
                return None

    # ---- general path: arbitrary layouts, dtypes, devices and element counts
    scalar_f = None if is_ts else float(scalar)
    groups = {}
    nd = []
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        if _is_dense(t):
            key = (t.device, t.dtype)
            g = groups.get(key)
            if g is None:
                groups[key] = [t]
            else:
                g.append(t)
        else:
            nd.append(t)

    cur = torch.cuda.current_device()
    for (dev, dtype), chunk in groups.items():
        idx = dev.index
        if idx is not None and idx != cur:
            torch.cuda.set_device(idx)
            cur = idx
        block = _BLOCK_BY_ITEM.get(chunk[0].element_size(), 1024)
        if is_ts:
            for t in chunk:
                n = t.numel()
                _clamp_flat_ts[(triton.cdiv(n, block),)](
                    t, n, scalar, BLOCK=block, num_warps=4
                )
        else:
            arg = int(scalar) if dtype in _INT_DTYPES else scalar_f
            for i in range(0, len(chunk), FUSE_MAX):
                _launch_group(chunk[i : i + FUSE_MAX], block, arg)

    if nd:
        for t in nd:
            idx = t.device.index
            if idx is not None and idx != cur:
                torch.cuda.set_device(idx)
                cur = idx
            if is_ts:
                arg = scalar
            else:
                arg = int(scalar) if t.dtype in _INT_DTYPES else scalar_f
            n = t.numel()
            block = _BLOCK_BY_ITEM.get(t.element_size(), 1024)
            grid = (triton.cdiv(n, block),)
            shp, strd = _collapse(t.shape, t.stride())
            ndim = len(shp)
            if ndim == 0:
                shp, strd, ndim = [1], [1], 1
            if ndim > 4:
                raise NotImplementedError("unsupported layout rank %d" % ndim)
            pad_s = shp + [1] * (4 - ndim)
            pad_t = strd + [1] * (4 - ndim)
            if is_ts:
                _clamp_nd_ts[grid](
                    t,
                    n,
                    arg,
                    pad_s[0],
                    pad_s[1],
                    pad_s[2],
                    pad_s[3],
                    pad_t[0],
                    pad_t[1],
                    pad_t[2],
                    pad_t[3],
                    BLOCK=block,
                    NDIM=ndim,
                    num_warps=4,
                )
            else:
                _clamp_nd_s[grid](
                    t,
                    n,
                    arg,
                    pad_s[0],
                    pad_s[1],
                    pad_s[2],
                    pad_s[3],
                    pad_t[0],
                    pad_t[1],
                    pad_t[2],
                    pad_t[3],
                    BLOCK=block,
                    NDIM=ndim,
                    num_warps=4,
                )
    return None
