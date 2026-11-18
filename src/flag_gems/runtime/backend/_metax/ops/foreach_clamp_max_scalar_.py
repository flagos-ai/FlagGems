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

"""In-place foreach clamp_max (scalar bound) using Triton kernels.

For every floating tensor T in the list every element is replaced by
min(element, scalar), written back into T's own storage (arbitrary strides).
The call returns None (the authoritative oracle for this op returns no value).

Fast path: every tensor shares one non-fp64 floating dtype, is contiguous,
16-byte aligned and small enough for int32 offsets, and the list holds at most
_K tensors -> a single fused launch.  Each program belongs to exactly one tensor
of the group, located from in-kernel prefix block counts, so per-tensor launch
overhead is paid once for the whole list.  A pre-bound compiled-kernel handle is
reused per specialization to remove the per-call JIT binding work.  Everything
else falls back to one launch per tensor with a flat or collapsed-dimension
strided kernel.
"""

import struct

import torch
import triton
import triton.language as tl

try:
    # The flagtree fork prepends two distributed-runtime pointers to the launch
    # argument list in "lite" mode; the pre-bound runner below cannot add them,
    # so only use it on the ordinary path.
    from triton.runtime.jit import DistributedRtContext as _DistCtx

    _DIRECT_OK = not bool(_DistCtx().is_lite_mode)
except Exception:  # noqa: BLE001
    _DIRECT_OK = False


_BLOCK = 1024
_STREAM_THREADS = 256
_STREAM_WARPS = 8
_STREAM_MIN = 1 << 16
_MAXDIM = 4
_K = 4
_I32_MAX = 0x7FFFFFFF


@triton.jit
def _cm_body(x_ptr, b, blk, n, ar, BLOCK: tl.constexpr):
    offs = blk * BLOCK + ar
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    bb = b.to(x.dtype)
    tl.store(x_ptr + offs, tl.where(x > bb, bb, x), mask=mask)


@triton.jit(do_not_specialize=["b"])
def _cm_fused_kernel(x0, x1, x2, x3, b, n0, n1, n2, n3, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nb0 = (n0 + BLOCK - 1) // BLOCK
    nb1 = nb0 + (n1 + BLOCK - 1) // BLOCK
    nb2 = nb1 + (n2 + BLOCK - 1) // BLOCK
    nb3 = nb2 + (n3 + BLOCK - 1) // BLOCK
    ar = tl.arange(0, BLOCK)
    if pid < nb0:
        _cm_body(x0, b, pid, n0, ar, BLOCK)
    elif pid < nb1:
        _cm_body(x1, b, pid - nb0, n1, ar, BLOCK)
    elif pid < nb2:
        _cm_body(x2, b, pid - nb1, n2, ar, BLOCK)
    elif pid < nb3:
        _cm_body(x3, b, pid - nb2, n3, ar, BLOCK)


@triton.jit(do_not_specialize=["b"])
def _cm_flat_kernel(
    x_ptr, b, n, BLOCK: tl.constexpr, I64: tl.constexpr, BF64: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if BF64:
        bb = tl.cast(b, tl.float64, bitcast=True).to(x.dtype)
    else:
        bb = b.to(x.dtype)
    tl.store(x_ptr + offs, tl.where(x > bb, bb, x), mask=mask)


@triton.jit(do_not_specialize=["b"])
def _cm_strided_kernel(
    x_ptr,
    b,
    n,
    d0,
    d1,
    d2,
    d3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
    BF64: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM >= 1:
        c = idx % d0
        idx = idx // d0
        off += c * s0
    if NDIM >= 2:
        c = idx % d1
        idx = idx // d1
        off += c * s1
    if NDIM >= 3:
        c = idx % d2
        idx = idx // d2
        off += c * s2
    if NDIM >= 4:
        c = idx % d3
        idx = idx // d3
        off += c * s3
    p = x_ptr + off
    x = tl.load(p, mask=mask)
    if BF64:
        bb = tl.cast(b, tl.float64, bitcast=True).to(x.dtype)
    else:
        bb = b.to(x.dtype)
    tl.store(p, tl.where(x > bb, bb, x), mask=mask)


def _collapse(shape, stride):
    """Collapse a (shape, stride) view into ordered dims, innermost-first.

    Size-1 dims are dropped and adjacent dims whose strides are compatible are
    merged.  The result is a bijection over the same address set, so the dims
    may be reordered by stride to make the innermost dim unit-stride.
    """
    dims = []
    for i in range(len(shape) - 1, -1, -1):
        sz = shape[i]
        if sz == 1:
            continue
        st = stride[i]
        if dims:
            psz, pst = dims[-1]
            if st == pst * psz:
                dims[-1] = (psz * sz, pst)
                continue
        dims.append((sz, st))
    if len(dims) > 1:
        dims = sorted(dims, key=lambda d: (d[1] == 0, d[1]))
    return dims


def _warps(blk):
    w = blk // 256
    if w < 1:
        w = 1
    if w > 8:
        w = 8
    return w


def _fused_tile(mx, isz):
    """(BLOCK, num_warps) for one fused specialization.

    For the streaming size classes the tile is chosen so that every thread moves
    16 bytes, i.e. exactly one 128-bit access per thread in the tensor's own
    element size (8 elements for fp16/bf16, 4 for fp32); smaller tiles collapse
    the vector width and cost bandwidth.
    """
    if mx >= _STREAM_MIN:
        ept = 16 // isz
        if ept < 1:
            ept = 1
        return _STREAM_THREADS * ept, _STREAM_WARPS
    if mx >= _BLOCK:
        return _BLOCK, _warps(_BLOCK)
    b = 32 if mx < 32 else triton.next_power_of_2(mx)
    return b, _warps(b)


# Pre-bound launch plans for the fused fast path.  The key covers every input
# that affects the compiled specialization (element dtype and each runtime numel,
# because Triton specializes ints equal to 1 or divisible by 16); BLOCK and
# num_warps are pure functions of those numels plus the dtype, and pointer
# alignment is a hard gate on this path.  The value is (runner-or-False,
# total_blocks, BLOCK, num_warps).  Never tensor identity or tensor data.
_fused_cache = {}
_FUSED_CACHE_MAX = 64


def _bind_fused(t0, t1, t2, t3, b, n0, n1, n2, n3, blk, nw, total):
    """Bind a direct launch handle for one exact specialization.

    On this stack CompiledKernel.run takes the whole kernel parameter list, i.e.
    the nine runtime arguments plus the constexpr BLOCK value, so the bound
    runner is called positionally with the same ten values.  One validation
    launch is issued here; clamp_max is idempotent, so re-running it on the same
    tensors is numerically a no-op.
    """
    if not _DIRECT_OK:
        return False
    try:
        ck = _cm_fused_kernel.warmup(
            t0,
            t1,
            t2,
            t3,
            b,
            n0,
            n1,
            n2,
            n3,
            BLOCK=blk,
            num_warps=nw,
            grid=(total, 1, 1),
        )
        runner = ck[(total, 1, 1)]
        runner(t0, t1, t2, t3, b, n0, n1, n2, n3, blk)
        return runner
    except Exception:  # noqa: BLE001
        return False


def _fused_plan(t0, t1, t2, t3, b, n0, n1, n2, n3, numels, isz):
    """First call for a specialization: launch through the ordinary JIT path
    and bind the direct handle used by later calls."""
    blk, nw = _fused_tile(max(numels), isz)
    total = 0
    for n in numels:
        total += (n + blk - 1) // blk
    _cm_fused_kernel[(total,)](
        t0, t1, t2, t3, b, n0, n1, n2, n3, BLOCK=blk, num_warps=nw
    )
    return (
        _bind_fused(t0, t1, t2, t3, b, n0, n1, n2, n3, blk, nw, total),
        total,
        blk,
        nw,
    )


def _run_one(t, scalar):
    dtype = t.dtype
    numel = t.numel()
    if numel == 0:
        return
    if dtype == torch.float64:
        b = struct.unpack("<q", struct.pack("<d", float(scalar)))[0]
        bf64 = True
    else:
        b = float(scalar)
        bf64 = False
    if t.is_contiguous():
        i64 = numel > _I32_MAX
        blk = _BLOCK if numel >= _BLOCK else max(32, triton.next_power_of_2(numel))
        grid = ((numel + blk - 1) // blk,)
        _cm_flat_kernel[grid](
            t, b, numel, BLOCK=blk, I64=i64, BF64=bf64, num_warps=_warps(blk)
        )
        return
    dims = _collapse(t.shape, t.stride())
    if not dims:
        return
    if len(dims) > _MAXDIM:
        for i in range(t.shape[0]):
            _run_one(t.narrow(0, i, 1).squeeze(0), scalar)
        return
    ds = [d[0] for d in dims]
    ss = [d[1] for d in dims]
    while len(ds) < _MAXDIM:
        ds.append(0)
        ss.append(0)
    grid = ((numel + _BLOCK - 1) // _BLOCK,)
    _cm_strided_kernel[grid](
        t,
        b,
        numel,
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        BLOCK=_BLOCK,
        NDIM=len(dims),
        BF64=bf64,
        num_warps=_warps(_BLOCK),
    )


def foreach_clamp_max_scalar_(self, scalar):
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_clamp_max_() received an invalid combination of "
            "arguments - got (Tensor, float), but expected "
            "(tuple of Tensors self, Number scalar)"
        )
    cnt = len(self)
    if cnt == 0:
        return None

    # Validate every element first (mirrors the reference's dtype rejection)
    # and collect what the fused fast path needs.
    dt = None
    numels = []
    fast = cnt <= _K
    for i in range(cnt):
        t = self[i]
        if not isinstance(t, torch.Tensor):
            raise TypeError("_foreach_clamp_max_() expected a sequence of Tensors")
        d = t.dtype
        if not d.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired "
                "output type " + str(d).replace("torch.", "")
            )
        if i == 0:
            dt = d
            if d is torch.float64:
                fast = False
        elif d is not dt:
            fast = False
        if fast:
            n = t.numel()
            if n == 0 or n > _I32_MAX or not t.is_contiguous():
                fast = False
            else:
                numels.append(n)

    if fast:
        t0 = self[0]
        last = self[cnt - 1]
        t1 = self[1] if cnt > 1 else last
        t2 = self[2] if cnt > 2 else last
        # 16-byte aligned pointers keep one compiled specialization valid for the
        # whole class; anything else takes the per-tensor JIT path below.
        if not ((t0.data_ptr() | t1.data_ptr() | t2.data_ptr() | last.data_ptr()) & 15):
            n0 = numels[0]
            n1 = numels[1] if cnt > 1 else 0
            n2 = numels[2] if cnt > 2 else 0
            n3 = numels[3] if cnt > 3 else 0
            key = (dt, n0, n1, n2, n3)
            plan = _fused_cache.get(key)
            if plan is None:
                plan = _fused_plan(
                    t0, t1, t2, last, float(scalar), n0, n1, n2, n3, numels, dt.itemsize
                )
                if len(_fused_cache) >= _FUSED_CACHE_MAX:
                    _fused_cache.clear()
                _fused_cache[key] = plan
                return None
            blk = plan[2]
            if plan[0] is False:
                _cm_fused_kernel[(plan[1],)](
                    t0,
                    t1,
                    t2,
                    last,
                    float(scalar),
                    n0,
                    n1,
                    n2,
                    n3,
                    BLOCK=blk,
                    num_warps=plan[3],
                )
            else:
                plan[0](t0, t1, t2, last, float(scalar), n0, n1, n2, n3, blk)
            return None

    for i in range(cnt):
        _run_one(self[i], scalar)
    return None
