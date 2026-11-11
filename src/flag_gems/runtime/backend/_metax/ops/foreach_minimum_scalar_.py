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

"""In-place foreach minimum with a scalar (aten::_foreach_minimum_.Scalar).

foreach_minimum_scalar_(tensors, scalar) mutates every tensor of the list in place and returns None.
Each element becomes torch.minimum(x, cast(scalar, x.dtype)) with ATen's
NaN-propagation rule, and no input is ever re-laid-out.

A list of 2..4 contiguous tensors sharing dtype and numel (the common case) takes
one launch on a 2-D grid (blocks x tensor).  After the first launch of a given
specialization the compiled launcher is cached, so later calls skip Triton's
per-call JIT binding.  Other lists fall back to per-(dtype, numel) bucketing, and
noncontiguous tensors are written through their collapsed strides.
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Element body
# ---------------------------------------------------------------------------
@triton.jit
def _apply(x_ptr, base, n, s, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offs = base * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        sx = s.to(x.dtype)
        r = tl.where(x != x, x, tl.where(x < sx, x, sx))
        tl.store(x_ptr + offs, r)
    else:
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        sx = s.to(x.dtype)
        r = tl.where(x != x, x, tl.where(x < sx, x, sx))
        tl.store(x_ptr + offs, r, mask=m)


# ---------------------------------------------------------------------------
# Single contiguous tensor
# ---------------------------------------------------------------------------
@triton.jit
def _min_flat(x_ptr, s, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    _apply(x_ptr, tl.program_id(0), n, s, BLOCK, EVEN)


# ---------------------------------------------------------------------------
# 2..4 contiguous tensors sharing dtype and numel: grid = (blocks, K)
# ---------------------------------------------------------------------------
@triton.jit
def _min_uniform(
    x0,
    x1,
    x2,
    x3,
    s,
    n,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
    EVEN: tl.constexpr,
):
    base = tl.program_id(0)
    k = tl.program_id(1)
    if k == 0:
        _apply(x0, base, n, s, BLOCK, EVEN)
    else:
        if k == 1:
            _apply(x1, base, n, s, BLOCK, EVEN)
        else:
            if K > 2:
                if k == 2:
                    _apply(x2, base, n, s, BLOCK, EVEN)
                else:
                    if K > 3:
                        _apply(x3, base, n, s, BLOCK, EVEN)


# ---------------------------------------------------------------------------
# Strided kernel: up to 8 collapsed dimensions, innermost first.
# ---------------------------------------------------------------------------
@triton.jit
def _min_strided(
    x_ptr,
    s,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    idx = offs
    off = tl.zeros([BLOCK], tl.int64)
    c = idx % d0
    idx = idx // d0
    off += c * t0
    c = idx % d1
    idx = idx // d1
    off += c * t1
    c = idx % d2
    idx = idx // d2
    off += c * t2
    c = idx % d3
    idx = idx // d3
    off += c * t3
    c = idx % d4
    idx = idx // d4
    off += c * t4
    c = idx % d5
    idx = idx // d5
    off += c * t5
    c = idx % d6
    idx = idx // d6
    off += c * t6
    off += idx * t7
    x = tl.load(x_ptr + off, mask=m)
    sx = s.to(x.dtype)
    r = tl.where(x != x, x, tl.where(x < sx, x, sx))
    tl.store(x_ptr + off, r, mask=m)


_BLOCK = 4096
_WARPS = 8
_SMALL_BLOCK = 1024
_SMALL_WARPS = 4
_SMALL_LIMIT = 65536
_MAX_DIMS = 8
_MAX_FUSE = 4

# (dtype, numel, block, k, even, warps, device) -> prebuilt CompiledKernel runner
_LAUNCHERS = {}


def _collapse(shape, strides):
    """Merge neighbouring dims that are contiguous in memory (outer first)."""
    dims = [(int(sz), int(st)) for sz, st in zip(shape, strides) if int(sz) != 1]
    if not dims:
        return []
    out = [dims[0]]
    for sz, st in dims[1:]:
        psz, pst = out[-1]
        if pst == sz * st:
            out[-1] = (psz * sz, pst)
        else:
            out.append((sz, st))
    return out


def _launch_strided(t, scalar):
    n = t.numel()
    dims = _collapse(t.shape, t.stride())
    if not dims:
        return
    if len(dims) > _MAX_DIMS:
        # Extremely rare: walk the leading dimension on the host.
        for i in range(dims[0][0]):
            _launch_strided(t.select(0, i), scalar)
        return
    # innermost dimension first, unused slots -> size 1, stride 0
    inner_first = dims[::-1]
    ds = [d for d, _ in inner_first] + [1] * (_MAX_DIMS - len(inner_first))
    ts = [s for _, s in inner_first] + [0] * (_MAX_DIMS - len(inner_first))
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _min_strided[grid](
        t,
        scalar,
        n,
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        ts[0],
        ts[1],
        ts[2],
        ts[3],
        ts[4],
        ts[5],
        ts[6],
        ts[7],
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def foreach_minimum_scalar_(tensors, scalar=None, **kwargs):
    """aten::_foreach_minimum_.Scalar — in place, returns None."""
    if tensors is None:
        tensors = kwargs.get("self")
    if scalar is None and "scalar" in kwargs:
        scalar = kwargs["scalar"]

    cnt = len(tensors)
    # Fast path: 2..4 contiguous tensors with identical dtype and numel.
    if 1 < cnt <= _MAX_FUSE:
        t0 = tensors[0]
        n = t0.numel()
        if n:
            dt = t0.dtype
            same = t0.is_contiguous()
            if same:
                for t in tensors:
                    if t.numel() != n or t.dtype is not dt or not t.is_contiguous():
                        same = False
                        break
            if same:
                if n < _SMALL_LIMIT:
                    blk, warps = _SMALL_BLOCK, _SMALL_WARPS
                else:
                    blk, warps = _BLOCK, _WARPS
                even = n % blk == 0
                nblk = (n + blk - 1) // blk
                a1 = tensors[1]
                a2 = tensors[2] if cnt > 2 else t0
                a3 = tensors[3] if cnt > 3 else t0
                key = None
                if type(scalar) is float and not (
                    (t0.data_ptr() | a1.data_ptr() | a2.data_ptr() | a3.data_ptr()) & 15
                ):
                    key = (dt, n, blk, cnt, even, warps, t0.get_device())
                    runner = _LAUNCHERS.get(key)
                    if runner is not None:
                        try:
                            runner(t0, a1, a2, a3, scalar, n, blk, cnt, even)
                            return None
                        except Exception:
                            _LAUNCHERS.pop(key, None)
                ker = _min_uniform[(nblk, cnt)](
                    t0,
                    a1,
                    a2,
                    a3,
                    scalar,
                    n,
                    blk,
                    cnt,
                    even,
                    num_warps=warps,
                )
                if key is not None and ker is not None:
                    try:
                        _LAUNCHERS[key] = ker[(nblk, cnt, 1)]
                    except Exception:
                        pass
                return None

    groups = {}
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            key = (t.dtype, n)
            g = groups.get(key)
            if g is None:
                groups[key] = [t]
            else:
                g.append(t)
        else:
            _launch_strided(t, scalar)
    for key, lst in groups.items():
        n = key[1]
        if n < _SMALL_LIMIT:
            blk, warps = _SMALL_BLOCK, _SMALL_WARPS
        else:
            blk, warps = _BLOCK, _WARPS
        nblk = (n + blk - 1) // blk
        even = n % blk == 0
        if len(lst) == 1:
            _min_flat[(nblk,)](lst[0], scalar, n, blk, even, num_warps=warps)
            continue
        for i in range(0, len(lst), _MAX_FUSE):
            chunk = lst[i : i + _MAX_FUSE]
            k = len(chunk)
            fill = chunk[0]
            a1 = chunk[1] if k > 1 else fill
            a2 = chunk[2] if k > 2 else fill
            a3 = chunk[3] if k > 3 else fill
            _min_uniform[(nblk, k)](
                chunk[0],
                a1,
                a2,
                a3,
                scalar,
                n,
                blk,
                k,
                even,
                num_warps=warps,
            )
    return None
