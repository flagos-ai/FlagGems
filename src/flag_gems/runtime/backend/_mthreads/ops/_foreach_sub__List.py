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
"""In-place foreach sub (``aten::_foreach_sub_``) implemented with Triton kernels.

Observable contract: every tensor in ``self`` is mutated in place
(``self[i] -= alpha * other[i]``) and the function returns ``None``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_sub__List as default__foreach_sub__List,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_HALF_DTYPES = (torch.float16, torch.bfloat16)
_MAX_RANK = 8
_SLOTS = 4


@triton.jit
def _sub_body(
    a_ptr,
    b_ptr,
    n,
    pid,
    alpha,
    IS_HALF: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = tl.load(b_ptr + offs, mask=mask, other=0.0)
    if IS_HALF:
        acc = x.to(tl.float32) - alpha * y.to(tl.float32)
    else:
        acc = x - alpha * y
    tl.store(a_ptr + offs, acc, mask=mask)


@triton.jit
def _sub_contig_kernel(
    a_ptr,
    b_ptr,
    n,
    alpha,
    IS_HALF: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _sub_body(a_ptr, b_ptr, n, tl.program_id(0), alpha, IS_HALF, I64, BLOCK)


@triton.jit
def _sub_fused4_kernel(
    a0,
    b0,
    a1,
    b1,
    a2,
    b2,
    a3,
    b3,
    n0,
    n1,
    n2,
    n3,
    k1,
    k2,
    k3,
    alpha,
    IS_HALF: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < k1:
        _sub_body(a0, b0, n0, pid, alpha, IS_HALF, I64, BLOCK)
    elif pid < k2:
        _sub_body(a1, b1, n1, pid - k1, alpha, IS_HALF, I64, BLOCK)
    elif pid < k3:
        _sub_body(a2, b2, n2, pid - k2, alpha, IS_HALF, I64, BLOCK)
    else:
        _sub_body(a3, b3, n3, pid - k3, alpha, IS_HALF, I64, BLOCK)


@triton.jit
def _sub_strided_kernel(
    a_ptr,
    b_ptr,
    n,
    alpha,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    sa0,
    sa1,
    sa2,
    sa3,
    sa4,
    sa5,
    sa6,
    sa7,
    sb0,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    sb6,
    sb7,
    IS_HALF: tl.constexpr,
    SCALAR: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    sizes = (d0, d1, d2, d3, d4, d5, d6, d7)
    divs = (v0, v1, v2, v3, v4, v5, v6, v7)
    sas = (sa0, sa1, sa2, sa3, sa4, sa5, sa6, sa7)
    sbs = (sb0, sb1, sb2, sb3, sb4, sb5, sb6, sb7)
    oa = tl.zeros([BLOCK], dtype=tl.int64)
    for i in tl.static_range(RANK):
        oa += ((offs // divs[i]) % sizes[i]) * sas[i]
    x = tl.load(a_ptr + oa, mask=mask, other=0.0)
    if SCALAR:
        if IS_HALF:
            acc = x.to(tl.float32) - alpha
        else:
            acc = x - alpha
    else:
        ob = tl.zeros([BLOCK], dtype=tl.int64)
        for i in tl.static_range(RANK):
            ob += ((offs // divs[i]) % sizes[i]) * sbs[i]
        y = tl.load(b_ptr + ob, mask=mask, other=0.0)
        if IS_HALF:
            acc = x.to(tl.float32) - alpha * y.to(tl.float32)
        else:
            acc = x - alpha * y
    tl.store(a_ptr + oa, acc, mask=mask)


@triton.jit
def _sub_scalar_contig_kernel(
    a_ptr, n, beta, IS_HALF: tl.constexpr, I64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if IS_HALF:
        acc = x.to(tl.float32) - beta
    else:
        acc = x - beta
    tl.store(a_ptr + offs, acc, mask=mask)


def _contig_launch_params(n):
    if n <= 1 << 16:
        return 1024, 4
    if n <= 1 << 23:
        return 2048, 4
    return 4096, 8


_RUNNERS = {}


def _spec_bits(values):
    # Triton specialises each integer argument on both "== 1" and "% 16 == 0",
    # so the replayed variant must be keyed on both, not just divisibility.
    return tuple((v == 1, v % 16 == 0) for v in values)


def _runner_for(jit_fn, key, args, constexpr, grid, warps):
    """Compiled-kernel launcher for one specialization, built on first use.

    ``warmup`` compiles without launching, so building the cache entry has no
    side effect on the operands.  The current stream is part of the cache key so
    a replay issued from a different stream still lands on the right one.
    """
    stream = torch.musa.current_stream().musa_stream
    skey = (key, stream)
    runner = _RUNNERS.get(skey)
    if runner is not None:
        return runner
    runner = False
    try:
        compiled = jit_fn.warmup(*args, grid=grid, num_warps=warps, **constexpr)
        if compiled is None:
            raise RuntimeError("warmup returned no kernel")
        g0, g1, g2 = (tuple(grid) + (1, 1))[:3]
        run = compiled.run
        fn = compiled.function
        pm = compiled.packed_metadata
        is_half = constexpr["IS_HALF"]
        i64 = constexpr["I64"]
        block = constexpr["BLOCK"]

        def launch(*a):
            run(g0, g1, g2, stream, fn, pm, None, None, None, *a, is_half, i64, block)

        runner = launch
    except Exception:
        runner = False
    _RUNNERS[skey] = runner
    return runner


def _launch_contig(a, b, alpha, n):
    block, warps = _contig_launch_params(n)
    grid = (triton.cdiv(n, block),)
    args = (a, b, n, alpha)
    constexpr = {
        "IS_HALF": a.dtype in _HALF_DTYPES,
        "I64": n >= 1 << 31,
        "BLOCK": block,
    }
    if (
        n != 1
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
        and a.data_ptr() != b.data_ptr()
    ):
        key = ("c", a.dtype, block, warps, n % 16 == 0)
        runner = _runner_for(_sub_contig_kernel, key, args, constexpr, grid, warps)
        if runner is not False:
            try:
                runner(*args)
                return
            except Exception:
                _RUNNERS[key] = False
    _sub_contig_kernel[grid](*args, num_warps=warps, **constexpr)


def _launch_group(group, alpha):
    """One launch for up to ``_SLOTS`` contiguous, same-dtype entries."""
    m = len(group)
    if m == 1:
        a, b, n = group[0]
        _launch_contig(a, b, alpha, n)
        return
    nmax = 0
    for _, _, n in group:
        if n > nmax:
            nmax = n
    block, warps = _contig_launch_params(nmax)
    a0, b0, n0 = group[0]
    a1, b1, n1 = group[1] if m > 1 else group[0]
    a2, b2, n2 = group[2] if m > 2 else group[0]
    a3, b3, n3 = group[3] if m > 3 else group[0]
    if m < 3:
        n2 = 0
    if m < 4:
        n3 = 0
    k1 = -(-n0 // block)
    k2 = k1 + (-(-n1 // block) if n1 else 0)
    k3 = k2 + (-(-n2 // block) if n2 else 0)
    total = k3 + (-(-n3 // block) if n3 else 0)
    args = (a0, b0, a1, b1, a2, b2, a3, b3, n0, n1, n2, n3, k1, k2, k3, alpha)
    grid = (total,)
    constexpr = {
        "IS_HALF": a0.dtype in _HALF_DTYPES,
        "I64": nmax >= 1 << 31,
        "BLOCK": block,
    }
    ints = (n0, n1, n2, n3, k1, k2, k3)
    used = ((a0, b0), (a1, b1), (a2, b2), (a3, b3))[:m]
    ptrs = [p for pair in used for p in (pair[0].data_ptr(), pair[1].data_ptr())]
    # Replay only the alias-free, 16-byte-aligned variant: the compiled kernel is
    # cached without the per-call pointer-alias metadata Triton's own launcher
    # adds, so a session where the operands overlap must take the ordinary path.
    if not any(p & 15 for p in ptrs) and len(set(ptrs)) == len(ptrs):
        key = ("f4", a0.dtype, block, warps, m, _spec_bits(ints))
        runner = _runner_for(_sub_fused4_kernel, key, args, constexpr, grid, warps)
        if runner is not False:
            try:
                runner(*args)
                return
            except Exception:
                _RUNNERS[key] = False
    _sub_fused4_kernel[grid](*args, num_warps=warps, **constexpr)


def _broadcast_dims(a, b):
    """Collapsed (size, self_stride, other_stride) plan, outer -> inner."""
    ashape = a.shape
    bshape = b.shape
    ra, rb = len(ashape), len(bshape)
    r = ra if ra > rb else rb
    sizes_a = (1,) * (r - ra) + tuple(ashape)
    sizes_b = (1,) * (r - rb) + tuple(bshape)
    strides_a = (0,) * (r - ra) + tuple(a.stride())
    strides_b = (0,) * (r - rb) + tuple(b.stride())

    dims = []
    for i in range(r):
        sa = sizes_a[i]
        sb = sizes_b[i]
        if sa == sb:
            o = sa
        elif sb == 1:
            o = sa
        elif sa == 1:
            raise RuntimeError(
                "output with shape %s doesn't match the broadcast shape %s"
                % (list(ashape), list(bshape))
            )
        else:
            raise RuntimeError(
                "The size of tensor a (%d) must match the size of tensor b (%d) "
                "at non-singleton dimension %d" % (sa, sb, i)
            )
        if o == 1:
            continue
        st_b = strides_b[i] if sb != 1 else 0
        dims.append((o, strides_a[i], st_b))

    if not dims:
        return [(1, 0, 0)]

    # largest self-stride outermost, unit self-stride innermost (coalesced stores)
    dims = sorted(dims, key=lambda d: -d[1])
    merged = []
    for size, sta, stb in dims:
        if merged:
            psize, psta, pstb = merged[-1]
            if psta == sta * size and pstb == stb * size:
                merged[-1] = (psize * size, sta, stb)
                continue
        merged.append((size, sta, stb))
    return merged


def _launch_ranked(a, b, alpha, dims, n, scalar=False):
    rank = len(dims)
    sizes = [1] * _MAX_RANK
    divs = [1] * _MAX_RANK
    sas = [0] * _MAX_RANK
    sbs = [0] * _MAX_RANK
    suffix = 1
    for k in range(rank - 1, -1, -1):
        size, sta, stb = dims[k]
        sizes[k] = size
        divs[k] = suffix
        sas[k] = sta
        sbs[k] = stb
        suffix *= size
    block = 1024
    grid = (triton.cdiv(n, block),)
    _sub_strided_kernel[grid](
        a,
        a if b is None else b,
        n,
        alpha,
        *sizes,
        *divs,
        *sas,
        *sbs,
        IS_HALF=a.dtype in _HALF_DTYPES,
        SCALAR=scalar,
        RANK=rank,
        BLOCK=block,
        num_warps=4,
    )


def _sub_pair(a, b, alpha):
    n = a.numel()
    if n == 0:
        return
    if a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
        _launch_contig(a, b, alpha, n)
        return
    dims = _broadcast_dims(a, b)
    if len(dims) > _MAX_RANK:
        raise RuntimeError("foreach_sub: unsupported operand rank")
    _launch_ranked(a, b, alpha, dims, n)


def _sub_pair_scalar(a, beta):
    n = a.numel()
    if n == 0:
        return
    if a.is_contiguous():
        block, warps = _contig_launch_params(n)
        grid = (triton.cdiv(n, block),)
        _sub_scalar_contig_kernel[grid](
            a,
            n,
            beta,
            a.dtype in _HALF_DTYPES,
            n >= 1 << 31,
            BLOCK=block,
            num_warps=warps,
        )
        return
    dims = _broadcast_dims(a, a)
    if len(dims) > _MAX_RANK:
        raise RuntimeError("foreach_sub: unsupported operand rank")
    _launch_ranked(a, None, beta, dims, n, scalar=True)


def _foreach_sub__List(self, other, alpha=1):
    logger.debug("GEMS_MTHREADS _FOREACH_SUB__LIST")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_sub__List(self, other, alpha)
    """In-place ``self[i] -= alpha * other[i]`` for every entry of ``self``."""
    if isinstance(other, torch.Tensor):
        if other.numel() == 1:
            beta = alpha * other.item()
            for a in self:
                _sub_pair_scalar(a, beta)
            return None
        for a in self:
            _sub_pair(a, other, alpha)
        return None

    if isinstance(other, (list, tuple)):
        if len(other) == 0 and len(self) == 0:
            raise RuntimeError("Tensor list must have at least one tensor.")
        if len(other) != len(self):
            raise RuntimeError(
                "Tensor lists must have the same number of tensors, got %d and %d"
                % (len(self), len(other))
            )
        group = []
        dtype = None
        for a, b in zip(self, other):
            if type(b) is not torch.Tensor:
                if group:
                    _launch_group(group, alpha)
                    group = []
                    dtype = None
                _sub_pair_scalar(a, alpha * b)
                continue
            n = a.numel()
            if n == 0:
                continue
            if a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
                if dtype is not None and (a.dtype is not dtype or len(group) >= _SLOTS):
                    _launch_group(group, alpha)
                    group = []
                dtype = a.dtype
                group.append((a, b, n))
            else:
                if group:
                    _launch_group(group, alpha)
                    group = []
                    dtype = None
                _sub_pair(a, b, alpha)
        if group:
            _launch_group(group, alpha)
        return None

    # scalar other: self[i] -= alpha * other
    beta = alpha * other
    for a in self:
        _sub_pair_scalar(a, beta)
    return None
