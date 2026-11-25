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

"""In-place foreach add (list overload) implemented with Triton kernels.

Semantics: for every pair (self[i], other[i]) -> self[i] += alpha * other[i]
(written into self[i]'s existing storage, call returns None).

Layout: homogeneous runs of identical, contiguous pairs are fused into a single
launch (grid = (tiles, n_pairs)); anything else is dispatched pair-by-pair
through a flat or a strided kernel.  Repeat launches of an already-compiled
kernel reuse the compiled binary directly, because the Python-side JIT
bookkeeping costs far more than these elementwise kernels take to run.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 2048
_NUM_WARPS = 4
_MAX_DIMS = 4
_PACK = 4
_I64_LIMIT = 2**31 - 1
_MISMATCH_MSG = "Tensor list must have same number of elements as other tensor list."
_BCAST_MSG = "output with shape [{}] doesn't match the broadcast shape [{}]"


@triton.jit
def _flat_kernel(
    a_ptr,
    b_ptr,
    b_val,
    alpha,
    n,
    BLOCK: tl.constexpr,
    IS_16BIT: tl.constexpr,
    SCALAR_B: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    if SCALAR_B:
        b = b_val
    else:
        b = tl.load(b_ptr + offs, mask=mask)
    if IS_16BIT:
        r = a.to(tl.float32) + alpha * b.to(tl.float32)
    else:
        r = a + alpha * b
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _packed_flat_kernel(
    a0,
    b0,
    a1,
    b1,
    a2,
    b2,
    a3,
    b3,
    alpha,
    n,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_16BIT: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a_ptr = a0
    b_ptr = b0
    if NT > 1:
        if tid == 1:
            a_ptr = a1
            b_ptr = b1
        elif NT > 2:
            if tid == 2:
                a_ptr = a2
                b_ptr = b2
            elif NT > 3:
                if tid == 3:
                    a_ptr = a3
                    b_ptr = b3
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    if IS_16BIT:
        r = a.to(tl.float32) + alpha * b.to(tl.float32)
    else:
        r = a + alpha * b
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _strided_kernel(
    a_ptr,
    b_ptr,
    b_val,
    alpha,
    n,
    n0,
    n1,
    n2,
    n3,
    as0,
    as1,
    as2,
    as3,
    bs0,
    bs1,
    bs2,
    bs3,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_16BIT: tl.constexpr,
    SCALAR_B: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    if NDIM == 1:
        a_off = off * as0
        b_off = off * bs0
    elif NDIM == 2:
        i1 = off % n1
        i0 = off // n1
        a_off = i0 * as0 + i1 * as1
        b_off = i0 * bs0 + i1 * bs1
    elif NDIM == 3:
        i2 = off % n2
        t = off // n2
        i1 = t % n1
        i0 = t // n1
        a_off = i0 * as0 + i1 * as1 + i2 * as2
        b_off = i0 * bs0 + i1 * bs1 + i2 * bs2
    else:
        i3 = off % n3
        t = off // n3
        i2 = t % n2
        t = t // n2
        i1 = t % n1
        i0 = t // n1
        a_off = i0 * as0 + i1 * as1 + i2 * as2 + i3 * as3
        b_off = i0 * bs0 + i1 * bs1 + i2 * bs2 + i3 * bs3
    a = tl.load(a_ptr + a_off, mask=mask)
    if SCALAR_B:
        b = b_val
    else:
        b = tl.load(b_ptr + b_off, mask=mask)
    if IS_16BIT:
        r = a.to(tl.float32) + alpha * b.to(tl.float32)
    else:
        r = a + alpha * b
    tl.store(a_ptr + a_off, r, mask=mask)


def _collapse(a, b):
    """Jointly collapse dims of a (in-place target) and b (broadcast source).

    Returns groups (size, a_stride, b_stride), left to right, such that a linear
    index maps to a_offset = sum(idx_k * a_stride_k).
    """
    nd = a.dim() if a.dim() > b.dim() else b.dim()
    ash, ast = a.shape, a.stride()
    bsh, bst = b.shape, b.stride()
    pa = nd - a.dim()
    pb = nd - b.dim()
    groups = []
    for d in range(nd - 1, -1, -1):
        ai = d - pa
        bi = d - pb
        if ai >= 0:
            sz = ash[ai]
            sa = ast[ai]
        else:
            sz = 1
            sa = 0
        if bi >= 0:
            bsz = bsh[bi]
            sb = bst[bi]
        else:
            bsz = 1
            sb = 0
        if sz == 1 and bsz == 1:
            continue
        if sz == 1 or (bsz != 1 and bsz != sz):
            raise RuntimeError(_BCAST_MSG.format(sz, bsz))
        if bsz == 1:
            sb = 0
        if groups:
            total, la, lb, lsize = groups[-1]
            if sa == la * lsize and sb == lb * lsize:
                groups[-1] = (total * sz, sa, sb, sz)
                continue
        groups.append((sz, sa, sb, sz))
    groups.reverse()
    return groups


def _is_16bit(a, b):
    return (
        a.dtype == torch.float16
        or a.dtype == torch.bfloat16
        or b.dtype == torch.float16
        or b.dtype == torch.bfloat16
    )


# ---------------------------------------------------------------------------
# Launch path
#
# JITFunction.run re-derives argument specialization, the cache key, compile
# options, global-value checks and launch-binding validation on every call; that
# Python work costs far more than these elementwise kernels take to execute.
# After the first (ordinary) launch has produced the exact CompiledKernel for a
# given (kernel variant, dtypes, constexpr flags, numel), the same binary is
# relaunched through its own low-level runner.  The reuse is only taken for
# 16-byte aligned data pointers (the alignment class Triton compiled for), the
# cache key carries every property Triton specializes on, and a one-off probe on
# private scratch data must first show that the runner executes exactly once with
# the expected result; otherwise everything falls back to the ordinary path.
# The generated launcher parses one argument slot per declared parameter
# (constexpr slots included, as ignored objects), so positional arguments are
# always passed as runtime args followed by the constexpr values.
# ---------------------------------------------------------------------------

_KCACHE = {}
_FAST = [None]


def _probe_fast_launch():
    try:
        a = torch.zeros(4096, device="cuda", dtype=torch.float32)
        b = torch.ones(4096, device="cuda", dtype=torch.float32)
        pos = (a, b, 0.0, 2.0, 4096, _BLOCK, False, False, False)
        ck = _flat_kernel[(2, 1, 1)](
            a,
            b,
            0.0,
            2.0,
            4096,
            BLOCK=_BLOCK,
            IS_16BIT=False,
            SCALAR_B=False,
            USE_I64=False,
            num_warps=_NUM_WARPS,
        )
        ck[(2, 1, 1)](*pos)
        # The ordinary launch above already applied alpha*other once; a working
        # direct relaunch applies it exactly once more, so zeros + 1.0*2.0 twice
        # must read back as exactly 4.0.
        return a[0].item() == 4.0 and a[4095].item() == 4.0
    except Exception:
        return False


def _fast_ok():
    ok = _FAST[0]
    if ok is None:
        ok = _probe_fast_launch()
        _FAST[0] = ok
    return ok


def _submit(kernel, grid, pos_args, n_rt, const_kwargs, key):
    """Launch `kernel`, reusing the compiled binary when that is safe."""
    if key is not None:
        runner = _KCACHE.get(key)
        if runner is not None:
            try:
                runner(*pos_args)
                return
            except Exception:
                _KCACHE.pop(key, None)
    ck = kernel[grid](*pos_args[:n_rt], **const_kwargs)
    if key is not None and ck is not None:
        _KCACHE[key] = ck[(grid[0], grid[1] if len(grid) > 1 else 1, 1)]


def _launch_flat(a, b, alpha, n, scalar_b, b_val):
    is16 = _is_16bit(a, b)
    use_i64 = n > _I64_LIMIT
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    pos = (a, b, b_val, alpha, n, _BLOCK, is16, scalar_b, use_i64)
    kw = dict(
        BLOCK=_BLOCK,
        IS_16BIT=is16,
        SCALAR_B=scalar_b,
        USE_I64=use_i64,
        num_warps=_NUM_WARPS,
    )
    key = None
    if _fast_ok() and not ((a.data_ptr() | b.data_ptr()) & 15):
        key = ("f", a.dtype, b.dtype, n, is16, scalar_b, use_i64)
    _submit(_flat_kernel, grid, pos, 5, kw, key)


def _launch_strided(a, b, alpha, n, scalar_b, b_val, groups):
    if len(groups) <= 1 and (
        len(groups) == 0 or (groups[0][1] == 1 and groups[0][2] == 1)
    ):
        _launch_flat(a, b, alpha, n, scalar_b, b_val)
        return
    if len(groups) > _MAX_DIMS:
        # Rare high-rank non-contiguous fallback: peel the first non-singleton
        # dim of `a` with zero-copy narrow/squeeze and recurse.
        d = None
        for k in range(a.dim()):
            if a.shape[k] > 1:
                d = k
                break
        if d is None:
            return
        use_b = b.dim() == a.dim() and b.shape[d] == a.shape[d]
        for i in range(a.shape[d]):
            ai = a.narrow(d, i, 1).squeeze(d)
            bi = b.narrow(d, i, 1).squeeze(d) if use_b else b
            _launch_pair(ai, bi, alpha)
        return
    use_i64 = n > _I64_LIMIT
    if not use_i64:
        sa_max = max(a.stride()) if a.dim() > 0 else 0
        sb_max = max(b.stride()) if b.dim() > 0 else 0
        use_i64 = max(sa_max, sb_max) * n > _I64_LIMIT
    pad = _MAX_DIMS - len(groups)
    gs = groups + [(1, 0, 0, 1)] * pad
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _strided_kernel[grid](
        a,
        b,
        b_val,
        alpha,
        n,
        gs[0][0],
        gs[1][0],
        gs[2][0],
        gs[3][0],
        gs[0][1],
        gs[1][1],
        gs[2][1],
        gs[3][1],
        gs[0][2],
        gs[1][2],
        gs[2][2],
        gs[3][2],
        NDIM=len(groups),
        BLOCK=_BLOCK,
        IS_16BIT=_is_16bit(a, b),
        SCALAR_B=scalar_b,
        USE_I64=use_i64,
        num_warps=_NUM_WARPS,
    )


def _launch_pair(a, b, alpha, scalar_b=False):
    n = a.numel()
    if n == 0:
        return
    if not scalar_b:
        if a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            _launch_flat(a, b, alpha, n, False, 0.0)
            return
        _launch_strided(a, b, alpha, n, False, 0.0, _collapse(a, b))
        return
    # scalar overload: self += alpha * scalar
    if a.is_contiguous():
        _launch_flat(a, a, 1.0, n, True, alpha)
        return
    _launch_strided(a, a, 1.0, n, True, alpha, _collapse(a, a))


def _packed_run(tensors, others, start, end, alpha):
    """Fuse pairs [start, end) of identical contiguous layout into one launch."""
    cnt = end - start
    a0 = tensors[start]
    n = a0.numel()
    tiles = (n + _BLOCK - 1) // _BLOCK
    sa = [tensors[start + k] for k in range(cnt)]
    sb = [others[start + k] for k in range(cnt)]
    while len(sa) < _PACK:
        sa.append(sa[-1])
        sb.append(sb[-1])
    is16 = _is_16bit(sa[0], sb[0])
    use_i64 = n > _I64_LIMIT
    pos = (
        sa[0],
        sb[0],
        sa[1],
        sb[1],
        sa[2],
        sb[2],
        sa[3],
        sb[3],
        alpha,
        n,
        cnt,
        _BLOCK,
        is16,
        use_i64,
    )
    kw = dict(
        NT=cnt, BLOCK=_BLOCK, IS_16BIT=is16, USE_I64=use_i64, num_warps=_NUM_WARPS
    )
    key = None
    if _fast_ok():
        mask = 0
        for k in range(cnt):
            mask |= sa[k].data_ptr() | sb[k].data_ptr()
        if not (mask & 15):
            key = ("p", cnt, sa[0].dtype, n, is16, use_i64)
    _submit(_packed_flat_kernel, (tiles, cnt), pos, 10, kw, key)


def _packed_span(tensors, others, start):
    """Return the exclusive end index of a homogeneous run starting at `start`.

    A run has at least 2 pairs, at most _PACK, all with identical shape, dtype
    and full contiguity (so a single flat index space addresses every pair).
    """
    limit = len(tensors)
    a0 = tensors[start]
    if a0.numel() == 0 or not a0.is_contiguous():
        return None
    shp = a0.shape
    dt = a0.dtype
    b0 = others[start]
    if not b0.is_contiguous() or b0.shape != shp or b0.dtype != dt:
        return None
    end = start + 1
    while end < limit and end - start < _PACK:
        a = tensors[end]
        if a.shape != shp or a.dtype != dt or not a.is_contiguous():
            break
        b = others[end]
        if b.shape != shp or b.dtype != dt or not b.is_contiguous():
            break
        end += 1
    return end


def _foreach_add__List(self, other, alpha=1):
    logger.debug("GEMS_HYGON _FOREACH_ADD__LIST")
    if isinstance(other, (list, tuple)):
        if len(self) != len(other):
            raise RuntimeError(_MISMATCH_MSG)
        if isinstance(alpha, (list, tuple)):
            for i in range(len(self)):
                _launch_pair(self[i], other[i], float(alpha[i]))
            return None
        al = float(alpha)
        i = 0
        total = len(self)
        while i < total:
            end = _packed_span(self, other, i)
            if end is not None and end - i >= 2:
                _packed_run(self, other, i, end, al)
                i = end
            else:
                a = self[i]
                if a.numel() != 0:
                    _launch_pair(a, other[i], al)
                i += 1
        return None
    al = float(alpha) * float(other)
    for i in range(len(self)):
        _launch_pair(self[i], self[i], al, scalar_b=True)
    return None
