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

"""Triton implementation of ``aten::_foreach_minimum_`` (List / Scalar overloads).

Every tensor in ``self`` is updated in place with the NaN-propagating elementwise
minimum against the paired tensor (or scalar).  ``run`` returns ``None``.
"""

import torch
import triton
import triton.language as tl

_MAX_D = 8
_BLOCK = 1024
_WARPS = 4
_I32_MAX = 2147483647
_RUNNER_CACHE_MAX = 256
_FUSED_RUNNERS = {}
_BLOCK_BY_ITEMSIZE = {1: 4096, 2: 2048, 4: 1024, 8: 512}


@triton.jit
def _pair_min(a, b):
    # torch.minimum semantics: NaN always wins, otherwise the smaller value.
    return tl.where(a != a, a, tl.where(b != b, b, tl.minimum(a, b)))


@triton.jit
def _do_pair(A, B, offs, mask):
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask).to(a.dtype)
    tl.store(A + offs, _pair_min(a, b), mask=mask)


@triton.jit
def _fused4(A0, B0, A1, B1, A2, B2, A3, B3, n, BLOCK: tl.constexpr):
    """One launch for up to four independent flat pairs (row = pair index)."""
    pid = tl.program_id(0)
    which = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if which == 0:
        _do_pair(A0, B0, offs, mask)
    elif which == 1:
        _do_pair(A1, B1, offs, mask)
    elif which == 2:
        _do_pair(A2, B2, offs, mask)
    else:
        _do_pair(A3, B3, offs, mask)


@triton.jit
def _flat_kernel(A, B, n, b_step, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs * b_step, mask=mask).to(a.dtype)
    tl.store(A + offs, _pair_min(a, b), mask=mask)


@triton.jit
def _flat_kernel_i64(A, B, n, b_step, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs * b_step, mask=mask).to(a.dtype)
    tl.store(A + offs, _pair_min(a, b), mask=mask)


@triton.jit
def _flat_scalar_kernel(A, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask)
    sv = s.to(a.dtype)
    r = tl.where(a != a, a, tl.where(sv != sv, sv, tl.minimum(a, sv)))
    tl.store(A + offs, r, mask=mask)


@triton.jit
def _gen_kernel(
    A,
    B,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
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
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # dN is the N-th (fastest-varying first) logical dimension.
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    oa = tl.zeros([BLOCK], dtype=tl.int64)
    ob = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        i = rem % d0
        rem = rem // d0
        oa += i * s0
        ob += i * t0
    if RANK > 1:
        i = rem % d1
        rem = rem // d1
        oa += i * s1
        ob += i * t1
    if RANK > 2:
        i = rem % d2
        rem = rem // d2
        oa += i * s2
        ob += i * t2
    if RANK > 3:
        i = rem % d3
        rem = rem // d3
        oa += i * s3
        ob += i * t3
    if RANK > 4:
        i = rem % d4
        rem = rem // d4
        oa += i * s4
        ob += i * t4
    if RANK > 5:
        i = rem % d5
        rem = rem // d5
        oa += i * s5
        ob += i * t5
    if RANK > 6:
        i = rem % d6
        rem = rem // d6
        oa += i * s6
        ob += i * t6
    if RANK > 7:
        i = rem % d7
        oa += i * s7
        ob += i * t7
    a = tl.load(A + oa, mask=mask)
    b = tl.load(B + ob, mask=mask).to(a.dtype)
    tl.store(A + oa, _pair_min(a, b), mask=mask)


@triton.jit
def _gen_scalar_kernel(
    A,
    n,
    s,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    oa = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        i = rem % d0
        rem = rem // d0
        oa += i * s0
    if RANK > 1:
        i = rem % d1
        rem = rem // d1
        oa += i * s1
    if RANK > 2:
        i = rem % d2
        rem = rem // d2
        oa += i * s2
    if RANK > 3:
        i = rem % d3
        rem = rem // d3
        oa += i * s3
    if RANK > 4:
        i = rem % d4
        rem = rem // d4
        oa += i * s4
    if RANK > 5:
        i = rem % d5
        rem = rem // d5
        oa += i * s5
    if RANK > 6:
        i = rem % d6
        rem = rem // d6
        oa += i * s6
    if RANK > 7:
        i = rem % d7
        oa += i * s7
    a = tl.load(A + oa, mask=mask)
    sv = s.to(a.dtype)
    r = tl.where(a != a, a, tl.where(sv != sv, sv, tl.minimum(a, sv)))
    tl.store(A + oa, r, mask=mask)


def _collapse(dims, sas, sbs):
    """Drop size-1 dims and merge adjacent dense runs for both operands."""
    cd = []
    csa = []
    csb = []
    for i in range(len(dims)):
        if dims[i] == 1:
            continue
        cd.append(dims[i])
        csa.append(sas[i])
        csb.append(sbs[i])
    if not cd:
        return [1], [0], [0]
    out_d = [cd[0]]
    out_a = [csa[0]]
    out_b = [csb[0]]
    for i in range(1, len(cd)):
        d = cd[i]
        if out_a[-1] == csa[i] * d and out_b[-1] == csb[i] * d:
            out_d[-1] *= d
        else:
            out_d.append(d)
            out_a.append(csa[i])
            out_b.append(csb[i])
    return out_d, out_a, out_b


def _fastest_first(dims, sas, sbs):
    """Put the operand-a unit-stride dim first (fastest varying) when present."""
    for i in range(len(sas)):
        if sas[i] == 1 and dims[i] > 1:
            if i == 0:
                return dims, sas, sbs
            rest = [j for j in range(len(dims)) if j != i]
            order = [i] + rest
            return (
                [dims[j] for j in order],
                [sas[j] for j in order],
                [sbs[j] for j in order],
            )
    return dims, sas, sbs


def _generic_plan(a, b, n):
    sha = a.shape
    shb = b.shape
    ra = len(sha)
    rb = len(shb)
    shift = ra - rb
    dims = []
    sas = []
    sbs = []
    for k in range(ra):
        j = k - shift
        if j >= 0 and shb[j] != 1:
            sb = b.stride(j)
        else:
            sb = 0
        dims.append(sha[k])
        sas.append(a.stride(k))
        sbs.append(sb)
    dims, sas, sbs = _collapse(dims, sas, sbs)
    dims, sas, sbs = _fastest_first(dims, sas, sbs)
    if len(dims) > _MAX_D:
        raise RuntimeError(
            "_foreach_minimum_: tensors with more than 8 broadcast dims are unsupported"
        )
    return (1, a, b, n, len(dims), tuple(dims), tuple(sas), tuple(sbs))


def _plan(a, b):
    n = a.numel()
    if n == 0:
        return (2,)
    sha = a.shape
    shb = b.shape
    if sha == shb:
        if a.is_contiguous() and b.is_contiguous():
            return (0, a, b, n, 1)
        return _generic_plan(a, b, n)
    if b.numel() == 1:
        if a.is_contiguous():
            return (0, a, b, n, 0)
        return _generic_plan(a, b, n)
    if torch.broadcast_shapes(sha, shb) != tuple(sha):
        raise RuntimeError(
            "_foreach_minimum_: output tensor shape does not match the broadcast shape"
        )
    return _generic_plan(a, b, n)


def _block_for(itemsize):
    """BLOCK giving 16 bytes per thread at 4 warps of 64 lanes (256 threads)."""
    return _BLOCK_BY_ITEMSIZE.get(itemsize, 512)


def _launch_flat(a_ptr, b_ptr, n, b_step):
    block = _block_for(a_ptr.element_size())
    grid = ((n + block - 1) // block,)
    if n > (_I32_MAX - 2 * block):
        _flat_kernel_i64[grid](a_ptr, b_ptr, n, b_step, BLOCK=block, num_warps=_WARPS)
    else:
        _flat_kernel[grid](a_ptr, b_ptr, n, b_step, BLOCK=block, num_warps=_WARPS)


def _launch_gen(a_ptr, b_ptr, n, rank, dims, sas, sbs):
    pad = _MAX_D - rank
    dd = dims + (1,) * pad
    aa = sas + (0,) * pad
    bb = sbs + (0,) * pad
    block = _block_for(a_ptr.element_size())
    grid = ((n + block - 1) // block,)
    _gen_kernel[grid](
        a_ptr,
        b_ptr,
        n,
        dd[0],
        dd[1],
        dd[2],
        dd[3],
        dd[4],
        dd[5],
        dd[6],
        dd[7],
        aa[0],
        aa[1],
        aa[2],
        aa[3],
        aa[4],
        aa[5],
        aa[6],
        aa[7],
        bb[0],
        bb[1],
        bb[2],
        bb[3],
        bb[4],
        bb[5],
        bb[6],
        bb[7],
        RANK=rank,
        BLOCK=block,
        num_warps=_WARPS,
    )


def _fused_numel(pairs, others):
    """Return (common flat numel, 16B-aligned) when the list can share one launch."""
    a0 = pairs[0]
    n = a0.numel()
    if n == 0 or not a0.is_contiguous():
        return 0, False
    b0 = others[0]
    if not b0.is_contiguous() or b0.shape != a0.shape:
        return 0, False
    pa = a0.data_ptr()
    pb = b0.data_ptr()
    acc = pa | pb
    ptrs = [pa, pb]
    for i in range(1, len(pairs)):
        a = pairs[i]
        b = others[i]
        if not a.is_contiguous() or a.numel() != n:
            return 0, False
        if not b.is_contiguous() or b.shape != a.shape:
            return 0, False
        pa = a.data_ptr()
        pb = b.data_ptr()
        acc |= pa | pb
        ptrs.append(pa)
        ptrs.append(pb)
    if acc & 15:
        # A misaligned base would make a cached (vectorized) launch unsafe.
        return 0, False
    # Cross-pair aliasing would make the fused write order observable.
    if len(set(ptrs)) != len(ptrs):
        return 0, False
    return n, True


def _launch_fused(pairs, others, n):
    k = len(pairs)
    a0 = pairs[0]
    b0 = others[0]
    a1 = pairs[1] if k > 1 else a0
    b1 = others[1] if k > 1 else b0
    a2 = pairs[2] if k > 2 else a0
    b2 = others[2] if k > 2 else b0
    a3 = pairs[3] if k > 3 else a0
    b3 = others[3] if k > 3 else b0
    block = _block_for(a0.element_size())
    grid = ((n + block - 1) // block, k, 1)
    key = (
        n,
        k,
        a0.dtype,
        b0.dtype,
        a1.dtype,
        b1.dtype,
        a2.dtype,
        b2.dtype,
        a3.dtype,
        b3.dtype,
        block,
    )
    runner = _FUSED_RUNNERS.get(key)
    if runner is None:
        runner = _build_fused_runner(
            key, a0, b0, a1, b1, a2, b2, a3, b3, n, block, grid
        )
    if runner is not None:
        try:
            runner(a0, b0, a1, b1, a2, b2, a3, b3, n, block)
            return
        except Exception:  # noqa: BLE001 - fall back to the generic launch
            _FUSED_RUNNERS.pop(key, None)
    _fused4[(grid[0], k)](
        a0,
        b0,
        a1,
        b1,
        a2,
        b2,
        a3,
        b3,
        n,
        BLOCK=block,
        num_warps=_WARPS,
    )


def _build_fused_runner(key, a0, b0, a1, b1, a2, b2, a3, b3, n, block, grid):
    try:
        compiled = _fused4.warmup(
            a0,
            b0,
            a1,
            b1,
            a2,
            b2,
            a3,
            b3,
            n,
            BLOCK=block,
            num_warps=_WARPS,
            grid=(1, 1, 1),
        )
    except Exception:  # noqa: BLE001 - pre-compiled launch unavailable
        return None
    gx, gy, gz = grid
    runner = _direct_runner(compiled, gx, gy, gz)
    if runner is None:
        try:
            runner = compiled[grid]
        except Exception:  # noqa: BLE001
            return None
    if len(_FUSED_RUNNERS) >= _RUNNER_CACHE_MAX:
        _FUSED_RUNNERS.clear()
    _FUSED_RUNNERS[key] = runner
    return runner


def _direct_runner(compiled, gx, gy, gz):
    """Skip the per-call launcher closure and its metadata build when possible."""
    try:
        launch = compiled.run
        function = compiled.function
        packed = compiled.packed_metadata
        stream_now = torch.cuda.current_stream

        def runner(
            a0,
            b0,
            a1,
            b1,
            a2,
            b2,
            a3,
            b3,
            n,
            block,
            _launch=launch,
            _fn=function,
            _pm=packed,
            _stream_now=stream_now,
            _grid=(gx, gy, gz),
        ):
            _launch(
                _grid[0],
                _grid[1],
                _grid[2],
                _stream_now().cuda_stream,
                _fn,
                _pm,
                None,
                None,
                None,
                a0,
                b0,
                a1,
                b1,
                a2,
                b2,
                a3,
                b3,
                n,
                block,
            )

        runner(a0, b0, a1, b1, a2, b2, a3, b3, n, block)  # noqa: F821
    except Exception:  # noqa: BLE001 - keep the closure launcher instead
        return None
    return runner


def _run_tensor_list(pairs, others):
    count = len(pairs)
    if count > 1:
        n, _fast = _fused_numel(pairs, others)
        if n:
            if count <= 4:
                _launch_fused(pairs, others, n)
            else:
                for s in range(0, count, 4):
                    _launch_fused(pairs[s : s + 4], others[s : s + 4], n)
            return None
    plans = []
    for i in range(count):
        plans.append(_plan(pairs[i], others[i]))
    for p in plans:
        kind = p[0]
        if kind == 0:
            _launch_flat(p[1], p[2], p[3], p[4])
        elif kind == 1:
            _launch_gen(p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    return None


def _generic_scalar_plan(a, n):
    sha = a.shape
    dims = list(sha)
    sas = [a.stride(k) for k in range(len(sha))]
    cd = []
    csa = []
    for i in range(len(dims)):
        if dims[i] == 1:
            continue
        cd.append(dims[i])
        csa.append(sas[i])
    if not cd:
        cd = [1]
        csa = [0]
    merged_d = [cd[0]]
    merged_a = [csa[0]]
    for i in range(1, len(cd)):
        d = cd[i]
        if merged_a[-1] == csa[i] * d:
            merged_d[-1] *= d
        else:
            merged_d.append(d)
            merged_a.append(csa[i])
    if len(merged_d) > _MAX_D:
        raise RuntimeError(
            "_foreach_minimum_: tensors with more than 8 dims are unsupported"
        )
    for i in range(len(merged_a)):
        if merged_a[i] == 1 and merged_d[i] > 1:
            if i != 0:
                order = [i] + [j for j in range(len(merged_d)) if j != i]
                merged_d = [merged_d[j] for j in order]
                merged_a = [merged_a[j] for j in order]
            break
    return (len(merged_d), tuple(merged_d), tuple(merged_a))


def _run_scalar_list(pairs, value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("_foreach_minimum_: expected a list of tensors or a scalar")
        value = value.item()
    scalar = float(value)
    plans = []
    for a in pairs:
        n = a.numel()
        if n == 0:
            plans.append((2,))
        elif a.is_contiguous():
            plans.append((0, a, n))
        else:
            rank, dims, sas = _generic_scalar_plan(a, n)
            plans.append((1, a, n, rank, dims, sas))
    for p in plans:
        kind = p[0]
        if kind == 0:
            n = p[2]
            block = _block_for(p[1].element_size())
            grid = ((n + block - 1) // block,)
            _flat_scalar_kernel[grid](p[1], n, scalar, BLOCK=block, num_warps=_WARPS)
        elif kind == 1:
            n = p[2]
            pad = _MAX_D - p[3]
            dd = p[4] + (1,) * pad
            aa = p[5] + (0,) * pad
            block = _block_for(p[1].element_size())
            grid = ((n + block - 1) // block,)
            _gen_scalar_kernel[grid](
                p[1],
                n,
                scalar,
                dd[0],
                dd[1],
                dd[2],
                dd[3],
                dd[4],
                dd[5],
                dd[6],
                dd[7],
                aa[0],
                aa[1],
                aa[2],
                aa[3],
                aa[4],
                aa[5],
                aa[6],
                aa[7],
                RANK=p[3],
                BLOCK=block,
                num_warps=_WARPS,
            )
    return None


def foreach_minimum_list_(self, other):
    if isinstance(self, torch.Tensor):
        raise TypeError("_foreach_minimum_ expects a list of tensors")
    if not isinstance(other, (list, tuple)):
        return _run_scalar_list(self, other)
    if len(other) != len(self):
        raise RuntimeError(
            "_foreach_minimum_: both lists must have the same number of tensors"
        )
    if len(self) == 0:
        return None
    return _run_tensor_list(self, other)
