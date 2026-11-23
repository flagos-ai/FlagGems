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

"""foreach maximum between a Tensor list and a Scalar list (Triton implementation).

out[i] = maximum(self[i], scalars[i]) with ATen `maximum` semantics:
  isnan(a) ? a : isnan(b) ? b : (a > b ? a : b)
computed in the tensor dtype (weak scalar promotion: a Python Number scalar is
rounded to self[i].dtype; fp16/bf16 stay in their own dtype).

Device kernels:
  * fused flat kernel: ONE launch per (dtype, device) group. The tensors are
    non-overlapping and dense, so a linear physical index is correct and
    coalesced; the grid is the sum of the per-tensor block counts and each
    program resolves its tensor against host-computed cumulative bounds.
    When every slot numel is a multiple of BLOCK the launch uses a mask-free
    load/store (MASKED=False) so the backend keeps wide vector accesses.
  * strided kernel: generic dim unravelling, used only for rare non-dense views
    (strided slices, expand), one launch per such tensor.
No host-side tensor math and no per-call device tables.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_NUM_WARPS = 4
_MAX_RANK = 8
_NSLOT = 4  # max tensors fused in one launch (longer lists are chunked)


@triton.jit
def _mx(x, y):
    # ATen maximum: NaN-propagating, tie-break keeps the second operand.
    return tl.where(x != x, x, tl.where(y != y, y, tl.where(x > y, x, y)))


@triton.jit
def _fused_slot(
    pid, x_ptr, y_ptr, s, n, lo, hi, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    if (pid >= lo) & (pid < hi):
        offs = (pid - lo) * BLOCK + tl.arange(0, BLOCK)
        if MASKED:
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0)
            sv = s.to(x.dtype)
            tl.store(y_ptr + offs, _mx(x, sv), mask=mask)
        else:
            x = tl.load(x_ptr + offs)
            sv = s.to(x.dtype)
            tl.store(y_ptr + offs, _mx(x, sv))


@triton.jit(do_not_specialize=["n0", "n1", "n2", "n3", "e0", "e1", "e2", "e3"])
def _fused_flat_kernel(
    x0,
    y0,
    s0,
    n0,
    e0,
    x1,
    y1,
    s1,
    n1,
    e1,
    x2,
    y2,
    s2,
    n2,
    e2,
    x3,
    y3,
    s3,
    n3,
    e3,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    _fused_slot(pid, x0, y0, s0, n0, 0, e0, BLOCK, MASKED)
    _fused_slot(pid, x1, y1, s1, n1, e0, e1, BLOCK, MASKED)
    _fused_slot(pid, x2, y2, s2, n2, e1, e2, BLOCK, MASKED)
    _fused_slot(pid, x3, y3, s3, n3, e2, e3, BLOCK, MASKED)


@triton.jit
def _strided_max_kernel(
    x_ptr,
    o_ptr,
    s,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    shapes = (sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7)
    xst = (xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7)
    ost = (os0, os1, os2, os3, os4, os5, os6, os7)
    rem = offs
    xoff = tl.zeros([BLOCK], dtype=tl.int64)
    ooff = tl.zeros([BLOCK], dtype=tl.int64)
    for i in tl.static_range(RANK):
        d = shapes[i].to(tl.int64)
        idx = rem % d
        rem = rem // d
        xoff += idx * xst[i].to(tl.int64)
        ooff += idx * ost[i].to(tl.int64)
    x = tl.load(x_ptr + xoff, mask=mask, other=0)
    sv = s.to(x.dtype)
    tl.store(o_ptr + ooff, _mx(x, sv), mask=mask)


def _is_dense(t):
    """True when the logical element order matches a contiguous physical range.

    Equivalent to Tensor.is_non_overlapping_and_dense() (missing on this Torch
    build): no gaps, no overlaps, no negative or zero strides.
    """
    shape = t.shape
    strides = t.stride()
    dims = []
    for i in range(len(shape)):
        sz = int(shape[i])
        if sz == 1:
            continue
        dims.append((int(strides[i]), sz))
    if not dims:
        return True
    dims = sorted(dims)
    expected = 1
    for st, sz in dims:
        if st != expected:
            return False
        expected *= sz
    return True


def _collapse(shape, ist, ost):
    """Merge adjacent dims that are contiguous in both layouts, drop size-1 dims.

    Returned list is ordered outermost (largest stride) first.
    """
    dims = []
    for i in range(len(shape)):
        sz = int(shape[i])
        if sz == 1:
            continue
        dims.append([sz, int(ist[i]), int(ost[i])])
    changed = True
    while changed:
        changed = False
        for i in range(len(dims) - 1):
            s0, a0, b0 = dims[i]
            s1, a1, b1 = dims[i + 1]
            if a0 == a1 * s1 and b0 == b1 * s1:
                dims[i] = [s0 * s1, a1, b1]
                del dims[i + 1]
                changed = True
                break
    return sorted(dims, key=lambda d: d[1], reverse=True)


def _scalar_value(s, dtype):
    """Scalar handed to the kernel; it is rounded to the tensor dtype in-kernel."""
    if dtype.is_floating_point:
        return float(s)
    if dtype == torch.bool:
        return bool(s)
    return int(s)


def _launch_strided(x, out, sval, n):
    dims = _collapse(x.shape, x.stride(), out.stride())
    rank = len(dims)
    sh = [1] * _MAX_RANK
    xs = [0] * _MAX_RANK
    os = [0] * _MAX_RANK
    for i in range(rank):
        sh[i], xs[i], os[i] = dims[i]
    grid = (triton.cdiv(n, _BLOCK),)
    _strided_max_kernel[grid](
        x,
        out,
        sval,
        n,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        os[0],
        os[1],
        os[2],
        os[3],
        os[4],
        os[5],
        os[6],
        os[7],
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_group(group):
    """One fused launch per (up to _NSLOT) items of a same-dtype group."""
    for start in range(0, len(group), _NSLOT):
        chunk = group[start : start + _NSLOT]
        args = []
        prev = 0
        masked = False
        for t, out, sval, n in chunk:
            prev += (n + _BLOCK - 1) // _BLOCK
            if n % _BLOCK:
                masked = True
            args.append(t)
            args.append(out)
            args.append(sval)
            args.append(n)
            args.append(prev)
        pad_ptr = chunk[0][0]
        pad_s = chunk[0][2]
        for _ in range(_NSLOT - len(chunk)):
            args.extend((pad_ptr, pad_ptr, pad_s, 0, prev))
        _fused_flat_kernel[(prev,)](
            *args, BLOCK=_BLOCK, MASKED=masked, num_warps=_NUM_WARPS
        )


def foreach_maximum_scalar_list(self, scalars):
    n_t = len(self)
    if isinstance(scalars, (list, tuple)):
        slist = scalars
        if len(slist) != n_t:
            raise RuntimeError(
                "Tensor list and scalar list must have the same length, "
                "got {} tensors and {} scalars".format(n_t, len(slist))
            )
    else:
        slist = [scalars] * n_t

    if n_t == 0:
        return []

    outs = [torch.empty_like(t) for t in self]

    # Fast path: one uniform (dtype, device) group of flat-compatible tensors
    # that fits a single launch.
    if n_t <= _NSLOT:
        t0 = self[0]
        dt = t0.dtype
        dev = t0.device
        ok = True
        for t in self:
            if t.dtype is not dt or t.device != dev or t.numel() == 0:
                ok = False
                break
            if not (t.is_contiguous() or _is_dense(t)):
                ok = False
                break
        if ok:
            fscalar = float if dt.is_floating_point else int
            args = []
            prev = 0
            masked = False
            for t, out, s in zip(self, outs, slist):
                n = t.numel()
                prev += (n + _BLOCK - 1) // _BLOCK
                if n % _BLOCK:
                    masked = True
                args.append(t)
                args.append(out)
                args.append(fscalar(s))
                args.append(n)
                args.append(prev)
            pad_ptr = self[0]
            pad_s = args[2]
            for _ in range(_NSLOT - n_t):
                args.extend((pad_ptr, pad_ptr, pad_s, 0, prev))
            _fused_flat_kernel[(prev,)](
                *args, BLOCK=_BLOCK, MASKED=masked, num_warps=_NUM_WARPS
            )
            return outs

    groups = {}
    singles = []
    for i, t in enumerate(self):
        n = t.numel()
        if n == 0:
            continue
        sval = _scalar_value(slist[i], t.dtype)
        if t.is_contiguous() or _is_dense(t):
            key = (t.dtype, t.device)
            g = groups.get(key)
            if g is None:
                groups[key] = [(t, outs[i], sval, n)]
            else:
                g.append((t, outs[i], sval, n))
        else:
            singles.append((t, outs[i], sval, n))

    for group in groups.values():
        _launch_group(group)
    for t, out, sval, n in singles:
        _launch_strided(t, out, sval, n)
    return outs
