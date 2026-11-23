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

"""Triton implementation of aten::_foreach_clamp_min.ScalarList (out-of-place).

For every tensor ``t`` in the list, with its paired scalar ``s``::

    out = where(isnan(x), x, maximum_num(x, cast(s, x.dtype)))

which is exactly NaN-propagating ``clamp_min`` evaluated in the tensor's own
dtype (ATen converts the Scalar to the element dtype before comparing).

The wrapper only allocates outputs, reads metadata, collapses metadata for
non-contiguous inputs and launches the kernels.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_DIMS = 8
_WIDE_LIMIT = 1 << 31
# Grouped launch for short lists of small tensors: one launch instead of N.
_GROUP_K = 4
_GROUP_BLOCK = 512
_GROUP_WARPS = 4
_GROUP_MAX = 1 << 24


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _clamp_min_1d(x_ptr, out_ptr, n, s, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    sc = s.to(x.dtype)
    y = tl.where(x != x, x, tl.maximum(x, sc))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _clamp_min_bool_1d(x_ptr, out_ptr, n, SB: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = x | SB
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _clamp_min_slot(x_ptr, out_ptr, s, offs, mask):
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    sc = s.to(x.dtype)
    y = tl.where(x != x, x, tl.maximum(x, sc))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _clamp_min_group4(
    x0,
    x1,
    x2,
    x3,
    o0,
    o1,
    o2,
    o3,
    s0,
    s1,
    s2,
    s3,
    n0,
    n1,
    n2,
    n3,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if K > 0:
        _clamp_min_slot(x0, o0, s0, offs, offs < n0)
    if K > 1:
        _clamp_min_slot(x1, o1, s1, offs, offs < n1)
    if K > 2:
        _clamp_min_slot(x2, o2, s2, offs, offs < n2)
    if K > 3:
        _clamp_min_slot(x3, o3, s3, offs, offs < n3)


@triton.jit
def _clamp_min_strided4(
    x_ptr, out_ptr, n, s, n0, n1, n2, n3, t0, t1, t2, t3, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    i3 = offs % n3
    r = offs // n3
    i2 = r % n2
    r = r // n2
    i1 = r % n1
    i0 = r // n1
    xo = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3
    x = tl.load(x_ptr + xo, mask=mask, other=0)
    sc = s.to(x.dtype)
    y = tl.where(x != x, x, tl.maximum(x, sc))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _clamp_min_strided8(
    x_ptr,
    out_ptr,
    n,
    s,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
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
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    i7 = offs % n7
    r = offs // n7
    i6 = r % n6
    r = r // n6
    i5 = r % n5
    r = r // n5
    i4 = r % n4
    r = r // n4
    i3 = r % n3
    r = r // n3
    i2 = r % n2
    r = r // n2
    i1 = r % n1
    i0 = r // n1
    xo = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5 + i6 * t6 + i7 * t7
    x = tl.load(x_ptr + xo, mask=mask, other=0)
    sc = s.to(x.dtype)
    y = tl.where(x != x, x, tl.maximum(x, sc))
    tl.store(out_ptr + offs, y, mask=mask)


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #
def _scalar_for_dtype(value, dtype):
    """Reproduce ATen's ``Scalar -> scalar_t`` conversion exactly."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if dtype == torch.bool:
        return bool(value)
    if dtype.is_floating_point:
        if dtype in (torch.float32, torch.float64):
            return float(value)
        # fp16/bf16: round on the host so the in-kernel cast is exact.
        return float(torch.tensor(float(value), dtype=dtype).item())
    return int(value)


def _collapse(shape, strides):
    """Drop size-1 dims and merge stride-contiguous dim runs (metadata only)."""
    dims = [(int(sz), int(st)) for sz, st in zip(shape, strides) if sz != 1]
    merged = []
    for sz, st in reversed(dims):
        if merged and merged[-1][1] * merged[-1][0] == st:
            msz, mst = merged[-1]
            merged[-1] = (msz * sz, mst)
        else:
            merged.append((sz, st))
    merged.reverse()
    return merged


def foreach_clamp_min_scalar_list(self, scalars):
    tensors = list(self)
    count = len(tensors)

    if isinstance(scalars, (bool, int, float)) or isinstance(scalars, torch.Tensor):
        if isinstance(scalars, torch.Tensor) and scalars.dim() > 0:
            scalar_list = list(scalars)
        else:
            scalar_list = [scalars] * count
    else:
        scalar_list = list(scalars)
        if len(scalar_list) != count:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list, "
                "got {0} tensors, {1} scalars".format(count, len(scalar_list))
            )

    outs = []
    if count == 0:
        return outs

    for t in tensors:
        outs.append(torch.empty(t.shape, dtype=t.dtype, device=t.device))

    cur_device = torch.cuda.current_device()
    on_current = True
    for t in tensors:
        idx = t.device.index
        if idx is not None and idx != cur_device:
            on_current = False
            break

    if on_current and _try_grouped(tensors, outs, scalar_list):
        return outs

    for t, out, s in zip(tensors, outs, scalar_list):
        numel = t.numel()
        if numel == 0:
            continue
        dev_index = t.device.index
        if dev_index is not None and dev_index != cur_device:
            with torch.cuda.device(dev_index):
                _launch(t, out, numel, s)
        else:
            _launch(t, out, numel, s)

    return outs


def _try_grouped(tensors, outs, scalar_list):
    """One launch for a short list of small, same-dtype contiguous tensors."""
    count = len(tensors)
    if count < 2 or count > _GROUP_K:
        return False
    dtype = tensors[0].dtype
    if dtype == torch.bool:
        return False

    sizes = []
    largest = 0
    for t in tensors:
        if t.dtype != dtype or not t.is_contiguous():
            return False
        numel = t.numel()
        if numel == 0:
            return False
        if numel > largest:
            largest = numel
        sizes.append(numel)
    if largest > _GROUP_MAX:
        return False

    pad = _GROUP_K - count
    xs = list(tensors) + [tensors[0]] * pad
    os_ = list(outs) + [outs[0]] * pad
    ss = [_scalar_for_dtype(s, dtype) for s in scalar_list]
    ss += [ss[0]] * pad
    ns = sizes + [0] * pad

    grid = (triton.cdiv(largest, _GROUP_BLOCK),)
    _clamp_min_group4[grid](
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        os_[0],
        os_[1],
        os_[2],
        os_[3],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        K=count,
        BLOCK=_GROUP_BLOCK,
        num_warps=_GROUP_WARPS,
    )
    return True


def _launch(t, out, numel, scalar):
    grid = (triton.cdiv(numel, _BLOCK),)
    if t.is_contiguous():
        if t.dtype == torch.bool:
            _clamp_min_bool_1d[grid](
                t, out, numel, SB=bool(scalar), BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
        else:
            _clamp_min_1d[grid](
                t,
                out,
                numel,
                _scalar_for_dtype(scalar, t.dtype),
                BLOCK=_BLOCK,
                WIDE=numel >= _WIDE_LIMIT,
                num_warps=_NUM_WARPS,
            )
        return

    dims = _collapse(t.shape, t.stride())
    ndim = len(dims)
    if ndim > _MAX_DIMS:
        raise RuntimeError(
            "foreach_clamp_min: input with {0} non-collapsible dims is not "
            "supported".format(ndim)
        )
    sv = _scalar_for_dtype(scalar, t.dtype)

    if ndim <= 4:
        pad = 4 - ndim
        sizes = [1] * pad + [d[0] for d in dims]
        steps = [0] * pad + [d[1] for d in dims]
        _clamp_min_strided4[grid](
            t,
            out,
            numel,
            sv,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            steps[0],
            steps[1],
            steps[2],
            steps[3],
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    else:
        pad = _MAX_DIMS - ndim
        sizes = [1] * pad + [d[0] for d in dims]
        steps = [0] * pad + [d[1] for d in dims]
        _clamp_min_strided8[grid](
            t,
            out,
            numel,
            sv,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            sizes[6],
            sizes[7],
            steps[0],
            steps[1],
            steps[2],
            steps[3],
            steps[4],
            steps[5],
            steps[6],
            steps[7],
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
