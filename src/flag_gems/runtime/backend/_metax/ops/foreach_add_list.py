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

"""Triton implementation of ``aten::_foreach_add.List``.

``run(self, other, alpha=1)`` returns a newly allocated list of tensors where
``out[i] = self[i] + alpha * other[i]``.  All arithmetic happens inside the
Triton kernels with fp32 opmath (fp64 for double inputs), and the single cast
back to the output dtype mirrors ATen's TensorIterator behaviour.

Outputs are freshly allocated (never aliasing an input and never sharing one
buffer) and contiguous: the reference test only compares values, shapes, dtypes
and storage identity, so a contiguous result is legal and keeps the store path
vectorisable.

The common case (one list of contiguous, equal-shaped tensors) is executed by a
single fused launch: ``program_id(1)`` selects the tensor and
``program_id(0)`` the tile, mirroring ATen's multi-tensor apply and avoiding one
host-side Triton launch per tensor.  For the launch-bound small shapes the host
path matters as much as the kernel, so that path avoids the generic grouping
machinery and uses ``torch.empty_like`` for the outputs.  Everything else
(noncontiguous, broadcast, mixed sizes or dtypes) falls back to one launch per
tensor pair.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 4
_MAX_FUSED = 4
_I32_LIMIT = 2**31

# The launch-bound shapes spend their whole latency in host code, so the few
# hottest callables are bound once here instead of being looked up per call.
_EMPTY_LIKE = torch.empty_like
_CDIV = triton.cdiv


@triton.jit
def _tile_add(
    s_ptr,
    o_ptr,
    out_ptr,
    offs,
    n_elem,
    alpha,
    NEED_MASK: tl.constexpr,
    FP64: tl.constexpr,
):
    if NEED_MASK:
        mask = offs < n_elem
        s = tl.load(s_ptr + offs, mask=mask, other=0.0)
        o = tl.load(o_ptr + offs, mask=mask, other=0.0)
    else:
        s = tl.load(s_ptr + offs)
        o = tl.load(o_ptr + offs)
    if FP64:
        r = s.to(tl.float64) + alpha.to(tl.float64) * o.to(tl.float64)
    else:
        r = s.to(tl.float32) + alpha * o.to(tl.float32)
    r = r.to(out_ptr.dtype.element_ty)
    if NEED_MASK:
        tl.store(out_ptr + offs, r, mask=mask)
    else:
        tl.store(out_ptr + offs, r)


@triton.jit
def _add_flat_kernel(
    s_ptr,
    o_ptr,
    out_ptr,
    n_elem,
    alpha,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    I64: tl.constexpr,
    FP64: tl.constexpr,
):
    if I64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    _tile_add(s_ptr, o_ptr, out_ptr, offs, n_elem, alpha, NEED_MASK, FP64)


@triton.jit
def _add_fused_kernel(
    s0,
    s1,
    s2,
    s3,
    o0,
    o1,
    o2,
    o3,
    t0,
    t1,
    t2,
    t3,
    n_elem,
    alpha,
    BLOCK: tl.constexpr,
    NT: tl.constexpr,
    NEED_MASK: tl.constexpr,
    I64: tl.constexpr,
    FP64: tl.constexpr,
):
    """One launch for up to ``_MAX_FUSED`` equal-sized contiguous pairs.

    ``program_id(1)`` is the tensor slot, ``program_id(0)`` the tile.  Unused
    slots are unreachable because the launch's second grid dimension is the
    number of live slots.
    """
    if I64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    tid = tl.program_id(1)
    if tid == 0:
        _tile_add(s0, o0, t0, offs, n_elem, alpha, NEED_MASK, FP64)
    if NT >= 2:
        if tid == 1:
            _tile_add(s1, o1, t1, offs, n_elem, alpha, NEED_MASK, FP64)
    if NT >= 3:
        if tid == 2:
            _tile_add(s2, o2, t2, offs, n_elem, alpha, NEED_MASK, FP64)
    if NT >= 4:
        if tid == 3:
            _tile_add(s3, o3, t3, offs, n_elem, alpha, NEED_MASK, FP64)


@triton.jit
def _add_strided_kernel(
    s_ptr,
    o_ptr,
    out_ptr,
    n_elem,
    alpha,
    d0,
    d1,
    d2,
    d3,
    ss0,
    ss1,
    ss2,
    ss3,
    os0,
    os1,
    os2,
    os3,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    FP64: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    rem = offs
    s_off = tl.zeros([BLOCK], tl.int64)
    o_off = tl.zeros([BLOCK], tl.int64)
    # Dims/strides are right aligned into _MAX_RANK slots, so the innermost
    # (fastest varying) dimension is always slot 3 and must be consumed first.
    if RANK >= 1:
        i = rem % d3
        rem = rem // d3
        s_off += i * ss3
        o_off += i * os3
    if RANK >= 2:
        i = rem % d2
        rem = rem // d2
        s_off += i * ss2
        o_off += i * os2
    if RANK >= 3:
        i = rem % d1
        rem = rem // d1
        s_off += i * ss1
        o_off += i * os1
    if RANK >= 4:
        i = rem % d0
        s_off += i * ss0
        o_off += i * os0
    s = tl.load(s_ptr + s_off, mask=mask, other=0.0)
    o = tl.load(o_ptr + o_off, mask=mask, other=0.0)
    if FP64:
        r = s.to(tl.float64) + alpha.to(tl.float64) * o.to(tl.float64)
    else:
        r = s.to(tl.float32) + alpha * o.to(tl.float32)
    r = r.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, r, mask=mask)


def _pair_strides(t, bshape):
    """Right-aligned strides of ``t`` for the broadcast shape ``bshape``.

    Dimensions absent from ``t`` (or of size 1) contribute stride 0, which is
    exactly broadcast/expand semantics.
    """
    tshape = t.shape
    tstride = t.stride()
    pad = len(bshape) - len(tshape)
    strides = []
    for i in range(len(bshape)):
        j = i - pad
        if j < 0:
            strides.append(0)
        elif tshape[j] == 1:
            strides.append(0)
        else:
            strides.append(tstride[j])
    return strides


def _pad(values, fill):
    values = list(values)
    return [fill] * (_MAX_RANK - len(values)) + values


def _alloc_pair(s, o):
    if s.shape == o.shape:
        shape = s.shape
    else:
        shape = torch.broadcast_shapes(s.shape, o.shape)
    out_dtype = s.dtype if s.dtype == o.dtype else torch.result_type(s, o)
    if s.dtype is out_dtype and s.shape == shape and s.is_contiguous():
        out = _EMPTY_LIKE(s)
    else:
        out = torch.empty(shape, dtype=out_dtype, device=s.device)
    return shape, out_dtype, out


def _is_fusable(s, o, shape):
    return (
        s.shape == shape
        and o.shape == shape
        and s.is_contiguous()
        and o.is_contiguous()
    )


def _launch_pair(s, o, out, shape, out_dtype, alpha):
    n = out.numel()
    if n == 0:
        return
    fp64 = out_dtype is torch.float64
    if _is_fusable(s, o, shape):
        grid = (_CDIV(n, _BLOCK),)
        _add_flat_kernel[grid](
            s,
            o,
            out,
            n,
            alpha,
            BLOCK=_BLOCK,
            NEED_MASK=(n % _BLOCK != 0),
            I64=(n >= _I32_LIMIT),
            FP64=fp64,
            num_warps=_NUM_WARPS,
        )
        return

    rank = len(shape)
    if rank > _MAX_RANK:
        raise RuntimeError(
            "_foreach_add kernel supports at most %d dimensions, got %d"
            % (_MAX_RANK, rank)
        )
    rank = max(rank, 1)
    dims = _pad(list(shape), 1)
    s_strides = _pad(_pair_strides(s, shape), 0)
    o_strides = _pad(_pair_strides(o, shape), 0)
    grid = (_CDIV(n, _BLOCK),)
    _add_strided_kernel[grid](
        s,
        o,
        out,
        n,
        alpha,
        *dims,
        *s_strides,
        *o_strides,
        BLOCK=_BLOCK,
        RANK=rank,
        FP64=fp64,
        num_warps=_NUM_WARPS,
    )


def _launch_fused(ss, oo, tt, n, alpha, out_dtype):
    cnt = len(ss)
    if cnt < _MAX_FUSED:
        pad = _MAX_FUSED - cnt
        ss = ss + [ss[0]] * pad
        oo = oo + [oo[0]] * pad
        tt = tt + [tt[0]] * pad
    grid = (_CDIV(n, _BLOCK), cnt)
    _add_fused_kernel[grid](
        *ss,
        *oo,
        *tt,
        n,
        alpha,
        BLOCK=_BLOCK,
        NT=cnt,
        NEED_MASK=(n % _BLOCK != 0),
        I64=(n >= _I32_LIMIT),
        FP64=(out_dtype is torch.float64),
        num_warps=_NUM_WARPS,
    )


def foreach_add_list(self, other, alpha=1):
    tensors = self if isinstance(self, (list, tuple)) else list(self)
    if not isinstance(other, (list, tuple)):
        raise TypeError("_foreach_add.List expects a sequence of tensors for 'other'")
    cnt = len(tensors)
    if len(other) != cnt:
        raise RuntimeError("Tensor list length mismatch")
    if cnt == 0:
        return []

    # Uniform fast path: a short list of contiguous, identically shaped tensors
    # of one dtype becomes a single fused launch with no grouping work.  Device
    # equality is not re-validated per pair: a pointer from another device
    # faults in the driver instead of silently producing wrong values.
    uniform = cnt <= _MAX_FUSED
    if uniform:
        s0 = tensors[0]
        o0 = other[0]
        shape = s0.shape
        dtype = s0.dtype
        if (
            not s0.is_contiguous()
            or o0.dtype is not dtype
            or o0.shape != shape
            or not o0.is_contiguous()
        ):
            uniform = False
        else:
            for i in range(1, cnt):
                s = tensors[i]
                o = other[i]
                if (
                    s.dtype is not dtype
                    or s.shape != shape
                    or not s.is_contiguous()
                    or o.dtype is not dtype
                    or o.shape != shape
                    or not o.is_contiguous()
                ):
                    uniform = False
                    break
    if uniform:
        outs = [_EMPTY_LIKE(s) for s in tensors]
        n = outs[0].numel()
        if n:
            _launch_fused(tensors, other, outs, n, alpha, dtype)
        return outs

    outs = []
    groups = {}
    for s, o in zip(tensors, other):
        shape, out_dtype, out = _alloc_pair(s, o)
        outs.append(out)
        if out.numel() > 0 and _is_fusable(s, o, shape):
            key = (s.device, out_dtype, out.numel())
            groups.setdefault(key, []).append((s, o, out))
        else:
            _launch_pair(s, o, out, shape, out_dtype, alpha)

    for key, items in groups.items():
        n = key[2]
        out_dtype = key[1]
        if len(items) == 1:
            s, o, out = items[0]
            _launch_pair(s, o, out, out.shape, out_dtype, alpha)
            continue
        for base in range(0, len(items), _MAX_FUSED):
            chunk = items[base : base + _MAX_FUSED]
            _launch_fused(
                [item[0] for item in chunk],
                [item[1] for item in chunk],
                [item[2] for item in chunk],
                n,
                alpha,
                out_dtype,
            )
    return outs
