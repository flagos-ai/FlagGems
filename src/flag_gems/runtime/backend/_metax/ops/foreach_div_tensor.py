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

"""Triton implementation of aten::_foreach_div.Tensor (tensor list / scalar tensor).

Reference semantics: for every tensor ``t`` in ``self``, ``out = t / other`` with the
result returned as a freshly allocated tensor list (no aliasing of ``self``).  A
0-dimensional (single-element) ``other`` participates as a scalar, so the output
dtype follows ``t.dtype``; half/bfloat16 inputs are computed with float32 op-math
and rounded once to the storage dtype, matching ATen's TensorIterator behaviour.

The hot path fuses the whole (same-dtype, contiguous) tensor list into a single
grid: each program maps its linear program id onto one tensor/block pair through a
compile-time unrolled compare chain, so a list of L tensors costs one launch
instead of L.  Block size 2048 with 256 threads gives 8 elements per thread, i.e.
a full 128-bit vector load and store for 2-byte dtypes and 2 x 128-bit for fp32,
which reaches the device's streaming plateau.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_NUM_WARPS = 4
_MAX_RANK = 8
_FUSED_MAX = 8
_FUSED4 = 4


@triton.jit
def _seg_div(x_ptr, y_ptr, base, n, o, offs0, BLOCK: tl.constexpr):
    offs = base * BLOCK + offs0
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = (x.to(tl.float32) / o).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _div_fused4_kernel(
    x0,
    x1,
    x2,
    x3,
    y0,
    y1,
    y2,
    y3,
    n0,
    n1,
    n2,
    n3,
    o_ptr,
    o_val,
    SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch over a contiguous, same-dtype tensor list of up to four entries."""
    pid = tl.program_id(0)
    offs0 = tl.arange(0, BLOCK)
    c0 = tl.cdiv(n0, BLOCK)
    c1 = c0 + tl.cdiv(n1, BLOCK)
    c2 = c1 + tl.cdiv(n2, BLOCK)
    if SCALAR:
        o = o_val
    else:
        o = tl.load(o_ptr).to(tl.float32)
    if pid < c0:
        _seg_div(x0, y0, pid, n0, o, offs0, BLOCK)
    elif pid < c1:
        _seg_div(x1, y1, pid - c0, n1, o, offs0, BLOCK)
    elif pid < c2:
        _seg_div(x2, y2, pid - c1, n2, o, offs0, BLOCK)
    elif pid < c2 + tl.cdiv(n3, BLOCK):
        _seg_div(x3, y3, pid - c2, n3, o, offs0, BLOCK)


@triton.jit
def _div_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    o_ptr,
    o_val,
    SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch over a contiguous, same-dtype tensor list of five to eight entries."""
    pid = tl.program_id(0)
    offs0 = tl.arange(0, BLOCK)
    c0 = tl.cdiv(n0, BLOCK)
    c1 = c0 + tl.cdiv(n1, BLOCK)
    c2 = c1 + tl.cdiv(n2, BLOCK)
    c3 = c2 + tl.cdiv(n3, BLOCK)
    c4 = c3 + tl.cdiv(n4, BLOCK)
    c5 = c4 + tl.cdiv(n5, BLOCK)
    c6 = c5 + tl.cdiv(n6, BLOCK)
    if SCALAR:
        o = o_val
    else:
        o = tl.load(o_ptr).to(tl.float32)
    if pid < c0:
        _seg_div(x0, y0, pid, n0, o, offs0, BLOCK)
    elif pid < c1:
        _seg_div(x1, y1, pid - c0, n1, o, offs0, BLOCK)
    elif pid < c2:
        _seg_div(x2, y2, pid - c1, n2, o, offs0, BLOCK)
    elif pid < c3:
        _seg_div(x3, y3, pid - c2, n3, o, offs0, BLOCK)
    elif pid < c4:
        _seg_div(x4, y4, pid - c3, n4, o, offs0, BLOCK)
    elif pid < c5:
        _seg_div(x5, y5, pid - c4, n5, o, offs0, BLOCK)
    elif pid < c6:
        _seg_div(x6, y6, pid - c5, n6, o, offs0, BLOCK)
    elif pid < c6 + tl.cdiv(n7, BLOCK):
        _seg_div(x7, y7, pid - c6, n7, o, offs0, BLOCK)


@triton.jit
def _div_flat_kernel(
    x_ptr,
    out_ptr,
    o_ptr,
    o_val,
    n,
    SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Contiguous flat elementwise division by one scalar value."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if SCALAR:
        o = o_val
    else:
        o = tl.load(o_ptr).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = (x.to(tl.float32) / o).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _div_strided_kernel(
    x_ptr,
    out_ptr,
    o_ptr,
    o_val,
    n,
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
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Layout-aware elementwise division for non-contiguous tensors (rank <= 8)."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n

    i7 = idx % s7
    idx = idx // s7
    i6 = idx % s6
    idx = idx // s6
    i5 = idx % s5
    idx = idx // s5
    i4 = idx % s4
    idx = idx // s4
    i3 = idx % s3
    idx = idx // s3
    i2 = idx % s2
    idx = idx // s2
    i1 = idx % s1
    i0 = idx // s1

    x_off = (
        i0 * p0 + i1 * p1 + i2 * p2 + i3 * p3 + i4 * p4 + i5 * p5 + i6 * p6 + i7 * p7
    )
    out_off = (
        i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5 + i6 * t6 + i7 * t7
    )

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    if SCALAR:
        o = o_val
    else:
        o_off = (
            i0 * q0
            + i1 * q1
            + i2 * q2
            + i3 * q3
            + i4 * q4
            + i5 * q5
            + i6 * q6
            + i7 * q7
        )
        o = tl.load(o_ptr + o_off, mask=mask, other=0.0).to(tl.float32)
    y = (x.to(tl.float32) / o).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_off, y, mask=mask)


def _collapse(shape, strides):
    """Drop size-1 dims and fuse adjacent dims that are contiguous in memory."""
    s = [int(v) for v in shape]
    st = [int(v) for v in strides]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(s):
            if s[i] == 1:
                del s[i]
                del st[i]
                changed = True
                continue
            if i + 1 < len(s) and st[i] == st[i + 1] * s[i + 1]:
                s[i] = s[i] * s[i + 1]
                del s[i + 1]
                del st[i + 1]
                changed = True
                continue
            i += 1
    return s, st


def _slots(shape, strides, zero_broadcast_strides=False):
    """Right-align a (shape, strides) pair into _MAX_RANK slots."""
    sizes = [1] * _MAX_RANK
    strs = [0] * _MAX_RANK
    k = len(shape)
    for j in range(k):
        sizes[_MAX_RANK - k + j] = int(shape[j])
        strs[_MAX_RANK - k + j] = int(strides[j])
    if zero_broadcast_strides:
        strs = [0 if sz == 1 else st for sz, st in zip(sizes, strs)]
    return sizes, strs


def _make_out(t, dtype):
    if dtype == t.dtype:
        return torch.empty_like(t)
    return torch.empty_strided(t.shape, t.stride(), dtype=dtype, device=t.device)


def _launch_fused4(tensors, outputs, sizes, o_ptr, o_val, scalar_flag):
    """One launch for a contiguous, same-dtype list of up to four tensors."""
    pad = _FUSED4 - len(tensors)
    if pad:
        xs = list(tensors) + [tensors[0]] * pad
        ys = list(outputs) + [outputs[0]] * pad
        ns = list(sizes) + [0] * pad
    else:
        xs, ys, ns = tensors, outputs, sizes
    blocks = 0
    for n in ns:
        blocks += (n + _BLOCK - 1) // _BLOCK
    _div_fused4_kernel[(blocks,)](
        *xs,
        *ys,
        *ns,
        o_ptr,
        o_val,
        scalar_flag,
        _BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_fused8(tensors, outputs, sizes, o_ptr, o_val, scalar_flag):
    """One launch for a contiguous, same-dtype list of five to eight tensors."""
    pad = _FUSED_MAX - len(tensors)
    xs = list(tensors) + [tensors[0]] * pad
    ys = list(outputs) + [outputs[0]] * pad
    ns = list(sizes) + [0] * pad
    blocks = 0
    for n in ns:
        blocks += (n + _BLOCK - 1) // _BLOCK
    _div_fused_kernel[(blocks,)](
        *xs,
        *ys,
        *ns,
        o_ptr,
        o_val,
        scalar_flag,
        _BLOCK,
        num_warps=_NUM_WARPS,
    )


def foreach_div_tensor(self, other):
    tensors = list(self) if not isinstance(self, list) else self
    k = len(tensors)
    if k == 0:
        return []

    is_tensor = isinstance(other, torch.Tensor)
    scalar_other = (other.numel() == 1) if is_tensor else True

    if scalar_other:
        if is_tensor and other.is_cuda:
            o_ptr = other
            o_val = 0.0
            use_ptr = True
        elif is_tensor:
            o_ptr = None
            o_val = float(other.item())
            use_ptr = False
        else:
            o_ptr = None
            o_val = float(other)
            use_ptr = False
        other_dtype = other.dtype if is_tensor else None
    else:
        use_ptr = True
        o_ptr = other
        o_val = 0.0
        other_dtype = other.dtype

    if scalar_other:
        outputs = [torch.empty_like(t) for t in tensors]
    else:
        outputs = [
            _make_out(t, torch.promote_types(t.dtype, other_dtype)) for t in tensors
        ]

    scalar_flag = not use_ptr

    # Single pass: element counts and fused eligibility (contiguous, one dtype,
    # one device) decided together, so the fused path costs no extra host work.
    head = tensors[0]
    dtype0 = head.dtype
    device0 = head.device
    sizes = []
    fused = scalar_other and k <= _FUSED_MAX
    for t in tensors:
        sizes.append(t.numel())
        if fused and (
            t.dtype != dtype0 or t.device != device0 or not t.is_contiguous()
        ):
            fused = False

    if fused:
        ptr = o_ptr if use_ptr else head
        if k <= _FUSED4:
            _launch_fused4(tensors, outputs, sizes, ptr, o_val, scalar_flag)
        else:
            _launch_fused8(tensors, outputs, sizes, ptr, o_val, scalar_flag)
        return outputs

    for t, out, n in zip(tensors, outputs, sizes):
        if n == 0:
            continue
        ptr = o_ptr if use_ptr else t
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        if t.is_contiguous():
            _div_flat_kernel[grid](
                t,
                out,
                ptr,
                o_val,
                n,
                scalar_flag,
                _BLOCK,
                num_warps=_NUM_WARPS,
            )
        else:
            shape, _ = _collapse(t.shape, t.stride())
            if len(shape) > _MAX_RANK:
                raise NotImplementedError("rank too high for strided division kernel")
            sizes_slots, out_strides = _slots(t.shape, out.stride())
            _, x_strides = _slots(t.shape, t.stride())
            if use_ptr:
                if other.dim() > _MAX_RANK:
                    raise NotImplementedError("other rank too high")
                _, o_strides = _slots(
                    other.shape, other.stride(), zero_broadcast_strides=True
                )
            else:
                _, o_strides = _slots((1,), (0,), zero_broadcast_strides=True)
            _div_strided_kernel[grid](
                t,
                out,
                ptr,
                o_val,
                n,
                *sizes_slots,
                *out_strides,
                *x_strides,
                *o_strides,
                scalar_flag,
                _BLOCK,
                num_warps=_NUM_WARPS,
            )
    return outputs
