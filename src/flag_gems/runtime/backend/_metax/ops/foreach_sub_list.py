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

"""Triton implementation of the foreach sub list form.

foreach_sub_list(self, other, alpha=1) -> tuple of freshly allocated tensors, each
out_i = self_i - alpha * other_i (paired) or self_i - alpha (scalar overload),
computed entirely by @triton.jit kernels. The host wrapper performs metadata
reads, output allocation and launches only.
"""

import numpy as np
import torch
import triton
import triton.language as tl

_MAX_RANK = 8
_BLOCK = 1024
_WARPS = 4
_INT32_LIMIT = 1 << 31


@triton.jit
def _sub_flat(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    alpha,
    alpha_ptr,
    ALPHA_PTR: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ALPHA_PTR:
        al = tl.load(alpha_ptr)
    else:
        al = alpha
    if NEED_MASK:
        m = offs < n
        a = tl.load(a_ptr + offs, mask=m, other=0)
        if HAS_B:
            b = tl.load(b_ptr + offs, mask=m, other=0)
            r = a - al * b
        else:
            r = a - al
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)
    else:
        a = tl.load(a_ptr + offs)
        if HAS_B:
            b = tl.load(b_ptr + offs)
            r = a - al * b
        else:
            r = a - al
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty))


@triton.jit
def _sub_nd(
    a_ptr,
    b_ptr,
    o_ptr,
    n,
    alpha,
    alpha_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    ta0,
    ta1,
    ta2,
    ta3,
    ta4,
    ta5,
    ta6,
    ta7,
    tb0,
    tb1,
    tb2,
    tb3,
    tb4,
    tb5,
    tb6,
    tb7,
    to0,
    to1,
    to2,
    to3,
    to4,
    to5,
    to6,
    to7,
    ALPHA_PTR: tl.constexpr,
    HAS_B: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    rem = offs.to(tl.int64)

    idx = rem // d0
    rem = rem - idx * d0
    ao = idx * ta0
    bo = idx * tb0
    oo = idx * to0
    if RANK > 1:
        idx = rem // d1
        rem = rem - idx * d1
        ao += idx * ta1
        bo += idx * tb1
        oo += idx * to1
    if RANK > 2:
        idx = rem // d2
        rem = rem - idx * d2
        ao += idx * ta2
        bo += idx * tb2
        oo += idx * to2
    if RANK > 3:
        idx = rem // d3
        rem = rem - idx * d3
        ao += idx * ta3
        bo += idx * tb3
        oo += idx * to3
    if RANK > 4:
        idx = rem // d4
        rem = rem - idx * d4
        ao += idx * ta4
        bo += idx * tb4
        oo += idx * to4
    if RANK > 5:
        idx = rem // d5
        rem = rem - idx * d5
        ao += idx * ta5
        bo += idx * tb5
        oo += idx * to5
    if RANK > 6:
        idx = rem // d6
        rem = rem - idx * d6
        ao += idx * ta6
        bo += idx * tb6
        oo += idx * to6
    if RANK > 7:
        idx = rem // d7
        ao += idx * ta7
        bo += idx * tb7
        oo += idx * to7

    if ALPHA_PTR:
        al = tl.load(alpha_ptr)
    else:
        al = alpha

    m = offs < n
    a = tl.load(a_ptr + ao, mask=m, other=0)
    if HAS_B:
        b = tl.load(b_ptr + bo, mask=m, other=0)
        r = a - al * b
    else:
        r = a - al
    tl.store(o_ptr + oo, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _sub_flat4(
    a0,
    b0,
    o0,
    a1,
    b1,
    o1,
    a2,
    b2,
    o2,
    a3,
    b3,
    o3,
    numel,
    alpha,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    if tid == 0:
        if NEED_MASK:
            x = tl.load(a0 + offs, mask=m, other=0)
            y = tl.load(b0 + offs, mask=m, other=0)
            tl.store(o0 + offs, (x - alpha * y).to(o0.dtype.element_ty), mask=m)
        else:
            x = tl.load(a0 + offs)
            y = tl.load(b0 + offs)
            tl.store(o0 + offs, (x - alpha * y).to(o0.dtype.element_ty))
    if NT > 1:
        if tid == 1:
            if NEED_MASK:
                x = tl.load(a1 + offs, mask=m, other=0)
                y = tl.load(b1 + offs, mask=m, other=0)
                tl.store(o1 + offs, (x - alpha * y).to(o1.dtype.element_ty), mask=m)
            else:
                x = tl.load(a1 + offs)
                y = tl.load(b1 + offs)
                tl.store(o1 + offs, (x - alpha * y).to(o1.dtype.element_ty))
    if NT > 2:
        if tid == 2:
            if NEED_MASK:
                x = tl.load(a2 + offs, mask=m, other=0)
                y = tl.load(b2 + offs, mask=m, other=0)
                tl.store(o2 + offs, (x - alpha * y).to(o2.dtype.element_ty), mask=m)
            else:
                x = tl.load(a2 + offs)
                y = tl.load(b2 + offs)
                tl.store(o2 + offs, (x - alpha * y).to(o2.dtype.element_ty))
    if NT > 3:
        if tid == 3:
            if NEED_MASK:
                x = tl.load(a3 + offs, mask=m, other=0)
                y = tl.load(b3 + offs, mask=m, other=0)
                tl.store(o3 + offs, (x - alpha * y).to(o3.dtype.element_ty), mask=m)
            else:
                x = tl.load(a3 + offs)
                y = tl.load(b3 + offs)
                tl.store(o3 + offs, (x - alpha * y).to(o3.dtype.element_ty))


def _collapse(a, b, o):
    """Collapse adjacent dims that are contiguous for all three operands."""
    shape = a.shape
    if len(shape) == 0:
        return None
    sa = a.stride()
    sb = b.stride()
    so = o.stride()
    dims = []
    for k in range(len(shape)):
        s = shape[k]
        if s == 1:
            continue
        if (
            dims
            and dims[-1][1] * s == sa[k]
            and dims[-1][2] * s == sb[k]
            and dims[-1][3] * s == so[k]
        ):
            dims[-1][0] *= s
        else:
            dims.append([s, sa[k], sb[k], so[k]])
    if not dims or len(dims) > _MAX_RANK:
        return None
    # Put the axis that is unit-stride for the freshly allocated output last so
    # the flat index walks memory in output store order.
    pick = -1
    for i, d in enumerate(dims):
        if d[3] == 1:
            if pick < 0 or (dims[pick][1] != 1 and d[1] == 1):
                pick = i
    order = [i for i in range(len(dims)) if i != pick]
    if pick >= 0:
        order.append(pick)
    ordered = [dims[i] for i in order]
    rank = len(ordered)
    divs = [1] * rank
    acc = 1
    for k in range(rank - 1, -1, -1):
        divs[k] = acc
        acc *= ordered[k][0]
    return ordered, divs


def _pad(vals):
    out = list(vals) + [1] * (_MAX_RANK - len(vals))
    return out


def _scalar_alpha(value, out_dtype):
    if out_dtype == torch.float64:
        return np.float64(value)
    if out_dtype in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ):
        return int(value)
    return float(value)


def _launch_pair(a, b, alpha, alpha_tensor, out, n):
    """Launch one kernel for a single (a, b-or-scalar) pair writing into out."""
    use_ptr = alpha_tensor is not None
    if use_ptr:
        alpha_arg = 0.0
        alpha_ptr = alpha_tensor
    else:
        alpha_arg = alpha
        alpha_ptr = a

    flat = (
        a.is_contiguous()
        and out.is_contiguous()
        and (b is None or b.is_contiguous())
        and n < _INT32_LIMIT
    )
    if flat:
        grid = (triton.cdiv(n, _BLOCK),)
        _sub_flat[grid](
            a,
            a if b is None else b,
            out,
            n,
            alpha_arg,
            alpha_ptr,
            ALPHA_PTR=use_ptr,
            HAS_B=b is not None,
            BLOCK=_BLOCK,
            NEED_MASK=(n % _BLOCK) != 0,
            num_warps=_WARPS,
        )
        return

    plan = _collapse(a, a if b is None else b, out)
    if plan is None:
        raise RuntimeError("unsupported layout: too many non-collapsible dims")
    ordered, divs = plan
    rank = len(ordered)
    grid = (triton.cdiv(n, _BLOCK),)
    _sub_nd[grid](
        a,
        a if b is None else b,
        out,
        n,
        alpha_arg,
        alpha_ptr,
        *_pad(divs),
        *_pad([d[1] for d in ordered]),
        *_pad([d[2] for d in ordered]),
        *_pad([d[3] for d in ordered]),
        ALPHA_PTR=use_ptr,
        HAS_B=b is not None,
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _launch_multi(group):
    """Fuse up to four equal-length contiguous pairs into one launch."""
    cnt = len(group)
    a0, b0, o0 = group[0][0], group[0][1], group[0][2]
    a1, b1, o1 = (group[1][0], group[1][1], group[1][2]) if cnt > 1 else (a0, b0, o0)
    a2, b2, o2 = (group[2][0], group[2][1], group[2][2]) if cnt > 2 else (a0, b0, o0)
    a3, b3, o3 = (group[3][0], group[3][1], group[3][2]) if cnt > 3 else (a0, b0, o0)
    numel = group[0][5]
    blocks = (numel + _BLOCK - 1) // _BLOCK
    _sub_flat4[(blocks, cnt)](
        a0,
        b0,
        o0,
        a1,
        b1,
        o1,
        a2,
        b2,
        o2,
        a3,
        b3,
        o3,
        numel,
        group[0][3],
        NT=cnt,
        BLOCK=_BLOCK,
        NEED_MASK=(numel % _BLOCK) != 0,
        num_warps=_WARPS,
    )


def foreach_sub_list(self, other, alpha=1, inplace=False, **kwargs):
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "foreach_sub_list() expects a sequence of tensors as the first argument"
        )
    n = len(self)

    other_is_ptr = isinstance(other, torch.Tensor)
    if isinstance(other, (list, tuple)):
        if len(other) and isinstance(other[0], torch.Tensor):
            mode = "list"
        else:
            mode = "scalarlist"
        if len(other) != n:
            raise RuntimeError(
                "Tensor lists must have the same number of tensors, got "
                f"{n} and {len(other)}"
            )
    elif other_is_ptr or isinstance(other, (int, float, bool)):
        mode = "scalar"
    else:
        mode = "scalar"

    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    items = []
    empty_like = torch.empty_like
    scalar_cache = {} if (mode == "scalar" and not other_is_ptr) else None
    for i in range(n):
        a = self[i]
        b = None
        alpha_tensor = None
        alpha_val = alpha
        if mode == "list":
            b = other[i]
            out_dtype = a.dtype if a.dtype is b.dtype else torch.result_type(a, b)
        elif mode == "scalarlist":
            alpha_val = other[i]
            out_dtype = torch.result_type(a, alpha_val)
        elif other_is_ptr:
            alpha_tensor = other
            out_dtype = torch.result_type(a, other)
        else:
            alpha_val = other
            if a.is_floating_point() or isinstance(alpha_val, int):
                out_dtype = a.dtype
            else:
                out_dtype = torch.result_type(a, alpha_val)

        if inplace:
            out = a
        elif out_dtype is a.dtype:
            out = empty_like(a)
        else:
            out = empty_like(a, dtype=out_dtype)

        numel = a.numel()
        if alpha_tensor is not None:
            scalar = 0.0
        elif scalar_cache is None:
            scalar = _scalar_alpha(alpha_val, out.dtype)
        else:
            scalar = scalar_cache.get(out.dtype)
            if scalar is None:
                scalar = _scalar_alpha(alpha_val, out.dtype)
                scalar_cache[out.dtype] = scalar

        # Fusible when every operand is a contiguous dense buffer of the same
        # length, so one launch can serve up to four list elements.
        key = None
        if (
            not inplace
            and b is not None
            and alpha_tensor is None
            and numel > 0
            and numel < _INT32_LIMIT
            and a.is_contiguous()
            and b.is_contiguous()
        ):
            key = (a.dtype, b.dtype, out.dtype, numel, scalar)

        items.append([a, b, out, scalar, alpha_tensor, numel, key])

    i = 0
    total = len(items)
    while i < total:
        it = items[i]
        if it[6] is None:
            if it[5] > 0:
                _launch_pair(it[0], it[1], it[3], it[4], it[2], it[5])
            i += 1
            continue
        j = i + 1
        while j < total and items[j][6] == it[6]:
            j += 1
        if j - i == 1:
            _launch_pair(it[0], it[1], it[3], it[4], it[2], it[5])
        else:
            k = i
            while k < j:
                e = min(j, k + 4)
                _launch_multi(items[k:e])
                k = e
        i = j

    if inplace:
        return self
    return tuple(it[2] for it in items)
