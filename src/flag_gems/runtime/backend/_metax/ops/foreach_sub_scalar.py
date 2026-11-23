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

"""foreach_sub_scalar: out_i = self_i - scalar for a list of tensors.

Round 2 (retained): the flat (dense) path has an unmasked "exact" specialization for
numel % BLOCK == 0 with 16B-aligned pointers; dropping the mask lets the backend emit
128-bit vector accesses instead of four scalar predicated accesses per thread.

Round 3: dense tensors of one call that share (input dtype, output dtype) are fused
into a SINGLE launch (up to 8 slots, per-slot block starts, one uniform branch per
program). The launch-bound small-tensor workloads were paying ~4.9 us of host launch
overhead per tensor while their streaming work is a few microseconds.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_MAX_DIMS = 8
_FUSED = 8

_BOOL_ERR = (
    "Subtraction, the `-` operator, with a bool tensor is not supported. "
    "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
)


# ---------------------------------------------------------------- flat kernels
@triton.jit(do_not_specialize=["n"])
def _sub_flat_f(
    x_ptr, o_ptr, n, s, BLOCK: tl.constexpr, EXACT: tl.constexpr, I64: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        nlim = n.to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        nlim = n
    if EXACT:
        x = tl.load(x_ptr + offs)
        dt = o_ptr.dtype.element_ty
        if dt == tl.float32:
            r = x.to(tl.float32) - s
        elif dt == tl.float16 or dt == tl.bfloat16:
            r = (x.to(tl.float32) - s).to(dt)
        elif dt == tl.float64:
            r = x.to(tl.float64) - s.to(tl.float64)
        else:
            r = x.to(dt) - s
        tl.store(o_ptr + offs, r)
    else:
        mask = offs < nlim
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        dt = o_ptr.dtype.element_ty
        if dt == tl.float32:
            r = x.to(tl.float32) - s
        elif dt == tl.float16 or dt == tl.bfloat16:
            r = (x.to(tl.float32) - s).to(dt)
        elif dt == tl.float64:
            r = x.to(tl.float64) - s.to(tl.float64)
        else:
            r = x.to(dt) - s
        tl.store(o_ptr + offs, r, mask=mask)


@triton.jit(do_not_specialize=["n", "s"])
def _sub_flat_i(
    x_ptr, o_ptr, n, s, BLOCK: tl.constexpr, EXACT: tl.constexpr, I64: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        nlim = n.to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        nlim = n
    dt = o_ptr.dtype.element_ty
    if EXACT:
        x = tl.load(x_ptr + offs)
        tl.store(o_ptr + offs, x.to(dt) - s)
    else:
        mask = offs < nlim
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        tl.store(o_ptr + offs, x.to(dt) - s, mask=mask)


# ------------------------------------------------------- fused (multi-tensor)
@triton.jit
def _slot_f(xp, op, base, n, s, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    offs = base + tl.arange(0, BLOCK)
    dt = op.dtype.element_ty
    if EXACT:
        x = tl.load(xp + offs)
        if dt == tl.float32:
            r = x.to(tl.float32) - s
        elif dt == tl.float16 or dt == tl.bfloat16:
            r = (x.to(tl.float32) - s).to(dt)
        elif dt == tl.float64:
            r = x.to(tl.float64) - s.to(tl.float64)
        else:
            r = x.to(dt) - s
        tl.store(op + offs, r)
    else:
        mask = offs < n
        x = tl.load(xp + offs, mask=mask, other=0)
        if dt == tl.float32:
            r = x.to(tl.float32) - s
        elif dt == tl.float16 or dt == tl.bfloat16:
            r = (x.to(tl.float32) - s).to(dt)
        elif dt == tl.float64:
            r = x.to(tl.float64) - s.to(tl.float64)
        else:
            r = x.to(dt) - s
        tl.store(op + offs, r, mask=mask)


@triton.jit
def _slot_i(xp, op, base, n, s, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    offs = base + tl.arange(0, BLOCK)
    dt = op.dtype.element_ty
    if EXACT:
        x = tl.load(xp + offs)
        tl.store(op + offs, x.to(dt) - s)
    else:
        mask = offs < n
        x = tl.load(xp + offs, mask=mask, other=0)
        tl.store(op + offs, x.to(dt) - s, mask=mask)


@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "sb1",
        "sb2",
        "sb3",
        "sb4",
        "sb5",
        "sb6",
        "sb7",
    ]
)
def _sub_fused_f(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    x4,
    o4,
    n4,
    x5,
    o5,
    n5,
    x6,
    o6,
    n6,
    x7,
    o7,
    n7,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    sb6,
    sb7,
    s,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < sb1:
        _slot_f(x0, o0, pid * BLOCK, n0, s, BLOCK, EXACT)
    elif pid < sb2:
        _slot_f(x1, o1, (pid - sb1) * BLOCK, n1, s, BLOCK, EXACT)
    elif pid < sb3:
        _slot_f(x2, o2, (pid - sb2) * BLOCK, n2, s, BLOCK, EXACT)
    elif pid < sb4:
        _slot_f(x3, o3, (pid - sb3) * BLOCK, n3, s, BLOCK, EXACT)
    elif pid < sb5:
        _slot_f(x4, o4, (pid - sb4) * BLOCK, n4, s, BLOCK, EXACT)
    elif pid < sb6:
        _slot_f(x5, o5, (pid - sb5) * BLOCK, n5, s, BLOCK, EXACT)
    elif pid < sb7:
        _slot_f(x6, o6, (pid - sb6) * BLOCK, n6, s, BLOCK, EXACT)
    else:
        _slot_f(x7, o7, (pid - sb7) * BLOCK, n7, s, BLOCK, EXACT)


@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "sb1",
        "sb2",
        "sb3",
        "sb4",
        "sb5",
        "sb6",
        "sb7",
        "s",
    ]
)
def _sub_fused_i(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    x4,
    o4,
    n4,
    x5,
    o5,
    n5,
    x6,
    o6,
    n6,
    x7,
    o7,
    n7,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    sb6,
    sb7,
    s,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < sb1:
        _slot_i(x0, o0, pid * BLOCK, n0, s, BLOCK, EXACT)
    elif pid < sb2:
        _slot_i(x1, o1, (pid - sb1) * BLOCK, n1, s, BLOCK, EXACT)
    elif pid < sb3:
        _slot_i(x2, o2, (pid - sb2) * BLOCK, n2, s, BLOCK, EXACT)
    elif pid < sb4:
        _slot_i(x3, o3, (pid - sb3) * BLOCK, n3, s, BLOCK, EXACT)
    elif pid < sb5:
        _slot_i(x4, o4, (pid - sb4) * BLOCK, n4, s, BLOCK, EXACT)
    elif pid < sb6:
        _slot_i(x5, o5, (pid - sb5) * BLOCK, n5, s, BLOCK, EXACT)
    elif pid < sb7:
        _slot_i(x6, o6, (pid - sb6) * BLOCK, n6, s, BLOCK, EXACT)
    else:
        _slot_i(x7, o7, (pid - sb7) * BLOCK, n7, s, BLOCK, EXACT)


# ------------------------------------------------------------ strided kernels
@triton.jit(
    do_not_specialize=[
        "n",
        "sf",
        "si",
        "sh0",
        "sh1",
        "sh2",
        "sh3",
        "sh4",
        "sh5",
        "sh6",
        "sh7",
        "xs0",
        "xs1",
        "xs2",
        "xs3",
        "xs4",
        "xs5",
        "xs6",
        "xs7",
        "os0",
        "os1",
        "os2",
        "os3",
        "os4",
        "os5",
        "os6",
        "os7",
    ]
)
def _sub_strided(
    x_ptr,
    o_ptr,
    n,
    sf,
    si,
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
    IS_INT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n.to(tl.int64)
    rem = idx
    xo = tl.zeros([BLOCK], tl.int64)
    oo = tl.zeros([BLOCK], tl.int64)
    if RANK > 0:
        c = rem % sh0
        rem = rem // sh0
        xo += c * xs0
        oo += c * os0
    if RANK > 1:
        c = rem % sh1
        rem = rem // sh1
        xo += c * xs1
        oo += c * os1
    if RANK > 2:
        c = rem % sh2
        rem = rem // sh2
        xo += c * xs2
        oo += c * os2
    if RANK > 3:
        c = rem % sh3
        rem = rem // sh3
        xo += c * xs3
        oo += c * os3
    if RANK > 4:
        c = rem % sh4
        rem = rem // sh4
        xo += c * xs4
        oo += c * os4
    if RANK > 5:
        c = rem % sh5
        rem = rem // sh5
        xo += c * xs5
        oo += c * os5
    if RANK > 6:
        c = rem % sh6
        rem = rem // sh6
        xo += c * xs6
        oo += c * os6
    if RANK > 7:
        c = rem % sh7
        rem = rem // sh7
        xo += c * xs7
        oo += c * os7
    x = tl.load(x_ptr + xo, mask=mask, other=0)
    dt = o_ptr.dtype.element_ty
    if IS_INT:
        r = x.to(dt) - si
    else:
        if dt == tl.float32:
            r = x.to(tl.float32) - sf
        elif dt == tl.float16 or dt == tl.bfloat16:
            r = (x.to(tl.float32) - sf).to(dt)
        elif dt == tl.float64:
            r = x.to(tl.float64) - sf.to(tl.float64)
        else:
            r = x.to(dt) - sf
    tl.store(o_ptr + oo, r, mask=mask)


# ------------------------------------------------------------------ host side
def _is_dense(t):
    """True when storage[data_ptr() : data_ptr()+numel) is exactly this tensor."""
    if t.is_contiguous():
        return True
    dims = sorted([(st, sh) for sh, st in zip(t.shape, t.stride()) if sh != 1])
    running = 1
    for st, sh in dims:
        if st != running:
            return False
        running *= sh
    return running == t.numel()


def _out_dtype(t, scalar):
    # ATen weak-scalar promotion: floating tensors keep their dtype; integral
    # tensors promote to the default float dtype only for a non-integral scalar.
    if t.dtype.is_floating_point or t.dtype.is_complex:
        return t.dtype
    if isinstance(scalar, int) and not isinstance(scalar, bool):
        return t.dtype
    return torch.get_default_dtype()


def _launch_strided(t, out, n, scalar, is_int):
    shape = t.shape
    rank = len(shape)
    xs = t.stride()
    os_ = out.stride()
    pad = _MAX_DIMS - rank
    sh = [shape[i] for i in range(rank - 1, -1, -1)] + [1] * pad
    xsr = [xs[i] for i in range(rank - 1, -1, -1)] + [0] * pad
    osr = [os_[i] for i in range(rank - 1, -1, -1)] + [0] * pad
    grid = (triton.cdiv(n, _BLOCK),)
    _sub_strided[grid](
        t,
        out,
        n,
        float(scalar),
        int(scalar),
        *sh,
        *xsr,
        *osr,
        RANK=rank,
        IS_INT=is_int,
        BLOCK=_BLOCK,
        num_warps=4,
    )


def _fill(t, out, scalar, is_int):
    n = t.numel()
    if n == 0:
        return
    if _is_dense(t):
        grid = (triton.cdiv(n, _BLOCK),)
        i64 = n > 0x7FFFFFFF
        exact = (
            (n % _BLOCK == 0)
            and (t.data_ptr() & 15) == 0
            and (out.data_ptr() & 15) == 0
        )
        if is_int:
            _sub_flat_i[grid](
                t, out, n, int(scalar), BLOCK=_BLOCK, EXACT=exact, I64=i64, num_warps=4
            )
        else:
            _sub_flat_f[grid](
                t,
                out,
                n,
                float(scalar),
                BLOCK=_BLOCK,
                EXACT=exact,
                I64=i64,
                num_warps=4,
            )
    elif t.dim() <= _MAX_DIMS:
        _launch_strided(t, out, n, scalar, is_int)
    else:
        # rank above the kernel's static limit: recurse over leading-dim views
        for i in range(t.shape[0]):
            _fill(t[i], out[i], scalar, is_int)


def _single(t, scalar):
    out_dtype = _out_dtype(t, scalar)
    out = torch.empty_like(t, dtype=out_dtype)
    _fill(t, out, scalar, not out_dtype.is_floating_point)
    return out


def _launch_slot_group(chunk, scalar, is_int):
    """One fused launch for up to _FUSED dense tensors sharing in/out dtype."""
    starts = [0]
    for t, out, n in chunk:
        starts.append(starts[-1] + triton.cdiv(n, _BLOCK))
    total = starts[-1]
    exact = True
    for t, out, n in chunk:
        if (n % _BLOCK) or (t.data_ptr() & 15) or (out.data_ptr() & 15):
            exact = False
            break
    k = len(chunk)
    pad_x = chunk[0][0]
    pad_o = chunk[0][1]
    xs = [c[0] for c in chunk] + [pad_x] * (_FUSED - k)
    os_ = [c[1] for c in chunk] + [pad_o] * (_FUSED - k)
    ns = [c[2] for c in chunk] + [0] * (_FUSED - k)
    sb = starts[1:_FUSED]
    sb = sb + [total] * (_FUSED - 1 - len(sb))
    args = []
    for j in range(_FUSED):
        args.append(xs[j])
        args.append(os_[j])
        args.append(ns[j])
    args.extend(sb)
    args.append(int(scalar) if is_int else float(scalar))
    grid = (total,)
    if is_int:
        _sub_fused_i[grid](*args, BLOCK=_BLOCK, EXACT=exact, num_warps=4)
    else:
        _sub_fused_f[grid](*args, BLOCK=_BLOCK, EXACT=exact, num_warps=4)


def foreach_sub_scalar(self, scalar):
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_sub() received an invalid combination of arguments - got ("
            + type(self).__name__
            + ", "
            + type(scalar).__name__
            + "), but expected (tuple of Tensors self, Number scalar)"
        )
    tensors = list(self)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in tensors:
        if t.dtype == torch.bool:
            raise RuntimeError(_BOOL_ERR)
    if isinstance(scalar, (list, tuple)):
        if len(scalar) != len(tensors):
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
        return [_single(t, s) for t, s in zip(tensors, scalar)]

    outs = []
    groups = {}
    loose = []
    for t in tensors:
        out_dtype = _out_dtype(t, scalar)
        out = torch.empty_like(t, dtype=out_dtype)
        outs.append(out)
        n = t.numel()
        if n == 0:
            continue
        if n <= 0x7FFFFFFF and _is_dense(t):
            groups.setdefault((t.dtype, out_dtype), []).append((t, out, n))
        else:
            loose.append((t, out, n, out_dtype))
    for key, items in groups.items():
        is_int = not key[1].is_floating_point
        if len(items) == 1:
            t, out, n = items[0]
            _fill(t, out, scalar, is_int)
        else:
            for c in range(0, len(items), _FUSED):
                _launch_slot_group(items[c : c + _FUSED], scalar, is_int)
    for t, out, n, out_dtype in loose:
        _fill(t, out, scalar, not out_dtype.is_floating_point)
    return outs
