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

"""In-place foreach clamp_min (TensorList / ScalarList / Scalar / Tensor overloads).

foreach_clamp_min_list_(self, other) mutates every tensor in ``self`` as ``self[i] = max(self[i], bound_i)``
and returns ``None`` (ATen schema ``_foreach_clamp_min_`` -> ``()``).
"""

import struct

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


@triton.jit
def _clamp_op(x, y, IS_BOOL: tl.constexpr, IS_FLOAT: tl.constexpr):
    if IS_BOOL:
        r = x | y
    else:
        if IS_FLOAT:
            r = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
        else:
            r = tl.maximum(x, y)
    return r


@triton.jit
def _cast_clamp(x, y, IS_BOOL: tl.constexpr, IS_FLOAT: tl.constexpr, CD: tl.constexpr):
    # CD: 0 -> compute in x.dtype, 1 -> fp32, 2 -> fp64, 3 -> int64
    if IS_BOOL:
        r = x | (y != 0)
    else:
        if CD == 0:
            r = _clamp_op(x, y.to(x.dtype), IS_BOOL, IS_FLOAT)
        else:
            if CD == 1:
                r = _clamp_op(x.to(tl.float32), y.to(tl.float32), False, True).to(
                    x.dtype
                )
            else:
                if CD == 2:
                    r = _clamp_op(x.to(tl.float64), y.to(tl.float64), False, True).to(
                        x.dtype
                    )
                else:
                    r = _clamp_op(x.to(tl.int64), y.to(tl.int64), False, False).to(
                        x.dtype
                    )
    return r


# ---------------------------------------------------------------------------
# contiguous kernels (flat 1-D indexing)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["b"])
def _k_scalar_flat(
    x_ptr,
    n,
    b,
    IS_BOOL: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    B_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if IS_BOOL:
        y = b != 0
    else:
        if B_BITS:
            y = b.to(tl.float64, bitcast=True).to(x.dtype)
        else:
            y = b.to(x.dtype)
    r = _clamp_op(x, y, IS_BOOL, IS_FLOAT)
    tl.store(x_ptr + offs, r, mask=mask)


@triton.jit
def _k_scalar_tensor_flat(
    x_ptr,
    b_ptr,
    n,
    IS_BOOL: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    CD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(b_ptr)
    r = _cast_clamp(x, y, IS_BOOL, IS_FLOAT, CD)
    tl.store(x_ptr + offs, r, mask=mask)


@triton.jit
def _k_tensor_flat(
    x_ptr,
    b_ptr,
    n,
    IS_BOOL: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    CD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(b_ptr + offs, mask=mask)
    r = _cast_clamp(x, y, IS_BOOL, IS_FLOAT, CD)
    tl.store(x_ptr + offs, r, mask=mask)


# ---------------------------------------------------------------------------
# fused multi-tensor kernel: one launch handles up to 4 homogeneous pairs.
# Axis 0 = flat block index inside one tensor, axis 1 = tensor index (uniform
# per-CTA branch), so no global concatenated offset is ever formed.
# ---------------------------------------------------------------------------


@triton.jit
def _ft_body(xp, bp, offs, mask, IS_FLOAT: tl.constexpr):
    x = tl.load(xp + offs, mask=mask)
    y = tl.load(bp + offs, mask=mask)
    if IS_FLOAT:
        r = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.maximum(x, y)
    tl.store(xp + offs, r, mask=mask)


@triton.jit
def _fused_t(
    x0,
    b0,
    x1,
    b1,
    x2,
    b2,
    x3,
    b3,
    n,
    T: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(1)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if t == 0:
        _ft_body(x0, b0, offs, mask, IS_FLOAT)
    else:
        if T > 1:
            if t == 1:
                _ft_body(x1, b1, offs, mask, IS_FLOAT)
            else:
                if T > 2:
                    if t == 2:
                        _ft_body(x2, b2, offs, mask, IS_FLOAT)
                    else:
                        if T > 3:
                            _ft_body(x3, b3, offs, mask, IS_FLOAT)


@triton.jit
def _fs_body(xp, offs, mask, bnd, B_BITS: tl.constexpr, IS_FLOAT: tl.constexpr):
    x = tl.load(xp + offs, mask=mask)
    if B_BITS:
        y = bnd.to(tl.float64, bitcast=True).to(x.dtype)
    else:
        y = bnd.to(x.dtype)
    if IS_FLOAT:
        r = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    else:
        r = tl.maximum(x, y)
    tl.store(xp + offs, r, mask=mask)


@triton.jit(do_not_specialize=["b0", "b1", "b2", "b3"])
def _fused_s(
    x0,
    x1,
    x2,
    x3,
    n,
    b0,
    b1,
    b2,
    b3,
    T: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    B_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(1)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if t == 0:
        _fs_body(x0, offs, mask, b0, B_BITS, IS_FLOAT)
    else:
        if T > 1:
            if t == 1:
                _fs_body(x1, offs, mask, b1, B_BITS, IS_FLOAT)
            else:
                if T > 2:
                    if t == 2:
                        _fs_body(x2, offs, mask, b2, B_BITS, IS_FLOAT)
                    else:
                        if T > 3:
                            _fs_body(x3, offs, mask, b3, B_BITS, IS_FLOAT)


# ---------------------------------------------------------------------------
# strided kernel (collapsed rank, offset math for both operands)
# ---------------------------------------------------------------------------

_STRIDED_DNS = [
    "n",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "xs0",
    "xs1",
    "xs2",
    "xs3",
    "xs4",
    "xs5",
    "xs6",
    "xs7",
    "bs0",
    "bs1",
    "bs2",
    "bs3",
    "bs4",
    "bs5",
    "bs6",
    "bs7",
    "b_scalar",
]


@triton.jit(do_not_specialize=_STRIDED_DNS)
def _k_strided(
    x_ptr,
    b_ptr,
    n,
    b_scalar,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    USE_SCALAR: tl.constexpr,
    IS_BOOL: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    CD: tl.constexpr,
    B_BITS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < n
    rem = offs
    ox = tl.zeros([BLOCK], tl.int64)
    ob = tl.zeros([BLOCK], tl.int64)
    if RANK >= 8:
        t = rem % s7
        rem = rem // s7
        ox += t * xs7
        ob += t * bs7
    if RANK >= 7:
        t = rem % s6
        rem = rem // s6
        ox += t * xs6
        ob += t * bs6
    if RANK >= 6:
        t = rem % s5
        rem = rem // s5
        ox += t * xs5
        ob += t * bs5
    if RANK >= 5:
        t = rem % s4
        rem = rem // s4
        ox += t * xs4
        ob += t * bs4
    if RANK >= 4:
        t = rem % s3
        rem = rem // s3
        ox += t * xs3
        ob += t * bs3
    if RANK >= 3:
        t = rem % s2
        rem = rem // s2
        ox += t * xs2
        ob += t * bs2
    if RANK >= 2:
        t = rem % s1
        rem = rem // s1
        ox += t * xs1
        ob += t * bs1
    ox += rem * xs0
    x = tl.load(x_ptr + ox, mask=mask)
    if USE_SCALAR:
        if IS_BOOL:
            y = b_scalar != 0
        else:
            if B_BITS:
                y = b_scalar.to(tl.float64, bitcast=True).to(x.dtype)
            else:
                y = b_scalar.to(x.dtype)
        r = _clamp_op(x, y, IS_BOOL, IS_FLOAT)
    else:
        ob += rem * bs0
        y = tl.load(b_ptr + ob, mask=mask)
        r = _cast_clamp(x, y, IS_BOOL, IS_FLOAT, CD)
    tl.store(x_ptr + ox, r, mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

_MAX_RANK = 8


def _block_for(dtype):
    try:
        itemsize = dtype.itemsize
    except Exception:
        itemsize = 4
    if itemsize <= 2:
        return 2048, 4
    if itemsize == 4:
        return 1024, 4
    return 512, 4


def _is_scalar_bound(b):
    return not isinstance(b, torch.Tensor)


def _cd_code(xd, yd):
    if xd == yd:
        return 0
    try:
        p = torch.promote_types(xd, yd)
    except Exception:
        return 0
    if p == torch.float64:
        return 2
    if p == torch.float32:
        return 1
    if p == torch.int64:
        return 3
    return 0


def _scalar_arg(b, dtype):
    """Return (value, is_bitcast) for a python scalar bound."""
    if dtype == torch.float64:
        f = float(b)
        return struct.unpack("<q", struct.pack("<d", f))[0], True
    if dtype.is_floating_point:
        return float(b), False
    return int(b), False


def _collapse(shape, xs, bs):
    """Drop size-1 dims and merge adjacent dims valid for both operands."""
    dims = []
    for d in range(len(shape)):
        s = shape[d]
        if s == 1:
            continue
        dims.append([s, xs[d], bs[d]])
    i = len(dims) - 1
    while i > 0:
        s_in, x_in, b_in = dims[i]
        s_out, x_out, b_out = dims[i - 1]
        if x_out == x_in * s_in and b_out == b_in * s_in:
            dims[i - 1] = [s_out * s_in, x_in, b_in]
            del dims[i]
        i -= 1
    return dims


def _apply_one(x, b):
    n = x.numel()
    if n == 0:
        return

    is_bool = x.dtype == torch.bool
    if is_bool:
        try:
            if isinstance(b, torch.Tensor):
                if b.is_contiguous():
                    b = b.view(torch.uint8)
                else:
                    b = b.contiguous().view(torch.uint8)
            x = x.view(torch.uint8)
        except Exception:
            pass  # fall back to native bool pointers

    block, warps = _block_for(x.dtype)
    is_float = x.dtype.is_floating_point
    grid = (triton.cdiv(n, block),)

    if _is_scalar_bound(b):
        bval, b_bits = _scalar_arg(b, x.dtype) if not is_bool else (int(bool(b)), False)
        if x.is_contiguous():
            _k_scalar_flat[grid](
                x,
                n,
                bval,
                IS_BOOL=is_bool,
                IS_FLOAT=is_float,
                B_BITS=b_bits,
                BLOCK=block,
                num_warps=warps,
            )
            return
        xshape = x.shape
        xst = x.stride()
        dims = _collapse(xshape, xst, (0,) * len(xshape))
        if len(dims) > _MAX_RANK:
            raise RuntimeError("foreach_clamp_min_: rank too large")
        _launch_strided(
            x, None, n, bval, dims, True, is_bool, is_float, 0, b_bits, block, warps
        )
        return

    if not isinstance(b, torch.Tensor):
        return

    if is_bool:
        try:
            b = b.view(torch.uint8)
        except Exception:
            pass

    cd = _cd_code(x.dtype, b.dtype)

    if b.numel() == 1:
        if x.is_contiguous():
            _k_scalar_tensor_flat[grid](
                x,
                b,
                n,
                IS_BOOL=is_bool,
                IS_FLOAT=is_float,
                CD=cd,
                BLOCK=block,
                num_warps=warps,
            )
            return
    elif x.is_contiguous() and b.is_contiguous() and b.shape == x.shape:
        _k_tensor_flat[grid](
            x,
            b,
            n,
            IS_BOOL=is_bool,
            IS_FLOAT=is_float,
            CD=cd,
            BLOCK=block,
            num_warps=warps,
        )
        return

    be = b.expand(x.shape)
    dims = _collapse(x.shape, x.stride(), be.stride())
    if len(dims) > _MAX_RANK:
        raise RuntimeError("foreach_clamp_min_: rank too large")
    _launch_strided(
        x, be, n, 0.0, dims, False, is_bool, is_float, cd, False, block, warps
    )


def _launch_strided(
    x, be, n, bval, dims, use_scalar, is_bool, is_float, cd, b_bits, block, warps
):
    shape_extra = [1] * 7
    for i in range(1, len(dims)):
        shape_extra[i - 1] = dims[i][0]
    xs = [0] * 8
    bs = [0] * 8
    for i, d in enumerate(dims):
        xs[i] = d[1]
        bs[i] = d[2]
    grid = (triton.cdiv(n, block),)
    _k_strided[grid](
        x,
        x if be is None else be,
        n,
        bval,
        *shape_extra,
        *xs,
        *bs,
        USE_SCALAR=use_scalar,
        IS_BOOL=is_bool,
        IS_FLOAT=is_float,
        CD=cd,
        B_BITS=b_bits,
        RANK=len(dims),
        BLOCK=block,
        num_warps=warps,
    )


_PLAN_CACHE = {}


def _plan_for(dtype):
    plan = _PLAN_CACHE.get(dtype)
    if plan is None:
        plan = _block_for(dtype)
        _PLAN_CACHE[dtype] = plan
    return plan


def _fused_tensor_list(tensors, bounds):
    """One launch for 2..4 homogeneous contiguous tensor-bound pairs."""
    x0 = tensors[0]
    dtype = x0.dtype
    if dtype is torch.bool:
        return False
    if not (dtype.is_floating_point or dtype.is_signed):
        return False
    n = x0.numel()
    if n == 0 or not x0.is_contiguous():
        return False
    t_count = len(tensors)
    for i in range(1, t_count):
        x = tensors[i]
        if x.dtype is not dtype or x.numel() != n or not x.is_contiguous():
            return False
    b0 = bounds[0]
    if not isinstance(b0, torch.Tensor):
        return False
    if b0.dtype is not dtype or not b0.is_contiguous() or b0.shape != x0.shape:
        return False
    for i in range(1, t_count):
        b = bounds[i]
        if not isinstance(b, torch.Tensor):
            return False
        if b.dtype is not dtype or not b.is_contiguous() or b.shape != tensors[i].shape:
            return False

    block, warps = _plan_for(dtype)
    grid = ((n + block - 1) // block, t_count)
    is_float = dtype.is_floating_point
    if t_count == 2:
        _fused_t[grid](
            x0,
            b0,
            tensors[1],
            bounds[1],
            x0,
            b0,
            x0,
            b0,
            n,
            2,
            is_float,
            block,
            num_warps=warps,
        )
    elif t_count == 3:
        _fused_t[grid](
            x0,
            b0,
            tensors[1],
            bounds[1],
            tensors[2],
            bounds[2],
            x0,
            b0,
            n,
            3,
            is_float,
            block,
            num_warps=warps,
        )
    else:
        _fused_t[grid](
            x0,
            b0,
            tensors[1],
            bounds[1],
            tensors[2],
            bounds[2],
            tensors[3],
            bounds[3],
            n,
            4,
            is_float,
            block,
            num_warps=warps,
        )
    return True


def _fused_scalar_list(tensors, bounds):
    """One launch for 2..4 homogeneous contiguous pairs with python scalars."""
    x0 = tensors[0]
    dtype = x0.dtype
    if dtype is torch.bool:
        return False
    is_float = dtype.is_floating_point
    if not (is_float or dtype.is_signed):
        return False
    n = x0.numel()
    if n == 0 or not x0.is_contiguous():
        return False
    t_count = len(tensors)
    for i in range(1, t_count):
        x = tensors[i]
        if x.dtype is not dtype or x.numel() != n or not x.is_contiguous():
            return False
    vals = []
    if dtype is torch.float64:
        for b in bounds:
            if isinstance(b, torch.Tensor):
                return False
            vals.append(struct.unpack("<q", struct.pack("<d", float(b)))[0])
        b_bits = True
    elif is_float:
        for b in bounds:
            if isinstance(b, torch.Tensor):
                return False
            vals.append(float(b))
        b_bits = False
    else:
        for b in bounds:
            if isinstance(b, torch.Tensor):
                return False
            vals.append(int(b))
        b_bits = False

    block, warps = _plan_for(dtype)
    grid = ((n + block - 1) // block, t_count)
    if t_count == 2:
        _fused_s[grid](
            x0,
            tensors[1],
            x0,
            x0,
            n,
            vals[0],
            vals[1],
            vals[0],
            vals[1],
            2,
            is_float,
            b_bits,
            block,
            num_warps=warps,
        )
    elif t_count == 3:
        _fused_s[grid](
            x0,
            tensors[1],
            tensors[2],
            x0,
            n,
            vals[0],
            vals[1],
            vals[2],
            vals[0],
            3,
            is_float,
            b_bits,
            block,
            num_warps=warps,
        )
    else:
        _fused_s[grid](
            x0,
            tensors[1],
            tensors[2],
            tensors[3],
            n,
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            4,
            is_float,
            b_bits,
            block,
            num_warps=warps,
        )
    return True


def foreach_clamp_min_list_(self, other):
    tensors = self if type(self) is list else list(self)
    n_t = len(tensors)
    if n_t == 0:
        return None
    if isinstance(other, (list, tuple)):
        if len(other) != n_t:
            raise RuntimeError(
                "Expected lists of tensors of same length, but found list 0 of "
                "length %d and list 1 of length %d" % (n_t, len(other))
            )
        if 2 <= n_t <= 4:
            if _fused_tensor_list(tensors, other):
                return None
            if _fused_scalar_list(tensors, other):
                return None
        for i in range(n_t):
            _apply_one(tensors[i], other[i])
        return None

    if isinstance(other, torch.Tensor) and 2 <= n_t <= 4:
        if _fused_tensor_list(tensors, (other,) * n_t):
            return None
    for t in tensors:
        _apply_one(t, other)
    return None
