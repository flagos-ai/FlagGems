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

import logging
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.triton_lang_extension import div_rn, trunc

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# libdevice fma on gcu400; used to get the exact division residual.
fma = tl_extra_shim.fma


@triton.jit
def _rne_div(x, y):
    # gcu400's f32 `/` and libdevice div_rn are only accurate to ~1-2 ulp,
    # which flips trunc() when the exact quotient is within a few ulp of an
    # integer (e.g. 1.5639446 / 0.3909861 is exactly 4.0 but divides to
    # 3.9999998).  One Newton step on the exact FMA residual restores IEEE
    # round-to-nearest, matching PyTorch's trunc_divide semantics.
    q = div_rn(x, y)
    # Skip the residual for non-finite quotients (x/0 -> +-inf, 0/0 -> nan):
    # fma(-inf, 0, x) is nan and would clobber the correct inf/nan result.
    refined = q + fma(-q, y, x) / y
    return tl.where(q - q == 0.0, refined, q)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


def true_divide(A, B):
    logger.debug("GEMS TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_(A, B):
    logger.debug("GEMS TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return trunc(_rne_div(x, y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc(_rne_div(x, tl.cast(y, x.dtype)))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc(_rne_div(tl.cast(x, y.dtype), y))


# Integer truncation division: Triton's // on integers is C-style (truncates
# toward zero), matching aten.
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y):
    return x // y


def trunc_divide(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE")
    # Integer types: dedicated int kernels (Triton // is C-style truncation)
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        return _trunc_divide_scalars(A, B)


def _trunc_divide_scalars(A, B):
    """Both operands are Python/NumPy scalars: mirror aten's wrapped-number
    semantics for ``div(..., rounding_mode="trunc")``."""
    if isinstance(A, bool) or isinstance(B, bool):
        raise NotImplementedError("\"div_trunc_cpu\" not implemented for 'Bool'")
    if isinstance(A, numbers.Integral) and isinstance(B, numbers.Integral):
        if B == 0:
            raise RuntimeError("ZeroDivisionError")
        quot = abs(int(A)) // abs(int(B))
        return torch.tensor(-quot if (A < 0) != (B < 0) else quot)
    a = torch.tensor(float(A), dtype=torch.get_default_dtype())
    b = torch.tensor(float(B), dtype=torch.get_default_dtype())
    return torch.trunc(a / b)


def trunc_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE_")
    # Integer types: dedicated int kernels (Triton // is C-style truncation)
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    # TODO: request Triton to add an integer remainder builtin
    # The semantic of Triton floordiv differs from Pytorch/Numpy
    # Triton floordiv equates to
    #     (x - np.fmod(x, y)) / y
    # whereas Pytorch floordiv is
    #     (x - np.remainder(x, y)) y
    # The results show a one off difference when
    #     C1) x and y have opposite signs
    # and C2) x is not multiples of y.
    # Apart from the above, there's an erroneous case x // 0 returns -1
    # whereas in Pytorch x // 0 returns -1 if x >=0 and -2 if x < 0
    # but this special case is coalesced into the c1 and c2 check so
    # there's extra handling.
    orig_dtype = x.dtype
    if orig_dtype == tl.int8 or orig_dtype == tl.int16:
        # Narrow integer division is lowered through i32 and cast back with
        # saturation, so INT_MIN // -1 becomes INT_MAX on this backend.
        # PyTorch wraps that single overflowing quotient back to INT_MIN
        # (which is x itself); it is the only quotient above the dtype max.
        xw = x.to(tl.int32)
        yw = y.to(tl.int32)
    else:
        xw = x
        yw = y
    r = xw % yw
    c1 = r != 0
    c2 = (xw < 0) ^ (yw < 0)
    q = tl.where(c1 & c2, xw // yw - 1, xw // yw)
    if orig_dtype == tl.int8 or orig_dtype == tl.int16:
        if orig_dtype == tl.int8:
            limit = 128
        else:
            limit = 32768
        q = tl.where(q >= limit, xw, q)
        res = q.to(orig_dtype)
    else:
        res = q
    return res


@triton.jit
def _trunc_to_int(q):
    return tl.where(q >= 0.0, tl.math.floor(q), tl.math.ceil(q))


@triton.jit
def _exact_fmod(x, y):
    """IEEE-exact fmod built from ``fma``.

    gcu400's libdevice ``fmod`` is not exact once x/y is large: for
    ``x=1.47, y=1.39e-7`` it returns -7.7e-7 instead of 1.09e-7 (and even
    violates ``|r| < |y|``), which moves ``floor(x/y)`` by several units.
    ``fma`` is exact on this device (verified bit-for-bit against an fp64
    reference), so recover the integral quotient ``k`` one step at a time and
    keep ``r = x - k*y`` exact via fma.  Each reduction removes an approximate
    quotient <= a few ulp, so two passes plus a magnitude safety net leave
    ``|r| < |y|``.
    """
    k = _trunc_to_int(div_rn(x, y))
    r = fma(-k, y, x)
    k = _trunc_to_int(div_rn(r, y))
    r = fma(-k, y, r)
    k = _trunc_to_int(div_rn(r, y))
    r = fma(-k, y, r)

    ay = tl.abs(y)
    ar = tl.abs(r)
    adj = tl.where(ar >= ay, tl.where(r >= 0.0, ay, -ay), 0.0)
    r = r - adj
    ar = tl.abs(r)
    adj = tl.where(ar >= ay, tl.where(r >= 0.0, ay, -ay), 0.0)
    r = r - adj

    # fmod's result carries the dividend's sign.  When the true quotient sits
    # just below an integer, div_rn can round up and leave a tiny r of the
    # wrong sign; the canonical remainder is then r + sign(x)*|y|.
    wrong_sign = (r != 0.0) & ((r < 0.0) != (x < 0.0))
    return tl.where(wrong_sign, r + tl.where(x < 0.0, -ay, ay), r)


# TO be consistent with python, numpy and torch, we have to implement it in the
# following way.
# CPython
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636
# numpy
# https://github.com/numpy/numpy/blob/a4ad142aa1282a77bbb05acd706cb57c9cc29846/numpy/_core/src/npymath/npy_math_internal.h.src#L532
# torch
# https://github.com/pytorch/pytorch/blob/d6d9183456cd07ca0b361a194b98c2fb196e7c36/c10/util/generic_math.h#L23
@triton.jit
def _float_floordiv(x, y):
    # gcu400: triton_gcu's fmod/div_rn only expose signatures for matching
    # fp32/fp64 operands.  A tensor+scalar pair can arrive as (fp16, fp32)
    # and used to raise KeyError in libdevice's arg_type_symbol_dict.
    # Cast both operands to fp32 first, then restore the input dtype.
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    remainder = _exact_fmod(x, y)
    imperfect = remainder != 0.0
    # A bool `^` between a tensor and a scalar-broadcast operand miscompiles in
    # the tensor/scalar specialisation: `(x < 0) ^ (y < 0)` collapses to
    # `x < 0`, which drops the divisor's sign and returns only {-1, 0, 1}
    # (floor_divide(x, -2.5) yielded {-1, 0, 1} instead of the quotient).  The
    # int32 form of the same test is exact on this backend.
    different_sign = ((x < 0.0).to(tl.int32) ^ (y < 0.0).to(tl.int32)) != 0

    # (x - remainder) / y is exact in real arithmetic; one Newton step on the
    # fma residual turns the ~1-2 ulp div_rn into the correctly rounded
    # division the reference uses.
    q = _rne_div(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out.to(orig_dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


# ---------------------------------------------------------------------------
# Flat floor-division fast path.
#
# ``pointwise_dynamic`` picks a 64K-element tile and caps the grid at 48 CTAs,
# which is close to optimal for the grid, but its per-call wrapping costs an
# extra pass and its tile is far too wide for the float kernel.  A dedicated
# 1-D launch over dense storage measured, on the benchmark shapes:
#
#   dtype / shape        pointwise   flat(48 CTAs)   native
#   int8   4096x4096       471us        284us         204us
#   int16  4096x4096       568us        173us         188us
#   int32  4096x4096       221us        156us         411us
#   int16  1G              24.0ms       12.3ms        11.0ms
#   int32  1G              10.7ms       11.1ms        25.0ms
#   fp32   4096x4096      6815us       4172us         432us
#
# The kernel bodies reuse ``_int_floordiv`` / ``_float_floordiv`` unchanged, so
# the output is bit-identical to the pointwise path -- verified over magnitude
# pairs from 1e-30 to 1e30 (including overflow to inf/nan) by comparing the
# raw bit patterns, plus randn, exact multiples, and quotients sitting 1 ulp
# off an integer.
#
# The float kernel is still ~10x behind aten because a single correctly rounded
# ``div_rn`` costs ~1200us on 16M elements here (12x a plain ``/``) and exact
# floor division needs several of them.  That is a property of the backend's
# libdevice lowering, not of this kernel; the cheaper substitutes (plain ``/``
# plus Newton refinement, or a single ``div_rn`` plus exact-``fma`` magnitude
# adjustments) were each measured and disagree with aten on 0.2-5% of inputs.
# ---------------------------------------------------------------------------

_FLAT_BLOCK_INT = 4096
_FLAT_BLOCK_FLOAT = 2048
# 48 CTAs measured fastest for every dtype and shape tried; larger grids lost
# 10-25%.  Kept under the 65535 grid.x limit by the grid-stride loop.
_FLAT_GRID = 48
_FLAT_NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _floor_div_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        if x.type.scalar.is_int() & y.type.scalar.is_int():
            result = _int_floordiv(x, y)
        else:
            result = _float_floordiv(x, y)
        tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "y"])
def _floor_div_scalar_flat_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    y,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # The Python side only takes this path when the scalar is exactly
        # representable in the element type, so this cast never rounds.  It has
        # to happen: letting the scalar arrive as int64 forces gcu400's
        # emulated 64-bit integer division, which measured 366ms for an int16
        # 4096x4096 tensor against 0.17ms in the element width.
        scalar = y if x.type.scalar.is_int() else y.to(tl.float32)
        if x.type.scalar.is_int():
            result = _int_floordiv(x, tl.full(x.shape, 0, x.dtype) + scalar)
        else:
            result = _float_floordiv(x, scalar)
        tl.store(out_ptr + offsets, result, mask=mask)


def _dense(result_dtype, *tensors):
    """Whether every listed operand is dense and already at ``result_dtype``."""
    for tensor in tensors:
        if tensor is None:
            continue
        if not tensor.is_contiguous() or tensor.dtype != result_dtype:
            return False
    return True


def _int_scalar_fits(value, dtype):
    """Whether a Python int survives an exact cast to ``dtype``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return False
    info = torch.iinfo(dtype)
    return info.min <= int(value) <= info.max


def _try_flat_tensor(A, B, out):
    """Launch the flat kernel, or return None to keep the pointwise path."""
    dtype = torch.promote_types(A.dtype, B.dtype)
    if dtype.is_complex or dtype == torch.bool:
        return None
    if dtype.is_floating_point and dtype.itemsize > 4:
        # fp64 arrives silently downconverted on this backend anyway.
        return None
    if not _dense(dtype, A, B, out):
        return None
    # The flat kernel indexes both operands linearly, so it cannot broadcast.
    # A 0-d operand against a larger tensor (floor_divide(scalar_tensor, x))
    # has to keep the pointwise path, which does.
    if A.shape != B.shape:
        return None
    n_elements = A.numel()
    buffer = out if out is not None else torch.empty_like(A)
    if n_elements == 0:
        return buffer
    BLOCK = _FLAT_BLOCK_FLOAT if dtype.is_floating_point else _FLAT_BLOCK_INT
    grid_size = min(triton.cdiv(n_elements, BLOCK), _FLAT_GRID)
    _floor_div_flat_kernel[(grid_size,)](
        A,
        B,
        buffer,
        n_elements,
        BLOCK=BLOCK,
        GRID_SIZE=grid_size,
        num_warps=_FLAT_NUM_WARPS,
    )
    return buffer


def _try_flat_scalar(A, B, out):
    """Launch the flat kernel for a dense tensor and a Python scalar."""
    dtype = A.dtype
    if dtype.is_complex or dtype == torch.bool:
        return None
    if dtype.is_floating_point:
        if dtype.itemsize > 4 or not isinstance(B, float):
            return None
        block = _FLAT_BLOCK_FLOAT
        scalar = float(B)
    else:
        if not _int_scalar_fits(B, dtype):
            return None
        block = _FLAT_BLOCK_INT
        scalar = int(B)
    if not _dense(dtype, A, out):
        return None
    n_elements = A.numel()
    buffer = out if out is not None else torch.empty_like(A)
    if n_elements == 0:
        return buffer
    grid_size = min(triton.cdiv(n_elements, block), _FLAT_GRID)
    _floor_div_scalar_flat_kernel[(grid_size,)](
        A,
        buffer,
        n_elements,
        scalar,
        BLOCK=block,
        GRID_SIZE=grid_size,
        num_warps=_FLAT_NUM_WARPS,
    )
    return buffer


def floor_divide(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    # gcu400: pointwise_dynamic allocates its output with ``empty_like`` of the
    # first input, so a non-contiguous input makes the store use a
    # non-contiguous block_ptr.  The backend's block_ptr store lowering writes
    # the wrong addresses for several stride orders (see
    # docs/bug_reports/gcu400_blockptr_store_nondense.md), which corrupts the
    # result on every pointwise op.  Copy non-contiguous operands to dense
    # storage first: the returned tensor is contiguous, but correct.
    if isinstance(A, torch.Tensor) and not A.is_contiguous():
        A = A.contiguous()
    if isinstance(B, torch.Tensor) and not B.is_contiguous():
        B = B.contiguous()
    # gcu400: pointwise_dynamic promotes two tensors but leaves a scalar or a
    # 0-d tensor at its own dtype.  A mixed int/float pair then reaches the
    # kernel as (int32, float32), where _int_floordiv's `x // y` raises
    # "unexpected type fp32" and _float_floordiv's fmod has no mixed-type
    # signature.  Normalise with torch's promotion rules before dispatch.
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(A, B, None)
        if flat is not None:
            return flat
        if A.dtype != B.dtype:
            dtype = torch.promote_types(A.dtype, B.dtype)
            if A.dtype != dtype:
                A = A.to(dtype)
            if B.dtype != dtype:
                B = B.to(dtype)
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        # The scalar specialisation of _float_floordiv used to be wrong on this
        # backend for a Python-float divisor (measured: floor_divide(x, -2.5)
        # returned only {-1, 0, 1}); that was the tensor/scalar bool `^`
        # miscompile fixed in _float_floordiv.  Keep the scalar in the kernel
        # rather than materialising a 0-d device tensor: the operand comes
        # through as fp32 (or int64 for an integer scalar), so no widening of
        # the input is needed, and dropping the extra tensor plus the
        # broadcasted tensor/tensor kernel took the 4096x4096 int cases from
        # 4.8ms/11.5ms to 0.24ms/0.15ms.
        flat = _try_flat_scalar(A, B, None)
        if flat is not None:
            return flat
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        A = torch.as_tensor(A, dtype=B.dtype, device=B.device)
        return floor_div_func(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE_")
    if isinstance(A, torch.Tensor) and not A.is_contiguous():
        # Same block_ptr store issue as floor_divide: writing into a
        # non-contiguous destination corrupts the result, so compute into a
        # dense buffer and copy back into the view.
        A.copy_(floor_divide(A, B))
        return A
    if isinstance(B, torch.Tensor) and not B.is_contiguous():
        B = B.contiguous()
    if isinstance(B, torch.Tensor):
        flat = _try_flat_tensor(A, B, A)
        if flat is not None:
            return flat
        return floor_div_func(A, B, out0=A)
    flat = _try_flat_scalar(A, B, A)
    if flat is not None:
        return flat
    return floor_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    # trunc and floor both need the gcu400-specific IEEE arithmetic above;
    # only None (plain true division) keeps the generic implementation.
    if rounding_mode == "trunc":
        return trunc_divide(A, B)
    if rounding_mode == "floor":
        return floor_divide(A, B)
    from flag_gems.ops.div import div_mode as _generic_div_mode

    return _generic_div_mode(A, B, rounding_mode)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode == "trunc":
        return trunc_divide_(A, B)
    if rounding_mode == "floor":
        return floor_divide_(A, B)
    from flag_gems.ops.div import div_mode_ as _generic_div_mode_

    return _generic_div_mode_(A, B, rounding_mode)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        return rem_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_ENFLAME REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)


import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(_generic_ops_module, "div_mode", div_mode)
    setattr(_generic_ops_module, "div_mode_", div_mode_)
