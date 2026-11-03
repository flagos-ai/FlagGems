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

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import broadcast_shapes
from flag_gems.utils.triton_lang_extension import trunc

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


def _to_kernel_contiguous(x: torch.Tensor, task_shape):
    if tuple(x.shape) != tuple(task_shape):
        x = x.expand(task_shape)
    if not x.is_contiguous():
        return x.contiguous()
    return x


def _prepare_out_of_place_operands(A, B):
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)):
        return A, B
    task_shape = broadcast_shapes([A.shape, B.shape])
    return _to_kernel_contiguous(A, task_shape), _to_kernel_contiguous(B, task_shape)


def _run_pointwise_inplace(kernel_fn, A, B):
    task_shape = broadcast_shapes([A.shape, B.shape])
    B = _to_kernel_contiguous(B, task_shape)
    if A.is_contiguous():
        return kernel_fn(A, B, out0=A)
    A_work = A.contiguous()
    kernel_fn(A_work, B, out0=A_work)
    A.copy_(A_work)
    return A


def _is_float64(*args):
    return any(isinstance(a, torch.Tensor) and a.dtype == torch.float64 for a in args)


def _cpu_div(A, B, rounding_mode=None):
    device = A.device if isinstance(A, torch.Tensor) else B.device
    A_cpu = A.cpu() if isinstance(A, torch.Tensor) else A
    B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
    return torch.div(A_cpu, B_cpu, rounding_mode=rounding_mode).to(device)


def _cpu_div_inplace(A, B, rounding_mode=None):
    A_cpu = A.cpu()
    B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
    result = torch.div(A_cpu, B_cpu, rounding_mode=rounding_mode)
    A.copy_(result)
    return A


def _cpu_remainder(A, B):
    device = A.device if isinstance(A, torch.Tensor) else B.device
    A_cpu = A.cpu() if isinstance(A, torch.Tensor) else A
    B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
    return torch.remainder(A_cpu, B_cpu).to(device)


def _cpu_remainder_inplace(A, B):
    A_cpu = A.cpu()
    B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
    result = torch.remainder(A_cpu, B_cpu)
    A.copy_(result)
    return A


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
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE")
    if _is_float64(A, B):
        return _cpu_div(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        A, B = _prepare_out_of_place_operands(A, B)
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE_")
    if _is_float64(A, B):
        return _cpu_div_inplace(A, B)
    if isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return _run_pointwise_inplace(true_div_func, A, B)
    else:
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return true_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _float_div_quotients(x, y):
    """Return ``(floor(x / y), trunc(x / y))`` for fp32 ``x`` and ``y``.

    gcu300's fp32 ``/`` is an approximate (fast) division and its libdevice
    div_rn/div_rz round the quotient to an integer, so neither can be used
    directly.  Refine ``x / y`` to the correctly rounded quotient with one FMA
    residual step, then truncate and correct floor from the fp32 remainder.
    """
    q0 = x / y
    residual = tl.math.fma(-y, q0, x)
    q = q0 + residual / y
    q = tl.where(y == 0.0, q0, q)

    trunc_q = trunc(q)
    # fma keeps the residual exact; `x - y * trunc_q` loses the low bits when
    # |q| is large, which flips the floor decision for quotients that sit within
    # half an ulp of an integer.
    remainder = tl.math.fma(-y, trunc_q, x)
    different_sign = (remainder < 0) != (y < 0)
    floor_q = tl.where((remainder != 0.0) & different_sign, trunc_q - 1.0, trunc_q)
    # The rounded quotient can also land one representable step *below* the true
    # one, which happens once |q| exceeds 2**24 and the step becomes 2. The
    # residual is exact, so `|remainder| >= |y|` proves the true quotient is at
    # least `trunc_q + 1`.
    overshoot = (remainder * y > 0.0) & (tl.abs(remainder) >= tl.abs(y))
    floor_q = tl.where(overshoot, trunc_q + 1.0, floor_q)
    return floor_q, trunc_q


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    # gcu300 has no fp16/bf16 div lowering: compute the float path in fp32 and
    # narrow the integer-valued result back to the input dtype.
    _, trunc_q = _float_div_quotients(x.to(tl.float32), y.to(tl.float32))
    return trunc_q.to(x.dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    _, trunc_q = _float_div_quotients(x.to(tl.float32), y.to(tl.float32))
    return trunc_q.to(x.dtype)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    _, trunc_q = _float_div_quotients(x.to(tl.float32), y.to(tl.float32))
    return trunc_q.to(y.dtype)


# Integer truncation division: Triton's // on integers truncates toward zero.
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
    if _is_float64(A, B):
        return _cpu_div(A, B, rounding_mode="trunc")
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A, B = _prepare_out_of_place_operands(A, B)
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.div(A, B, rounding_mode="trunc")


def trunc_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE_")
    if _is_float64(A, B):
        return _cpu_div_inplace(A, B, rounding_mode="trunc")
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return _run_pointwise_inplace(trunc_div_func, A, B)
    else:
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
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
    # return x
    x = x.to(tl.int32)
    y = y.to(tl.int32)
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


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
    # gcu300 has no fp16/bf16 div lowering: compute in fp32 and narrow the
    # integer-valued result back to the promoted input dtype.
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    y = y.to(tl.float32)

    floor_q, _ = _float_div_quotients(x, y)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out.to(orig_dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    elif x.type.scalar.is_floating() & y.type.scalar.is_int():
        return _float_floordiv(x, y.to(tl.float32))
    elif x.type.scalar.is_int() & y.type.scalar.is_floating():
        return _float_floordiv(x.to(tl.float32), y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    if _is_float64(A, B):
        return _cpu_div(A, B, rounding_mode="floor")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        A, B = _prepare_out_of_place_operands(A, B)
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return floor_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE_")
    if _is_float64(A, B):
        return _cpu_div_inplace(A, B, rounding_mode="floor")
    if isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return _run_pointwise_inplace(floor_div_func, A, B)
    else:
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return floor_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


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
    if _is_float64(A, B):
        return _cpu_remainder(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        A, B = _prepare_out_of_place_operands(A, B)
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return rem_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_ENFLAME REMAINDER_")
    if _is_float64(A, B):
        return _cpu_remainder_inplace(A, B)
    if isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return _run_pointwise_inplace(rem_tt, A, B)
    else:
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return rem_ts(A, B, out0=A)


import sys as _sys  # noqa: E402

# The generic div_mode path truncates with libdevice div_rn, which gcu300
# implements as round-to-nearest-integer instead of IEEE division, so trunc and
# floor results land one integer step away.  flag_gems.ops still exposes those
# generic helpers, so rebind them to the gcu300 implementations (same pattern as
# the _kunlunxin backend specializations).
_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(_generic_ops_module, "div_mode", div_mode)
    setattr(_generic_ops_module, "div_mode_", div_mode_)
