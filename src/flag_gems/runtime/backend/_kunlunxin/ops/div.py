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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn
div_rz = tl_extra_shim.div_rz
fmod = tl_extra_shim.fmod
trunc = tl_extra_shim.trunc
xpu_trunc_div = tl_extra_shim.xpu_trunc_div  # use it if we need to cmp result with xpu

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
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


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "DEFAULT"),
        (0, 1, 2, 3, "DEFAULT"),
    ],
)
@triton.jit
def div_complex_kernel(ar, ai, br, bi):
    # Compute in fp32 for both complex32 and complex64. Smith's method avoids
    # overflowing br*br + bi*bi and is stable when one denominator component is
    # much larger than the other.
    ar = ar.to(tl.float32)
    ai = ai.to(tl.float32)
    br = br.to(tl.float32)
    bi = bi.to(tl.float32)

    use_br = tl.abs(br) >= tl.abs(bi)
    safe_br = tl.where(br == 0, 1.0, br)
    safe_bi = tl.where(bi == 0, 1.0, bi)

    ratio_br = tl.where(br == 0, 0.0, bi / safe_br)
    denom_br = br + bi * ratio_br
    zero_denominator = (br == 0) & (bi == 0)
    real_br = (ar + ai * ratio_br) / tl.where(use_br, denom_br, 1.0)
    imag_br = (ai - ar * ratio_br) / tl.where(use_br, denom_br, 1.0)

    ratio_bi = tl.where(bi == 0, 0.0, br / safe_bi)
    denom_bi = bi + br * ratio_bi
    real_bi = (ar * ratio_bi + ai) / tl.where(use_br, 1.0, denom_bi)
    imag_bi = (ai * ratio_bi - ar) / tl.where(use_br, 1.0, denom_bi)

    real = tl.where(use_br, real_br, real_bi)
    imag = tl.where(use_br, imag_br, imag_bi)

    # The tensor CPU reference used by the accuracy suite returns a complex NaN
    # for an exact zero denominator (its vectorized complex path differs from
    # scalar division here).  Half-precision random tensors hit exact zero often
    # enough for large shapes, so make this edge explicit and deterministic.
    zero = tl.abs(br) + tl.abs(bi)
    nan = zero / zero
    return tl.where(zero_denominator, nan, real), tl.where(zero_denominator, nan, imag)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def div_complex_real_tensor_kernel(ar, ai, denominator):
    ar = ar.to(tl.float32)
    ai = ai.to(tl.float32)
    denominator = denominator.to(tl.float32)
    real = ar / denominator
    imag = ai / denominator

    # Match the vectorized CPU complex-tensor reference for an exact zero real
    # denominator (complex NaN in both channels).
    zero = tl.abs(denominator)
    nan = zero / zero
    is_zero = denominator == 0
    return tl.where(is_zero, nan, real), tl.where(is_zero, nan, imag)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def div_complex_real_scalar_kernel(ar, ai, denominator):
    # Keeping a Python scalar as a scalar avoids broadcasting two zero-dimensional
    # tensors through the rank-N pointwise wrapper. That path can trigger an XPU
    # strided-slice kernel exception for million-element complex tensors.
    return ar.to(tl.float32) / denominator, ai.to(tl.float32) / denominator


def _complex_components(value, dtype, device):
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            parts = torch.view_as_real(value.resolve_conj())
            return (
                parts[..., 0].to(dtype).contiguous(),
                parts[..., 1].to(dtype).contiguous(),
            )
        real = value.to(device=device, dtype=dtype)
        return real, torch.zeros_like(real)

    scalar = complex(value)
    return (
        torch.tensor(scalar.real, device=device, dtype=dtype),
        torch.tensor(scalar.imag, device=device, dtype=dtype),
    )


def _complex_divide(A, B):
    tensor = A if isinstance(A, torch.Tensor) else B
    device = tensor.device
    result_dtype = torch.result_type(A, B)
    component_dtype = (
        torch.float16 if result_dtype == torch.complex32 else torch.float32
    )
    if (
        isinstance(A, torch.Tensor)
        and A.is_complex()
        and not (
            (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(B, complex)
        )
    ):
        ar, ai = _complex_components(A, component_dtype, device)
        if isinstance(B, torch.Tensor):
            real, imag = div_complex_real_tensor_kernel(ar, ai, B)
        else:
            real, imag = div_complex_real_scalar_kernel(ar, ai, B)
        result = torch.view_as_complex(torch.stack((real, imag), dim=-1).contiguous())
        return result.to(result_dtype)

    ar, ai = _complex_components(A, component_dtype, device)
    br, bi = _complex_components(B, component_dtype, device)
    real, imag = div_complex_kernel(ar, ai, br, bi)
    result = torch.view_as_complex(torch.stack((real, imag), dim=-1).contiguous())
    return result.to(result_dtype)


def _check_complex_destination(A, B, destination):
    result_dtype = torch.result_type(A, B)
    if not torch.can_cast(result_dtype, destination.dtype):
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {destination.dtype}"
        )


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
            return torch.tensor(A / B)
        return _complex_divide(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_OUT")
    if (
        (isinstance(A, torch.Tensor) and A.is_complex())
        or isinstance(A, complex)
        or (isinstance(B, torch.Tensor) and B.is_complex())
        or isinstance(B, complex)
    ):
        if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
            return out.fill_(A / B)
        _check_complex_destination(A, B, out)
        out.copy_(_complex_divide(A, B))
        return out
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        # Both scalar
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_")
    if (
        A.is_complex()
        or (isinstance(B, torch.Tensor) and B.is_complex())
        or isinstance(B, complex)
    ):
        _check_complex_destination(A, B, A)
        A.copy_(_complex_divide(A, B))
        return A
    if isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return xpu_trunc_div(x, y)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return xpu_trunc_div(x, y)


@pointwise_dynamic(
    is_tensor=[False, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return xpu_trunc_div(x, y)


# Integer truncation division: Triton's // on integers is C-style (truncates toward zero)
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
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
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
        # Both scalar
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE_")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
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
    # Kunlunxin's libdevice fmod accepts fp32/fp64 only. The pointwise wrapper
    # casts the fp32 result back to the promoted output dtype.
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    # NOTE: fmod's sign is the same as the dividend
    remainder = fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    # NOTE: we have to use div_rn explicitly here
    q = div_rn(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out


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


def floor_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        # PyTorch treats a Python floating-point scalar as a wrapped scalar.  If
        # the tensor is fp16/bf16, the scalar is first rounded to that dtype and
        # the actual division is then evaluated in fp32.  Passing the Python
        # value directly as a Triton scalar skips that first rounding step (for
        # example, 0.001 remains 0.001 instead of fp16 0.001000404...), which
        # creates many off-by-one floor results.  Materialize the wrapped scalar
        # only for the low-precision cases that need this semantic distinction.
        if A.dtype in (torch.float16, torch.bfloat16) and isinstance(B, float):
            B = torch.tensor(B, dtype=A.dtype, device=A.device)
            return floor_div_func(A, B)
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        # Apply the same wrapped-scalar rule for scalar-first division.  Keeping
        # a Python float at fp32 precision while the tensor is fp16/bf16 can
        # change the integer selected by floor (for example, 0.001 // -0.0002).
        if B.dtype in (torch.float16, torch.bfloat16) and isinstance(A, float):
            A = torch.tensor(A, dtype=B.dtype, device=B.device)
            return floor_div_func(A, B)
        return floor_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    else:
        if A.dtype in (torch.float16, torch.bfloat16) and isinstance(B, float):
            B = torch.tensor(B, dtype=A.dtype, device=A.device)
            return floor_div_func(A, B, out0=A)
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
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
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
    logger.debug("GEMS_KUNLUNXIN REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)
