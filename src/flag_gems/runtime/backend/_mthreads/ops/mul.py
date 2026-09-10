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

"""
Specialized mul for mthreads backend.

Problem:
  The common op (src/flag_gems/ops/mul.py) handles complex multiplication via
  mul_complex_broadcast_func, which calls _complex_output to create an empty
  complex64 tensor. This triggers empty_kernel with dtype=complex64, but mthreads
  Triton does not support complex64 in canonicalize_dtype (KeyError: 'complex64').

Fix:
  Decompose complex multiplication into real/imag part operations directly,
  avoiding the creation of complex64 output tensors via empty_kernel.

  - complex * complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  - complex * real:    (a+bi)*c     = ac + bci
  - real * real:       delegate to common mul_broadcast_func (no change needed)
"""

import logging

import torch

from flag_gems.ops.mul import mul_broadcast_func

logger = logging.getLogger(__name__)


def _is_complex_op(A, B):
    """Check if either operand involves complex dtype."""
    if isinstance(A, torch.Tensor) and A.is_complex():
        return True
    if isinstance(B, torch.Tensor) and B.is_complex():
        return True
    if isinstance(A, complex) or isinstance(B, complex):
        return True
    return False


def _get_real_dtype(complex_dtype):
    """Map complex dtype to its corresponding real dtype."""
    if complex_dtype == torch.complex64:
        return torch.float32
    elif complex_dtype == torch.complex128:
        return torch.float64
    return torch.float32


def _complex_mul(A, B):
    """
    Complex multiplication decomposed into real/imag operations.

    Handles all combinations:
    1. complex tensor * complex tensor
    2. complex tensor * complex scalar
    3. complex tensor * real tensor
    4. complex tensor * real scalar (int/float)
    5. real tensor * complex tensor (symmetric cases)

    All paths avoid creating complex64 tensors via torch.empty/torch.tensor on
    device, which would trigger the unsupported empty_kernel path on mthreads.
    Instead, we decompose into real-valued arithmetic on the real/imag parts.
    """
    # Verify at least one operand is a tensor
    if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        raise TypeError("At least one operand must be a tensor")

    # Determine result complex dtype
    if isinstance(A, torch.Tensor) and A.is_complex():
        result_dtype = A.dtype
    elif isinstance(B, torch.Tensor) and B.is_complex():
        result_dtype = B.dtype
    else:
        result_dtype = torch.complex64

    real_dtype = _get_real_dtype(result_dtype)

    # Extract real/imag parts for A
    if isinstance(A, torch.Tensor) and A.is_complex():
        a_view = torch.view_as_real(A.to(dtype=result_dtype))
        ar, ai = a_view[..., 0], a_view[..., 1]
    elif isinstance(A, torch.Tensor):
        # Real tensor: imag part is zero
        ar = A.to(dtype=real_dtype)
        ai = torch.zeros_like(ar)
    elif isinstance(A, complex):
        # Complex scalar: extract real/imag as Python floats, broadcast later
        ar = A.real
        ai = A.imag
    else:
        # Real scalar (int/float): imag is zero
        ar = float(A)
        ai = 0.0

    # Extract real/imag parts for B
    if isinstance(B, torch.Tensor) and B.is_complex():
        b_view = torch.view_as_real(B.to(dtype=result_dtype))
        br, bi = b_view[..., 0], b_view[..., 1]
    elif isinstance(B, torch.Tensor):
        # Real tensor: imag part is zero
        br = B.to(dtype=real_dtype)
        bi = torch.zeros_like(br)
    elif isinstance(B, complex):
        # Complex scalar
        br = B.real
        bi = B.imag
    else:
        # Real scalar (int/float): imag is zero
        br = float(B)
        bi = 0.0

    # Optimized path: complex * real (imag part of one operand is zero)
    # (a+bi)*c = ac + bci  — avoids unnecessary multiplications
    if not isinstance(A, complex) and isinstance(A, (int, float)):
        # A is real scalar, B must be complex tensor
        out_r = ar * br - ai * bi
        out_i = ar * bi + ai * br
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    if not isinstance(B, complex) and isinstance(B, (int, float)):
        # B is real scalar: (a+bi)*c = ac + bci
        # ar, ai are tensors; br is float, bi is 0.0
        out_r = ar * br
        out_i = ai * br
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    if isinstance(B, torch.Tensor) and not B.is_complex():
        # B is real tensor: (a+bi)*c = ac + bci
        out_r = ar * br
        out_i = ai * br
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    if isinstance(A, torch.Tensor) and not A.is_complex():
        # A is real tensor: c*(b+di) = cb + cdi
        out_r = ar * br
        out_i = ar * bi
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    # General case: complex * complex
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    out_r = ar * br - ai * bi
    out_i = ar * bi + ai * br
    out = torch.stack((out_r, out_i), dim=-1)
    return torch.view_as_complex(out.contiguous()).to(result_dtype)


def mul(A, B, *, out=None):
    """
    mthreads specialized mul.

    For complex operations, decomposes into real/imag part arithmetic to avoid
    triggering empty_kernel with complex64 dtype (unsupported by mthreads Triton).
    For real*real, delegates to the common mul_broadcast_func.
    """
    logger.debug("GEMS_MTHREADS MUL")
    if isinstance(A, torch.Tensor) or isinstance(B, torch.Tensor):
        if _is_complex_op(A, B):
            result = _complex_mul(A, B)
            if out is not None:
                out.copy_(result)
                return out
            return result
        return mul_broadcast_func(A, B, out=out)
    return torch.tensor(A * B)


def mul_(A, B):
    """
    mthreads specialized mul_ (inplace).

    Same complex decomposition strategy as mul().
    """
    logger.debug("GEMS_MTHREADS MUL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("mul_ expects the first argument to be a tensor")
    if _is_complex_op(A, B):
        result = _complex_mul(A, B)
        A.copy_(result)
        return A
    return mul_broadcast_func(A, B, out=A)
