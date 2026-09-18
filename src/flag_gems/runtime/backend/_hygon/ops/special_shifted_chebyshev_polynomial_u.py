# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.ops.special_shifted_chebyshev_polynomial_u import (
    shifted_chebyshev_polynomial_u_kernel,
    shifted_chebyshev_polynomial_u_kernel_scalar_n,
)
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# On DCU the hardware v_sin_f32 instruction is ~2x faster than the libdevice
# sinf for this kernel's workload and stays within this op's f32 accuracy
# budget (tests use atol=5e-3). The generic implementation in
# flag_gems/ops/special_shifted_chebyshev_polynomial_u.py uses libdevice sin
# for portability; this Hygon override uses the hardware instruction.
#
# NB: AMD's v_sin_f32 computes sin(2*pi*x) -- the argument is in turns, not
# radians -- so scale by 1/(2*pi) first.
_INV_2PI: float = 0.15915494309189535


@triton.jit
def _hw_sin(x):
    return tl.inline_asm_elementwise(
        "v_sin_f32 $0, $1",
        "=v,v",
        [x * 0.15915494309189535],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cheby_u_math_hygon(x_f32, n_f32):
    # Same formula and boundary handling as the generic implementation:
    # U_n^*(x) = sin((n+1) * acos(2x-1)) / sin(acos(2x-1))
    x_shifted = x_f32 * 2.0 - 1.0
    x_shifted = tl.where(x_shifted > 1.0, 1.0, x_shifted)
    x_shifted = tl.where(x_shifted < -1.0, -1.0, x_shifted)

    acos_val = tl_extra_shim.acos(x_shifted)
    sin_acos = _hw_sin(acos_val)

    near_boundary = tl.abs(sin_acos) < 1e-6
    n_mod_2 = n_f32 - 2.0 * tl_extra_shim.floor(n_f32 / 2.0)
    is_odd = tl.abs(n_mod_2 - 1.0) < 0.5
    boundary_val = tl.where(
        x_shifted < 0.0, tl.where(is_odd, -1.0 - n_f32, n_f32 + 1.0), n_f32 + 1.0
    )

    numerator = _hw_sin((n_f32 + 1.0) * acos_val)
    result = numerator / sin_acos
    return tl.where(near_boundary, boundary_val, result)


@libentry()
@triton.jit
def _cheby_u_tt_kernel_hygon(
    x_ptr,
    n_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.load(n_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = _cheby_u_math_hygon(x, n)
    tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _cheby_u_ts_kernel_hygon(
    x_ptr,
    out_ptr,
    n_scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.full([BLOCK_SIZE], 0.0, tl.float32) + n_scalar
    r = _cheby_u_math_hygon(x, n)
    tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty), mask=mask)


def _cheby_u_can_fast(x, n):
    return x.is_contiguous() and (
        not isinstance(n, torch.Tensor)
        or (n.is_contiguous() and n.numel() == x.numel())
    )


def _cheby_u_fast_hygon(x, n, out):
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 4096),)
    if isinstance(n, torch.Tensor):
        _cheby_u_tt_kernel_hygon[grid](
            x, n, out, n_elements, BLOCK_SIZE=4096, num_warps=8
        )
    else:
        _cheby_u_ts_kernel_hygon[grid](
            x, out, float(n), n_elements, BLOCK_SIZE=4096, num_warps=8
        )
    return out


def special_shifted_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_HYGON SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_U")
    if x.dtype not in (torch.float32,):
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if _cheby_u_can_fast(x, n):
        return _cheby_u_fast_hygon(x, n, torch.empty_like(x))
    if not isinstance(n, torch.Tensor):
        return shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n)
    return shifted_chebyshev_polynomial_u_kernel(x, n)


def special_shifted_chebyshev_polynomial_u_(x, n):
    logger.debug("GEMS_HYGON SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_U_")
    if x.dtype not in (torch.float32,):
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if _cheby_u_can_fast(x, n):
        return _cheby_u_fast_hygon(x, n, x)
    if not isinstance(n, torch.Tensor):
        return shifted_chebyshev_polynomial_u_kernel_scalar_n(x, n, out0=x)
    return shifted_chebyshev_polynomial_u_kernel(x, n, out0=x)
