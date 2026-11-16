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

"""Kunlunxin override for ``aten::poisson_nll_loss``.

The generic implementation handles the complex dtype through a
``@pointwise_dynamic``-generated kernel.  On XPU3 that template fails to lower
for large element counts (``65537``) because ``TritonXPUUnrollControl`` refuses
to narrow a hoisted non-splat operand of the unrolled region
(``_complex_loss_kernel_..._t512``).  Every real dtype path is fine.

This override keeps the generic real-dtype path untouched and replaces *only*
the complex branch with hand-written, fixed-block real (``real``/``imag``) two
lane kernels, so no large-tile unroll is generated.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_
from flag_gems.ops.poisson_nll_loss import (
    _broadcast_metadata,
    _broadcast_offset,
    _complex_parts,
    _result_dtype,
)
from flag_gems.ops.poisson_nll_loss import poisson_nll_loss as _generic_poisson_nll_loss
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_PARTIAL_BLOCK_SIZE = 4096


@triton.jit
def _complex_loss_value(
    xr,
    xi,
    yr,
    yi,
    eps,
    LOG_INPUT: tl.constexpr,
    COMPLEX_INPUT: tl.constexpr,
    FULL: tl.constexpr,
):
    if LOG_INPUT:
        ex = tl_extra_shim.exp(xr)
        real = ex * tl_extra_shim.cos(xi) - (yr * xr - yi * xi)
        imag = ex * tl_extra_shim.sin(xi) - (yr * xi + yi * xr)
    else:
        shifted = xr + eps
        if COMPLEX_INPUT:
            scale = tl.maximum(tl.abs(shifted), tl.abs(xi))
            safe_scale = tl.where(scale == 0, 1, scale)
            a = shifted / safe_scale
            b = xi / safe_scale
            lr = tl_extra_shim.log(safe_scale) + 0.5 * tl_extra_shim.log(a * a + b * b)
            li = tl_extra_shim.atan2(xi, shifted)
        else:
            lr = tl_extra_shim.log(shifted)
            li = tl.full(xr.shape, 0, xr.dtype)
        real = xr - (yr * lr - yi * li)
        imag = xi - (yr * li + yi * lr)
    if FULL:
        stirling = (
            yr * tl_extra_shim.log(yr)
            - yr
            + 0.5 * tl_extra_shim.log(2.0 * math.pi * yr)
        )
        real = real + tl.where(yr > 1, stirling, 0.0)
    return real, imag


@libentry()
@triton.jit
def _complex_loss_none_kernel(
    xr_ptr,
    xi_ptr,
    yr_ptr,
    yi_ptr,
    out_r_ptr,
    out_i_ptr,
    n_elements,
    eps,
    OUT_SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    LINEAR: tl.constexpr,
    LOG_INPUT: tl.constexpr,
    COMPLEX_INPUT: tl.constexpr,
    FULL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Clamped unmasked loads: a masked tail load returns garbage on this
    # backend, so read a valid element and drop it at the store mask.
    safe_offsets = tl.minimum(tl.maximum(offsets, 0), n_elements - 1)
    if LINEAR:
        x_offsets = safe_offsets
        y_offsets = safe_offsets
    else:
        x_offsets = _broadcast_offset(safe_offsets, OUT_SHAPE, X_STRIDES)
        y_offsets = _broadcast_offset(safe_offsets, OUT_SHAPE, Y_STRIDES)
    xr = tl.load(xr_ptr + x_offsets)
    xi = tl.load(xi_ptr + x_offsets)
    yr = tl.load(yr_ptr + y_offsets)
    yi = tl.load(yi_ptr + y_offsets)
    real, imag = _complex_loss_value(xr, xi, yr, yi, eps, LOG_INPUT, COMPLEX_INPUT, FULL)
    tl.store(out_r_ptr + offsets, real, mask=mask)
    tl.store(out_i_ptr + offsets, imag, mask=mask)


@libentry()
@triton.jit
def _complex_loss_partial_kernel(
    xr_ptr,
    xi_ptr,
    yr_ptr,
    yi_ptr,
    partial_r_ptr,
    partial_i_ptr,
    n_elements,
    eps,
    OUT_SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    LINEAR: tl.constexpr,
    LOG_INPUT: tl.constexpr,
    COMPLEX_INPUT: tl.constexpr,
    FULL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    safe_offsets = tl.minimum(tl.maximum(offsets, 0), n_elements - 1)
    if LINEAR:
        x_offsets = safe_offsets
        y_offsets = safe_offsets
    else:
        x_offsets = _broadcast_offset(safe_offsets, OUT_SHAPE, X_STRIDES)
        y_offsets = _broadcast_offset(safe_offsets, OUT_SHAPE, Y_STRIDES)
    xr = tl.load(xr_ptr + x_offsets)
    xi = tl.load(xi_ptr + x_offsets)
    yr = tl.load(yr_ptr + y_offsets)
    yi = tl.load(yi_ptr + y_offsets)
    real, imag = _complex_loss_value(xr, xi, yr, yi, eps, LOG_INPUT, COMPLEX_INPUT, FULL)
    pid = tl.program_id(0)
    tl.store(partial_r_ptr + pid, tl.sum(tl.where(mask, real, 0.0)))
    tl.store(partial_i_ptr + pid, tl.sum(tl.where(mask, imag, 0.0)))


@libentry()
@triton.jit
def _complex_loss_final_kernel(
    partial_r_ptr,
    partial_i_ptr,
    out_r_ptr,
    out_i_ptr,
    partial_size,
    n_elements,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < partial_size
    total_r = tl.sum(tl.load(partial_r_ptr + offsets, mask=mask, other=0.0))
    total_i = tl.sum(tl.load(partial_i_ptr + offsets, mask=mask, other=0.0))
    if REDUCTION == 1:
        total_r /= n_elements
        total_i /= n_elements
    tl.store(out_r_ptr, total_r)
    tl.store(out_i_ptr, total_i)


def _complex_poisson(input, target, dtype, log_input, full, eps, reduction):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(input, real_dtype)
    yr, yi = _complex_parts(target, real_dtype)
    output_shape, x_strides, y_strides = _broadcast_metadata(xr, yr)
    n_elements = math.prod(output_shape)
    linear = (
        tuple(xr.shape) == output_shape
        and tuple(yr.shape) == output_shape
        and xr.is_contiguous()
        and yr.is_contiguous()
    )
    complex_input = bool(input.is_complex())
    eps = float(eps)
    none_reduction = reduction not in (1, 2)

    if none_reduction:
        out_real = torch.empty(output_shape, dtype=real_dtype, device=input.device)
        out_imag = torch.empty(output_shape, dtype=real_dtype, device=input.device)
    else:
        partial_size = triton.cdiv(n_elements, _PARTIAL_BLOCK_SIZE)
        partial_real = torch.empty(
            partial_size, dtype=real_dtype, device=input.device
        )
        partial_imag = torch.empty(
            partial_size, dtype=real_dtype, device=input.device
        )
        out_real = torch.empty((), dtype=real_dtype, device=input.device)
        out_imag = torch.empty((), dtype=real_dtype, device=input.device)

    with torch_device_fn.device(input.device):
        if none_reduction:
            grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
            _complex_loss_none_kernel[grid](
                xr,
                xi,
                yr,
                yi,
                out_real,
                out_imag,
                n_elements,
                eps,
                OUT_SHAPE=output_shape,
                X_STRIDES=x_strides,
                Y_STRIDES=y_strides,
                LINEAR=linear,
                LOG_INPUT=bool(log_input),
                COMPLEX_INPUT=complex_input,
                FULL=bool(full),
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        else:
            grid = (partial_size,)
            _complex_loss_partial_kernel[grid](
                xr,
                xi,
                yr,
                yi,
                partial_real,
                partial_imag,
                n_elements,
                eps,
                OUT_SHAPE=output_shape,
                X_STRIDES=x_strides,
                Y_STRIDES=y_strides,
                LINEAR=linear,
                LOG_INPUT=bool(log_input),
                COMPLEX_INPUT=complex_input,
                FULL=bool(full),
                BLOCK_SIZE=_PARTIAL_BLOCK_SIZE,
            )
            _complex_loss_final_kernel[(1,)](
                partial_real,
                partial_imag,
                out_real,
                out_imag,
                partial_size,
                n_elements,
                REDUCTION=reduction,
                BLOCK_SIZE=triton.next_power_of_2(partial_size),
            )

    output = torch.empty(
        out_real.shape, dtype=dtype, device=input.device
    )
    parts = torch.view_as_real(output)
    copy_(parts[..., 0], out_real)
    copy_(parts[..., 1], out_imag)
    return output


def poisson_nll_loss(input, target, log_input, full, eps, reduction):
    logger.debug("GEMS_KUNLUNXIN POISSON_NLL_LOSS")
    if not (input.dtype.is_complex or target.dtype.is_complex):
        # Real dtypes use the bit-exact generic Triton path.
        return _generic_poisson_nll_loss(
            input, target, log_input, full, eps, reduction
        )
    # Mirror the generic front-end validation order.
    if input.device != target.device:
        raise RuntimeError("input and target must be on the same device")
    if full and target.dtype == torch.bool:
        raise RuntimeError("Subtraction with a bool target is not supported")
    if not log_input and input.dtype == torch.bool:
        raise RuntimeError("Subtraction with a bool input is not supported")
    if full and target.is_complex():
        raise RuntimeError("Comparisons with a complex target are not supported")
    dtype = _result_dtype(input, target)
    if not input.dtype.is_floating_point:
        input = input.to(dtype)
    if not target.dtype.is_floating_point:
        target = target.to(dtype)
    output_shape, _, _ = _broadcast_metadata(input, target)
    if math.prod(output_shape) == 0:
        # Empty: the generic path returns the empty / zero result with no kernel.
        return _generic_poisson_nll_loss(
            input, target, log_input, full, eps, reduction
        )
    return _complex_poisson(input, target, dtype, log_input, full, eps, reduction)
