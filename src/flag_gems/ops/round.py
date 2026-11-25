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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_INTEGER_DTYPES = frozenset(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)


@triton.jit
def round_half_to_even_impl(x):
    """Round to nearest integer with ties to even (IEEE-754 roundTiesToEven).
    Supports both fp32 and fp64 using core Triton primitives."""
    if x.dtype == tl.float64:
        limit = 4503599627370496.0
    else:
        limit = 8388608.0
    ax = tl.abs(x)
    in_range = ax < limit
    ax_safe = tl.where(in_range, ax, 0.0)
    r = tl.floor(ax_safe)
    d = ax_safe - r
    half_r = r * 0.5
    is_odd = half_r != tl.floor(half_r)
    rounded_abs = tl.where((d > 0.5) | ((d == 0.5) & is_odd), r + 1.0, r)
    # Use `* -1.0` rather than unary `-`: Triton lowers `-a` as `0.0 - a`, which
    # yields `+0.0` instead of `-0.0` when `a == +0.0` (e.g. x in [-0.5, 0)).
    signed = tl.where(x < 0.0, rounded_abs * -1.0, tl.where(x == 0.0, x, rounded_abs))
    return tl.where(in_range, signed, x)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _round_zero_decimals_kernel(x):
    if x.dtype == tl.float64:
        return round_half_to_even_impl(x)
    elif x.dtype.is_floating():
        return round_half_to_even_impl(x.to(tl.float32)).to(x.dtype)
    else:
        return x


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _round_pos_decimals_kernel(x, scale):
    if x.dtype == tl.float64:
        return round_half_to_even_impl(x * scale) / scale
    else:
        x_fp32 = x.to(tl.float32)
        return (round_half_to_even_impl(x_fp32 * scale) / scale).to(x.dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _round_neg_decimals_kernel(x, inv_scale):
    if x.dtype == tl.float64:
        return round_half_to_even_impl(x / inv_scale) * inv_scale
    else:
        x_fp32 = x.to(tl.float32)
        return (round_half_to_even_impl(x_fp32 / inv_scale) * inv_scale).to(x.dtype)


def _pow10(exp: int) -> float:
    try:
        return float(10.0**exp)
    except OverflowError:
        return float("inf")


def _check_round_input(input, decimals: int):
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype == torch.bool:
        raise NotImplementedError("round is not implemented for 'Bool'")
    if input.dtype in _INTEGER_DTYPES and decimals != 0:
        raise NotImplementedError(f"round is not implemented for '{input.dtype}'")


def round_func(input, decimals=0):
    _check_round_input(input, decimals)
    if decimals == 0:
        return _round_zero_decimals_kernel(input)
    if decimals > 0:
        return _round_pos_decimals_kernel(input, _pow10(decimals))
    return _round_neg_decimals_kernel(input, _pow10(-decimals))


def round(input, decimals=0):
    logger.debug("GEMS ROUND")
    return round_func(input, decimals=decimals)


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS ROUND_OUT")
    if out is None:
        return round_func(input, decimals=decimals)
    _check_round_input(input, decimals)
    if out.numel() == 0 and out.shape != input.shape:
        out.resize_(input.shape)
    if decimals == 0:
        _round_zero_decimals_kernel(input, out0=out)
    elif decimals > 0:
        _round_pos_decimals_kernel(input, _pow10(decimals), out0=out)
    else:
        _round_neg_decimals_kernel(input, _pow10(-decimals), out0=out)
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS ROUND_")
    _check_round_input(input, decimals)
    if input.dtype in _INTEGER_DTYPES:
        return input
    if decimals == 0:
        _round_zero_decimals_kernel(input, out0=input)
    elif decimals > 0:
        _round_pos_decimals_kernel(input, _pow10(decimals), out0=input)
    else:
        _round_neg_decimals_kernel(input, _pow10(-decimals), out0=input)
    return input
