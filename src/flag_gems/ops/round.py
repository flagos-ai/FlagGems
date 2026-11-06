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


@triton.jit
def round_half_to_even_impl(x):
    """Round to nearest with ties to even (round half to even).
    x must be fp32."""
    r = tl.floor(x)
    d = x - r  # fractional part, in [0, 1) for positive, in (-1, 0] for negative

    # is_odd = (r % 2 == 1), i.e., r is odd
    # In Triton: r - 2 * floor(r/2) for odd r in [-2.5, 2.5] range is close to 1
    is_odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) > 0.5

    # For d > 0.5: always round up
    # For d == 0.5 and r is odd: round up (to make result even)
    # For d == 0.5 and r is even: stay at r (already even)
    # For d < 0.5: stay at r
    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


# #7756: the kernel used to require a contiguous tensor and raised otherwise,
# which rejects layouts ATen accepts (a transposed matrix, a strided slice).
# pointwise_dynamic addresses through the tensor's own strides, so any layout
# works; fp64 keeps its width, narrower floats still compute in fp32.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def round_func(x):
    if x.dtype == tl.float64:
        return round_half_to_even_impl(x)
    return round_half_to_even_impl(x.to(tl.float32)).to(x.dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def round_decimals_func(x, scale):
    # scale = 10 ** decimals is computed on the host, so the kernel never
    # branches on decimals
    if x.dtype == tl.float64:
        return round_half_to_even_impl(x * scale) / scale
    return (round_half_to_even_impl(x.to(tl.float32) * scale) / scale).to(x.dtype)


_INTEGER_DTYPES = (torch.int32, torch.int64, torch.int16, torch.int8)


def _check_round_args(input):
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")


def round(input, decimals=0):
    logger.debug("GEMS ROUND")
    _check_round_args(input)

    # For integer types, return a copy (array-api convention)
    if input.dtype in _INTEGER_DTYPES:
        return input.clone()

    if input.numel() == 0:
        return input

    if decimals == 0:
        return round_func(input)
    return round_decimals_func(input, 10.0**decimals)


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS ROUND_OUT")
    if out is None:
        return round(input, decimals=decimals)

    _check_round_args(input)

    # For integer types, return a copy
    if input.dtype in _INTEGER_DTYPES:
        out.copy_(input)
        return out

    if input.numel() == 0:
        return out

    if decimals == 0:
        round_func(input, out0=out)
    else:
        round_decimals_func(input, 10.0**decimals, out0=out)
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS ROUND_")
    _check_round_args(input)

    # For integer types, return input unchanged (array-api convention for integer round)
    if input.dtype in _INTEGER_DTYPES:
        return input

    if input.numel() == 0:
        return input

    if decimals == 0:
        round_func(input, out0=input)
    else:
        round_decimals_func(input, 10.0**decimals, out0=input)
    return input
