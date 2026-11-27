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

logger = logging.getLogger(__name__)

_INTEGRAL_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def _round_half_to_even(x):
    """Round to nearest, ties to even. ``x`` must be fp32.

    Adds and subtracts 2**23 to force the mantissa through the hardware's
    round-to-nearest-even step. The magnitude is used so the trick stays
    exact for negatives (and -0.0 keeps its sign); for |x| >= 2**23 every
    fp32 value is already integral, so the input is returned unchanged.

    Applying the bias to a signed value (or zeroing it above 2**22, as an
    earlier revision did) leaves half-integers in [2**22, 2**23) unrounded,
    since fp32 still represents those exactly.
    """
    magic: tl.constexpr = 8388608.0  # 2**23
    magnitude = tl.abs(x)
    rounded = (magnitude + magic) - magic
    rounded = tl.where(magnitude >= magic, magnitude, rounded)
    return tl.where(x < 0.0, -rounded, rounded)


@triton.jit
def _round_kernel(
    x_ptr,
    n_elements,
    SCALE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    if HAS_SCALE:
        rounded = _round_half_to_even(x_fp32 * SCALE) / SCALE
    else:
        rounded = _round_half_to_even(x_fp32)
    tl.store(x_ptr + offsets, rounded.to(x.dtype), mask=mask)


def round_(input, *, decimals=0):
    logger.debug("GEMS ROUND_")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round_ expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round_ is not supported for complex tensors.")

    # Integral tensors are already rounded; ATen leaves them untouched. Casting
    # them through fp32 would corrupt magnitudes above 2**24.
    if input.dtype in _INTEGRAL_DTYPES or input.dtype == torch.bool:
        if decimals != 0:
            raise RuntimeError(
                f"\"round_cpu\" not implemented for '{input.dtype}' with decimals"
            )
        return input

    if input.dtype not in _FLOAT_DTYPES:
        raise RuntimeError(f"round_ is not implemented for '{input.dtype}'")

    # The kernel indexes storage linearly, so a non-dense view (e.g. x[::2])
    # would update the wrong elements.
    if not input.is_contiguous():
        raise ValueError(
            "round_ Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return input

    scale = float(10.0**decimals)
    has_scale = decimals != 0
    # Capped at 4096: the rounding needs several fp32 temporaries per element
    # and 8192 exhausts the Ascend local buffer (BiShengIR reports "some ops
    # need extra local buffer").
    block_size = 4096
    grid = (triton.cdiv(n_elements, block_size),)
    _round_kernel[grid](
        input,
        n_elements,
        scale,
        has_scale,
        block_size,
    )
    return input
