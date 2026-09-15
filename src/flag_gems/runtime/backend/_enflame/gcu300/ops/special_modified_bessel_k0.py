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

# GCU300 backend implementation of special_modified_bessel_k0 / _out.
#
# The generic (KernelGen-generated, CUDA-only) implementation in
# flag_gems/ops/special_modified_bessel_k0.py asserts `x.is_cuda` in its
# launch helper, which can never hold for GCU tensors (is_cuda is always
# False on the gcu device). Provide a device-agnostic backend override that
# launches the same Triton kernel on the current GCU device.

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def i0_approx(x):
    """Approximation for I0(x) - modified Bessel function of first kind, order 0"""
    ax = tl.abs(x)

    # Small region: |x| <= 3.75
    t = x / 3.75
    y = t * t
    p_small = 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
        )
    )

    # Large region: |x| > 3.75
    yb = 3.75 / ax
    p_big = 0.39894228 + yb * (
        0.01328592
        + yb
        * (
            0.00225319
            + yb
            * (
                -0.00157565
                + yb
                * (
                    0.00916281
                    + yb
                    * (
                        -0.02057706
                        + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                    )
                )
            )
        )
    )
    res_big = tl.exp(ax) * p_big / tl.sqrt(ax)

    use_small = ax <= 3.75
    return tl.where(use_small, p_small, res_big)


@triton.jit
def special_modified_bessel_k0_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # K0 is only defined for x > 0
    # For x < 0, return NaN (matching PyTorch behavior)
    # For x = 0, return inf (singularity)
    is_negative = x_f32 < 0.0
    is_zero = x_f32 == 0.0

    # Compute I0(x) for use in the small region formula
    i0_x = i0_approx(x_f32)

    # Small region: 0 < x <= 2.0
    # Use the relation: K0(x) = -ln(x/2) * I0(x) + poly(x^2/4)
    y = x_f32 * x_f32 / 4.0  # y = (x/2)^2

    # Fitted polynomial coefficients (from scipy's k0)
    p = -0.57721566
    p = p + 0.42278441 * y
    p = p + 0.23069500 * y * y
    p = p + 0.03488730 * y * y * y
    p = p + 0.00260380 * y * y * y * y
    p = p + 0.00012900 * y * y * y * y * y

    small_result = -tl.log(x_f32 * 0.5 + 1e-40) * i0_x + p

    # Large region: x > 2.0
    # Use asymptotic expansion: K0(x) ~ sqrt(pi/(2x)) * exp(-x) * Q(2/x)
    t = 2.0 / x_f32  # t = 2/x

    q = 1.25331414
    q = q - 0.07832324 * t
    q = q + 0.0218956 * t * t
    q = q - 0.01072842 * t * t * t
    q = q + 0.00162318 * t * t * t * t
    q = q - 0.00013259 * t * t * t * t * t

    large_result = q * tl.exp(-x_f32) / tl.sqrt(x_f32 + 1e-40)

    # Select based on x value
    use_small = x_f32 <= 2.0
    result = tl.where(use_small, small_result, large_result)

    # Handle edge cases: x < 0 -> NaN, x = 0 -> inf, x > 0 -> result
    result = tl.where(is_negative, float("nan"), result)
    result = tl.where(is_zero, float("inf"), result)

    # Cast back to input dtype and store
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


def _run_special_modified_bessel_k0(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {x.dtype}"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"

    x_c = x.contiguous()
    out_c = out.contiguous()
    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    # BLOCK_SIZE 1024 provides good occupancy for element-wise kernels
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        special_modified_bessel_k0_kernel[grid](
            x_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_modified_bessel_k0(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_K0")
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _run_special_modified_bessel_k0(x_c, out)
    if x.layout == torch.strided and x.is_contiguous():
        return out
    else:
        return out.view_as(x)


def special_modified_bessel_k0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_K0_OUT")
    if out.dtype != x.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != x.device:
        raise TypeError("out device must match input device")

    x_c = x.contiguous()
    out_c = out.contiguous()
    _run_special_modified_bessel_k0(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out
