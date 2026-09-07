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
"""Kunlunxin (XPU) special_logit / special_logit.out.

The generic src/flag_gems/ops/special_logit.py kernel guards the eps-clamp
with `tl.where(x_f32 != x_f32, ...)` to keep NaN inputs as NaN.  On the
Triton-XPU backend that unordered float compare lowers to an LLVM `setuo`
(`setcc ... setuo`) on v16f32, which the xdnn/xpu backend cannot select and
`make_elf` aborts the whole process (only the fp32/bf16 case is hit after the
fp16 variant was compiled; fp16 also crashes with more than one block).
The vendor kernel below replaces the unordered compare with two ordered
comparisons (`x < lo` / `x > hi`): for NaN both predicates are false, so the
value passes through unchanged and NaN is preserved, and the ordered
predicates are selectable on XPU.
"""
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def special_logit_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    eps,
    HAS_EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    if HAS_EPS:
        lo = eps
        hi = 1.0 - eps
        # Ordered comparisons only (see module docstring): for NaN both are
        # false, so x_f32 stays NaN; otherwise clamp into [lo, hi].
        x_c = tl.where(x_f32 < lo, lo, x_f32)
        x_c = tl.where(x_c > hi, hi, x_c)
        x_f32 = x_c

    y = tl.log(x_f32 / (1.0 - x_f32))
    tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)


def _special_logit_impl(input: torch.Tensor, eps=None, out: torch.Tensor = None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_floating_point():
        raise TypeError("special_logit expected a floating point tensor as input")

    if eps is not None:
        eps = float(eps)
        if not (0.0 <= eps <= 0.5):
            raise ValueError("eps must be in the range [0.0, 0.5].")

    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != input.dtype:
            raise TypeError("For special_logit.out, out.dtype must match input.dtype")
        result = out
    else:
        result = torch.empty_like(input)

    n_elements = result.numel()
    # 1024 is a common power-of-2 block size that balances occupancy and memory throughput.
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        special_logit_kernel[grid](
            input,
            result,
            n_elements,
            eps if eps is not None else 0.0,
            HAS_EPS=(eps is not None),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return result


def special_logit(input, eps=None, out=None):
    """Compute the logit of the input tensor (Kunlunxin XPU override).

    Args:
        input: Input tensor with values in [0, 1] (or outside if eps is None)
        eps: Optional epsilon for clamping. If provided, input is clamped to [eps, 1-eps]
        out: Optional output tensor

    Returns:
        Tensor with logit values
    """
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOGIT")
    return _special_logit_impl(input, eps=eps, out=out)


def special_logit_out(input, eps=None, out=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOGIT_OUT")
    if out is None:
        raise TypeError("special_logit_out requires an 'out' tensor.")
    return _special_logit_impl(input, eps=eps, out=out)