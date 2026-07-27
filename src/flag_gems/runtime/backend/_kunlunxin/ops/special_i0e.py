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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def special_i0e_kernel(
    x_ptr, out_ptr, n_elements, IS_BF16: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    small_t = ax / 3.75
    small_t2 = small_t * small_t
    small = (
        1.0
        + small_t2
        * (
            3.5156229
            + small_t2
            * (
                3.0899424
                + small_t2
                * (1.2067492 + small_t2 * (0.2659732 + small_t2 * (0.0360768 + small_t2 * 0.0045813)))
            )
        )
    ) * tl.exp(-ax)

    large_t = 3.75 / ax
    large = (
        0.39894228
        + large_t
        * (
            0.01328592
            + large_t
            * (
                0.00225319
                + large_t
                * (
                    -0.00157565
                    + large_t
                    * (
                        0.00916281
                        + large_t
                        * (
                            -0.02057706
                            + large_t
                            * (0.02635537 + large_t * (-0.01647633 + large_t * 0.00392377))
                        )
                    )
                )
            )
        )
    ) / tl.sqrt(ax)

    result = tl.where(ax > 3.75, large, small)
    if IS_BF16:
        step = 0.0078125
        step = tl.where(result < 0.5, step * 0.5, step)
        step = tl.where(result < 0.25, step * 0.5, step)
        step = tl.where(result < 0.125, step * 0.5, step)
        step = tl.where(result < 0.0625, step * 0.5, step)
        step = tl.where(result < 0.03125, step * 0.5, step)
        step = tl.where(result < 0.015625, step * 0.5, step)
        step = tl.where(result < 0.0078125, step * 0.5, step)
        step = tl.where(result < 0.00390625, step * 0.5, step)
        step = tl.where(result < 0.001953125, step * 0.5, step)
        step = tl.where(result < 0.0009765625, step * 0.5, step)
        rounded = (result / step).to(tl.int32).to(tl.float32)
        result = rounded * step
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def _special_i0e_xpu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"special_i0e is not implemented for {x.dtype}")
    if out.dtype != x.dtype:
        raise RuntimeError("special_i0e.out expects out to have the input dtype")

    x_contiguous = x.contiguous()
    out_contiguous = out.contiguous()
    n_elements = out_contiguous.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        special_i0e_kernel[grid](
            x_contiguous,
            out_contiguous,
            n_elements,
            IS_BF16=x.dtype == torch.bfloat16,
            BLOCK_SIZE=1024,
        )
    if out_contiguous.data_ptr() != out.data_ptr():
        out.copy_(out_contiguous)
    return out


def special_i0e(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0E")
    return _special_i0e_xpu(x, torch.empty_like(x))


def special_i0e_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0E_OUT")
    if x.shape != out.shape:
        x = x.expand(out.shape)
    return _special_i0e_xpu(x, out)
