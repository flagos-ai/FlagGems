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
def special_i1_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    y = x / 3.75
    y2 = y * y
    small = 0.5 + y2 * (
        0.87890594
        + y2
        * (
            0.51498869
            + y2
            * (0.15084934 + y2 * (0.02658733 + y2 * (0.00301532 + y2 * 0.00032411)))
        )
    )
    small = x * small

    t = 3.75 / tl.maximum(ax, 1e-20)
    large_poly = 0.39894228 + t * (
        -0.03988024
        + t
        * (
            -0.00362018
            + t
            * (
                0.00163801
                + t
                * (
                    -0.01031555
                    + t
                    * (
                        0.02282967
                        + t * (-0.02895312 + t * (0.01787654 + t * -0.00420059))
                    )
                )
            )
        )
    )
    large = tl.exp(ax) / tl.sqrt(tl.maximum(ax, 1e-20)) * large_poly
    large = tl.where(x < 0, -large, large)
    result = tl.where(ax <= 3.75, small, large)
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def _special_i1_xpu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise TypeError(f"special_i1 is not implemented for {x.dtype}")
    if out.dtype != x.dtype:
        raise RuntimeError("special_i1.out expects out to have the input dtype")

    x_contiguous = x.contiguous()
    out_contiguous = out.contiguous()
    n_elements = out_contiguous.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        special_i1_kernel[grid](
            x_contiguous, out_contiguous, n_elements, BLOCK_SIZE=1024
        )
    if out_contiguous.data_ptr() != out.data_ptr():
        out.copy_(out_contiguous)
    return out


def special_i1(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1")
    return _special_i1_xpu(x, torch.empty_like(x))


def special_i1_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1_OUT")
    if x.shape != out.shape:
        x = x.expand(out.shape)
    return _special_i1_xpu(x, out)
