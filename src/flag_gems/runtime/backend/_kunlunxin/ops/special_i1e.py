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
#
import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.copy import copy_ as _gems_copy_

logger = logging.getLogger(__name__)


@triton.jit
def special_i1e_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    ax = tl.abs(xf)

    # Small region: |x| <= 3.75. I1(x) = x * P(y^2), y = x / 3.75.
    y = xf / 3.75
    y2 = y * y
    p = 0.00032411
    p = 0.00301532 + y2 * p
    p = 0.02658733 + y2 * p
    p = 0.15084934 + y2 * p
    p = 0.51498869 + y2 * p
    p = 0.87890594 + y2 * p
    p = 0.5 + y2 * p
    small = xf * p * tl.exp(-ax)

    # Large region: |x| > 3.75. I1(x) ~ exp(|x|)/sqrt(|x|) * Q(3.75/|x|).
    # Multiplying by exp(-|x|) cancels the exponential prefactor.
    t = 3.75 / tl.maximum(ax, 1e-20)
    q = -0.00420059
    q = 0.01787654 + t * q
    q = -0.02895312 + t * q
    q = 0.02282967 + t * q
    q = -0.01031555 + t * q
    q = 0.00163801 + t * q
    q = -0.00362018 + t * q
    q = -0.03988024 + t * q
    q = 0.39894228 + t * q
    large = q / tl.sqrt(tl.maximum(ax, 1e-20))
    # I1 is odd, so i1e is odd as well.
    large = tl.where(xf < 0, -large, large)

    y_out = tl.where(ax <= 3.75, small, large)
    tl.store(out_ptr + offsets, y_out.to(x.dtype), mask=mask)


def _run_special_i1e_kernel(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    # Layout regularisation uses the gems Triton copy (no torch/aten fallback):
    # contiguous tensors are passed through untouched (the benchmark/test path),
    # non-contiguous ones are staged through a flat buffer.
    if x.is_contiguous():
        x_c = x
    else:
        x_c = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _gems_copy_(x_c, x)

    n_elements = out.numel()
    if n_elements == 0:
        return out
    if out.is_contiguous():
        out_c = out
    else:
        out_c = torch.empty(out.shape, dtype=out.dtype, device=out.device)

    grid = (triton.cdiv(n_elements, 1024),)
    special_i1e_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)

    if out_c is not out:
        _gems_copy_(out, out_c)
    return out


def special_i1e(x: torch.Tensor):
    """
    ATen wrapper: special_i1e(Tensor self) -> Tensor
    """
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1E")
    out = torch.empty_like(x)
    return _run_special_i1e_kernel(x, out)


def special_i1e_out(self: torch.Tensor, out: torch.Tensor):
    """
    ATen wrapper: special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    """
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1E_OUT")
    x = self
    return _run_special_i1e_kernel(x, out)
