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
import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.softplus import (
    softplus_backward as default_softplus_backward,  # fallback
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
exp = tl_extra_shim.exp


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "VEC": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "VEC": 1}, num_warps=16, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit(do_not_specialize=["beta", "threshold"])
def softplus_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements

    dy = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    z = x_fp32 * beta
    # d/dx softplus(x) = sigmoid(beta * x) when z <= threshold, else 1
    # sigmoid(z) = 1 / (1 + exp(-z))
    sig = 1.0 / (1.0 + exp(-z))
    dydx = tl.where(z > threshold, 1.0, sig)
    dx = (dy.to(tl.float32) * dydx).to(x.dtype)

    tl.store(out_ptr + offsets, dx, mask=mask)


def _coerce_scalar(value, name: str) -> Tuple[float, bool]:
    try:
        v = float(value) if not isinstance(value, torch.Tensor) else float(value.item())
    except Exception:
        return 0.0, False
    if not math.isfinite(v):
        return 0.0, False
    return v, True


def _use_triton_kernel(
    grad_output: torch.Tensor, x: torch.Tensor, beta, threshold
) -> Tuple[bool, float, float]:
    if not isinstance(grad_output, torch.Tensor) or not isinstance(x, torch.Tensor):
        return False, 0.0, 0.0
    if grad_output.device.type != "musa" or x.device.type != "musa":
        return False, 0.0, 0.0
    if grad_output.dtype != x.dtype or grad_output.dtype not in _SUPPORTED_DTYPES:
        return False, 0.0, 0.0
    if (
        grad_output.numel() != x.numel()
        or grad_output.numel() == 0
        or not grad_output.is_contiguous()
        or not x.is_contiguous()
    ):
        return False, 0.0, 0.0
    beta_value, ok_beta = _coerce_scalar(beta, "beta")
    threshold_value, ok_thr = _coerce_scalar(threshold, "threshold")
    if not ok_beta or not ok_thr:
        return False, 0.0, 0.0
    return True, beta_value, threshold_value


def _launch_softplus_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    beta: float,
    threshold: float,
    dtype_size: int,
):
    grad_flat = grad_output.view(-1)
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        softplus_backward_kernel[grid](
            grad_flat, x_flat, out_flat, n_elements, beta, threshold, dtype_size
        )
    return out


def softplus_backward(grad_output, self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_MTHREADS SOFTPLUS_BACKWARD")
    use_triton, beta_value, threshold_value = _use_triton_kernel(
        grad_output, self, beta, threshold
    )
    if not use_triton:
        return default_softplus_backward(
            grad_output, self, beta=beta, threshold=threshold
        )

    out = torch.empty_like(self)
    dtype_size = self.element_size()
    return _launch_softplus_backward(
        grad_output, self, out, beta_value, threshold_value, dtype_size
    )
