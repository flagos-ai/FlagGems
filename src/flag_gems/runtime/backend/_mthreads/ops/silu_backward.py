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

from flag_gems.ops.silu import silu_backward as default_silu_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def silu_bwd_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(grad_ptr + offs, mask=mask)
    x = tl.load(x_ptr + offs, mask=mask)
    # silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    gf = g.to(tl.float32)
    xf = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-xf))
    res = gf * (sig * (1.0 + xf * (1.0 - sig)))
    tl.store(out_ptr + offs, res.to(x.dtype), mask=mask)


def _use_triton_kernel(grad: torch.Tensor, x: torch.Tensor) -> bool:
    if not isinstance(grad, torch.Tensor) or not isinstance(x, torch.Tensor):
        return False
    if grad.device.type != "musa" or grad.dtype not in _SUPPORTED_DTYPES:
        return False
    if grad.dtype != x.dtype or grad.shape != x.shape:
        return False
    if not grad.is_contiguous() or not x.is_contiguous():
        return False
    if grad.numel() == 0:
        return False
    return True


def silu_backward(grad_output: torch.Tensor, self_input: torch.Tensor):
    logger.debug("GEMS_MTHREADS SILU_BACKWARD")
    if not _use_triton_kernel(grad_output, self_input):
        return default_silu_backward(grad_output, self_input)

    n = grad_output.numel()
    # dtype/size-tuned block: fp16/bf16 use BLOCK=512, large fp32 uses 4096,
    # otherwise 2048. Hardcoded (no autotune) — the kernel is out-of-place so
    # autotune would be safe, but the tuned bands above already cover the
    # working set regimes.
    if grad_output.dtype.itemsize == 2:
        BLOCK = 512
    elif n >= (1 << 26):
        BLOCK = 4096
    else:
        BLOCK = 2048
    grid = (triton.cdiv(n, BLOCK),)
    with torch_device_fn.device(grad_output.device):
        out = torch.empty_like(grad_output)
        silu_bwd_kernel[grid](grad_output, self_input, out, n, BLOCK=BLOCK, num_warps=4)
    return out
