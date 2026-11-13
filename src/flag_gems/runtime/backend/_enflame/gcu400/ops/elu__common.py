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
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# P02 flaky-compile shield: a failed compile is not cached, so retrying the
# launch recompiles and usually succeeds; once compiled the retry is a no-op.
_COMPILE_RETRY_ATTEMPTS = 6
_COMPILE_RETRY_DELAY_S = 1.0


def _launch_retry(launch):
    for attempt in range(_COMPILE_RETRY_ATTEMPTS):
        try:
            return launch()
        except Exception:
            if attempt == _COMPILE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_COMPILE_RETRY_DELAY_S)


@libentry()
@triton.jit(do_not_specialize=["N_total", "alpha", "scale", "input_scale"])
def elu_backward_self_kernel(
    grad_ptr, x_ptr, out_ptr, N_total, alpha, scale, input_scale, BLOCK: tl.constexpr
):
    """dx = where(x > 0, grad * scale * input_scale,
    grad * (scale * alpha * exp(x * input_scale) * input_scale))"""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        g = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(
            x > 0.0,
            g * scale * input_scale,
            g * (scale * alpha * tl.exp(x * input_scale) * input_scale),
        )
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "alpha", "scale", "input_scale"])
def elu_backward_result_kernel(
    grad_ptr, y_ptr, out_ptr, N_total, alpha, scale, input_scale, BLOCK: tl.constexpr
):
    """dx = where(y > 0, grad * scale * input_scale,
    grad * ((y + scale * alpha) * input_scale))"""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        g = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(
            y > 0.0,
            g * scale * input_scale,
            g * ((y + scale * alpha) * input_scale),
        )
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def elu_backward_self_kernel_default(
    grad_ptr, x_ptr, out_ptr, N_total, BLOCK: tl.constexpr
):
    """Specialized kernel for alpha=1.0, scale=1.0, input_scale=1.0:
    dx = grad if x > 0 else grad * exp(x)"""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        g = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(x > 0.0, g, g * tl.exp(x))
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def elu_backward_result_kernel_default(
    grad_ptr, y_ptr, out_ptr, N_total, BLOCK: tl.constexpr
):
    """Specialized kernel for alpha=1.0, scale=1.0, input_scale=1.0:
    dx = grad if y > 0 else grad * (y + 1)"""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        g = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(y > 0.0, g, g * (y + 1.0))
        tl.store(out_ptr + off, out, mask=mask)


def _launch_elu_backward(
    grad, self_or_result, out, N_total, alpha, scale, input_scale, is_result
):
    # 2R1W flat stream: keep the two load tiles within the 448KB DSM
    # dual-program budget (256KB per program at the cap: 2 loads x itemsize x
    # BLOCK), which pins BLOCK to 65536 lanes for 2-byte dtypes and 32768 for
    # fp32 at the large end.
    cap = 32768 if grad.dtype == torch.float32 else 65536
    if N_total <= 1024:
        BLOCK = 1024
    elif N_total <= 32768:
        BLOCK = triton.next_power_of_2(N_total)
    elif N_total <= 8388608:
        target_block = N_total // (NUM_SIPS * 2)
        BLOCK = max(1024, 1 << (target_block - 1).bit_length())
        BLOCK = min(BLOCK, cap)
    else:
        BLOCK = cap

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    is_default = alpha == 1.0 and scale == 1.0 and input_scale == 1.0

    with torch_device_fn.device(grad.device):
        if is_result:
            kernel = (
                elu_backward_result_kernel_default
                if is_default
                else elu_backward_result_kernel
            )
        else:
            kernel = (
                elu_backward_self_kernel_default
                if is_default
                else elu_backward_self_kernel
            )
        if is_default:
            _launch_retry(
                lambda: kernel[(grid_size,)](
                    grad, self_or_result, out, N_total, BLOCK=BLOCK, num_warps=2
                )
            )
        else:
            _launch_retry(
                lambda: kernel[(grid_size,)](
                    grad,
                    self_or_result,
                    out,
                    N_total,
                    alpha,
                    scale,
                    input_scale,
                    BLOCK=BLOCK,
                    num_warps=2,
                )
            )


def elu_backward(
    grad_output,
    alpha=1.0,
    scale=1.0,
    input_scale=1.0,
    is_result=False,
    self_or_result=None,
):
    logger.debug("GEMS_ENFLAME ELU_BACKWARD")
    grad = grad_output.contiguous()
    sor = self_or_result.contiguous()
    out_dtype = torch.result_type(grad, sor)
    out = torch.empty_like(grad, dtype=out_dtype)
    _launch_elu_backward(
        grad, sor, out, grad.numel(), alpha, scale, input_scale, is_result
    )
    return out
