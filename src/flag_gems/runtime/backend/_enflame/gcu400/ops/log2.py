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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N"])
def log2_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask, other=1.0).to(tl.float32)
        y = tl.log(x) * 1.4426950408889634
        tl.store(out_ptr + off, y, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_fp32:
        return 65536
    return 8192


def _launch(inp, out):
    N = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        log2_kernel[(grid_size,)](
            inp,
            out,
            N,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def _log2(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


_LN2 = 0.6931471805599453
_BACKWARD_MAX_BLOCK = 8192
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _log2_grad_kernel(grad_ptr, x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """d/dx log2(x) = grad / (x * ln2) in float32 opmath.

    Evaluating ``x * ln2`` in the operand dtype loses a full ulp for fp16
    inputs (the gcu400 forward vector unit rounds the product to fp16 before
    the divide), which overshoots the 1e-3 rtol of the accuracy test on ~2.5%
    of the elements.  A gcu400 probe over fp16/bf16/fp32 with the test
    tolerances shows the float32-opmath variant with zero violations for all
    three dtypes while the in-dtype variant fails fp16.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        grad = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        tl.store(out_ptr + off, grad / (x * 0.6931471805599453), mask=mask)


def _log2_backward(grad, x):
    N = x.numel()
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if N == 0:
        return out
    safe_grad = grad if grad.is_contiguous() else grad.contiguous()
    safe_x = x if x.is_contiguous() else x.contiguous()
    # This kernel only runs from the autograd engine thread, where the gcu400
    # Triton launcher charges per warp (one warp measured ~46us against ~117us
    # for four warps on the same 4K tile, probe 2026-11-24), so the backward
    # pins one warp and skips the device guard (another ~20-45us there).
    BLOCK = min(_choose_block(N, x.dtype == torch.float32), _BACKWARD_MAX_BLOCK)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    _log2_grad_kernel[(grid_size,)](
        safe_grad,
        safe_x,
        out,
        N,
        BLOCK=BLOCK,
        num_warps=1,
    )
    return out


class _Log2Function(torch.autograd.Function):
    """Autograd bridge for the out-of-place ``log2``.

    ``flag_gems.log2`` is a plain Python kernel launcher, so a requiring-grad
    input currently returns a tensor without a graph.  The bridge restores it
    with the Gem backward kernel (no aten math op on the backward path).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _log2(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return _log2_backward(grad, x)


def log2(A):
    logger.debug("GEMS_ENFLAME LOG2")
    if A.dtype in _FLOAT_DTYPES and torch.is_grad_enabled() and A.requires_grad:
        return _Log2Function.apply(A)
    return _log2(A)


def log2_(A):
    logger.debug("GEMS_ENFLAME LOG2_")
    if not A.is_contiguous():
        A.copy_(_log2(A))
        return A
    return _launch(A, A)
