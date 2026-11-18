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

"""rrelu_with_noise_backward candidate v1: simplified host wrapper + constexpr TRAINING branch.

The production gcu400 implementation (TRACKED) unconditionally does 4 `.contiguous()` calls, a
`torch_device_fn.device()` context, an `out_t is out` check and a possible `copy_` in the host layer.
This candidate:
- makes tensors contiguous only when they actually are not;
- drops the device context (Triton launch device is determined by the tensors);
- uses a constexpr TRAINING branch in the kernel, so training=True no longer loads `input`
  or computes grad_eval (production computes both branches);
- keeps the grid-stride loop, correct for any N; numeric path matches production (f32 intermediates).
TRACKED files are untouched; delivered additively by rebinding imports.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


__all__ = ["rrelu_with_noise_backward"]

NUM_SIPS = 24
BLOCK = 4096


@triton.jit(do_not_specialize=["N_total", "lower", "upper"])
def _rrwb_kernel(
    GO,
    X,
    NZ,
    OUT,
    N_total,
    lower,
    upper,
    TRAINING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    slope = (lower + upper) * 0.5
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        go = tl.load(GO + off, mask=mask, other=0)
        go_f32 = go.to(tl.float32)
        if TRAINING:
            nz = tl.load(NZ + off, mask=mask, other=0)
            grad_f32 = go_f32 * nz.to(tl.float32)
        else:
            x = tl.load(X + off, mask=mask, other=0)
            grad_f32 = go_f32 * tl.where(x.to(tl.float32) > 0, 1.0, slope)
        tl.store(OUT + off, grad_f32.to(go.dtype), mask=mask)


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool = False,
):
    # Explicit contiguous output: empty_like preserves non-contiguous strides,
    # while the kernel writes by linear offsets, so a non-contiguous out would scramble values.
    logger.debug("GEMS_ENFLAME_GCU400 RRELU_WITH_NOISE_BACKWARD")
    out = torch.empty(
        grad_output.shape, dtype=grad_output.dtype, device=grad_output.device
    )
    n = out.numel()
    if n == 0:
        return out
    go = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    x = input if input.is_contiguous() else input.contiguous()
    nz = noise if noise.is_contiguous() else noise.contiguous()
    grid = min(triton.cdiv(n, BLOCK), NUM_SIPS * 2)
    _rrwb_kernel[(grid,)](
        go,
        x,
        nz,
        out,
        n,
        float(lower),
        float(upper),
        TRAINING=bool(training),
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out
