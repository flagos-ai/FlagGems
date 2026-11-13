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

"""gcu400 specialization of silu_and_mul_with_clamp.

The common implementation uses @pointwise_dynamic, whose multi-dimensional
index decomposition is very expensive on GCU: for 16.7M elements a 2D
[4096,4096] input takes 1.32ms while an equivalent 3D [64,512,512] input
takes 14.1ms, and 1G elements in [1024,1024,1024] takes 870ms.  A flat
linear-index kernel removes that decomposition entirely; it also avoids the
per-call torch.tensor(limit, device=...) synchronization in the common
forward.  Measured 64x64: 190us -> 4us.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

BLOCK_SIZE = 8192
MAX_GRID_SIZE = 48


@libentry()
@triton.jit
def _smc_flat_kernel(X, Y, Out, N, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, num_programs * BLOCK):
        off = start + cols
        mask = off < N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        y = tl.load(Y + off, mask, other=0.0).to(tl.float32)
        gate = tl.minimum(x, LIMIT)
        up = tl.minimum(tl.maximum(y, -LIMIT), LIMIT)
        sig = tl.sigmoid(gate)
        tl.store(Out + off, gate * sig * up, mask)


@libentry()
@triton.jit
def _smc_flat_grad_kernel(
    X, Y, DGrad, DX, DY, N, LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, num_programs * BLOCK):
        off = start + cols
        mask = off < N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        y = tl.load(Y + off, mask, other=0.0).to(tl.float32)
        d = tl.load(DGrad + off, mask, other=0.0).to(tl.float32)
        gate = tl.minimum(x, LIMIT)
        up = tl.minimum(tl.maximum(y, -LIMIT), LIMIT)
        sig = tl.sigmoid(gate)
        gate_silu = gate * sig
        d_gate_silu = sig * (1.0 + gate * (1.0 - sig))
        gate_mask = (x <= LIMIT).to(tl.float32)
        up_mask = ((y >= -LIMIT) & (y <= LIMIT)).to(tl.float32)
        tl.store(DX + off, d * up * d_gate_silu * gate_mask, mask)
        tl.store(DY + off, d * gate_silu * up_mask, mask)


def _launch_flat(x, y, out, limit):
    n = x.numel()
    grid = (min(MAX_GRID_SIZE, triton.cdiv(n, BLOCK_SIZE)),)
    _smc_flat_kernel[grid](x, y, out, n, float(limit), BLOCK_SIZE, num_warps=4)


def _launch_flat_grad(x, y, dgrad, dx, dy, limit):
    n = x.numel()
    grid = (min(MAX_GRID_SIZE, triton.cdiv(n, BLOCK_SIZE)),)
    _smc_flat_grad_kernel[grid](
        x, y, dgrad, dx, dy, n, float(limit), BLOCK_SIZE, num_warps=4
    )


class SiluAndMulWithClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, limit):
        logger.debug("GEMS_ENFLAME SILU_AND_MUL_WITH_CLAMP_FORWARD")
        x = x.contiguous()
        y = y.contiguous()
        ctx.save_for_backward(x, y)
        ctx.limit = limit
        out = torch.empty_like(x)
        _launch_flat(x, y, out, limit)
        return out

    @staticmethod
    def backward(ctx, dgrad):
        x, y = ctx.saved_tensors
        dgrad = dgrad.contiguous()
        dx = torch.empty_like(x)
        dy = torch.empty_like(y)
        _launch_flat_grad(x, y, dgrad, dx, dy, ctx.limit)
        return dx, dy, None


def silu_and_mul_with_clamp(x, y, limit):
    # Inference path: avoid autograd.Function.apply graph construction.
    if not torch.is_grad_enabled() or not (x.requires_grad or y.requires_grad):
        out = torch.empty_like(x)
        _launch_flat(x.contiguous(), y.contiguous(), out, limit)
        return out
    return SiluAndMulWithClamp.apply(x, y, limit)


def silu_and_mul_with_clamp_out(x, y, out, limit):
    logger.debug("GEMS_ENFLAME SILU_AND_MUL_WITH_CLAMP_OUT")
    _launch_flat(x.contiguous(), y.contiguous(), out, limit)
    return out
