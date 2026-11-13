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
"""gcu400 nll_loss2d forward/backward specialisation.

The common ``flag_gems.ops.nllloss`` implementation feeds the int64 target
straight into a Triton kernel, which aborts the process on gcu400 (SIGABRT).
It also uses Python ``and`` / ``not`` on ``tl.tensor`` and an atomic counter
for the mean reduction.  Here we reuse the newer atomic-free three-stage
kernels from ``flag_gems.ops.nll_loss2d`` with gcu400-specific host handling.
"""

import importlib
import logging

import torch
import triton

import flag_gems
from flag_gems.ops.nllloss import nll_loss2d_backward as _common_nll_loss2d_backward

_common = importlib.import_module("flag_gems.ops.nll_loss2d")

logger = logging.getLogger(__name__)

_NUM_STAGES = 2


def _to_int32_target(target):
    # gcu400: Triton cannot issue 64-bit loads from the target tensor.
    # gcu300 already applies this downcast in its specialised implementation.
    if target.dtype == torch.int64:
        return target.to(torch.int32)
    return target


def _as_int(value):
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _launch_config(total):
    # R4 scan: tile size barely changes kernel time (launch-bound operator),
    # but a larger tile keeps the number of stage-1 partials -- and therefore
    # the single-block finalize kernel -- small for large outputs.
    if total <= 4096:
        return 512, 4
    if total <= 65536:
        return 1024, 8
    return 2048, 8


def _forward_impl(self, target, weight, reduction_val, ignore_index_val):
    N, C, H, W = self.shape
    total = N * H * W
    has_weight = weight is not None
    out_dtype = self.dtype
    BLOCK, num_warps = _launch_config(total)
    grid = (triton.cdiv(total, BLOCK),)

    if reduction_val == 0:
        # Store straight into the caller dtype.  tl.store casts the float32
        # accumulator, so the common implementation's extra `.to(out_dtype)`
        # kernel launch is not needed.
        out = torch.empty((N, H, W), dtype=out_dtype, device=self.device)
        _common._nll_loss2d_none_kernel[grid](
            self,
            target,
            weight,
            out,
            N,
            C,
            H,
            W,
            total,
            HAS_WEIGHT=has_weight,
            IGNORE_INDEX=ignore_index_val,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=_NUM_STAGES,
        )
        return out, torch.zeros((), dtype=out_dtype, device=self.device)

    num_blocks = grid[0]
    partials = torch.empty((2, num_blocks), dtype=torch.float32, device=self.device)
    partial_loss, partial_weight = partials[0], partials[1]
    _common._nll_loss2d_reduce_kernel[grid](
        self,
        target,
        weight,
        partial_loss,
        partial_weight,
        N,
        C,
        H,
        W,
        total,
        HAS_WEIGHT=has_weight,
        REDUCTION=reduction_val,
        IGNORE_INDEX=ignore_index_val,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
    )
    out = torch.empty((), dtype=out_dtype, device=self.device)
    _common._nll_loss2d_finalize_kernel[(1,)](
        partial_loss,
        partial_weight,
        out,
        num_blocks,
        REDUCTION=reduction_val,
        OUT_DTYPE=_common._TORCH_TO_TL[out_dtype],
        BLOCK=triton.next_power_of_2(num_blocks),
    )
    # Keep total_weight in float32: the backward kernel only reads it to
    # divide, so converting to out_dtype costs an extra kernel for no benefit.
    return out, flag_gems.sum(partial_weight)


def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_ENFLAME NLL_LOSS2D_FWD")
    if self.device.type != flag_gems.device or target.device.type != flag_gems.device:
        raise RuntimeError("nll_loss2d_forward: expected tensors on the GCU device")
    assert self.ndim == 4, "nll_loss2d_forward: expected 4D input (N, C, H, W)"
    target = _to_int32_target(target).contiguous()
    self = self.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    return _forward_impl(
        self, target, weight, _as_int(reduction), _as_int(ignore_index)
    )


def nll_loss2d_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_ENFLAME NLL_LOSS2D_BWD")
    return _common_nll_loss2d_backward(
        grad_output,
        self,
        _to_int32_target(target),
        weight,
        reduction,
        ignore_index,
        total_weight,
    )


# ---------------------------------------------------------------------------
# gcu400 nll_loss_forward/backward (1-D / 2-D input) specialisation (S0 migration)
#
# The generic implementation zeroes the whole grad_input with a separate
# memset kernel and then launches a 1-D grid of cdiv(N, 128) programs that
# each touch only one (n, target[n]) element.  Keeping that sparse-write shape
# is important: torch.zeros_like is a DMA-class memset and is far cheaper than
# writing every element from a Triton kernel on large shapes.  The gcu400
# specialisation keeps the same two-step structure but downcasts the int64
# target to int32 on the host (GCU400 cannot issue 64-bit loads), replaces the
# Python `not`/`and` on tl.tensor with explicit bitwise logic, and uses a
# smaller BLOCK_N / num_warps=1 so small-N launches get more programs.
# ---------------------------------------------------------------------------

import triton.language as tl  # noqa: E402

from flag_gems.ops.nllloss import (  # noqa: E402
    nll_loss_forward_raw as _common_forward_raw,
)
from flag_gems.runtime import torch_device_fn  # noqa: E402
from flag_gems.utils import libentry  # noqa: E402

_GCU400_BWD_BLOCK = 32
_GCU400_BWD_WARPS = 1


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def _nll_loss_backward_gcu400_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    REDUCTION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N
    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=ignore_index)
    hit = (tgt != ignore_index) & mask_n
    if HAS_WEIGHT:
        wgt = tl.load(wgt_ptr + tgt, mask=hit, other=0.0).to(tl.float32)
    else:
        wgt = hit.to(tl.float32)
    if REDUCTION == 0:
        grad = tl.load(out_grad_ptr + offsets_n, mask=mask_n, other=0.0).to(tl.float32)
    else:
        grad = tl.load(out_grad_ptr).to(tl.float32)
    if REDUCTION == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1.0
    val = tl.where(hit, -1.0 * grad * wgt / total_w, 0.0)
    tl.store(inp_grad_ptr + offsets_n * C + tgt, val, mask=hit)


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_ENFLAME NLL_LOSS_BWD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    reduction = _as_int(reduction)
    ignore_index = _as_int(ignore_index)
    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()
    grad_input = torch.zeros_like(self).contiguous()
    grid = (triton.cdiv(N, _GCU400_BWD_BLOCK),)
    with torch_device_fn.device(self.device):
        _nll_loss_backward_gcu400_kernel[grid](
            grad_output,
            target,
            weight if weight is not None else self,
            grad_input,
            ignore_index,
            total_weight if total_weight is not None else self,
            N,
            C,
            reduction,
            weight is not None,
            _GCU400_BWD_BLOCK,
            num_warps=_GCU400_BWD_WARPS,
        )
    return grad_input


class NllLossForwardFunctionGCU400(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index):
        output, total_weight = _common_forward_raw(
            inp, target, weight, reduction, ignore_index
        )
        ctx.save_for_backward(inp, target, total_weight)
        ctx.weight = weight
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        return output, total_weight

    @staticmethod
    def backward(ctx, grad_output, grad_total_weight):
        inp, target, total_weight = ctx.saved_tensors
        grad_input = nll_loss_backward(
            grad_output,
            inp,
            target,
            weight=ctx.weight,
            reduction=ctx.reduction,
            ignore_index=ctx.ignore_index,
            total_weight=total_weight,
        )
        return grad_input, None, None, None, None


def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_ENFLAME NLL_LOSS_FWD")
    return NllLossForwardFunctionGCU400.apply(
        self, target, weight, reduction, ignore_index
    )
