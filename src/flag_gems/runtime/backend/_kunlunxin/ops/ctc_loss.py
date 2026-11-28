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

from flag_gems.ops.ctc_loss import (
    _REDUCTION_MEAN,
    _REDUCTION_NONE,
    _REDUCTION_SUM,
    _compute_dtype,
    _is_integral_dtype,
    _length_stats,
    _lengths_to_tensor,
    _reduction_enum,
)
from flag_gems.ops.isinf import isinf
from flag_gems.ops.where import where_self
from flag_gems.runtime import torch_device_fn

from ._ctc_loss import _ctc_loss_forward_kernel
from ._ctc_loss_backward import _ctc_loss_backward_kernel, _ctc_loss_init_grad_kernel
from .cumsum import cumsum

logger = logging.getLogger(__name__)

# Kunlunxin XPU lowering notes (see harness KernelOpt_Knowledge):
#   * the generic public ctc_loss kernels rely on `tl.debug_barrier`, which is
#     NOT a reliable cross-thread barrier on this backend once the grid has
#     more than one program: the vectorised cross-lane alpha/beta recursion
#     races through the scratch buffer, corrupting every batch except the
#     first -- and, worst case, the device wedges into "wait for noc idle
#     timeout" (runtime error 299).
#   * the generic kernels also rely on `tl.load(..., mask=..., other=-inf)`,
#     but masked-out lanes read neighbouring memory instead of `other`, which
#     makes `exp(alpha + beta - log_likelihood)` explode to ~1e6 magnitudes.
#
# This module therefore re-implements the public `ctc_loss` entry point on top
# of the vendor kernels (scalar state recursion, clamped-address loads +
# `tl.where` value selects, atomic-free scatter) with the same host-side
# semantics as `flag_gems.ops.ctc_loss`.


class CtcLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    ):
        reduction = _reduction_enum(reduction)
        if reduction not in (_REDUCTION_NONE, _REDUCTION_MEAN, _REDUCTION_SUM):
            raise ValueError(f"ctc_loss got invalid reduction enum {reduction}")

        if log_probs.ndim not in (2, 3):
            raise RuntimeError(
                "ctc_loss expects log_probs to be a 2D or 3D tensor, "
                f"but got {log_probs.ndim}D"
            )
        if not torch.is_floating_point(log_probs):
            raise RuntimeError(f'"ctc_loss" not implemented for {log_probs.dtype}')
        if blank < 0 or blank >= log_probs.shape[-1]:
            raise RuntimeError("blank must be in label range")

        original_dtype = log_probs.dtype
        compute_dtype = _compute_dtype(original_dtype)
        unbatched = log_probs.ndim == 2
        batch_size = 1 if unbatched else log_probs.shape[1]

        work_log_probs = log_probs.unsqueeze(1) if unbatched else log_probs
        work_log_probs = work_log_probs.contiguous()
        if work_log_probs.dtype != compute_dtype:
            work_log_probs = work_log_probs.to(compute_dtype)

        if torch.is_floating_point(targets):
            work_targets = targets.to(dtype=torch.long).contiguous()
        elif _is_integral_dtype(targets.dtype):
            work_targets = targets.contiguous()
        else:
            raise RuntimeError("ctc_loss targets must be integral or floating point")
        work_input_lengths = _lengths_to_tensor(
            input_lengths, log_probs.device, "input_lengths"
        )
        work_target_lengths = _lengths_to_tensor(
            target_lengths, log_probs.device, "target_lengths"
        )
        if work_input_lengths.numel() != batch_size:
            raise RuntimeError(
                f"ctc_loss expected input_lengths to have size {batch_size}, "
                f"but got {work_input_lengths.numel()}"
            )
        if work_target_lengths.numel() != batch_size:
            raise RuntimeError(
                f"ctc_loss expected target_lengths to have size {batch_size}, "
                f"but got {work_target_lengths.numel()}"
            )
        min_input_length, max_input_length, _ = _length_stats(work_input_lengths)
        min_target_length, max_target, total_target_length = _length_stats(
            work_target_lengths
        )
        if min_input_length < 0 or max_input_length > work_log_probs.shape[0]:
            raise RuntimeError("ctc_loss input_lengths must be in [0, T]")
        if min_target_length < 0:
            raise RuntimeError("ctc_loss target_lengths must be non-negative")

        state_count_max = 2 * max_target + 1
        target_stride = max_target
        if work_targets.ndim == 1:
            target_1d = True
            if total_target_length != work_targets.numel():
                raise RuntimeError(
                    "ctc_loss expected concatenated targets length to equal "
                    "sum(target_lengths)"
                )
            work_target_offsets = (
                cumsum(work_target_lengths, 0) - work_target_lengths
            ).contiguous()
        elif work_targets.ndim == 2:
            target_1d = False
            if max_target > work_targets.shape[1]:
                raise RuntimeError(
                    "ctc_loss target_lengths cannot exceed padded target width"
                )
            target_stride = work_targets.shape[1]
            work_target_offsets = work_target_lengths
        else:
            raise RuntimeError(
                "ctc_loss expects targets to be a 1D concatenated tensor or a "
                f"2D padded tensor, but got {work_targets.ndim}D"
            )

        block_s = triton.next_power_of_2(state_count_max)
        raw_neg_log_likelihood = torch.empty(
            (batch_size,), dtype=torch.float32, device=log_probs.device
        )
        log_alpha = torch.empty(
            (batch_size, work_log_probs.shape[0], state_count_max),
            dtype=torch.float32,
            device=log_probs.device,
        )
        with torch_device_fn.device(log_probs.device):
            _ctc_loss_forward_kernel[(batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                raw_neg_log_likelihood,
                log_alpha,
                work_log_probs.shape[0],
                batch_size,
                work_log_probs.shape[2],
                target_stride,
                state_count_max,
                blank,
                target_1d,
                block_s,
            )
        neg_log_likelihood = raw_neg_log_likelihood
        if zero_infinity:
            neg_log_likelihood = where_self(
                isinf(neg_log_likelihood),
                torch.zeros(
                    (), dtype=neg_log_likelihood.dtype, device=log_probs.device
                ),
                neg_log_likelihood,
            )

        if reduction == _REDUCTION_NONE:
            output = neg_log_likelihood
            if unbatched:
                output = output.squeeze(0)
        elif reduction == _REDUCTION_SUM:
            output = neg_log_likelihood.sum()
        else:
            denom = work_target_lengths.clamp_min(1)
            output = (neg_log_likelihood / denom).mean()

        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        ctx.save_for_backward(
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            raw_neg_log_likelihood,
            log_alpha,
        )
        ctx.blank = blank
        ctx.reduction = reduction
        ctx.zero_infinity = zero_infinity
        ctx.unbatched = unbatched
        ctx.batch_size = batch_size
        ctx.original_dtype = original_dtype
        ctx.max_target = target_stride
        ctx.state_count_max = state_count_max
        ctx.target_1d = target_1d

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            neg_log_likelihood,
            log_alpha,
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_log_probs = torch.empty_like(work_log_probs)
        total = work_log_probs.numel()
        block = 256
        with torch_device_fn.device(work_log_probs.device):
            _ctc_loss_init_grad_kernel[(triton.cdiv(total, block),)](
                work_log_probs,
                work_input_lengths,
                work_target_lengths,
                neg_log_likelihood,
                grad_output,
                grad_log_probs,
                total,
                work_log_probs.shape[0],
                ctx.batch_size,
                work_log_probs.shape[2],
                ctx.reduction,
                ctx.zero_infinity,
                block,
            )

            scratch_beta = torch.empty(
                (ctx.batch_size, work_log_probs.shape[0], ctx.state_count_max),
                dtype=torch.float32,
                device=work_log_probs.device,
            )
            block_s = triton.next_power_of_2(ctx.state_count_max)
            _ctc_loss_backward_kernel[(ctx.batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                neg_log_likelihood,
                log_alpha,
                grad_output,
                grad_log_probs,
                scratch_beta,
                work_log_probs.shape[0],
                ctx.batch_size,
                work_log_probs.shape[2],
                ctx.max_target,
                ctx.state_count_max,
                ctx.blank,
                ctx.target_1d,
                ctx.reduction,
                ctx.zero_infinity,
                block_s,
            )

        if ctx.unbatched:
            grad_log_probs = grad_log_probs.squeeze(1)
        if grad_log_probs.dtype != ctx.original_dtype:
            grad_log_probs = grad_log_probs.to(ctx.original_dtype)

        return grad_log_probs, None, None, None, None, None, None


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    logger.debug("GEMS_KUNLUNXIN CTC_LOSS")
    return CtcLossFunction.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )


def _install():
    """Wire the XPU implementation into the ops submodule entrypoint.

    ``tests/test_ctc_loss.py`` calls the op through the ``flag_gems.ops``
    namespace (``flag_gems.ops.ctc_loss``) and through the top-level
    ``flag_gems`` attribute. ``SpecOpRegistrar`` swaps the top-level name as
    long as this module is imported from ``_kunlunxin.ops``; the explicit
    setattr mirrors ``_ctc_loss._install`` for direct submodule callers.
    """
    import sys

    mod = sys.modules.get("flag_gems.ops")
    if mod is not None:
        mod.ctc_loss = ctc_loss


_install()
