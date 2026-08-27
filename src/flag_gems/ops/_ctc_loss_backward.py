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
from typing import Optional

import torch
import triton

from flag_gems.ops.ctc_loss import (
    _REDUCTION_NONE,
    _compute_dtype,
    _ctc_loss_backward_kernel,
    _ctc_loss_init_grad_kernel,
    _is_integral_dtype,
    _length_stats,
    _lengths_to_tensor,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


class _CtcLossBackwardSetup:
    """Normalized inputs for _ctc_loss_backward wrappers."""

    __slots__ = (
        "batch_size",
        "block_s",
        "original_dtype",
        "state_count_max",
        "target_1d",
        "target_stride",
        "unbatched",
        "work_grad",
        "work_input_lengths",
        "work_log_alpha",
        "work_log_probs",
        "work_neg_log_likelihood",
        "work_target_lengths",
        "work_target_offsets",
        "work_targets",
    )


def _prepare_backward(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
):
    """Validate and normalize inputs for _ctc_loss_backward."""
    if log_probs.ndim not in (2, 3):
        raise RuntimeError(
            "ctc_loss_backward expects log_probs to be 2D or 3D, "
            f"but got {log_probs.ndim}D"
        )
    if not torch.is_floating_point(log_probs):
        raise RuntimeError(f'"ctc_loss_backward" not implemented for {log_probs.dtype}')
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
        raise RuntimeError("ctc_loss_backward targets must be integral or floating")

    work_input_lengths = _lengths_to_tensor(
        input_lengths, log_probs.device, "input_lengths"
    )
    work_target_lengths = _lengths_to_tensor(
        target_lengths, log_probs.device, "target_lengths"
    )

    if work_input_lengths.numel() != batch_size:
        raise RuntimeError(
            f"ctc_loss_backward expected input_lengths size {batch_size}, "
            f"got {work_input_lengths.numel()}"
        )
    if work_target_lengths.numel() != batch_size:
        raise RuntimeError(
            f"ctc_loss_backward expected target_lengths size {batch_size}, "
            f"got {work_target_lengths.numel()}"
        )

    min_input_length, max_input_length, _ = _length_stats(work_input_lengths)
    min_target_length, max_target, total_target_length = _length_stats(
        work_target_lengths
    )

    if min_input_length < 0 or max_input_length > work_log_probs.shape[0]:
        raise RuntimeError("ctc_loss_backward input_lengths must be in [0, T]")
    if min_target_length < 0:
        raise RuntimeError("ctc_loss_backward target_lengths must be non-negative")

    state_count_max = 2 * max_target + 1
    target_stride = max_target

    if work_targets.ndim == 1:
        target_1d = True
        if total_target_length != work_targets.numel():
            raise RuntimeError(
                "ctc_loss_backward expected concatenated targets length to equal "
                "sum(target_lengths)"
            )
        work_target_offsets = (
            work_target_lengths.cumsum(0) - work_target_lengths
        ).contiguous()
    elif work_targets.ndim == 2:
        target_1d = False
        if max_target > work_targets.shape[1]:
            raise RuntimeError(
                "ctc_loss_backward target_lengths cannot exceed padded target width"
            )
        target_stride = work_targets.shape[1]
        work_target_offsets = work_target_lengths
    else:
        raise RuntimeError(
            "ctc_loss_backward expects targets to be 1D concatenated or "
            f"2D padded, but got {work_targets.ndim}D"
        )

    work_neg_log_likelihood = neg_log_likelihood
    if unbatched and work_neg_log_likelihood.ndim == 0:
        work_neg_log_likelihood = work_neg_log_likelihood.unsqueeze(0)
    work_neg_log_likelihood = work_neg_log_likelihood.contiguous()
    if work_neg_log_likelihood.dtype != torch.float32:
        work_neg_log_likelihood = work_neg_log_likelihood.to(torch.float32)

    work_log_alpha = log_alpha.unsqueeze(0) if unbatched else log_alpha
    work_log_alpha = work_log_alpha.contiguous()
    if work_log_alpha.dtype != torch.float32:
        work_log_alpha = work_log_alpha.to(torch.float32)

    work_grad = grad.contiguous()
    if work_grad.dtype != compute_dtype:
        work_grad = work_grad.to(compute_dtype)

    setup = _CtcLossBackwardSetup()
    setup.batch_size = batch_size
    setup.block_s = triton.next_power_of_2(state_count_max)
    setup.original_dtype = original_dtype
    setup.state_count_max = state_count_max
    setup.target_1d = target_1d
    setup.target_stride = target_stride
    setup.unbatched = unbatched
    setup.work_grad = work_grad
    setup.work_input_lengths = work_input_lengths
    setup.work_log_alpha = work_log_alpha
    setup.work_log_probs = work_log_probs
    setup.work_neg_log_likelihood = work_neg_log_likelihood
    setup.work_target_lengths = work_target_lengths
    setup.work_target_offsets = work_target_offsets
    setup.work_targets = work_targets
    return setup


def _ctc_loss_backward(
    grad: torch.Tensor,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    neg_log_likelihood: torch.Tensor,
    log_alpha: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
):
    """
    Computes the gradient of CTC loss w.r.t. log_probs.

    This is the lower-level backward operator that computes gradients given
    the forward outputs (neg_log_likelihood and log_alpha).

    Args:
        grad: upstream gradient w.r.t. neg_log_likelihood
        log_probs: (T, N, C) log probabilities from forward
        targets: target sequences (1D concatenated or 2D padded)
        input_lengths: lengths of input sequences (int[] or Tensor)
        target_lengths: lengths of target sequences (int[] or Tensor)
        neg_log_likelihood: neg_log_likelihood from forward pass
        log_alpha: log_alpha from forward pass
        blank: blank label index
        zero_infinity: whether infinite losses should have zero gradient

    Returns:
        grad_log_probs: gradient w.r.t. log_probs
    """
    logger.debug("GEMS _CTC_LOSS_BACKWARD")

    setup = _prepare_backward(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank,
    )

    batch_size = setup.batch_size
    original_dtype = setup.original_dtype
    unbatched = setup.unbatched
    state_count_max = setup.state_count_max
    work_grad = setup.work_grad
    work_log_probs = setup.work_log_probs
    work_targets = setup.work_targets
    work_input_lengths = setup.work_input_lengths
    work_target_lengths = setup.work_target_lengths
    work_target_offsets = setup.work_target_offsets
    work_neg_log_likelihood = setup.work_neg_log_likelihood
    work_log_alpha = setup.work_log_alpha
    target_stride = setup.target_stride
    target_1d = setup.target_1d
    block_s = setup.block_s

    grad_log_probs = torch.empty_like(work_log_probs)
    total = work_log_probs.numel()
    block = 256

    with torch_device_fn.device(log_probs.device):
        _ctc_loss_init_grad_kernel[(triton.cdiv(total, block),)](
            work_log_probs,
            work_input_lengths,
            work_target_lengths,
            work_neg_log_likelihood,
            work_grad,
            grad_log_probs,
            total,
            work_log_probs.shape[0],
            batch_size,
            work_log_probs.shape[2],
            _REDUCTION_NONE,
            zero_infinity,
            block,
        )

        scratch_beta = torch.empty(
            (batch_size, 2, state_count_max),
            dtype=torch.float32,
            device=log_probs.device,
        )

        _ctc_loss_backward_kernel[(batch_size,)](
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            work_neg_log_likelihood,
            work_log_alpha,
            work_grad,
            grad_log_probs,
            scratch_beta,
            work_log_probs.shape[0],
            batch_size,
            work_log_probs.shape[2],
            target_stride,
            state_count_max,
            blank,
            target_1d,
            _REDUCTION_NONE,
            zero_infinity,
            block_s,
        )

    if unbatched:
        grad_log_probs = grad_log_probs.squeeze(1)
    if grad_log_probs.dtype != original_dtype:
        grad_log_probs = grad_log_probs.to(original_dtype)

    return grad_log_probs


def _ctc_loss_backward_out(
    grad: torch.Tensor,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    neg_log_likelihood: torch.Tensor,
    log_alpha: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
):
    """Out variant of _ctc_loss_backward."""
    logger.debug("GEMS _CTC_LOSS_BACKWARD OUT")

    grad_log_probs = _ctc_loss_backward(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank,
        zero_infinity,
    )

    if out is not None:
        if out.shape != grad_log_probs.shape:
            out.resize_(grad_log_probs.shape)
        out.copy_(grad_log_probs)
    else:
        out = grad_log_probs

    return out
