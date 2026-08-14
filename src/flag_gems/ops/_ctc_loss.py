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
    _compute_dtype,
    _ctc_loss_forward_kernel,
    _is_integral_dtype,
    _length_stats,
    _lengths_to_tensor,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


class _CtcLossSetup:
    """Normalized inputs shared by the _ctc_loss forward and backward wrappers."""

    __slots__ = (
        "batch_size",
        "block_s",
        "original_dtype",
        "state_count_max",
        "target_1d",
        "target_stride",
        "unbatched",
        "work_input_lengths",
        "work_log_probs",
        "work_target_lengths",
        "work_target_offsets",
        "work_targets",
    )


def _prepare(log_probs, targets, input_lengths, target_lengths, blank):
    """Validate and normalize the inputs common to both _ctc_loss directions."""
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
            work_target_lengths.cumsum(0) - work_target_lengths
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

    setup = _CtcLossSetup()
    setup.batch_size = batch_size
    setup.block_s = triton.next_power_of_2(state_count_max)
    setup.original_dtype = original_dtype
    setup.state_count_max = state_count_max
    setup.target_1d = target_1d
    setup.target_stride = target_stride
    setup.unbatched = unbatched
    setup.work_input_lengths = work_input_lengths
    setup.work_log_probs = work_log_probs
    setup.work_target_lengths = work_target_lengths
    setup.work_target_offsets = work_target_offsets
    setup.work_targets = work_targets
    return setup


def _ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
):
    """
    Computes the CTC (Connectionist Temporal Classification) loss.

    This is the lower-level operator that returns both neg_log_likelihood and log_alpha.
    Unlike ctc_loss which only returns the reduced loss, this returns the tuple
    (neg_log_likelihood, log_alpha) needed for custom loss processing and gradients.

    Args:
        log_probs: (T, N, C) log probabilities
        targets: target sequences (1D concatenated or 2D padded)
        input_lengths: lengths of input sequences
        target_lengths: lengths of target sequences
        blank: blank label index
        zero_infinity: whether to zero out infinite losses

    Returns:
        Tuple of (neg_log_likelihood, log_alpha)
    """
    logger.debug("GEMS _CTC_LOSS")

    setup = _prepare(log_probs, targets, input_lengths, target_lengths, blank)
    batch_size = setup.batch_size
    original_dtype = setup.original_dtype
    unbatched = setup.unbatched
    state_count_max = setup.state_count_max
    work_log_probs = setup.work_log_probs
    work_targets = setup.work_targets
    work_input_lengths = setup.work_input_lengths
    work_target_lengths = setup.work_target_lengths
    work_target_offsets = setup.work_target_offsets
    target_stride = setup.target_stride
    target_1d = setup.target_1d
    block_s = setup.block_s

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

    # aten::_ctc_loss returns the raw loss even when zero_infinity is set: the
    # zeroing is applied by the higher-level ctc_loss wrapper. Keeping inf here
    # also preserves the signal _ctc_loss_backward needs to zero those gradients.
    neg_log_likelihood = raw_neg_log_likelihood

    if unbatched:
        neg_log_likelihood = neg_log_likelihood.squeeze(0)
        log_alpha = log_alpha.squeeze(0)

    if neg_log_likelihood.dtype != original_dtype:
        neg_log_likelihood = neg_log_likelihood.to(original_dtype)
    if log_alpha.dtype != original_dtype:
        log_alpha = log_alpha.to(original_dtype)

    return neg_log_likelihood, log_alpha


def _ctc_loss_out(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
    *,
    out0: Optional[torch.Tensor] = None,
    out1: Optional[torch.Tensor] = None,
):
    """Out variant of _ctc_loss."""
    logger.debug("GEMS _CTC_LOSS OUT")

    neg_log_likelihood, log_alpha = _ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank, zero_infinity
    )

    if out0 is not None:
        if out0.shape != neg_log_likelihood.shape:
            out0.resize_(neg_log_likelihood.shape)
        out0.copy_(neg_log_likelihood)
    else:
        out0 = neg_log_likelihood

    if out1 is not None:
        if out1.shape != log_alpha.shape:
            out1.resize_(log_alpha.shape)
        out1.copy_(log_alpha)
    else:
        out1 = log_alpha

    return out0, out1
