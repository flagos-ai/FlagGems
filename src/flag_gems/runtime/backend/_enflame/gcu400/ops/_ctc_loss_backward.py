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
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.ctc_loss import (
    _REDUCTION_NONE,
    _compute_dtype,
    _ctc_loss_init_grad_kernel,
    _debug_barrier,
    _is_integral_dtype,
    _length_stats,
    _lengths_to_tensor,
    _logaddexp3,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Number of elements processed by the init kernel per step. On GCU400 a 256-element tile is too small (high launch
# counts
# for large batches), and measurements show 2048 elements / 4 warps is about 10x faster than 256 elements / 2 warps.
_INIT_GRAD_BLOCK = 2048
_INIT_GRAD_WARPS = 4


# num_warps must match the lane count: this backend computes incorrectly when 32*warps is much larger than BLOCK_S or
# equals 2
# (measured: BLOCK_S=16 with 4 warps and BLOCK_S=64 with 2 warps both give wrong results),
# so choose the value from three tiers based on BLOCK_S.
def _warps_for(block_s):
    if block_s <= 32:
        return 1
    return 4


@libentry()
@triton.jit
def _ctc_loss_backward_posterior_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    log_alpha,
    grad_output,
    posterior_out,
    scratch_beta,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    POST_STRIDE: tl.constexpr,
    POST_T: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """β (backward) recursion that only writes the posterior into scratch, no atomic accumulation.

    atomic_add on GCU400 is a slow ~0.5us per-instruction path, and the original did one atomic per
    T-loop step (256 lanes, about half of them blank states sharing addresses), letting atomic
    serialization latency dominate the whole kernel (59ms at T=1024). This switches to a plain
    store at a unique (t, state) address and moves all atomics into a crossing kernel outside the loop.

    POST_STRIDE / POST_T decouple the posterior scratch row width and row count from (SCM, T): the
    legacy path passes (STATE_COUNT_MAX, T), exactly matching the original implementation, while the
    GEMM-reduce path passes (BLOCK_S, cdiv(T, KT)*KT) with zero-fill, so the downstream dot consumer
    can use mask-free affine loads (a masked load fed to tl.dot yields wrong values at nw>1 here).
    """
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    nll = tl.load(neg_log_likelihood + batch).to(tl.float32)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX

    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch)
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    state1 = states + 1
    is_blank_state1 = (state1 % 2) == 0
    target_index1 = (state1 - 1) // 2
    target_mask1 = (target_index1 >= 0) & (target_index1 < target_len)
    target_safe_index1 = tl.where(target_mask1, target_index1, 0)
    target_ptrs1 = targets + target_origin + target_safe_index1
    target_value1 = tl.load(target_ptrs1, mask=target_mask1, other=BLANK)
    labels1 = tl.where(is_blank_state1, BLANK, target_value1)

    state2 = states + 2
    target_index2 = (state2 - 1) // 2
    target_mask2 = (target_index2 >= 0) & (target_index2 < target_len)
    target_safe_index2 = tl.where(target_mask2, target_index2, 0)
    target_ptrs2 = targets + target_origin + target_safe_index2
    target_value2 = tl.load(target_ptrs2, mask=target_mask2, other=BLANK)
    labels2 = target_value2

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            denom = tl.maximum(target_len, 1).to(tl.float32) * N
            scale = scale / denom

    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    beta_init = tl.where(
        ((states == state_count - 1) | ((states == state_count - 2) & (target_len > 0)))
        & valid_state
        & (input_len > 0),
        0.0,
        -float("inf"),
    )
    scratch_batch = scratch_beta + batch * 2 * STATE_COUNT_MAX
    tl.store(scratch_batch + states, beta_init, mask=stored_state)
    # Must synchronize before reading β: β's initial value may be written by another warp (omitting this causes -inf to
    # spread,
    # appearing as data-dependent random wrong values).
    _debug_barrier()
    log_likelihood = tl.where(scale != 0.0, -nll, 0.0)
    out_batch = posterior_out + batch * POST_T * POST_STRIDE

    for step in tl.range(0, T, num_stages=1):
        t = input_len - 1 - step
        active = t >= 0
        safe_t = tl.where(active, t, 0)
        beta_base = scratch_batch + (step % 2) * STATE_COUNT_MAX
        next_beta_base = scratch_batch + ((step + 1) % 2) * STATE_COUNT_MAX
        beta = tl.load(beta_base + states, mask=stored_state, other=-float("inf")).to(
            tl.float32
        )

        alpha_t = tl.load(
            log_alpha + batch * T * STATE_COUNT_MAX + safe_t * STATE_COUNT_MAX + states,
            mask=active & stored_state,
            other=-float("inf"),
        ).to(tl.float32)
        log_post = alpha_t + beta - log_likelihood
        posterior = tl.where(
            active & valid_state & (scale != 0.0),
            tl.exp(log_post),
            0.0,
        )
        # Unique-address store (no conflict with any other lane), handing -scale*posterior to the cross-cutting
        # kernel for per-class reduction.
        tl.store(
            out_batch + safe_t * POST_STRIDE + states,
            -scale * posterior,
            mask=active & stored_state,
        )

        stay = beta + tl.load(
            log_probs + safe_t * N * C + batch * C + labels,
            mask=active & valid_state,
            other=-float("inf"),
        ).to(tl.float32)
        next1 = tl.load(
            beta_base + states + 1,
            mask=(states + 1 < state_count) & stored_state,
            other=-float("inf"),
        ).to(tl.float32) + tl.load(
            log_probs + safe_t * N * C + batch * C + labels1,
            mask=active & (states + 1 < state_count) & stored_state,
            other=-float("inf"),
        ).to(
            tl.float32
        )
        skip_allowed = (
            (~is_blank_state)
            & (states + 2 < state_count)
            & (target_value != target_value2)
        )
        next2 = tl.load(
            beta_base + states + 2,
            mask=(states + 2 < state_count) & stored_state,
            other=-float("inf"),
        ).to(tl.float32) + tl.load(
            log_probs + safe_t * N * C + batch * C + labels2,
            mask=active & skip_allowed & stored_state,
            other=-float("inf"),
        ).to(
            tl.float32
        )

        beta_next = _logaddexp3(stay, next1, next2, skip_allowed)
        tl.store(
            next_beta_base + states,
            tl.where(active, beta_next, -float("inf")),
            mask=stored_state,
        )
        _debug_barrier()


@libentry()
@triton.jit
def _ctc_loss_backward_scatter_kernel(
    posterior,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    grad_input,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Reduces the posterior of one (batch, t) into grad_input[t, batch, :].

    After flattening into a 1-D grid (x cap 65535) a stride loop covers all (batch, t): each program
    runs a constant number of instructions with no cross dependencies, so the fixed atomic latency
    is fully hidden by parallelism -- the key to moving atomics out of the serial T loop.
    """
    states = tl.arange(0, BLOCK_S)
    total_pairs = N * T
    for pair_index in tl.range(tl.program_id(0), total_pairs, tl.num_programs(0)):
        batch = pair_index // T
        t = pair_index % T
        input_len = tl.load(input_lengths + batch)
        if t < input_len:
            target_len = tl.load(target_lengths + batch)
            state_count = target_len * 2 + 1
            valid_state = states < state_count
            stored_state = states < STATE_COUNT_MAX
            is_blank_state = (states % 2) == 0

            target_index = (states - 1) // 2
            target_mask = (target_index >= 0) & (target_index < target_len)
            target_safe_index = tl.where(target_mask, target_index, 0)
            if TARGET_1D:
                target_origin = tl.load(target_offsets + batch)
            else:
                target_origin = batch * MAX_TARGET
            target_value = tl.load(
                targets + target_origin + target_safe_index,
                mask=target_mask,
                other=BLANK,
            )
            labels = tl.where(is_blank_state, BLANK, target_value)  # noqa: F841
            pair_index_all = tl.arange(0, BLOCK_P)
            stored_pair_1 = (2 * pair_index_all + 1) < STATE_COUNT_MAX
            stored_pair_2 = (2 * pair_index_all + 2) < STATE_COUNT_MAX  # noqa: F841

            post = tl.load(
                posterior + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states,
                mask=valid_state & stored_state,
                other=0.0,
            ).to(tl.float32)

            # State layout: even states = blank, odd state 2j+1 = target[j]. Therefore pick lanes by
            # target index (BLOCK_P lanes), each lane issuing only one atomic,
            # reducing the number of atomic lanes from 2*target_len+1 to target_len (GCU400's
            # atomic throughput is about 3.6ns/lane, the sole bottleneck of this kernel).
            pair_lanes = tl.arange(0, BLOCK_P)
            pair_valid = pair_lanes < target_len
            base_post = posterior + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX
            odd_state = 2 * pair_lanes + 1
            odd_post = tl.load(
                base_post + odd_state,
                mask=pair_valid & (odd_state < state_count) & stored_pair_1,
                other=0.0,
            ).to(tl.float32)
            pair_label = tl.load(
                targets + target_origin + pair_lanes, mask=pair_valid, other=BLANK
            )
            blank_target = pair_label == BLANK

            row = grad_input + t * N * C + batch * C
            # The blank class has only one address per row: reduce the even states and the odd states whose target is
            # exactly blank
            # to a scalar first, then issue a single atomic from just 1 lane.
            blank_sum = tl.sum(
                tl.where(is_blank_state & valid_state, post, 0.0)
            ) + tl.sum(tl.where(pair_valid & blank_target, odd_post, 0.0))
            lane0 = states == 0
            tl.atomic_add(
                row + BLANK + states * 0,
                blank_sum + states * 0.0,
                sem="relaxed",
                mask=lane0 & stored_state,
            )
            tl.atomic_add(
                row + pair_label,
                odd_post,
                sem="relaxed",
                mask=pair_valid & (~blank_target),
            )


_GEMM_REDUCE_KT = 64


def _use_gemm_reduce(block_s, block_c):
    # tl.dot needs all fragment dims >= 16; the one-hot tile
    # [BLOCK_S, BLOCK_C] fp32 plus operands must stay under the 448KB DSM
    # concurrency line, so cap both at 256x128.
    return 16 <= block_s <= 256 and 16 <= block_c <= 128


@libentry()
@triton.jit
def _ctc_loss_backward_gemm_reduce_kernel(
    posterior,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    grad_output,
    grad_input,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    POST_STRIDE: tl.constexpr,
    POST_T: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KT: tl.constexpr,
):
    """Per-(batch, KT-timestep-chunk) row-block reduction, replacing the scatter+init atomic path.

    Each (batch, t) row gradient is written exclusively by this program (no atomics): the posterior
    scratch already carries the -scale factor, and the per-class delta is computed in one shot with
    a one-hot matrix `tl.dot(post[KT, S], onehot[S, C])`; the base row (exp(logp)*scale and the NaN
    semantics) replicates the init kernel, and rows with t >= input_len are written as 0, so this arm
    no longer needs an init kernel. The posterior scratch is host zero-filled and padded in both the
    state and t dims (POST_STRIDE/POST_T), so the loads feeding dot are pure affine and mask-free --
    a masked load fed straight to tl.dot drags garbage from masked rows into the result at nw>1 here.
    """
    num_chunks = (T + KT - 1) // KT
    total_work = N * num_chunks
    states = tl.arange(0, BLOCK_S)
    c_range = tl.arange(0, BLOCK_C)
    c_mask = c_range < C

    for work in tl.range(tl.program_id(0), total_work, tl.num_programs(0)):
        batch = work // num_chunks
        chunk = work % num_chunks
        t_idx = chunk * KT + tl.arange(0, KT)
        t_valid = t_idx < T

        input_len = tl.load(input_lengths + batch)
        t_active = t_valid & (t_idx < input_len)

        target_len = tl.load(target_lengths + batch)
        state_count = target_len * 2 + 1  # noqa: F841

        is_blank_state = (states % 2) == 0
        target_index = (states - 1) // 2
        target_mask = (target_index >= 0) & (target_index < target_len)
        target_safe_index = tl.where(target_mask, target_index, 0)
        if TARGET_1D:
            target_origin = tl.load(target_offsets + batch)
        else:
            target_origin = batch * MAX_TARGET
        target_value = tl.load(
            targets + target_origin + target_safe_index,
            mask=target_mask,
            other=BLANK,
        )
        labels = tl.where(is_blank_state, BLANK, target_value).to(tl.int32)

        onehot = tl.where(labels[:, None] == c_range[None, :], 1.0, 0.0)

        post_ptrs = (
            posterior
            + batch * POST_T * POST_STRIDE
            + t_idx[:, None] * POST_STRIDE
            + states[None, :]
        )
        post = tl.load(post_ptrs).to(tl.float32)

        delta = tl.dot(post, onehot)

        nll = tl.load(neg_log_likelihood + batch).to(tl.float32)
        if REDUCTION == 0:
            scale = tl.load(grad_output + batch).to(tl.float32)
        else:
            scale = tl.load(grad_output).to(tl.float32)
            if REDUCTION == 1:
                denom = tl.maximum(target_len, 1).to(tl.float32) * N
                scale = scale / denom

        logp_ptrs = log_probs + t_idx[:, None] * (N * C) + batch * C + c_range[None, :]
        logp = tl.load(
            logp_ptrs,
            mask=t_valid[:, None] & c_mask[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        base = tl.where(t_active[:, None] & c_mask[None, :], tl.exp(logp) * scale, 0.0)
        base = tl.where(
            t_active[:, None]
            & c_mask[None, :]
            & (scale != 0.0)
            & (logp == -float("inf")),
            float("nan"),
            base,
        )
        if not ZERO_INFINITY:
            base = tl.where(
                t_active[:, None] & c_mask[None, :] & (nll == float("inf")),
                float("nan"),
                base,
            )

        grad_ptrs = grad_input + t_idx[:, None] * (N * C) + batch * C + c_range[None, :]
        tl.store(
            grad_ptrs,
            base + delta,
            mask=t_valid[:, None] & c_mask[None, :],
        )


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
    # The GCU400 Triton compile pipeline does not support fp64 kernels: promote to fp32 on the host,
    # work_grad / work_log_probs are converted to fp32 accordingly, then converted back to the caller's dtype on return.
    if compute_dtype == torch.float64:
        compute_dtype = torch.float32
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

    input_stats, target_stats = _length_stats(
        work_input_lengths, work_target_lengths, batch_size
    )
    min_input_length, max_input_length, _ = input_stats
    min_target_length, max_target, total_target_length = target_stats

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
    logger.debug("GEMS_ENFLAME _CTC_LOSS_BACKWARD")

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
    block_p = triton.next_power_of_2(max(target_stride, 1))

    grad_log_probs = torch.empty_like(work_log_probs)
    total = work_log_probs.numel()
    block = _INIT_GRAD_BLOCK
    time_steps = work_log_probs.shape[0]
    classes = work_log_probs.shape[2]
    block_c = triton.next_power_of_2(classes)
    use_gemm = _use_gemm_reduce(block_s, block_c)

    with torch_device_fn.device(log_probs.device):
        if use_gemm:
            # GEMM-reduce arm: the posterior scratch uses BLOCK_S as row width, with the t dimension padded to KT
            # alignment and zero-filled — loads feeding dot therefore need no mask and no clamping;
            # grad rows are written a whole row at a time exclusively by the gemm kernel (including zero rows for
            # t>=input_len),
            # so the init kernel is no longer needed.
            num_chunks = triton.cdiv(time_steps, _GEMM_REDUCE_KT)
            post_t = num_chunks * _GEMM_REDUCE_KT
            post_stride = block_s
            posterior = torch.zeros(
                (batch_size, post_t, post_stride),
                dtype=torch.float32,
                device=log_probs.device,
            )
        else:
            post_t = time_steps
            post_stride = state_count_max
            posterior = torch.empty(
                (batch_size, time_steps, state_count_max),
                dtype=torch.float32,
                device=log_probs.device,
            )
            _ctc_loss_init_grad_kernel[(triton.cdiv(total, block),)](
                work_log_probs,
                work_input_lengths,
                work_target_lengths,
                work_neg_log_likelihood,
                work_grad,
                grad_log_probs,
                total,
                time_steps,
                batch_size,
                classes,
                _REDUCTION_NONE,
                zero_infinity,
                block,
                num_warps=_INIT_GRAD_WARPS,
            )

        scratch_beta = torch.empty(
            (batch_size, 2, state_count_max),
            dtype=torch.float32,
            device=log_probs.device,
        )

        _ctc_loss_backward_posterior_kernel[(batch_size,)](
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            work_neg_log_likelihood,
            work_log_alpha,
            work_grad,
            posterior,
            scratch_beta,
            time_steps,
            batch_size,
            classes,
            target_stride,
            state_count_max,
            post_stride,
            post_t,
            blank,
            target_1d,
            _REDUCTION_NONE,
            zero_infinity,
            block_s,
            num_warps=_warps_for(block_s),
        )

        if use_gemm:
            gemm_grid = min(batch_size * num_chunks, 65535)
            _ctc_loss_backward_gemm_reduce_kernel[(gemm_grid,)](
                posterior,
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                work_neg_log_likelihood,
                work_grad,
                grad_log_probs,
                time_steps,
                batch_size,
                classes,
                target_stride,
                state_count_max,
                post_stride,
                post_t,
                blank,
                target_1d,
                _REDUCTION_NONE,
                zero_infinity,
                block_s,
                block_c,
                _GEMM_REDUCE_KT,
                num_warps=1,
            )
        else:
            # 1D grid: the x dimension is capped at 65535, and the kernel loops by stride internally to cover all
            # (batch, t)
            scatter_grid = min(batch_size * time_steps, 65535)
            _ctc_loss_backward_scatter_kernel[(scatter_grid,)](
                posterior,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                grad_log_probs,
                time_steps,
                batch_size,
                classes,
                target_stride,
                state_count_max,
                blank,
                target_1d,
                block_s,
                block_p,
                num_warps=_warps_for(block_s),
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
    logger.debug("GEMS_ENFLAME _CTC_LOSS_BACKWARD_OUT")

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


# Same as ``_ctc_loss``: callers that import from ``flag_gems.ops`` bypass the
# backend registration, so publish the gcu400 wrapper there as well.
import flag_gems.ops as _common_ops  # noqa: E402

_common_ops._ctc_loss_backward = _ctc_loss_backward
_common_ops._ctc_loss_backward_out = _ctc_loss_backward_out
