import logging
from typing import Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2
_TRITON_CTC_LOSS_DTYPES = {
    torch.float16,
    torch.float32,
    torch.bfloat16,
}


@triton.jit
def _ctc_loss_forward_kernel(
    log_probs_ptr,
    targets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    losses_ptr,
    label_prev_ptr,
    alpha_blank_ptr,
    alpha_label_ptr,
    cum_log_scale_ptr,
    stride_t,
    stride_n,
    stride_c,
    target_stride_n,
    label_stride_n,
    N_batch,
    T: tl.constexpr,
    STATE_PAD: tl.constexpr,
    BLANK: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    STORE_ALPHA: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, STATE_PAD)

    input_length = tl.load(input_lengths_ptr + pid).to(tl.int32)
    target_length = tl.load(target_lengths_ptr + pid).to(tl.int32)
    valid_blanks = lanes <= target_length
    valid_labels = lanes < target_length
    prev_lanes = tl.where(lanes > 0, lanes - 1, 0)

    labels = tl.load(
        targets_ptr + pid * target_stride_n + lanes,
        mask=valid_labels,
        other=BLANK,
    ).to(tl.int32)
    prev_labels = tl.load(
        targets_ptr + pid * target_stride_n + prev_lanes,
        mask=valid_labels & (lanes > 0),
        other=BLANK,
    ).to(tl.int32)
    can_skip = valid_labels & (lanes > 0) & (labels != prev_labels)

    if input_length == 0:
        loss = tl.where(target_length == 0, 0.0, float("inf"))
        if ZERO_INFINITY:
            loss = tl.where(loss == float("inf"), 0.0, loss)
        tl.store(losses_ptr + pid, loss)
        return

    base_addr = log_probs_ptr + pid * stride_n
    log_scale_sum = 0.0

    blank = tl.where(
        lanes == 0,
        tl.exp(tl.load(base_addr + BLANK * stride_c)),
        0.0,
    )
    label = tl.full((STATE_PAD,), 0.0, tl.float32)
    if STATE_PAD > 0:
        first_target = tl.load(
            targets_ptr + pid * target_stride_n,
            mask=target_length > 0,
            other=BLANK,
        )
        label = tl.where(
            (lanes == 0) & (target_length > 0),
            tl.exp(tl.load(base_addr + first_target * stride_c)),
            label,
        )

    if STORE_ALPHA:
        a_off = pid * STATE_PAD
        tl.store(alpha_blank_ptr + a_off + lanes, blank, mask=valid_blanks)
        tl.store(alpha_label_ptr + a_off + lanes, label, mask=valid_labels)
        tl.store(cum_log_scale_ptr + pid, 0.0)

    for t in range(1, T):
        if t < input_length:
            tl.store(
                label_prev_ptr + pid * label_stride_n + lanes,
                label,
                mask=valid_labels,
            )
            prev_label = tl.load(
                label_prev_ptr + pid * label_stride_n + prev_lanes,
                mask=valid_blanks & (lanes > 0),
                other=0.0,
            )

            p_blank = tl.exp(tl.load(base_addr + t * stride_t + BLANK * stride_c))
            label_emit_log = tl.load(
                base_addr + t * stride_t + labels * stride_c,
                mask=valid_labels,
                other=0.0,
            )
            p_label = tl.where(valid_labels, tl.exp(label_emit_log), 0.0)

            blank_next = tl.where(valid_blanks, (blank + prev_label) * p_blank, 0.0)
            skip = tl.where(can_skip, prev_label, 0.0)
            label = tl.where(valid_labels, (label + blank + skip) * p_label, 0.0)
            blank = blank_next

            if t % 4 == 0:
                scale = tl.maximum(
                    tl.max(blank, axis=0),
                    tl.max(label, axis=0),
                )
                inv_scale = tl.where(scale > 0.0, 1.0 / scale, 1.0)
                blank = blank * inv_scale
                label = label * inv_scale
                log_scale_sum += tl.where(scale > 0.0, tl.log(scale), 0.0)

            if STORE_ALPHA:
                a_off = (t * N_batch + pid) * STATE_PAD
                tl.store(alpha_blank_ptr + a_off + lanes, blank, mask=valid_blanks)
                tl.store(alpha_label_ptr + a_off + lanes, label, mask=valid_labels)
                tl.store(cum_log_scale_ptr + t * N_batch + pid, log_scale_sum)

    last_blank = tl.max(tl.where(lanes == target_length, blank, 0.0), axis=0)
    last_label = tl.max(
        tl.where((target_length > 0) & (lanes == (target_length - 1)), label, 0.0),
        axis=0,
    )
    total_prob = last_blank + last_label
    log_likelihood = tl.where(
        total_prob > 0.0,
        tl.log(total_prob) + log_scale_sum,
        float("-inf"),
    )
    loss = -log_likelihood
    if ZERO_INFINITY:
        loss = tl.where(loss == float("inf"), 0.0, loss)
    tl.store(losses_ptr + pid, loss)


@triton.jit
def _ctc_loss_forward_concat_kernel(
    log_probs_ptr,
    targets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    losses_ptr,
    label_prev_ptr,
    alpha_blank_ptr,
    alpha_label_ptr,
    cum_log_scale_ptr,
    stride_t,
    stride_n,
    stride_c,
    label_stride_n,
    N_batch,
    T: tl.constexpr,
    STATE_PAD: tl.constexpr,
    BATCH_PAD: tl.constexpr,
    BLANK: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    STORE_ALPHA: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, STATE_PAD)

    batch_idx = tl.arange(0, BATCH_PAD)
    all_tgt_len = tl.load(
        target_lengths_ptr + batch_idx, mask=batch_idx < pid, other=0
    ).to(tl.int32)
    target_base = tl.sum(all_tgt_len, axis=0)

    input_length = tl.load(input_lengths_ptr + pid).to(tl.int32)
    target_length = tl.load(target_lengths_ptr + pid).to(tl.int32)

    valid_blanks = lanes <= target_length
    valid_labels = lanes < target_length
    prev_lanes = tl.where(lanes > 0, lanes - 1, 0)

    labels = tl.load(
        targets_ptr + target_base + lanes,
        mask=valid_labels,
        other=BLANK,
    ).to(tl.int32)
    prev_labels = tl.load(
        targets_ptr + target_base + prev_lanes,
        mask=valid_labels & (lanes > 0),
        other=BLANK,
    ).to(tl.int32)
    can_skip = valid_labels & (lanes > 0) & (labels != prev_labels)

    if input_length == 0:
        loss = tl.where(target_length == 0, 0.0, float("inf"))
        if ZERO_INFINITY:
            loss = tl.where(loss == float("inf"), 0.0, loss)
        tl.store(losses_ptr + pid, loss)
        return

    base_addr = log_probs_ptr + pid * stride_n
    log_scale_sum = 0.0

    blank = tl.where(
        lanes == 0,
        tl.exp(tl.load(base_addr + BLANK * stride_c)),
        0.0,
    )
    label = tl.full((STATE_PAD,), 0.0, tl.float32)
    if STATE_PAD > 0:
        first_target = tl.load(
            targets_ptr + target_base, mask=target_length > 0, other=BLANK
        )
        label = tl.where(
            (lanes == 0) & (target_length > 0),
            tl.exp(tl.load(base_addr + first_target * stride_c)),
            label,
        )

    if STORE_ALPHA:
        a_off = pid * STATE_PAD
        tl.store(alpha_blank_ptr + a_off + lanes, blank, mask=valid_blanks)
        tl.store(alpha_label_ptr + a_off + lanes, label, mask=valid_labels)
        tl.store(cum_log_scale_ptr + pid, 0.0)

    for t in range(1, T):
        if t < input_length:
            tl.store(
                label_prev_ptr + pid * label_stride_n + lanes,
                label,
                mask=valid_labels,
            )
            prev_label = tl.load(
                label_prev_ptr + pid * label_stride_n + prev_lanes,
                mask=valid_blanks & (lanes > 0),
                other=0.0,
            )

            p_blank = tl.exp(tl.load(base_addr + t * stride_t + BLANK * stride_c))
            label_emit_log = tl.load(
                base_addr + t * stride_t + labels * stride_c,
                mask=valid_labels,
                other=0.0,
            )
            p_label = tl.where(valid_labels, tl.exp(label_emit_log), 0.0)

            blank_next = tl.where(valid_blanks, (blank + prev_label) * p_blank, 0.0)
            skip = tl.where(can_skip, prev_label, 0.0)
            label = tl.where(valid_labels, (label + blank + skip) * p_label, 0.0)
            blank = blank_next

            if t % 4 == 0:
                scale = tl.maximum(
                    tl.max(blank, axis=0),
                    tl.max(label, axis=0),
                )
                inv_scale = tl.where(scale > 0.0, 1.0 / scale, 1.0)
                blank = blank * inv_scale
                label = label * inv_scale
                log_scale_sum += tl.where(scale > 0.0, tl.log(scale), 0.0)

            if STORE_ALPHA:
                a_off = (t * N_batch + pid) * STATE_PAD
                tl.store(alpha_blank_ptr + a_off + lanes, blank, mask=valid_blanks)
                tl.store(alpha_label_ptr + a_off + lanes, label, mask=valid_labels)
                tl.store(cum_log_scale_ptr + t * N_batch + pid, log_scale_sum)

    last_blank = tl.max(tl.where(lanes == target_length, blank, 0.0), axis=0)
    last_label = tl.max(
        tl.where((target_length > 0) & (lanes == (target_length - 1)), label, 0.0),
        axis=0,
    )
    total_prob = last_blank + last_label
    log_likelihood = tl.where(
        total_prob > 0.0,
        tl.log(total_prob) + log_scale_sum,
        float("-inf"),
    )
    loss = -log_likelihood
    if ZERO_INFINITY:
        loss = tl.where(loss == float("inf"), 0.0, loss)
    tl.store(losses_ptr + pid, loss)


# ---------------------------------------------------------------------------
# Backward kernels: beta DP (reverse time) + gradient scatter
# ---------------------------------------------------------------------------


@triton.jit
def _ctc_loss_backward_kernel(
    log_probs_ptr,
    targets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    alpha_blank_ptr,
    alpha_label_ptr,
    cum_log_scale_ptr,
    grad_scale_ptr,
    grad_log_probs_ptr,
    scratch_ptr,
    stride_t,
    stride_n,
    stride_c,
    grad_stride_t,
    grad_stride_n,
    grad_stride_c,
    target_stride_n,
    scratch_stride_n,
    N_batch,
    T: tl.constexpr,
    STATE_PAD: tl.constexpr,
    BLANK: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, STATE_PAD)

    input_length = tl.load(input_lengths_ptr + pid).to(tl.int32)
    target_length = tl.load(target_lengths_ptr + pid).to(tl.int32)

    if input_length == 0:
        return

    g_scale = tl.load(grad_scale_ptr + pid)

    valid_blanks = lanes <= target_length
    valid_labels = lanes < target_length

    # Load target labels
    labels = tl.load(
        targets_ptr + pid * target_stride_n + lanes,
        mask=valid_labels,
        other=BLANK,
    ).to(tl.int32)

    # For backward shift: labels[k+1] and can_skip[k+1]
    next_labels = tl.load(
        targets_ptr + pid * target_stride_n + (lanes + 1),
        mask=(lanes + 1) < target_length,
        other=BLANK,
    ).to(tl.int32)
    can_skip_next = (lanes < target_length - 1) & (next_labels != labels)

    base_addr = log_probs_ptr + pid * stride_n
    grad_base = grad_log_probs_ptr + pid * grad_stride_n

    # Compute P_r from stored alpha at t = input_length - 1
    t_last = input_length - 1
    a_off_last = (t_last * N_batch + pid) * STATE_PAD
    alpha_b_last = tl.load(
        alpha_blank_ptr + a_off_last + lanes, mask=valid_blanks, other=0.0
    )
    alpha_l_last = tl.load(
        alpha_label_ptr + a_off_last + lanes, mask=valid_labels, other=0.0
    )
    cum_scale_a_last = tl.load(cum_log_scale_ptr + t_last * N_batch + pid)

    P_r = tl.sum(tl.where(lanes == target_length, alpha_b_last, 0.0), axis=0) + tl.sum(
        tl.where(
            (target_length > 0) & (lanes == (target_length - 1)), alpha_l_last, 0.0
        ),
        axis=0,
    )

    if P_r <= 0.0:
        return

    log_P_r = tl.log(P_r)

    # Initialize beta at t = input_length - 1
    beta_blank = tl.where(lanes == target_length, 1.0, 0.0)
    beta_label = tl.where(
        (target_length > 0) & (lanes == (target_length - 1)), 1.0, 0.0
    )
    log_scale_beta = 0.0

    # Gradient at t = input_length - 1
    factor = tl.exp(cum_scale_a_last + log_scale_beta - cum_scale_a_last - log_P_r)
    blank_grad = (
        -g_scale
        * factor
        * tl.sum(tl.where(valid_blanks, alpha_b_last * beta_blank, 0.0), axis=0)
    )
    tl.atomic_add(
        grad_base + t_last * grad_stride_t + BLANK * grad_stride_c, blank_grad
    )
    label_grad = (
        -g_scale * factor * tl.where(valid_labels, alpha_l_last * beta_label, 0.0)
    )
    tl.atomic_add(
        grad_base + t_last * grad_stride_t + labels * grad_stride_c,
        label_grad,
        mask=valid_labels,
    )

    # Backward loop
    for t_offset in range(1, T):
        t = T - 1 - t_offset
        if t >= 0 and t < input_length - 1:
            # Emission probs at t+1 (for transition t -> t+1)
            p_blank_t1 = tl.exp(
                tl.load(base_addr + (t + 1) * stride_t + BLANK * stride_c)
            )
            label_emit_t1 = tl.load(
                base_addr + (t + 1) * stride_t + labels * stride_c,
                mask=valid_labels,
                other=0.0,
            )
            p_label_t1 = tl.where(valid_labels, tl.exp(label_emit_t1), 0.0)

            next_label_emit_t1 = tl.load(
                base_addr + (t + 1) * stride_t + next_labels * stride_c,
                mask=can_skip_next,
                other=0.0,
            )
            p_label_next_t1 = tl.where(can_skip_next, tl.exp(next_label_emit_t1), 0.0)

            # Shift beta left: store both to separate scratch regions, then load
            scratch_base = scratch_ptr + pid * scratch_stride_n
            tl.store(scratch_base + lanes, beta_blank, mask=valid_blanks)
            tl.store(scratch_base + STATE_PAD + lanes, beta_label, mask=valid_labels)
            tl.debug_barrier()
            next_beta_blank = tl.load(
                scratch_base + (lanes + 1),
                mask=lanes < target_length,
                other=0.0,
            )
            next_beta_label = tl.load(
                scratch_base + STATE_PAD + (lanes + 1),
                mask=(lanes + 1) < target_length,
                other=0.0,
            )

            # Beta transitions
            beta_blank_new = tl.where(
                valid_blanks,
                beta_blank * p_blank_t1
                + tl.where(valid_labels, beta_label * p_label_t1, 0.0),
                0.0,
            )
            skip_term = tl.where(can_skip_next, next_beta_label * p_label_next_t1, 0.0)
            beta_label_new = tl.where(
                valid_labels,
                beta_label * p_label_t1 + next_beta_blank * p_blank_t1 + skip_term,
                0.0,
            )

            beta_blank = beta_blank_new
            beta_label = beta_label_new

            # Rescale every 4 steps
            if t_offset % 4 == 0:
                scale = tl.maximum(
                    tl.max(beta_blank, axis=0),
                    tl.max(beta_label, axis=0),
                )
                inv_scale = tl.where(scale > 0.0, 1.0 / scale, 1.0)
                beta_blank = beta_blank * inv_scale
                beta_label = beta_label * inv_scale
                log_scale_beta += tl.where(scale > 0.0, tl.log(scale), 0.0)

            # Gradient at time t
            a_off = (t * N_batch + pid) * STATE_PAD
            alpha_b = tl.load(
                alpha_blank_ptr + a_off + lanes, mask=valid_blanks, other=0.0
            )
            alpha_l = tl.load(
                alpha_label_ptr + a_off + lanes, mask=valid_labels, other=0.0
            )
            cum_scale_a = tl.load(cum_log_scale_ptr + t * N_batch + pid)

            factor = tl.exp(cum_scale_a + log_scale_beta - cum_scale_a_last - log_P_r)

            blank_grad = (
                -g_scale
                * factor
                * tl.sum(tl.where(valid_blanks, alpha_b * beta_blank, 0.0), axis=0)
            )
            tl.atomic_add(
                grad_base + t * grad_stride_t + BLANK * grad_stride_c, blank_grad
            )

            label_grad = (
                -g_scale * factor * tl.where(valid_labels, alpha_l * beta_label, 0.0)
            )
            tl.atomic_add(
                grad_base + t * grad_stride_t + labels * grad_stride_c,
                label_grad,
                mask=valid_labels,
            )


@triton.jit
def _ctc_loss_backward_concat_kernel(
    log_probs_ptr,
    targets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    alpha_blank_ptr,
    alpha_label_ptr,
    cum_log_scale_ptr,
    grad_scale_ptr,
    grad_log_probs_ptr,
    scratch_ptr,
    stride_t,
    stride_n,
    stride_c,
    grad_stride_t,
    grad_stride_n,
    grad_stride_c,
    scratch_stride_n,
    N_batch,
    T: tl.constexpr,
    STATE_PAD: tl.constexpr,
    BATCH_PAD: tl.constexpr,
    BLANK: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, STATE_PAD)

    # Compute target offset (same as concat forward)
    batch_idx = tl.arange(0, BATCH_PAD)
    all_tgt_len = tl.load(
        target_lengths_ptr + batch_idx, mask=batch_idx < pid, other=0
    ).to(tl.int32)
    target_base = tl.sum(all_tgt_len, axis=0)

    input_length = tl.load(input_lengths_ptr + pid).to(tl.int32)
    target_length = tl.load(target_lengths_ptr + pid).to(tl.int32)

    if input_length == 0:
        return

    g_scale = tl.load(grad_scale_ptr + pid)

    valid_blanks = lanes <= target_length
    valid_labels = lanes < target_length

    labels = tl.load(
        targets_ptr + target_base + lanes,
        mask=valid_labels,
        other=BLANK,
    ).to(tl.int32)

    next_labels = tl.load(
        targets_ptr + target_base + (lanes + 1),
        mask=(lanes + 1) < target_length,
        other=BLANK,
    ).to(tl.int32)
    can_skip_next = (lanes < target_length - 1) & (next_labels != labels)

    base_addr = log_probs_ptr + pid * stride_n
    grad_base = grad_log_probs_ptr + pid * grad_stride_n

    # Compute P_r from stored alpha at t = input_length - 1
    t_last = input_length - 1
    a_off_last = (t_last * N_batch + pid) * STATE_PAD
    alpha_b_last = tl.load(
        alpha_blank_ptr + a_off_last + lanes, mask=valid_blanks, other=0.0
    )
    alpha_l_last = tl.load(
        alpha_label_ptr + a_off_last + lanes, mask=valid_labels, other=0.0
    )
    cum_scale_a_last = tl.load(cum_log_scale_ptr + t_last * N_batch + pid)

    P_r = tl.sum(tl.where(lanes == target_length, alpha_b_last, 0.0), axis=0) + tl.sum(
        tl.where(
            (target_length > 0) & (lanes == (target_length - 1)), alpha_l_last, 0.0
        ),
        axis=0,
    )

    if P_r <= 0.0:
        return

    log_P_r = tl.log(P_r)

    # Initialize beta at t = input_length - 1
    beta_blank = tl.where(lanes == target_length, 1.0, 0.0)
    beta_label = tl.where(
        (target_length > 0) & (lanes == (target_length - 1)), 1.0, 0.0
    )
    log_scale_beta = 0.0

    # Gradient at t = input_length - 1
    factor = tl.exp(cum_scale_a_last + log_scale_beta - cum_scale_a_last - log_P_r)
    blank_grad = (
        -g_scale
        * factor
        * tl.sum(tl.where(valid_blanks, alpha_b_last * beta_blank, 0.0), axis=0)
    )
    tl.atomic_add(
        grad_base + t_last * grad_stride_t + BLANK * grad_stride_c, blank_grad
    )
    label_grad = (
        -g_scale * factor * tl.where(valid_labels, alpha_l_last * beta_label, 0.0)
    )
    tl.atomic_add(
        grad_base + t_last * grad_stride_t + labels * grad_stride_c,
        label_grad,
        mask=valid_labels,
    )

    # Backward loop
    for t_offset in range(1, T):
        t = T - 1 - t_offset
        if t >= 0 and t < input_length - 1:
            p_blank_t1 = tl.exp(
                tl.load(base_addr + (t + 1) * stride_t + BLANK * stride_c)
            )
            label_emit_t1 = tl.load(
                base_addr + (t + 1) * stride_t + labels * stride_c,
                mask=valid_labels,
                other=0.0,
            )
            p_label_t1 = tl.where(valid_labels, tl.exp(label_emit_t1), 0.0)

            next_label_emit_t1 = tl.load(
                base_addr + (t + 1) * stride_t + next_labels * stride_c,
                mask=can_skip_next,
                other=0.0,
            )
            p_label_next_t1 = tl.where(can_skip_next, tl.exp(next_label_emit_t1), 0.0)

            # Shift beta left: store both to separate scratch regions, then load
            scratch_base = scratch_ptr + pid * scratch_stride_n
            tl.store(scratch_base + lanes, beta_blank, mask=valid_blanks)
            tl.store(scratch_base + STATE_PAD + lanes, beta_label, mask=valid_labels)
            tl.debug_barrier()
            next_beta_blank = tl.load(
                scratch_base + (lanes + 1),
                mask=lanes < target_length,
                other=0.0,
            )
            next_beta_label = tl.load(
                scratch_base + STATE_PAD + (lanes + 1),
                mask=(lanes + 1) < target_length,
                other=0.0,
            )

            beta_blank_new = tl.where(
                valid_blanks,
                beta_blank * p_blank_t1
                + tl.where(valid_labels, beta_label * p_label_t1, 0.0),
                0.0,
            )
            skip_term = tl.where(can_skip_next, next_beta_label * p_label_next_t1, 0.0)
            beta_label_new = tl.where(
                valid_labels,
                beta_label * p_label_t1 + next_beta_blank * p_blank_t1 + skip_term,
                0.0,
            )

            beta_blank = beta_blank_new
            beta_label = beta_label_new

            if t_offset % 4 == 0:
                scale = tl.maximum(
                    tl.max(beta_blank, axis=0),
                    tl.max(beta_label, axis=0),
                )
                inv_scale = tl.where(scale > 0.0, 1.0 / scale, 1.0)
                beta_blank = beta_blank * inv_scale
                beta_label = beta_label * inv_scale
                log_scale_beta += tl.where(scale > 0.0, tl.log(scale), 0.0)

            a_off = (t * N_batch + pid) * STATE_PAD
            alpha_b = tl.load(
                alpha_blank_ptr + a_off + lanes, mask=valid_blanks, other=0.0
            )
            alpha_l = tl.load(
                alpha_label_ptr + a_off + lanes, mask=valid_labels, other=0.0
            )
            cum_scale_a = tl.load(cum_log_scale_ptr + t * N_batch + pid)

            factor = tl.exp(cum_scale_a + log_scale_beta - cum_scale_a_last - log_P_r)

            blank_grad = (
                -g_scale
                * factor
                * tl.sum(tl.where(valid_blanks, alpha_b * beta_blank, 0.0), axis=0)
            )
            tl.atomic_add(
                grad_base + t * grad_stride_t + BLANK * grad_stride_c, blank_grad
            )

            label_grad = (
                -g_scale * factor * tl.where(valid_labels, alpha_l * beta_label, 0.0)
            )
            tl.atomic_add(
                grad_base + t * grad_stride_t + labels * grad_stride_c,
                label_grad,
                mask=valid_labels,
            )


# ---------------------------------------------------------------------------
# Python helpers (mostly unchanged)
# ---------------------------------------------------------------------------


def _normalize_targets(
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    targets = targets.to(device=device, dtype=torch.long)
    if targets.ndim == 2:
        if targets.size(0) != batch_size:
            raise RuntimeError(
                "Expected padded targets to have the same batch dimension as log_probs"
            )
        return targets

    if targets.ndim == 1:
        return targets

    raise RuntimeError(
        f"ctc_loss expects targets to be 1D or 2D, but got {targets.ndim}D"
    )


def _normalize_reduction(reduction: int | str) -> int:
    if isinstance(reduction, str):
        reduction_map = {
            "none": _REDUCTION_NONE,
            "mean": _REDUCTION_MEAN,
            "sum": _REDUCTION_SUM,
        }
        if reduction not in reduction_map:
            raise RuntimeError(f"Unsupported reduction for ctc_loss: {reduction}")
        return reduction_map[reduction]
    return int(reduction)


def _normalize_log_probs(log_probs: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if log_probs.ndim == 2:
        return log_probs.unsqueeze(1), False
    if log_probs.ndim == 3:
        return log_probs, True
    raise RuntimeError(
        f"ctc_loss expects log_probs to be 2D or 3D, but got {log_probs.ndim}D"
    )


def _normalize_lengths(
    lengths: torch.Tensor | Sequence[int] | int,
    batch_size: int,
    device: torch.device,
    arg_name: str,
) -> torch.Tensor:
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(device=device, dtype=torch.long)
    elif isinstance(lengths, (list, tuple)):
        lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    elif isinstance(lengths, int):
        lengths = torch.tensor([lengths], device=device, dtype=torch.long)
    else:
        raise TypeError(f"{arg_name} must be a Tensor, list, tuple, or int")

    if lengths.ndim == 0:
        lengths = lengths.reshape(1)
    else:
        lengths = lengths.reshape(-1)

    if lengths.numel() != batch_size:
        raise RuntimeError(
            f"Expected {arg_name} to have {batch_size} elements, but got {lengths.numel()}"
        )
    return lengths


def _prepare_triton_forward_inputs(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor | Sequence[int] | int,
    target_lengths: torch.Tensor | Sequence[int] | int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    log_probs, is_batched = _normalize_log_probs(log_probs)
    batch_size = log_probs.shape[1]
    input_lengths = _normalize_lengths(
        input_lengths, batch_size, log_probs.device, "input_lengths"
    )
    target_lengths = _normalize_lengths(
        target_lengths, batch_size, log_probs.device, "target_lengths"
    )
    if targets.dtype != torch.long:
        raise RuntimeError("ctc_loss expects targets to have dtype torch.long")
    normalized_targets = _normalize_targets(targets, batch_size, log_probs.device)
    return log_probs, normalized_targets, input_lengths, target_lengths, is_batched


def _supports_triton_ctc_loss(log_probs: torch.Tensor, blank: int) -> bool:
    return (
        log_probs.device.type == "cuda"
        and log_probs.dtype in _TRITON_CTC_LOSS_DTYPES
        and 0 <= blank < log_probs.shape[-1]
    )


def _to_triton_compute_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs if log_probs.dtype == torch.float32 else log_probs.float()


def _maybe_prepare_triton_forward_inputs(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool] | None:
    try:
        prepared = _prepare_triton_forward_inputs(
            log_probs, targets, input_lengths, target_lengths
        )
        normalized_log_probs, _, _, _, _ = prepared
    except Exception:
        return None

    return prepared if _supports_triton_ctc_loss(normalized_log_probs, blank) else None


def _can_use_triton_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
    blank: int,
) -> bool:
    return (
        _maybe_prepare_triton_forward_inputs(
            log_probs, targets, input_lengths, target_lengths, blank
        )
        is not None
    )


def _compute_kernel_params(targets, target_lengths, batch_size):
    """Compute state_pad, batch_pad, num_warps for the kernels."""
    if targets.ndim == 2:
        max_target_len = targets.shape[1]
        state_pad = triton.next_power_of_2(max_target_len + 1)
    else:
        max_target_len_t = target_lengths.max()
        state_pad = None

    if state_pad is None:
        max_target_len = int(max_target_len_t.item())
        state_pad = triton.next_power_of_2(max_target_len + 1)

    if state_pad <= 32:
        num_warps = 1
    elif state_pad <= 64:
        num_warps = 2
    else:
        num_warps = 4

    batch_pad = triton.next_power_of_2(batch_size) if targets.ndim == 1 else 0
    return state_pad, batch_pad, num_warps


def _backward_scratch_cols(state_pad: int) -> int:
    return 2 * state_pad


def _allocate_forward_buffers(
    log_probs: torch.Tensor,
    batch_size: int,
    state_pad: int,
    T: int,
    *,
    store_alpha: bool,
):
    device = log_probs.device
    losses = torch.empty((batch_size,), device=device, dtype=torch.float32)
    label_prev = torch.empty(
        (batch_size, state_pad), device=device, dtype=torch.float32
    )
    if not store_alpha:
        dummy = torch.empty(0, device=device, dtype=torch.float32)
        return losses, label_prev, dummy, dummy, dummy

    alpha_blank = torch.empty(
        (T, batch_size, state_pad), device=device, dtype=torch.float32
    )
    alpha_label = torch.empty(
        (T, batch_size, state_pad), device=device, dtype=torch.float32
    )
    cum_log_scale = torch.empty((T, batch_size), device=device, dtype=torch.float32)
    return losses, label_prev, alpha_blank, alpha_label, cum_log_scale


def _launch_forward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    losses,
    label_prev,
    alpha_blank,
    alpha_label,
    cum_log_scale,
    batch_size,
    state_pad,
    batch_pad,
    num_warps,
    blank,
    zero_infinity,
    store_alpha,
):
    """Launch forward kernel (padded or concat variant)."""
    T = log_probs.shape[0]
    if targets.ndim == 2:
        _ctc_loss_forward_kernel[(batch_size,)](
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            losses,
            label_prev,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            log_probs.stride(0),
            log_probs.stride(1),
            log_probs.stride(2),
            targets.stride(0),
            label_prev.stride(0),
            batch_size,
            T=T,
            STATE_PAD=state_pad,
            BLANK=blank,
            ZERO_INFINITY=zero_infinity,
            STORE_ALPHA=store_alpha,
            num_warps=num_warps,
        )
    else:
        _ctc_loss_forward_concat_kernel[(batch_size,)](
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            losses,
            label_prev,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            log_probs.stride(0),
            log_probs.stride(1),
            log_probs.stride(2),
            label_prev.stride(0),
            batch_size,
            T=T,
            STATE_PAD=state_pad,
            BATCH_PAD=batch_pad,
            BLANK=blank,
            ZERO_INFINITY=zero_infinity,
            STORE_ALPHA=store_alpha,
            num_warps=num_warps,
        )


def _launch_backward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    alpha_blank,
    alpha_label,
    cum_log_scale,
    grad_scale,
    grad_log_probs,
    scratch,
    batch_size,
    state_pad,
    batch_pad,
    num_warps,
    blank,
):
    T = log_probs.shape[0]
    if targets.ndim == 2:
        _ctc_loss_backward_kernel[(batch_size,)](
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            grad_scale,
            grad_log_probs,
            scratch,
            log_probs.stride(0),
            log_probs.stride(1),
            log_probs.stride(2),
            grad_log_probs.stride(0),
            grad_log_probs.stride(1),
            grad_log_probs.stride(2),
            targets.stride(0),
            scratch.stride(0),
            batch_size,
            T=T,
            STATE_PAD=state_pad,
            BLANK=blank,
            num_warps=num_warps,
        )
    else:
        _ctc_loss_backward_concat_kernel[(batch_size,)](
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            grad_scale,
            grad_log_probs,
            scratch,
            log_probs.stride(0),
            log_probs.stride(1),
            log_probs.stride(2),
            grad_log_probs.stride(0),
            grad_log_probs.stride(1),
            grad_log_probs.stride(2),
            scratch.stride(0),
            batch_size,
            T=T,
            STATE_PAD=state_pad,
            BATCH_PAD=batch_pad,
            BLANK=blank,
            num_warps=num_warps,
        )


def _apply_reduction(losses, target_lengths, reduction, is_batched):
    """Apply reduction to per-sample losses."""
    if reduction == _REDUCTION_NONE:
        return losses if is_batched else losses.squeeze(0)
    if reduction == _REDUCTION_SUM:
        return losses.sum()
    mean_denominator = target_lengths.to(device=losses.device, dtype=losses.dtype)
    mean_denominator = mean_denominator.clamp_min(1)
    return (losses / mean_denominator).mean()


def _ctc_loss_forward_triton(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    is_batched: bool,
    blank: int,
    reduction: int,
    zero_infinity: bool,
) -> torch.Tensor:
    """Forward-only path (no gradient). Used when requires_grad is False."""
    T = log_probs.shape[0]
    batch_size = log_probs.shape[1]
    state_pad, batch_pad, num_warps = _compute_kernel_params(
        targets, target_lengths, batch_size
    )
    (
        losses,
        label_prev,
        alpha_blank,
        alpha_label,
        cum_log_scale,
    ) = _allocate_forward_buffers(
        log_probs,
        batch_size,
        state_pad,
        T,
        store_alpha=False,
    )
    _launch_forward_kernel(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        losses,
        label_prev,
        alpha_blank,
        alpha_label,
        cum_log_scale,
        batch_size,
        state_pad,
        batch_pad,
        num_warps,
        blank,
        zero_infinity,
        store_alpha=False,
    )
    return _apply_reduction(losses, target_lengths, reduction, is_batched)


class _CTCLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        is_batched,
        blank,
        reduction,
        zero_infinity,
    ):
        T, batch_size, C = log_probs.shape
        state_pad, batch_pad, num_warps = _compute_kernel_params(
            targets, target_lengths, batch_size
        )
        (
            losses,
            label_prev,
            alpha_blank,
            alpha_label,
            cum_log_scale,
        ) = _allocate_forward_buffers(
            log_probs,
            batch_size,
            state_pad,
            T,
            store_alpha=True,
        )

        _launch_forward_kernel(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            losses,
            label_prev,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            batch_size,
            state_pad,
            batch_pad,
            num_warps,
            blank,
            zero_infinity,
            store_alpha=True,
        )

        ctx.save_for_backward(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            losses,
        )
        ctx.blank = blank
        ctx.reduction = reduction
        ctx.zero_infinity = zero_infinity
        ctx.is_batched = is_batched
        ctx.state_pad = state_pad
        ctx.batch_pad = batch_pad
        ctx.num_warps = num_warps

        return _apply_reduction(losses, target_lengths, reduction, is_batched)

    @staticmethod
    def backward(ctx, grad_output):
        (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            losses,
        ) = ctx.saved_tensors

        T, N, C = log_probs.shape
        device = log_probs.device

        # Compute per-sample gradient scale based on reduction mode
        if ctx.reduction == _REDUCTION_NONE:
            grad_scale = grad_output.reshape(N).contiguous().to(dtype=torch.float32)
        elif ctx.reduction == _REDUCTION_SUM:
            grad_scale = grad_output.to(dtype=torch.float32).expand(N).contiguous()
        else:  # MEAN
            tl_clamped = target_lengths.to(
                dtype=torch.float32, device=device
            ).clamp_min(1)
            grad_scale = (
                grad_output.to(dtype=torch.float32) / (N * tl_clamped)
            ).contiguous()

        # For zero_infinity: zero gradient where loss was inf (stored as 0)
        if ctx.zero_infinity:
            # Detect samples that were clamped: nll was inf → zeroed to 0,
            # but target_length > 0 (P=1 only happens for target_length=0 with blank-only path)
            # A more robust check: the stored loss is 0 but target_length > 0
            # means it was clamped. When target_length=0, loss=0 is legitimate.
            inf_mask = (losses == 0.0) & (target_lengths > 0)
            grad_scale = grad_scale.masked_fill(inf_mask, 0.0)
        else:
            inf_mask = torch.isinf(losses)

        grad_log_probs = torch.empty_like(log_probs)
        torch.exp(log_probs.detach(), out=grad_log_probs)
        grad_log_probs.mul_(grad_scale[None, :, None])
        scratch = torch.empty(
            (N, _backward_scratch_cols(ctx.state_pad)),
            device=device,
            dtype=torch.float32,
        )
        _launch_backward_kernel(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            alpha_blank,
            alpha_label,
            cum_log_scale,
            grad_scale,
            grad_log_probs,
            scratch,
            N,
            ctx.state_pad,
            ctx.batch_pad,
            ctx.num_warps,
            ctx.blank,
        )

        # Match Torch behavior for impossible alignments when zero_infinity=False:
        # the loss is inf and the gradient is non-finite.
        if not ctx.zero_infinity and inf_mask.any():
            grad_log_probs[:, inf_mask, :] = float("nan")

        return grad_log_probs, None, None, None, None, None, None, None


def _ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
    blank: int = 0,
    reduction: int = _REDUCTION_MEAN,
    zero_infinity: bool = False,
    *,
    fallback_op,
    log_suffix: str,
) -> torch.Tensor:
    logger.debug("GEMS CTC_LOSS (%s)", log_suffix)
    reduction = _normalize_reduction(reduction)
    prepared = _maybe_prepare_triton_forward_inputs(
        log_probs, targets, input_lengths, target_lengths, blank
    )
    if prepared is not None:
        lp, tgt, il, tl_, is_batched = prepared
        triton_lp = _to_triton_compute_log_probs(lp)
        if lp.requires_grad:
            logger.debug("GEMS CTC_LOSS Triton fwd+bwd (%s)", log_suffix)
            return _CTCLossFunction.apply(
                triton_lp,
                tgt,
                il,
                tl_,
                is_batched,
                blank,
                reduction,
                zero_infinity,
            )
        logger.debug("GEMS CTC_LOSS Triton forward (%s)", log_suffix)
        return _ctc_loss_forward_triton(
            triton_lp,
            tgt,
            il,
            tl_,
            is_batched,
            blank,
            reduction,
            zero_infinity,
        )
    return fallback_op.redispatch(
        _FALLBACK_KEYSET,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: int = _REDUCTION_MEAN,
    zero_infinity: bool = False,
) -> torch.Tensor:
    return _ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
        fallback_op=torch.ops.aten.ctc_loss.Tensor,
        log_suffix="Tensor lengths",
    )


def ctc_loss_intlist(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: Sequence[int],
    target_lengths: Sequence[int],
    blank: int = 0,
    reduction: int = _REDUCTION_MEAN,
    zero_infinity: bool = False,
) -> torch.Tensor:
    return _ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
        fallback_op=torch.ops.aten.ctc_loss.IntList,
        log_suffix="IntList lengths",
    )
