import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ============================================================================
# CTC Loss - Connectionist Temporal Classification
#
# Reference: Graves et al., "Connectionist Temporal Classification:
# Labelling Unsegmented Sequence Data with Recurrent Neural Networks", 2006
#
# Algorithm:
# 1. Expand target labels by inserting blanks: [b, l1, b, l2, ..., lS, b]
#    (computed on-the-fly in kernels to avoid extra GPU launches)
# 2. Forward pass (alpha): DP computing log probability of all valid paths
# 3. NLL = -log_sum_exp(alpha[T-1][S'-1], alpha[T-1][S'-2])
# 4. Backward pass (beta): reverse DP computed on-the-fly during backward
# 5. Gradient: grad[t,c] = prob[t,c] - exp(ab_sum - log_probs[t,c] + nll)
#
# All DP computation (alpha, beta, NLL, gradient) is implemented as Triton
# kernels for GPU acceleration.
# ============================================================================

NEG_INF = float("-inf")


@triton.jit
def _tl_log_sum_exp(a, b):
    """Stable log(exp(a) + exp(b)) in Triton."""
    max_v = tl.maximum(a, b)
    result = max_v + tl.log(tl.exp(a - max_v) + tl.exp(b - max_v))
    neg_inf = float("-inf")
    both_neg = (a == neg_inf) & (b == neg_inf)
    return tl.where(both_neg, neg_inf, result)


# ============================================================================
# Forward-only kernel with double-buffered alpha (for ctc_loss benchmark path)
# Uses only 2 rows of alpha storage instead of T rows, improving cache perf.
# ============================================================================


@libentry()
@triton.jit
def ctc_alpha_nll_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C) float32
    targets_ptr,  # (N, S_max) int64
    input_lengths_ptr,  # (N,) int64
    target_lengths_ptr,  # (N,) int64
    # Outputs
    nll_ptr,  # (N,) float32
    # Scratch buffer
    alpha_buf_ptr,  # (N, 2, S_prime_max) float32 - double buffered
    # Dimensions
    T_max,
    N,
    C,
    S_prime_max: tl.constexpr,
    S_max: tl.constexpr,
    blank: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    ab_stride_n,
    ab_stride_row,
    ab_stride_s,
    tgt_stride_n,
    tgt_stride_s,
    # Block size
    BLOCK_S: tl.constexpr,
    IS_MEAN: tl.constexpr,  # True for mean reduction (atomic_add normalized NLL)
):
    """Forward-only CTC alpha kernel with double-buffered scratch.

    Grid: (N,) - one program per batch element.
    Uses 2-row circular buffer for alpha, outputting only NLL.
    For mean reduction, normalizes by target_length and accumulates via atomic_add.
    """
    batch_idx = tle.program_id(0)
    if batch_idx >= N:
        return

    neg_inf = float("-inf")

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int32)
    tgt_len = tl.load(target_lengths_ptr + batch_idx).to(tl.int32)
    S_prime_i = 2 * tgt_len + 1

    # Base pointers
    ab_base = alpha_buf_ptr + batch_idx * ab_stride_n
    lp_base = log_probs_ptr + batch_idx * lp_stride_n
    tgt_base = targets_ptr + batch_idx * tgt_stride_n

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S_prime_max

    # Compute expanded labels on-the-fly
    tgt_indices = s_offsets // 2
    tgt_load_mask = s_mask & ((s_offsets % 2) == 1) & (tgt_indices < S_max)
    tgt_vals = tl.load(
        tgt_base + tgt_indices * tgt_stride_s, mask=tgt_load_mask, other=0
    ).to(tl.int64)
    labels = tl.where((s_offsets % 2) == 1, tgt_vals, blank)

    # Precompute can_skip mask
    tgt_indices_m1 = tgt_indices - 1
    tgt_m1_load_mask = (
        s_mask
        & ((s_offsets % 2) == 1)
        & (s_offsets >= 2)
        & (tgt_indices_m1 >= 0)
        & (tgt_indices_m1 < S_max)
    )
    tgt_m1_vals = tl.load(
        tgt_base + tgt_indices_m1 * tgt_stride_s, mask=tgt_m1_load_mask, other=0
    ).to(tl.int64)
    can_skip = (s_offsets >= 2) & ((s_offsets % 2) == 1) & (tgt_vals != tgt_m1_vals)

    # ---- Timestep t=0 initialization ----
    emit_ptrs = lp_base + 0 * lp_stride_t + labels * lp_stride_c
    emit_vals = tl.load(emit_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    alpha_cur = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
    alpha_cur = tl.where(s_offsets == 0, emit_vals, alpha_cur)
    alpha_cur = tl.where((s_offsets == 1) & (tgt_len > 0), emit_vals, alpha_cur)

    valid_s = s_offsets < S_prime_i
    alpha_cur = tl.where(valid_s, alpha_cur, neg_inf)

    # Store to row 0
    alpha_store_ptrs = ab_base + 0 * ab_stride_row + s_offsets * ab_stride_s
    tl.store(alpha_store_ptrs, alpha_cur, mask=s_mask)

    # ---- Forward DP: t = 1 to T_max - 1 ----
    for t in range(1, T_max):
        a_s = alpha_cur  # alpha[t-1][s]

        prev_row = (t - 1) % 2
        cur_row = t % 2

        # Load alpha[t-1][s-1]
        a_s_m1 = tl.load(
            ab_base + prev_row * ab_stride_row + (s_offsets - 1) * ab_stride_s,
            mask=(s_offsets >= 1) & s_mask,
            other=neg_inf,
        ).to(tl.float32)

        # Load alpha[t-1][s-2] (only where can_skip)
        a_s_m2_raw = tl.load(
            ab_base + prev_row * ab_stride_row + (s_offsets - 2) * ab_stride_s,
            mask=(s_offsets >= 2) & s_mask,
            other=neg_inf,
        ).to(tl.float32)
        a_s_m2 = tl.where(can_skip, a_s_m2_raw, neg_inf)

        # 3-way log-sum-exp: log(exp(a_s) + exp(a_s_m1) + exp(a_s_m2))
        max_val = tl.maximum(a_s, tl.maximum(a_s_m1, a_s_m2))
        sum_exp = (
            tl.exp(a_s - max_val) + tl.exp(a_s_m1 - max_val) + tl.exp(a_s_m2 - max_val)
        )
        trans = tl.where(max_val > -1e30, max_val + tl.log(sum_exp), neg_inf)

        # Emission probability + validity mask
        emit_t = tl.load(
            lp_base + t * lp_stride_t + labels * lp_stride_c,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)
        alpha_cur = tl.where(
            (s_offsets < S_prime_i) & (t < T_i), trans + emit_t, neg_inf
        )

        # Store to current row (double-buffered)
        tl.store(
            ab_base + cur_row * ab_stride_row + s_offsets * ab_stride_s,
            alpha_cur,
            mask=s_mask,
        )

    # ---- Compute NLL ----
    last_row = (T_i - 1) % 2
    if tgt_len > 0:
        a_last = tl.load(
            ab_base + last_row * ab_stride_row + (S_prime_i - 1) * ab_stride_s
        ).to(tl.float32)
        a_second = tl.load(
            ab_base + last_row * ab_stride_row + (S_prime_i - 2) * ab_stride_s
        ).to(tl.float32)
        nll_val = -_tl_log_sum_exp(a_last, a_second)
    else:
        a_only = tl.load(ab_base + last_row * ab_stride_row + 0 * ab_stride_s).to(
            tl.float32
        )
        nll_val = -a_only

    # Output NLL
    if IS_MEAN:
        # mean: normalize by target_length and N, atomic accumulate
        tgt_len_f = tl.maximum(tgt_len.to(tl.float32), 1.0)
        tl.atomic_add(nll_ptr, nll_val / (tgt_len_f * N))
    else:
        # none/sum: store per-sample
        tl.store(nll_ptr + batch_idx, nll_val)


# ============================================================================
# Full alpha kernel (stores all T timesteps for backward)
# ============================================================================


@libentry()
@triton.jit
def ctc_alpha_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C) float32
    targets_ptr,  # (N, S_max) int64 - raw targets (no blanks)
    input_lengths_ptr,  # (N,) int64
    target_lengths_ptr,  # (N,) int64
    # Outputs
    log_alpha_ptr,  # (N, T, S_prime_max) float32
    nll_ptr,  # (N,) float32
    # Dimensions
    T_max,
    N,
    C,
    S_prime_max: tl.constexpr,
    S_max: tl.constexpr,
    blank: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    alpha_stride_n,
    alpha_stride_t,
    alpha_stride_s,
    tgt_stride_n,
    tgt_stride_s,
    # Block size
    BLOCK_S: tl.constexpr,
):
    """Compute forward (alpha) DP table and NLL for one batch element.

    Grid: (N,) - one program per batch element.
    Sequential loop over T timesteps, parallel across S' label positions.
    Labels are computed on-the-fly: labels[s] = blank if s%2==0, else targets[s//2].
    """
    batch_idx = tle.program_id(0)
    if batch_idx >= N:
        return

    neg_inf = float("-inf")

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int32)
    tgt_len = tl.load(target_lengths_ptr + batch_idx).to(tl.int32)
    S_prime_i = 2 * tgt_len + 1

    # Base pointers for this batch
    alpha_base = log_alpha_ptr + batch_idx * alpha_stride_n
    lp_base = log_probs_ptr + batch_idx * lp_stride_n
    tgt_base = targets_ptr + batch_idx * tgt_stride_n

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S_prime_max

    # Compute expanded labels on-the-fly for all positions
    tgt_indices = s_offsets // 2
    tgt_load_mask = s_mask & ((s_offsets % 2) == 1) & (tgt_indices < S_max)
    tgt_vals = tl.load(
        tgt_base + tgt_indices * tgt_stride_s, mask=tgt_load_mask, other=0
    ).to(tl.int64)
    labels = tl.where((s_offsets % 2) == 1, tgt_vals, blank)

    # Precompute can_skip mask for CTC skip transitions
    tgt_indices_m1 = tgt_indices - 1
    tgt_m1_load_mask = (
        s_mask
        & ((s_offsets % 2) == 1)
        & (s_offsets >= 2)
        & (tgt_indices_m1 >= 0)
        & (tgt_indices_m1 < S_max)
    )
    tgt_m1_vals = tl.load(
        tgt_base + tgt_indices_m1 * tgt_stride_s, mask=tgt_m1_load_mask, other=0
    ).to(tl.int64)
    can_skip = (s_offsets >= 2) & ((s_offsets % 2) == 1) & (tgt_vals != tgt_m1_vals)

    # ---- Timestep t=0 initialization ----
    emit_ptrs = lp_base + 0 * lp_stride_t + labels * lp_stride_c
    emit_vals = tl.load(emit_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    alpha_cur = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
    alpha_cur = tl.where(s_offsets == 0, emit_vals, alpha_cur)
    alpha_cur = tl.where((s_offsets == 1) & (tgt_len > 0), emit_vals, alpha_cur)

    valid_s = s_offsets < S_prime_i
    alpha_cur = tl.where(valid_s, alpha_cur, neg_inf)

    alpha_store_ptrs = alpha_base + 0 * alpha_stride_t + s_offsets * alpha_stride_s
    tl.store(alpha_store_ptrs, alpha_cur, mask=s_mask)

    # ---- Forward DP: t = 1 to T_i - 1 ----
    for t in range(1, T_max):
        a_s = alpha_cur  # alpha[t-1][s]

        # Load alpha[t-1][s-1]
        a_s_m1 = tl.load(
            alpha_base + (t - 1) * alpha_stride_t + (s_offsets - 1) * alpha_stride_s,
            mask=(s_offsets >= 1) & s_mask,
            other=neg_inf,
        ).to(tl.float32)

        # Load alpha[t-1][s-2] (only where can_skip)
        a_s_m2_raw = tl.load(
            alpha_base + (t - 1) * alpha_stride_t + (s_offsets - 2) * alpha_stride_s,
            mask=(s_offsets >= 2) & s_mask,
            other=neg_inf,
        ).to(tl.float32)
        a_s_m2 = tl.where(can_skip, a_s_m2_raw, neg_inf)

        # 3-way log-sum-exp
        max_val = tl.maximum(a_s, tl.maximum(a_s_m1, a_s_m2))
        sum_exp = (
            tl.exp(a_s - max_val) + tl.exp(a_s_m1 - max_val) + tl.exp(a_s_m2 - max_val)
        )
        trans = tl.where(max_val > -1e30, max_val + tl.log(sum_exp), neg_inf)

        emit_t = tl.load(
            lp_base + t * lp_stride_t + labels * lp_stride_c,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        alpha_cur = tl.where(
            (s_offsets < S_prime_i) & (t < T_i), trans + emit_t, neg_inf
        )

        tl.store(
            alpha_base + t * alpha_stride_t + s_offsets * alpha_stride_s,
            alpha_cur,
            mask=s_mask,
        )

    # ---- Compute NLL from alpha ----
    if tgt_len > 0:
        a_last = tl.load(
            alpha_base + (T_i - 1) * alpha_stride_t + (S_prime_i - 1) * alpha_stride_s
        ).to(tl.float32)
        a_second = tl.load(
            alpha_base + (T_i - 1) * alpha_stride_t + (S_prime_i - 2) * alpha_stride_s
        ).to(tl.float32)
        nll_val = -_tl_log_sum_exp(a_last, a_second)
    else:
        a_only = tl.load(
            alpha_base + (T_i - 1) * alpha_stride_t + 0 * alpha_stride_s
        ).to(tl.float32)
        nll_val = -a_only

    tl.store(nll_ptr + batch_idx, nll_val)


# ============================================================================
# Beta kernel (backward DP)
# ============================================================================


@libentry()
@triton.jit
def ctc_beta_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C) float32
    targets_ptr,  # (N, S_max) int64 - raw targets
    input_lengths_ptr,  # (N,) int64
    target_lengths_ptr,  # (N,) int64
    # Outputs
    log_beta_ptr,  # (N, T, S_prime_max) float32
    # Dimensions
    T_max,
    N,
    C,
    S_prime_max: tl.constexpr,
    S_max: tl.constexpr,
    blank: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    beta_stride_n,
    beta_stride_t,
    beta_stride_s,
    tgt_stride_n,
    tgt_stride_s,
    # Block size
    BLOCK_S: tl.constexpr,
):
    """Compute backward (beta) DP table for one batch element.

    Grid: (N,) - one program per batch element.
    Labels computed on-the-fly from raw targets.
    """
    batch_idx = tle.program_id(0)
    if batch_idx >= N:
        return

    neg_inf = float("-inf")

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int32)
    tgt_len = tl.load(target_lengths_ptr + batch_idx).to(tl.int32)
    S_prime_i = 2 * tgt_len + 1

    beta_base = log_beta_ptr + batch_idx * beta_stride_n
    lp_base = log_probs_ptr + batch_idx * lp_stride_n
    tgt_base = targets_ptr + batch_idx * tgt_stride_n

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S_prime_max

    # Compute expanded labels on-the-fly
    tgt_indices = s_offsets // 2
    tgt_load_mask = s_mask & ((s_offsets % 2) == 1) & (tgt_indices < S_max)
    tgt_vals = tl.load(
        tgt_base + tgt_indices * tgt_stride_s, mask=tgt_load_mask, other=0
    ).to(tl.int64)
    labels = tl.where((s_offsets % 2) == 1, tgt_vals, blank)

    # Precompute can_skip_fwd mask for beta (forward direction: s+2)
    tgt_indices_p1 = tgt_indices + 1
    tgt_p1_load_mask = s_mask & ((s_offsets % 2) == 1) & (tgt_indices_p1 < S_max)
    tgt_p1_vals = tl.load(
        tgt_base + tgt_indices_p1 * tgt_stride_s, mask=tgt_p1_load_mask, other=0
    ).to(tl.int64)
    can_skip_fwd = ((s_offsets % 2) == 1) & (tgt_vals != tgt_p1_vals)

    # ---- Initialize beta[T_i-1] ----
    emit_ptrs_last = lp_base + (T_i - 1) * lp_stride_t + labels * lp_stride_c
    emit_last = tl.load(emit_ptrs_last, mask=s_mask, other=0.0).to(tl.float32)

    beta_cur = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
    beta_cur = tl.where(
        (s_offsets == S_prime_i - 1) & (S_prime_i > 0), emit_last, beta_cur
    )
    beta_cur = tl.where(
        (s_offsets == S_prime_i - 2) & (S_prime_i > 1), emit_last, beta_cur
    )

    beta_store_ptrs_last = (
        beta_base + (T_i - 1) * beta_stride_t + s_offsets * beta_stride_s
    )
    tl.store(beta_store_ptrs_last, beta_cur, mask=s_mask)

    # ---- Backward DP: t = T_i - 2 down to 0 ----
    for t_offset in range(1, T_max):
        t = T_i - 1 - t_offset

        if t >= 0:
            b_s = beta_cur  # beta[t+1][s]

            # Load beta[t+1][s+1]
            b_s_p1 = tl.load(
                beta_base + (t + 1) * beta_stride_t + (s_offsets + 1) * beta_stride_s,
                mask=(s_offsets + 1 < S_prime_max) & s_mask,
                other=neg_inf,
            ).to(tl.float32)

            # Load beta[t+1][s+2] (only where can_skip_fwd)
            b_s_p2_raw = tl.load(
                beta_base + (t + 1) * beta_stride_t + (s_offsets + 2) * beta_stride_s,
                mask=(s_offsets + 2 < S_prime_max) & s_mask,
                other=neg_inf,
            ).to(tl.float32)
            b_s_p2 = tl.where(
                can_skip_fwd & (s_offsets + 2 < S_prime_i), b_s_p2_raw, neg_inf
            )

            # 3-way log-sum-exp
            max_val = tl.maximum(b_s, tl.maximum(b_s_p1, b_s_p2))
            sum_exp = (
                tl.exp(b_s - max_val)
                + tl.exp(b_s_p1 - max_val)
                + tl.exp(b_s_p2 - max_val)
            )
            trans = tl.where(max_val > -1e30, max_val + tl.log(sum_exp), neg_inf)

            emit_t = tl.load(
                lp_base + t * lp_stride_t + labels * lp_stride_c,
                mask=s_mask,
                other=0.0,
            ).to(tl.float32)

            beta_cur = tl.where(
                (s_offsets < S_prime_i) & (t < T_i), trans + emit_t, neg_inf
            )

            tl.store(
                beta_base + t * beta_stride_t + s_offsets * beta_stride_s,
                beta_cur,
                mask=s_mask,
            )


# ============================================================================
# Gradient kernel
# ============================================================================


@libentry()
@triton.jit
def ctc_grad_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C)
    alpha_ptr,  # (N, T, S')
    beta_ptr,  # (N, T, S')
    nll_ptr,  # (N,)
    targets_ptr,  # (N, S_max) - raw targets
    grad_out_ptr,  # (N,)
    target_lengths_ptr,  # (N,)
    input_lengths_ptr,  # (N,)
    # Outputs
    grad_ptr,  # (T, N, C)
    # Dimensions
    T_max: tl.constexpr,
    N,
    C: tl.constexpr,
    S_prime_max: tl.constexpr,
    S_max: tl.constexpr,
    blank: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    alpha_stride_n,
    alpha_stride_t,
    alpha_stride_s,
    tgt_stride_n,
    tgt_stride_s,
    grad_stride_t,
    grad_stride_n,
    grad_stride_c,
    # Block sizes
    BLOCK_C: tl.constexpr,
):
    """Compute CTC gradient for one (batch, time) pair.

    Grid: (N, T_max) - one program per (batch, time) pair.
    Labels computed on-the-fly from raw targets.
    """
    batch_idx = tle.program_id(0)
    t = tle.program_id(1)

    if batch_idx >= N:
        return

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int64)

    # Zero out gradients for padding timesteps
    if t >= T_i:
        for c_start in tl.static_range(0, C, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < C
            out_ptrs = (
                grad_ptr
                + t * grad_stride_t
                + batch_idx * grad_stride_n
                + c_offsets * grad_stride_c
            )
            tl.store(out_ptrs, tl.zeros([BLOCK_C], dtype=tl.float32), mask=c_mask)
        return

    tgt_len = tl.load(target_lengths_ptr + batch_idx).to(tl.int64)
    S_prime_i = 2 * tgt_len + 1
    nll = tl.load(nll_ptr + batch_idx).to(tl.float32)
    grad_scale = tl.load(grad_out_ptr + batch_idx).to(tl.float32)

    for c_start in tl.static_range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        lp_ptrs = (
            log_probs_ptr
            + t * lp_stride_t
            + batch_idx * lp_stride_n
            + c_offsets * lp_stride_c
        )
        lp_val = tl.load(lp_ptrs, mask=c_mask, other=0.0).to(tl.float32)

        ab_sum = tl.full([BLOCK_C], value=float("-inf"), dtype=tl.float32)

        for s in range(S_prime_max):
            if s < S_prime_i:
                # Compute label on-the-fly
                tgt_idx = s // 2
                tgt_val = tl.load(
                    targets_ptr + batch_idx * tgt_stride_n + tgt_idx * tgt_stride_s
                ).to(tl.int64)
                lbl = tl.where((s % 2) == 1, tgt_val, blank)

                a_val = tl.load(
                    alpha_ptr
                    + batch_idx * alpha_stride_n
                    + t * alpha_stride_t
                    + s * alpha_stride_s
                ).to(tl.float32)
                b_val = tl.load(
                    beta_ptr
                    + batch_idx * alpha_stride_n
                    + t * alpha_stride_t
                    + s * alpha_stride_s
                ).to(tl.float32)
                ab_single = a_val + b_val

                match_mask = (c_offsets == lbl) & c_mask
                max_v = tl.maximum(ab_sum, ab_single)
                new_sum = max_v + tl.log(
                    tl.exp(ab_sum - max_v) + tl.exp(ab_single - max_v)
                )
                both_neg_inf = (ab_sum == float("-inf")) & (ab_single == float("-inf"))
                new_sum = tl.where(both_neg_inf, float("-inf"), new_sum)
                ab_sum = tl.where(match_mask, new_sum, ab_sum)

        prob_val = tl.exp(lp_val)
        occupancy = tl.exp(ab_sum - lp_val + nll)
        occupancy = tl.where(ab_sum > float("-inf"), occupancy, 0.0)
        grad_val = (prob_val - occupancy) * grad_scale
        grad_val = tl.where(c_mask, grad_val, 0.0)

        out_ptrs = (
            grad_ptr
            + t * grad_stride_t
            + batch_idx * grad_stride_n
            + c_offsets * grad_stride_c
        )
        tl.store(out_ptrs, grad_val, mask=c_mask)


# ============================================================================
# Python wrapper helpers
# ============================================================================


def _ensure_2d_targets(targets, target_lengths, N, device):
    """Convert 1D concatenated targets to 2D (N, S_max) format."""
    if targets.dim() == 2:
        return targets
    S_max = target_lengths.max().item() if target_lengths.numel() > 0 else 0
    targets_2d = torch.zeros(N, S_max, dtype=torch.long, device=device)
    offsets = torch.zeros(N + 1, dtype=torch.long, device=device)
    offsets[1:] = target_lengths.cumsum(0)
    for n in range(N):
        tl_n = target_lengths[n].item()
        if tl_n > 0:
            targets_2d[n, :tl_n] = targets[offsets[n] : offsets[n] + tl_n]
    return targets_2d


# ============================================================================
# Forward (full alpha table for backward)
# ============================================================================


def _ctc_loss_forward(
    log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False
):
    """CTC loss forward pass - computes NLL and alpha DP table.

    Returns:
        Tuple of (neg_log_likelihood, log_alpha)
    """
    logger.debug("GEMS CTC_LOSS FWD")

    assert log_probs.dim() == 3, "log_probs must be (T, N, C)"
    T, N, C = log_probs.shape

    if input_lengths.dtype != torch.long:
        input_lengths = input_lengths.to(torch.long)
    if target_lengths.dtype != torch.long:
        target_lengths = target_lengths.to(torch.long)

    if targets.dim() == 2:
        S_max = targets.shape[1]
        targets_2d = targets
    else:
        S_max = target_lengths.max().item() if target_lengths.numel() > 0 else 0
        targets_2d = _ensure_2d_targets(targets, target_lengths, N, log_probs.device)
    S_prime_max = 2 * S_max + 1

    lp = log_probs if log_probs.dtype == torch.float32 else log_probs.float()

    log_alpha = torch.empty(
        (N, T, S_prime_max), dtype=torch.float32, device=log_probs.device
    )
    neg_log_likelihood = torch.empty(N, dtype=torch.float32, device=log_probs.device)

    BLOCK_S = triton.next_power_of_2(S_prime_max) if S_prime_max > 0 else 1

    with torch_device_fn.device(log_probs.device):
        ctc_alpha_kernel[(N,)](
            lp,
            targets_2d,
            input_lengths,
            target_lengths,
            log_alpha,
            neg_log_likelihood,
            T,
            N,
            C,
            S_prime_max,
            S_max,
            blank,
            lp.stride(0),
            lp.stride(1),
            lp.stride(2),
            log_alpha.stride(0),
            log_alpha.stride(1),
            log_alpha.stride(2),
            targets_2d.stride(0),
            targets_2d.stride(1),
            BLOCK_S=BLOCK_S,
        )

    return neg_log_likelihood, log_alpha


# ============================================================================
# Backward
# ============================================================================


def _ctc_loss_backward(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity=False,
):
    """CTC loss backward pass."""
    logger.debug("GEMS CTC_LOSS BWD")

    T, N, C = log_probs.shape

    lp_detached = log_probs.detach()
    log_probs_f = (
        lp_detached if lp_detached.dtype == torch.float32 else lp_detached.float()
    )
    if isinstance(input_lengths, (list, tuple)):
        input_lengths = torch.tensor(
            input_lengths, dtype=torch.long, device=log_probs.device
        )
    if isinstance(target_lengths, (list, tuple)):
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long, device=log_probs.device
        )
    if input_lengths.dtype != torch.long:
        input_lengths = input_lengths.to(torch.long)
    if target_lengths.dtype != torch.long:
        target_lengths = target_lengths.to(torch.long)

    if targets.dim() == 2:
        S_max = targets.shape[1]
        targets_2d = targets
    else:
        S_max = target_lengths.max().item() if target_lengths.numel() > 0 else 0
        targets_2d = _ensure_2d_targets(targets, target_lengths, N, log_probs.device)
    S_prime_max = 2 * S_max + 1

    BLOCK_S = triton.next_power_of_2(S_prime_max) if S_prime_max > 0 else 1

    # ---- Beta kernel ----
    log_beta = torch.empty(
        (N, T, S_prime_max), dtype=torch.float32, device=log_probs.device
    )
    with torch_device_fn.device(log_probs.device):
        ctc_beta_kernel[(N,)](
            log_probs_f,
            targets_2d,
            input_lengths,
            target_lengths,
            log_beta,
            T,
            N,
            C,
            S_prime_max,
            S_max,
            blank,
            log_probs_f.stride(0),
            log_probs_f.stride(1),
            log_probs_f.stride(2),
            log_beta.stride(0),
            log_beta.stride(1),
            log_beta.stride(2),
            targets_2d.stride(0),
            targets_2d.stride(1),
            BLOCK_S=BLOCK_S,
        )

    # ---- Gradient kernel ----
    grad_log_probs = torch.zeros(T, N, C, dtype=torch.float32, device=log_probs.device)
    BLOCK_C = triton.next_power_of_2(C)

    # Detach and copy autograd-tracked tensors (required for correctness
    # when called via autograd dispatch - saved tensors need clean copies)
    log_alpha_d = log_alpha.detach().clone()
    nll_d = neg_log_likelihood.detach().clone()
    grad_d = grad.detach().clone()

    with torch_device_fn.device(log_probs.device):
        ctc_grad_kernel[(N, T)](
            log_probs_f,
            log_alpha_d,
            log_beta,
            nll_d,
            targets_2d,
            grad_d,
            target_lengths,
            input_lengths,
            grad_log_probs,
            T,
            N,
            C,
            S_prime_max,
            S_max,
            blank,
            log_probs_f.stride(0),
            log_probs_f.stride(1),
            log_probs_f.stride(2),
            log_alpha_d.stride(0),
            log_alpha_d.stride(1),
            log_alpha_d.stride(2),
            targets_2d.stride(0),
            targets_2d.stride(1),
            grad_log_probs.stride(0),
            grad_log_probs.stride(1),
            grad_log_probs.stride(2),
            BLOCK_C=BLOCK_C,
        )

    # Handle zero_infinity
    if zero_infinity:
        nll_check = neg_log_likelihood.detach()
        inf_mask = torch.isinf(nll_check) | (nll_check != nll_check)
        if inf_mask.any():
            zero_mask = inf_mask.unsqueeze(0).unsqueeze(2).expand_as(grad_log_probs)
            grad_log_probs = torch.where(
                zero_mask, torch.zeros_like(grad_log_probs), grad_log_probs
            )

    return grad_log_probs.to(log_probs.dtype)


# ============================================================================
# Aten dispatch entry points
# ============================================================================


def _ctc_loss_impl(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    zero_infinity=False,
):
    """Compute _ctc_loss (dispatched from aten::_ctc_loss / _ctc_loss.Tensor)."""
    logger.debug("GEMS _CTC_LOSS")

    if isinstance(input_lengths, (list, tuple)):
        input_lengths = torch.tensor(
            input_lengths, dtype=torch.long, device=log_probs.device
        )
    if isinstance(target_lengths, (list, tuple)):
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long, device=log_probs.device
        )

    return _ctc_loss_forward(
        log_probs, targets, input_lengths, target_lengths, blank, zero_infinity
    )


def _ctc_loss_backward_impl(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity=False,
):
    """Compute _ctc_loss_backward (dispatched from aten::_ctc_loss_backward)."""
    logger.debug("GEMS _CTC_LOSS_BACKWARD")

    return _ctc_loss_backward(
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


# ============================================================================
# High-level ctc_loss (dispatched from aten::ctc_loss)
# Optimized: uses double-buffered kernel and minimizes Python overhead.
# ============================================================================


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction=1,
    zero_infinity=False,
):
    """CTC loss with reduction (dispatched from aten::ctc_loss).

    Args:
        log_probs: (T, N, C) log-softmax output
        targets: (N, S) or 1D concatenated target labels
        input_lengths: (N,) actual input sequence lengths
        target_lengths: (N,) actual target sequence lengths
        blank: blank label index (default: 0)
        reduction: 0=none, 1=mean, 2=sum (default: 1)
        zero_infinity: replace inf losses with zero (default: False)

    Returns:
        Scalar or (N,) tensor depending on reduction mode
    """
    logger.debug("GEMS CTC_LOSS")

    # Normalize string reduction to int
    if isinstance(reduction, str):
        reduction = {"none": 0, "mean": 1, "sum": 2}[reduction]

    device = log_probs.device
    orig_dtype = log_probs.dtype
    T, N, C = log_probs.shape

    # Promote to float32 for numerical stability
    lp = log_probs if orig_dtype == torch.float32 else log_probs.float()

    # Ensure tensor lengths (skip .to() if already correct dtype)
    if isinstance(input_lengths, (list, tuple)):
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    elif input_lengths.dtype != torch.long:
        input_lengths = input_lengths.to(torch.long)
    if isinstance(target_lengths, (list, tuple)):
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    elif target_lengths.dtype != torch.long:
        target_lengths = target_lengths.to(torch.long)

    # Get S_max and targets_2d
    if targets.dim() == 2:
        S_max = targets.shape[1]
        targets_2d = targets
    else:
        S_max = target_lengths.max().item() if target_lengths.numel() > 0 else 0
        targets_2d = _ensure_2d_targets(targets, target_lengths, N, device)
    S_prime_max = 2 * S_max + 1

    is_mean = reduction == 1
    BLOCK_S = triton.next_power_of_2(S_prime_max) if S_prime_max > 0 else 1

    # Allocate alpha scratch and NLL output
    alpha_buf = torch.empty((N, 2, S_prime_max), dtype=torch.float32, device=device)
    if is_mean:
        neg_log_likelihood = torch.zeros(1, dtype=torch.float32, device=device)
    else:
        neg_log_likelihood = torch.empty(N, dtype=torch.float32, device=device)

    with torch_device_fn.device(device):
        ctc_alpha_nll_kernel[(N,)](
            lp,
            targets_2d,
            input_lengths,
            target_lengths,
            neg_log_likelihood,
            alpha_buf,
            T,
            N,
            C,
            S_prime_max,
            S_max,
            blank,
            lp.stride(0),
            lp.stride(1),
            lp.stride(2),
            alpha_buf.stride(0),
            alpha_buf.stride(1),
            alpha_buf.stride(2),
            targets_2d.stride(0),
            targets_2d.stride(1),
            BLOCK_S=BLOCK_S,
            IS_MEAN=is_mean,
        )

    # Handle zero_infinity
    if zero_infinity:
        neg_log_likelihood = torch.where(
            torch.isinf(neg_log_likelihood),
            neg_log_likelihood.new_zeros(()),
            neg_log_likelihood,
        )

    # Apply reduction
    if reduction == 0:
        result = neg_log_likelihood
    elif reduction == 1:
        # Kernel accumulated mean(nll/tgt_len) directly
        result = neg_log_likelihood.squeeze(0)
    else:
        result = neg_log_likelihood.sum()

    return result.to(orig_dtype) if result.dtype != orig_dtype else result
