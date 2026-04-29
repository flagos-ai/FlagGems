"""
CTC Loss operator implementation for FlagGems - Optimized Global Memory Version.

Fix: Resolved Triton compilation error regarding type mismatch (scalar vs tensor)
in skip_allowed computation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ============================================================================
# Validation
# ============================================================================


def _validate_ctc_loss_input(
    log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity
):
    """Validate input tensors and parameters for ctc_loss."""
    if log_probs.dim() != 3 and log_probs.dim() != 2:
        raise ValueError(f"log_probs must be 2D or 3D, got {log_probs.dim()}D")

    supported_dtypes = [torch.float16, torch.float32, torch.bfloat16]
    if log_probs.dtype not in supported_dtypes:
        raise ValueError(
            f"log_probs dtype must be one of {supported_dtypes}, got {log_probs.dtype}"
        )

    T, N, C = (
        (log_probs.shape[0], 1, log_probs.shape[1])
        if log_probs.dim() == 2
        else log_probs.shape
    )

    if not isinstance(blank, int) or blank < 0 or blank >= C:
        raise ValueError(f"blank must be an integer in [0, {C}), got {blank}")

    valid_reductions = ["none", "mean", "sum"]
    if reduction not in valid_reductions:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Expected one of {valid_reductions}"
        )

    if not isinstance(zero_infinity, bool):
        raise ValueError(f"zero_infinity must be bool, got {type(zero_infinity)}")


# ============================================================================
# Optimized Triton Kernels
# ============================================================================


@libentry()
@triton.jit
def ctc_alpha_kernel_optimized(
    log_probs_ptr,
    targets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    alpha_ptr,
    losses_ptr,
    blank: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    S_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
):
    """
    Optimized CTC Alpha computation.
    One block per sample. Sequential timestep processing within block.
    """
    # 1. Batch Index
    n = tl.program_id(0)

    # 2. Load Lengths
    T_actual = tl.load(input_lengths_ptr + n).to(tl.int32)
    S_actual = tl.load(target_lengths_ptr + n).to(tl.int32)

    # Handle empty targets immediately
    if S_actual == 0:
        tl.store(losses_ptr + n, 0.0)
        return

    # 3. Setup Pointers
    log_probs_n = log_probs_ptr + n * C
    targets_n = targets_ptr + n * S_max

    # Strides for alpha buffer [N, T, S']
    stride_t = 2 * S_max + 1
    stride_n = T * stride_t

    alpha_base = alpha_ptr + n * stride_n

    # 4. Setup Thread Mapping
    thread_id = tl.arange(0, BLOCK_SIZE)
    num_states = 2 * S_actual + 1

    # 5. Timestep Loop
    for t in range(T_actual):
        s = thread_id
        valid_state_mask = s < num_states

        # --- Determine Labels ---
        is_blank_state = s % 2 == 0
        label_idx = s // 2

        # Load target label for current state s (needed for emission and skip logic)
        # Use blank as default for padding/out-of-bounds
        target_label = tl.load(
            targets_n + label_idx, mask=valid_state_mask, other=blank
        ).to(tl.int32)

        label_id = tl.where(is_blank_state, blank, target_label)

        # --- Load Emission Probability ---
        log_prob_offset = log_probs_n + t * N * C + label_id
        log_prob = tl.load(
            log_prob_offset, mask=valid_state_mask, other=-float("inf")
        ).to(tl.float32)

        # --- Compute Alpha ---
        if t == 0:
            # Initialization
            val = tl.where(s == 0, log_prob, -float("inf"))

            if S_actual > 0:
                first_target = tl.load(targets_n).to(tl.int32)
                log_prob_first = tl.load(log_probs_n + first_target).to(tl.float32)
                val = tl.where(s == 1, log_prob_first, val)

            # Handle NaN only (inf and -inf are legitimate in log-space)
            is_nan = val != val
            val = tl.where(is_nan, float("nan"), val)

            tl.store(alpha_base + t * stride_t + s, val, mask=valid_state_mask)

        else:
            # Recurrence
            prev_offset = alpha_base + (t - 1) * stride_t

            a_s = tl.load(prev_offset + s, mask=valid_state_mask, other=-float("inf"))
            a_s_m1 = tl.load(
                prev_offset + (s - 1),
                mask=(s > 0) & valid_state_mask,
                other=-float("inf"),
            )
            a_s_m2 = tl.load(
                prev_offset + (s - 2),
                mask=(s > 1) & valid_state_mask,
                other=-float("inf"),
            )

            # --- Skip connection logic (Fixed Type Mismatch) ---
            # We compute skip_allowed directly as a vector operation.
            # No scalar initialization 'skip_allowed = False' is used.

            # Load label for s-2
            # label[s-2] corresponds to index (s-2)//2 = label_idx - 1
            prev_label_idx = label_idx - 1
            # Safe load: if index < 0, returns -1 (or placeholder)
            target_label_m2 = tl.load(
                targets_n + prev_label_idx,
                mask=(prev_label_idx >= 0) & valid_state_mask,
                other=-1,
            ).to(tl.int32)

            # Condition for skip:
            # 1. s >= 2 (index valid)
            # 2. s is odd (current is target state, not blank)
            # 3. label[s] != label[s-2] (targets differ)
            skip_allowed = (
                valid_state_mask
                & (s >= 2)
                & (s % 2 == 1)
                & (target_label != target_label_m2)
            )

            a_s_m2 = tl.where(skip_allowed, a_s_m2, -float("inf"))

            # Log-sum-exp
            max_val = tl.maximum(a_s, tl.maximum(a_s_m1, a_s_m2))

            has_nan = (a_s != a_s) | (a_s_m1 != a_s_m1) | (a_s_m2 != a_s_m2)

            valid_sum = max_val > -1e30

            sum_exp = tl.where(
                valid_sum,
                tl.exp(a_s - max_val)
                + tl.exp(a_s_m1 - max_val)
                + tl.exp(a_s_m2 - max_val),
                0.0,
            )

            log_sum = tl.where(
                valid_sum, max_val + tl.log(sum_exp + 1e-10), -float("inf")
            )

            new_alpha = log_sum + log_prob

            # Propagate NaN only (Inf is legitimate in log-space)
            new_alpha = tl.where(has_nan, float("nan"), new_alpha)

            tl.store(alpha_base + t * stride_t + s, new_alpha, mask=valid_state_mask)

        # Synchronize threads within block
        tl.debug_barrier()

    # 6. Final Loss Calculation
    final_offset = alpha_base + (T_actual - 1) * stride_t

    s_final_1 = 2 * S_actual
    s_final_2 = 2 * S_actual - 1

    alpha_f1 = tl.load(final_offset + s_final_1)
    alpha_f2 = tl.load(final_offset + s_final_2)

    has_nan = (alpha_f1 != alpha_f1) | (alpha_f2 != alpha_f2)

    max_final = tl.maximum(alpha_f1, alpha_f2)
    valid = max_final > -1e30

    log_sum = tl.where(
        valid,
        max_final
        + tl.log(tl.exp(alpha_f1 - max_final) + tl.exp(alpha_f2 - max_final) + 1e-10),
        -float("inf"),
    )

    log_sum = tl.where(has_nan, float("nan"), log_sum)
    loss = -log_sum

    if ZERO_INFINITY:
        loss = tl.where(loss < 1e30, loss, 0.0)

    tl.store(losses_ptr + n, loss)


# ============================================================================
# Main API Function
# ============================================================================


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """
    Connectionist Temporal Classification loss (Optimized).
    """
    logger.debug("GEMS CTC LOSS FORWARD OPTIMIZED V2")

    _validate_ctc_loss_input(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )

    original_target_lengths = (
        target_lengths.clone() if target_lengths.dim() > 0 else target_lengths.clone()
    )

    # Handle dimensions
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(1)
        N = 1
        T, _, C = log_probs.shape
    else:
        T, N, C = log_probs.shape

    original_dtype = log_probs.dtype
    log_probs = log_probs.to(torch.float32)

    if input_lengths.dim() == 0:
        input_lengths = input_lengths.expand(N)
    if target_lengths.dim() == 0:
        target_lengths = target_lengths.expand(N)

    # Handle targets format
    if targets.dim() == 1:
        target_lengths_cpu = target_lengths.cpu()
        S_max = target_lengths.max().item()
        if S_max == 0:
            targets_2d = torch.zeros(N, 1, dtype=torch.long, device=targets.device)
            S_max = 1
        else:
            targets_2d = torch.full(
                (N, S_max), -1, dtype=torch.long, device=targets.device
            )
            offset = 0
            for n_idx in range(N):
                t_len = target_lengths_cpu[n_idx].item()
                if t_len > 0:
                    targets_2d[n_idx, :t_len] = targets[offset : offset + t_len]
                    offset += t_len
        targets = targets_2d
    else:
        S_max = targets.shape[1]

    # Move to device
    if targets.device != log_probs.device:
        targets = targets.to(log_probs.device)
    if input_lengths.device != log_probs.device:
        input_lengths = input_lengths.to(log_probs.device)
    if target_lengths.device != log_probs.device:
        target_lengths = target_lengths.to(log_probs.device)

    log_probs = log_probs.contiguous()
    targets = targets.contiguous()
    input_lengths = input_lengths.contiguous()
    target_lengths = target_lengths.contiguous()

    if (target_lengths == 0).all():
        losses = torch.zeros(N, dtype=torch.float32, device=log_probs.device)
    else:
        losses = torch.empty(N, dtype=torch.float32, device=log_probs.device)

        # Allocate alpha buffer
        alpha_buffer = torch.empty(
            N, T, 2 * S_max + 1, dtype=torch.float32, device=log_probs.device
        )

        # Determine Block Size
        BLOCK_SIZE = triton.next_power_of_2(min(2 * S_max + 1, 1024))

        # Grid: One block per batch item
        grid = (N,)

        logger.debug(
            f"Launching Optimized CTC kernel: grid={grid}, BLOCK_SIZE={BLOCK_SIZE}, S_max={S_max}"
        )

        with torch_device_fn.device(log_probs.device):
            ctc_alpha_kernel_optimized[grid](
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                alpha_buffer,
                losses,
                blank=blank,
                T=T,
                N=N,
                C=C,
                S_max=S_max,
                BLOCK_SIZE=BLOCK_SIZE,
                ZERO_INFINITY=zero_infinity,
            )

    # Apply reduction
    if reduction == "none":
        result = losses.to(original_dtype)
    elif reduction == "sum":
        result = losses.sum().to(original_dtype)
    else:  # "mean"
        if (original_target_lengths == 0).all():
            result = losses.mean().to(original_dtype)
        else:
            safe_target_lengths = original_target_lengths.to(losses.dtype)
            safe_target_lengths = torch.where(
                safe_target_lengths == 0, 1.0, safe_target_lengths
            )
            result = (losses / safe_target_lengths).mean().to(original_dtype)

    return result
