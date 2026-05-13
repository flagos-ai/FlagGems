"""CTC loss for FlagGems, log-space alpha/beta DP in Triton.

Reference: A. Graves et al., "Connectionist Temporal Classification: Labelling
Unsegmented Sequence Data with Recurrent Neural Networks", ICML 2006.

Notation (per batch element n):
    T_n   = input_lengths[n]                  effective time length
    L_n   = target_lengths[n]                 target length
    S'_n  = 2 * L_n + 1                       extended target length
    ext[s] for s in [0, S'_n):
        s even  -> BLANK
        s odd   -> targets[n, s // 2]
    log_probs: (T, N, C), already log-softmax'd along C

Forward DP (alpha, log-space):
    alpha[0, 0] = log_probs[0, n, BLANK]
    alpha[0, 1] = log_probs[0, n, target[0]]            if L_n > 0
    alpha[0, s] = -inf                                   for s >= 2
    For t >= 1:
        alpha[t, s] = log_probs[t, n, ext[s]] +
            logsumexp(alpha[t-1, s],
                      alpha[t-1, s-1]   if s >= 1,
                      alpha[t-1, s-2]   if s >= 2 and ext[s] != BLANK
                                                  and ext[s] != ext[s-2])
    loss[n] = -logsumexp(alpha[T_n-1, S'_n-1], alpha[T_n-1, S'_n-2])
    Edge case L_n = 0:  loss[n] = -alpha[T_n-1, 0]

Backward DP (beta, log-space; future-only convention):
    beta[t, s] = log P(observe x_{t+1:T_n-1} | state s at time t)
    beta[T_n-1, S'_n-1] = 0
    beta[T_n-1, S'_n-2] = 0                              if L_n > 0
    For t in (T_n-2, ..., 0):
        beta[t, s] = logsumexp(
            beta[t+1, s]   + log_probs[t+1, n, ext[s]],
            beta[t+1, s+1] + log_probs[t+1, n, ext[s+1]]   if s+1 < S'_n,
            beta[t+1, s+2] + log_probs[t+1, n, ext[s+2]]   if s+2 < S'_n
                                                              and ext[s] != BLANK
                                                              and ext[s] != ext[s+2])

Gradient (gamma, posterior):
    gamma[t, s] = exp(alpha[t, s] + beta[t, s] - log_like_n)
    posterior_sum[t, n, c] = sum_{s: ext[s] == c} gamma[t, s]
    dL/dlog_probs[t, n, c] = exp(log_probs[t, n, c]) - posterior_sum[t, n, c]

Implementation:
    * grid = (N,) for both forward and backward kernels; one program per batch.
    * Inside a program, the state dim (length S'_n) is held as a single
      BLOCK_S-wide register tile.
    * BLOCK_S = next_power_of_2(max(1, 2 * max(target_lengths) + 1)), capped at
      _MAX_BLOCK_STATES (2048 -> max target_length 1023).
    * alpha is saved as fp32 (N, T, BLOCK_S) for use in the backward pass.
    * beta only lives at the current and one shifted time slice; it is held in
      registers and round-tripped through a tiny scratch (N, BLOCK_S) so that
      register-level shifts (beta[s+1], beta[s+2]) become global loads with
      pointer offsets.
    * The backward kernel atomic-adds raw posterior masses into a scratch
      tensor; the final gradient is assembled Python-side as
          grad = scale * valid_mask * (exp(log_probs) - posterior_sum)
      where `scale` encodes reduction + zero_infinity, and `valid_mask` zeros
      out positions with t >= input_lengths[n].
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)

# Constructor-form constexpr so the @triton.jit kernels can read it as a
# module-level global.  Newer Triton (>= 3.4) rejects the annotation form
# `_NEG_INF: tl.constexpr = float("-inf")`; only the constructor form
# `_NEG_INF = tl.constexpr(float("-inf"))` is officially supported.
_NEG_INF = tl.constexpr(float("-inf"))
# Supports target_length up to 1023 (S' = 2L+1 = 2047).
_MAX_BLOCK_STATES = 2048

# Per-device cache: True if the GPU is sm_80+ (Ampere or newer), which is the
# minimum target for Triton's bf16 PTX path (`cvt.f32.bf16`).  On sm_70/sm_75
# (Volta / Turing) bf16 inputs must be upcast to fp32 Python-side before being
# handed to any kernel.  Caching avoids a CUDA-driver query per call.
_BF16_NATIVE_CACHE: dict = {}


def _bf16_native(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _BF16_NATIVE_CACHE.get(idx)
    if cached is None:
        cached = torch.cuda.get_device_capability(idx)[0] >= 8
        _BF16_NATIVE_CACHE[idx] = cached
    return cached


# ---------------------------------------------------------------------------
# Python-side helpers
# ---------------------------------------------------------------------------
def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        v = reduction.lower()
        if v == "none":
            return 0
        if v == "mean":
            return 1
        if v == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction!r}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _length_tensor(lengths, name, batch, device_obj):
    if torch.is_tensor(lengths):
        if lengths.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise RuntimeError(f"{name} must contain integral lengths")
        result = lengths.reshape(1) if lengths.ndim == 0 else lengths
        if result.numel() != batch:
            raise RuntimeError(
                f"{name} must have batch size {batch}, got {result.numel()}"
            )
        # Skip the .to() / .contiguous() dispatch when the tensor already
        # matches -- saves a few microseconds per call on the hot path.
        if result.dtype != torch.int64 or result.device != device_obj:
            result = result.to(device=device_obj, dtype=torch.int64)
        if not result.is_contiguous():
            result = result.contiguous()
        return result
    if not isinstance(lengths, (list, tuple)):
        raise RuntimeError(f"{name} must be a Tensor or a sequence of integers")
    if len(lengths) != batch:
        raise RuntimeError(f"{name} must have batch size {batch}, got {len(lengths)}")
    return torch.tensor(lengths, device=device_obj, dtype=torch.int64)


def _target_offsets_from_cpu(target_lengths_cpu, device_obj):
    """Build prefix-sum offsets from a CPU Python list.

    We already pay one CPU sync to fetch length stats in `_check_inputs`; doing
    the prefix-sum in Python from the same data avoids 3 extra small kernels
    (empty_like + assign + cumsum + assign) that `_target_offsets` used.
    """
    if not target_lengths_cpu:
        return torch.empty(0, device=device_obj, dtype=torch.int64)
    offsets = [0]
    running = 0
    for L in target_lengths_cpu[:-1]:
        running += L
        offsets.append(running)
    return torch.tensor(offsets, device=device_obj, dtype=torch.int64)


def _check_inputs(log_probs, targets, input_lengths, target_lengths, blank):
    if log_probs.ndim not in (2, 3):
        raise RuntimeError(
            f"ctc_loss expects 2D (unbatched) or 3D log_probs, got {log_probs.ndim}D"
        )
    if not torch.is_floating_point(log_probs):
        raise RuntimeError("ctc_loss not implemented for non-floating log_probs")
    if not torch.is_tensor(targets):
        raise RuntimeError("ctc_loss expects targets to be a Tensor")
    if log_probs.device.type != device:
        raise RuntimeError(f"ctc_loss expects log_probs on device {device}")

    unbatched = log_probs.ndim == 2
    if unbatched:
        t_steps, classes = log_probs.shape
        batch = 1
    else:
        t_steps, batch, classes = log_probs.shape

    if blank < 0 or blank >= classes:
        raise RuntimeError(f"blank index {blank} must be in [0, {classes})")

    input_lengths = _length_tensor(
        input_lengths, "input_lengths", batch, log_probs.device
    )
    target_lengths = _length_tensor(
        target_lengths, "target_lengths", batch, log_probs.device
    )
    # Avoid redundant .to() / .contiguous() dispatches when targets already
    # matches expectations.
    if targets.device != log_probs.device:
        targets = targets.to(device=log_probs.device)
    if not targets.is_contiguous():
        targets = targets.contiguous()
    if targets.ndim not in (1, 2):
        raise RuntimeError("targets must be 1D (concatenated) or 2D (padded)")
    if targets.ndim == 2 and targets.shape[0] != batch:
        raise RuntimeError("padded targets must have one row per batch element")

    # ----- single batched CPU<->GPU sync for all length validation -----
    # We sync the full target_lengths tensor (not just min/max/sum) so the
    # offsets prefix-sum for concatenated targets can be built in Python from
    # the same CPU snapshot, avoiding extra small kernel launches.
    if batch > 0:
        in_stats = torch.stack((input_lengths.amin(), input_lengths.amax())).cpu()
        in_min, in_max = int(in_stats[0]), int(in_stats[1])
        tl_cpu = target_lengths.cpu().tolist()
        tl_min = min(tl_cpu)
        tl_max = max(tl_cpu)
        tl_sum = sum(tl_cpu)
    else:
        in_min = in_max = tl_min = tl_max = tl_sum = 0
        tl_cpu = []

    if in_min < 0 or tl_min < 0:
        raise RuntimeError("ctc_loss lengths must be non-negative")
    if in_max > t_steps:
        raise RuntimeError("input_lengths must be <= log_probs.size(0)")
    full_input_lengths = (in_min == t_steps) and batch > 0

    if targets.ndim == 2:
        if tl_max > targets.shape[1]:
            raise RuntimeError("target_lengths exceed padded target width")
        target_ndim = 2
        max_target = targets.shape[1] if batch > 0 else 0
        # Single-element placeholder so the kernel signature accepts a pointer.
        offsets = torch.empty(1, device=log_probs.device, dtype=torch.int64)
    else:
        if tl_sum != targets.numel():
            raise RuntimeError(
                f"concatenated targets length {targets.numel()} != "
                f"sum(target_lengths) {tl_sum}"
            )
        target_ndim = 1
        max_target = tl_max
        offsets = _target_offsets_from_cpu(tl_cpu, log_probs.device)

    if targets.dtype != torch.int64:
        targets = targets.to(torch.int64)

    max_states = max(1, 2 * max_target + 1)
    block_states = max(triton.next_power_of_2(max_states), 16)
    if block_states > _MAX_BLOCK_STATES:
        raise RuntimeError(
            f"target_length {max_target} exceeds Triton state limit "
            f"(max={(_MAX_BLOCK_STATES - 1) // 2})"
        )

    return (
        unbatched,
        targets,
        input_lengths,
        target_lengths,
        offsets,
        target_ndim,
        max_target,
        max_states,
        block_states,
        full_input_lengths,
    )


# ---------------------------------------------------------------------------
# Triton helpers
# ---------------------------------------------------------------------------
@triton.jit
def _logaddexp2(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    m = tl.maximum(a, b)
    safe = tl.maximum(m, -1.0e30)
    return safe + tl.log(tl.exp(a - safe) + tl.exp(b - safe))


@triton.jit
def _logaddexp3(a, b, c, use_c):
    """Numerically stable log(exp(a) + exp(b) + (use_c ? exp(c) : 0))."""
    c_eff = tl.where(use_c, c, -float("inf"))
    m = tl.maximum(tl.maximum(a, b), c_eff)
    safe = tl.maximum(m, -1.0e30)
    s = tl.exp(a - safe) + tl.exp(b - safe)
    s = tl.where(use_c, s + tl.exp(c - safe), s)
    return safe + tl.log(s)


@triton.jit
def _fetch_target(
    targets,
    offsets,
    n,
    j,
    stride_n,
    stride_s,
    target_ndim: tl.constexpr,
    mask,
):
    """Load targets[n, j] under either padded (ndim=2) or concat (ndim=1) layout."""
    if target_ndim == 2:
        return tl.load(targets + n * stride_n + j * stride_s, mask=mask, other=0)
    start = tl.load(offsets + n)
    return tl.load(targets + (start + j) * stride_s, mask=mask, other=0)


@triton.jit
def _extended_labels(
    targets,
    offsets,
    n,
    states,
    target_len,
    blank,
    target_stride_n,
    target_stride_s,
    target_ndim: tl.constexpr,
):
    """Compute ext[s] for s in tile [0, BLOCK_S); out-of-range states return BLANK."""
    target_pos = (states - 1) // 2
    is_label = (states % 2) == 1
    valid = is_label & (target_pos >= 0) & (target_pos < target_len)
    raw = _fetch_target(
        targets,
        offsets,
        n,
        target_pos,
        target_stride_n,
        target_stride_s,
        target_ndim,
        valid,
    ).to(tl.int64)
    return tl.where(valid, raw, blank)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _ctc_forward_kernel(
    log_probs_ptr,  # (T, N, C) fp32
    targets_ptr,  # (N, S_pad) padded, or (sum_L,) concat, int64
    target_offsets_ptr,  # (N,) int64, valid only for concat layout
    input_lengths_ptr,  # (N,) int64
    target_lengths_ptr,  # (N,) int64
    losses_ptr,  # (N,) fp32 output -- raw losses (with potential +inf)
    normed_losses_ptr,  # (N,) fp32 output -- loss / max(target_len, 1) for mean
    log_alpha_ptr,  # (N, T, max_states) fp32 output (saved for backward)
    T,
    C,
    max_states,
    stride_t,
    stride_n,
    stride_c,
    target_stride_n,
    target_stride_s,
    BLANK: tl.constexpr,
    target_ndim: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    FULL_INPUT_LENGTHS: tl.constexpr,
    NEED_BARRIER: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    input_len = tl.load(input_lengths_ptr + n)
    target_len = tl.load(target_lengths_ptr + n)
    n_states = 2 * target_len + 1
    state_mask = states < n_states

    labels = _extended_labels(
        targets_ptr,
        target_offsets_ptr,
        n,
        states,
        target_len,
        BLANK,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    prev_labels = _extended_labels(
        targets_ptr,
        target_offsets_ptr,
        n,
        states - 2,
        target_len,
        BLANK,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    can_skip = (states >= 2) & state_mask & (labels != BLANK) & (labels != prev_labels)

    base = log_probs_ptr + n * stride_n
    alpha_base = log_alpha_ptr + n * T * max_states

    # ----- alpha at t = 0 -----
    # Explicit fp32 upcast on log_probs loads: log_probs may arrive in fp16,
    # and the subsequent tl.where merging into the fp32 alpha tile is cleaner
    # when the operands are already fp32.
    log_p_blank0 = tl.load(
        base + BLANK * stride_c, mask=input_len > 0, other=_NEG_INF
    ).to(tl.float32)
    first_target = _fetch_target(
        targets_ptr,
        target_offsets_ptr,
        n,
        0,
        target_stride_n,
        target_stride_s,
        target_ndim,
        target_len > 0,
    ).to(tl.int64)
    log_p_first = tl.load(
        base + first_target * stride_c,
        mask=(target_len > 0) & (input_len > 0),
        other=_NEG_INF,
    ).to(tl.float32)

    alpha = tl.full((BLOCK_S,), _NEG_INF, tl.float32)
    alpha = tl.where(states == 0, log_p_blank0, alpha)
    alpha = tl.where((states == 1) & (target_len > 0), log_p_first, alpha)
    alpha = tl.where(state_mask & (input_len > 0), alpha, _NEG_INF)
    tl.store(alpha_base + states, alpha, mask=state_mask)

    # ----- alpha at t = 1, ..., T - 1 -----
    # We keep alpha[t-1] in registers across iterations (saves one global load
    # per step vs. re-reading it).  The shifted versions a1, a2 must still come
    # from global memory because Triton lacks register-level shifts.
    #
    # When FULL_INPUT_LENGTHS=True (all batches have input_len == T), we can
    # skip the `t < input_len` mask entirely -- one less compare per iter.
    for t in tl.range(1, T):
        prev_base = alpha_base + (t - 1) * max_states
        cur_base = alpha_base + t * max_states

        # a0 is alpha[t-1, s], which we have in `alpha` from the previous iter.
        a1 = tl.load(
            prev_base + states - 1,
            mask=state_mask & (states >= 1),
            other=_NEG_INF,
        )
        a2 = tl.load(
            prev_base + states - 2,
            mask=state_mask & can_skip,
            other=_NEG_INF,
        )
        lse = _logaddexp3(alpha, a1, a2, can_skip)

        if FULL_INPUT_LENGTHS:
            lp = tl.load(
                base + t * stride_t + labels * stride_c,
                mask=state_mask,
                other=_NEG_INF,
            ).to(tl.float32)
            new_alpha = lse + lp
            new_alpha = tl.where(state_mask, new_alpha, _NEG_INF)
        else:
            active = t < input_len
            lp = tl.load(
                base + t * stride_t + labels * stride_c,
                mask=state_mask & active,
                other=_NEG_INF,
            ).to(tl.float32)
            new_alpha = lse + lp
            # When t >= input_len, freeze alpha (carry previous register value).
            new_alpha = tl.where(active & state_mask, new_alpha, alpha)
            new_alpha = tl.where(state_mask, new_alpha, _NEG_INF)

        alpha = new_alpha
        tl.store(cur_base + states, alpha, mask=state_mask)
        # Cross-warp ordering: the alpha[t] store must complete before the next
        # iter's alpha[t-1] load.  At <= 4 warps Triton's implicit scheduling
        # serializes this on its own, but at 8 warps (BLOCK_S >= 256) we
        # observed test_ctc_loss_long_target produce a non-deterministic
        # absolute error of ~6.6 against PyTorch.  Make the barrier a constexpr
        # so the 4-warp / 2-warp tiers (small shapes) skip it entirely and
        # avoid the ~1% per-iter sync overhead.
        if NEED_BARRIER:
            tl.debug_barrier()

    # ----- loss = -logsumexp(alpha[T_n-1, S'-1], alpha[T_n-1, S'-2]) -----
    last_t = tl.maximum(input_len - 1, 0)
    final_base = alpha_base + last_t * max_states
    last_state = n_states - 1
    end0 = tl.load(final_base + last_state, mask=input_len > 0, other=_NEG_INF)
    end1 = tl.load(
        final_base + last_state - 1,
        mask=(input_len > 0) & (target_len > 0),
        other=_NEG_INF,
    )
    log_like = tl.where(target_len > 0, _logaddexp2(end0, end1), end0)
    log_like = tl.where(input_len > 0, log_like, _NEG_INF)
    # Always emit the raw (possibly +inf) loss; zero_infinity is applied
    # Python-side so backward can still detect which samples were infinite.
    loss = -log_like
    tl.store(losses_ptr + n, loss)

    # Also emit pre-normalized loss for 'mean' reduction:
    #   normed[n] = (zero_infinity ? 0 : loss) / max(target_len, 1)
    # Folding the per-batch division into the kernel removes 3 Python ops
    # (clamp_min / cast / divide) from the forward fast path.  For 'none' /
    # 'sum' reductions Python ignores this buffer.
    safe_tl = tl.maximum(target_len, 1).to(tl.float32)
    normed = loss / safe_tl
    if ZERO_INFINITY:
        normed = tl.where(loss == float("inf"), 0.0, normed)
    tl.store(normed_losses_ptr + n, normed)


# ---------------------------------------------------------------------------
# Grad assembly kernel — fuses the Python-side
#   grad = scale[n] * valid_mask[t, n] * (exp(log_probs[t, n, c]) - posterior_sum[t, n, c])
# into a single pass to avoid four sequential elementwise launches and three
# intermediate (T, N, C) fp32 tensors.  Grid = (T, N); each program walks the
# C axis in BLOCK_C-wide tiles.
# ---------------------------------------------------------------------------
@triton.jit
def _ctc_grad_kernel(
    log_probs_ptr,  # (T, N, C) fp32 (post Python-side upcast)
    posterior_sum_ptr,  # (T, N, C) fp32
    scale_ptr,  # (N,) fp32 — reduction+zero_infinity baked in by Python
    input_lengths_ptr,  # (N,) int64
    grad_ptr,  # (T, N, C) fp32 output
    T,
    N,
    C,
    stride_t,
    stride_n,
    stride_c,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    scale = tl.load(scale_ptr + pid_n)
    input_len = tl.load(input_lengths_ptr + pid_n)
    active = pid_t < input_len

    base = pid_t * stride_t + pid_n * stride_n
    lp_base = log_probs_ptr + base
    ps_base = posterior_sum_ptr + base
    out_base = grad_ptr + base

    for c_offset in range(0, C, BLOCK_C):
        c_idx = c_offset + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        # Explicit fp32 upcast: log_probs may arrive in fp16; we want the
        # exp/sub to run at fp32 precision and we must avoid bf16 PTX on sm_75.
        log_p = tl.load(lp_base + c_idx * stride_c, mask=c_mask, other=0.0).to(
            tl.float32
        )
        post = tl.load(ps_base + c_idx * stride_c, mask=c_mask, other=0.0)
        out = scale * (tl.exp(log_p) - post)
        out = tl.where(active & c_mask, out, 0.0)
        tl.store(out_base + c_idx * stride_c, out, mask=c_mask)


# ---------------------------------------------------------------------------
# Backward kernel — beta DP + posterior scatter
# ---------------------------------------------------------------------------
@triton.jit
def _ctc_backward_kernel(
    log_probs_ptr,  # (T, N, C) fp32
    targets_ptr,
    target_offsets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    log_alpha_ptr,  # (N, T, max_states) fp32 (saved from forward)
    losses_ptr,  # (N,) fp32 (unused for value; kept for signature)
    scratch_beta_ptr,  # (N, max_states) fp32 — beta at current t, for shifted reads
    posterior_sum_ptr,  # (T, N, C) fp32 — atomic-add target
    T,
    C,
    max_states,
    stride_t,
    stride_n,
    stride_c,
    target_stride_n,
    target_stride_s,
    BLANK: tl.constexpr,
    target_ndim: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    NEED_BARRIER: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    input_len = tl.load(input_lengths_ptr + n)
    target_len = tl.load(target_lengths_ptr + n)
    n_states = 2 * target_len + 1
    state_mask = states < n_states

    if input_len <= 0:
        return

    labels = _extended_labels(
        targets_ptr,
        target_offsets_ptr,
        n,
        states,
        target_len,
        BLANK,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    labels_p1 = _extended_labels(
        targets_ptr,
        target_offsets_ptr,
        n,
        states + 1,
        target_len,
        BLANK,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    labels_p2 = _extended_labels(
        targets_ptr,
        target_offsets_ptr,
        n,
        states + 2,
        target_len,
        BLANK,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )

    # Forward "skip" rule from state s -> s+2 (used to build beta).
    # Requires: s+2 < S', s odd (i.e. ext[s] is a label not a blank),
    # ext[s] != BLANK (redundant with s odd if blank is distinct from labels,
    # but we keep both for safety) and ext[s] != ext[s+2].
    skip_fwd = (
        (states + 2 < n_states)
        & ((states % 2) == 1)
        & (labels != BLANK)
        & (labels != labels_p2)
    )

    base = log_probs_ptr + n * stride_n
    alpha_base = log_alpha_ptr + n * T * max_states
    sb_base = scratch_beta_ptr + n * max_states

    # ----- log_like_n (for gamma normalization) -----
    last_t = input_len - 1
    last_state = n_states - 1
    final_alpha_base = alpha_base + last_t * max_states
    end0 = tl.load(final_alpha_base + last_state, mask=input_len > 0, other=_NEG_INF)
    end1 = tl.load(
        final_alpha_base + last_state - 1,
        mask=(input_len > 0) & (target_len > 0),
        other=_NEG_INF,
    )
    log_like = tl.where(target_len > 0, _logaddexp2(end0, end1), end0)

    # zero_infinity: skip the whole batch when log_like == -inf.
    skip_batch = ZERO_INFINITY & (log_like == _NEG_INF)

    # ----- beta at t = T_n - 1 -----
    beta = tl.full((BLOCK_S,), _NEG_INF, tl.float32)
    beta = tl.where(states == last_state, 0.0, beta)
    beta = tl.where((states == last_state - 1) & (target_len > 0), 0.0, beta)
    beta = tl.where(state_mask, beta, _NEG_INF)
    # Persist beta into scratch so shifted reads (s+1, s+2) work next iteration.
    tl.store(sb_base + states, beta, mask=state_mask)
    if NEED_BARRIER:
        tl.debug_barrier()

    # Sweep t = last_t, last_t - 1, ..., 0.
    t = last_t
    while t >= 0:
        alpha_t = tl.load(
            alpha_base + t * max_states + states,
            mask=state_mask,
            other=_NEG_INF,
        )
        # gamma[t, s] = exp(alpha + beta - log_like).  -inf inputs give 0.
        gamma = tl.exp(alpha_t + beta - log_like)
        gamma = tl.where(state_mask, gamma, 0.0)

        if not skip_batch:
            scatter_ptr = (
                posterior_sum_ptr + t * stride_t + n * stride_n + labels * stride_c
            )
            tl.atomic_add(scatter_ptr, gamma, mask=state_mask, sem="relaxed")

        # ----- beta at t - 1 -----
        if t > 0:
            # Shifted beta from scratch.
            beta_p1 = tl.load(
                sb_base + states + 1,
                mask=state_mask & (states + 1 < n_states),
                other=_NEG_INF,
            )
            beta_p2 = tl.load(
                sb_base + states + 2,
                mask=state_mask & skip_fwd,
                other=_NEG_INF,
            )
            lp_s = tl.load(
                base + t * stride_t + labels * stride_c,
                mask=state_mask,
                other=_NEG_INF,
            ).to(tl.float32)
            lp_p1 = tl.load(
                base + t * stride_t + labels_p1 * stride_c,
                mask=state_mask & (states + 1 < n_states),
                other=_NEG_INF,
            ).to(tl.float32)
            lp_p2 = tl.load(
                base + t * stride_t + labels_p2 * stride_c,
                mask=state_mask & skip_fwd,
                other=_NEG_INF,
            ).to(tl.float32)

            stay = beta + lp_s
            move1 = beta_p1 + lp_p1
            move2 = beta_p2 + lp_p2
            new_beta = _logaddexp3(stay, move1, move2, skip_fwd)
            new_beta = tl.where(state_mask, new_beta, _NEG_INF)

            beta = new_beta
            tl.store(sb_base + states, beta, mask=state_mask)
            if NEED_BARRIER:
                tl.debug_barrier()

        t -= 1


# ---------------------------------------------------------------------------
# autograd.Function
# ---------------------------------------------------------------------------
class _CtcLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction=1,
        zero_infinity=False,
    ):
        reduction = _normalize_reduction(reduction)
        (
            unbatched,
            targets,
            input_lengths,
            target_lengths,
            offsets,
            target_ndim,
            max_target,
            max_states,
            block_states,
            full_input_lengths,
        ) = _check_inputs(log_probs, targets, input_lengths, target_lengths, blank)

        # Keep work_log_probs in its original dtype for fp16 inputs: all three
        # kernels (forward DP, backward DP, fused grad assembly) upcast to fp32
        # in-register on every tl.load, so a Python-side fp32 materialization
        # just doubles the kernel's HBM read of log_probs without changing the
        # math.  bf16 is a special case: Triton's bf16 -> fp32 conversion
        # emits `cvt.f32.bf16` PTX which requires sm_80+, so on older GPUs
        # (sm_75 = Turing 2080Ti / sm_70 = Volta V100) we still upcast bf16 to
        # fp32 here.  fp16 has been native since sm_53 so it works everywhere.
        work_log_probs = log_probs.contiguous()
        if unbatched:
            work_log_probs = work_log_probs.unsqueeze(1)
        if work_log_probs.dtype == torch.bfloat16 and not _bf16_native(
            work_log_probs.device
        ):
            work_log_probs = work_log_probs.to(torch.float32)

        t_steps, batch, classes = work_log_probs.shape

        target_stride_n = targets.stride(0) if targets.ndim == 2 else 0
        target_stride_s = targets.stride(1) if targets.ndim == 2 else targets.stride(0)

        losses = torch.empty((batch,), dtype=torch.float32, device=log_probs.device)
        normed_losses = torch.empty(
            (batch,), dtype=torch.float32, device=log_probs.device
        )
        log_alpha = torch.empty(
            (batch, t_steps, max_states),
            dtype=torch.float32,
            device=log_probs.device,
        )

        # Heuristic launch tuning: BLOCK_S controls per-program register/SMEM
        # footprint.  More warps help cover memory latency at large BLOCK_S; at
        # small BLOCK_S the inner DP is so cheap that adding warps just costs
        # synchronization.  num_stages=2 enables software pipelining of the
        # log_probs gather against the previous iteration's alpha update.
        if block_states >= 256:
            fwd_warps, fwd_stages = 8, 2
        elif block_states >= 128:
            fwd_warps, fwd_stages = 4, 2
        else:
            fwd_warps, fwd_stages = 2, 2

        with torch_device_fn.device(log_probs.device):
            _ctc_forward_kernel[(batch,)](
                work_log_probs,
                targets,
                offsets,
                input_lengths,
                target_lengths,
                losses,
                normed_losses,
                log_alpha,
                t_steps,
                classes,
                max_states,
                work_log_probs.stride(0),
                work_log_probs.stride(1),
                work_log_probs.stride(2),
                target_stride_n,
                target_stride_s,
                BLANK=int(blank),
                target_ndim=target_ndim,
                ZERO_INFINITY=bool(zero_infinity),
                FULL_INPUT_LENGTHS=bool(full_input_lengths),
                NEED_BARRIER=bool(fwd_warps >= 8),
                BLOCK_S=block_states,
                num_warps=fwd_warps,
                num_stages=fwd_stages,
            )

        ctx.save_for_backward(
            work_log_probs,
            targets,
            offsets,
            input_lengths,
            target_lengths,
            losses,
            log_alpha,
        )
        ctx.meta = (
            int(blank),
            int(reduction),
            bool(zero_infinity),
            bool(unbatched),
            int(target_ndim),
            int(max_states),
            int(block_states),
            int(target_stride_n),
            int(target_stride_s),
        )
        ctx.input_dtype = log_probs.dtype

        # ----- reduction -----
        # Mean: the kernel already wrote per-batch loss/max(target_len, 1) into
        # normed_losses, with zero_infinity zeroing applied if requested. So
        # mean is a single .mean() call here.
        # Sum/none: use raw losses; zero_infinity zeroes infs Python-side.
        if reduction == 1:  # mean
            out = normed_losses.mean()
        else:
            if zero_infinity:
                display_losses = torch.where(
                    torch.isinf(losses), torch.zeros_like(losses), losses
                )
            else:
                display_losses = losses
            if reduction == 0:  # none
                out = display_losses if not unbatched else display_losses[0]
            else:  # sum
                out = display_losses.sum()
        return out.to(log_probs.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (
            work_log_probs,
            targets,
            offsets,
            input_lengths,
            target_lengths,
            losses,
            log_alpha,
        ) = ctx.saved_tensors
        (
            blank,
            reduction,
            zero_infinity,
            unbatched,
            target_ndim,
            max_states,
            block_states,
            target_stride_n,
            target_stride_s,
        ) = ctx.meta

        t_steps, batch, classes = work_log_probs.shape
        dev = work_log_probs.device

        scratch_beta = torch.empty((batch, max_states), dtype=torch.float32, device=dev)
        posterior_sum = torch.zeros(
            (t_steps, batch, classes), dtype=torch.float32, device=dev
        )

        # Match forward's launch heuristic so backward gets the same occupancy
        # treatment.  Backward is even more memory-bound (writes posterior_sum
        # via atomic_add), so extra warps help hide latency on big shapes.
        if block_states >= 256:
            bwd_warps, bwd_stages = 8, 2
        elif block_states >= 128:
            bwd_warps, bwd_stages = 4, 2
        else:
            bwd_warps, bwd_stages = 2, 2

        with torch_device_fn.device(dev):
            _ctc_backward_kernel[(batch,)](
                work_log_probs,
                targets,
                offsets,
                input_lengths,
                target_lengths,
                log_alpha,
                losses,
                scratch_beta,
                posterior_sum,
                t_steps,
                classes,
                max_states,
                work_log_probs.stride(0),
                work_log_probs.stride(1),
                work_log_probs.stride(2),
                target_stride_n,
                target_stride_s,
                BLANK=int(blank),
                target_ndim=target_ndim,
                ZERO_INFINITY=bool(zero_infinity),
                NEED_BARRIER=bool(bwd_warps >= 8),
                BLOCK_S=block_states,
                num_warps=bwd_warps,
                num_stages=bwd_stages,
            )

        # ----- assemble per-batch scale -----
        # scale[n] folds the reduction (mean/sum/none) and zero_infinity into
        # a single (batch,) fp32 tensor that the fused grad kernel consumes.
        # The t < input_len mask is computed inside the kernel from
        # input_lengths to avoid materializing a (T, batch) helper tensor.
        go = grad_output.to(torch.float32)
        if reduction == 0:  # none
            if go.numel() == 1:
                scale = go.reshape(()).expand(batch).contiguous()
            else:
                scale = go.reshape(batch).contiguous()
        elif reduction == 2:  # sum
            scale = go.reshape(()).expand(batch).contiguous()
        else:  # mean
            denom = target_lengths.clamp_min(1).to(torch.float32) * batch
            scale = (go.reshape(()) / denom).contiguous()

        if zero_infinity:
            inf_mask = torch.isinf(losses)
            scale = torch.where(inf_mask, torch.zeros_like(scale), scale)

        # ----- fused grad kernel -----
        # Single (T, N) grid pass replaces 4 Python elementwise launches
        # (exp / sub / scale*valid*x / dtype-cast) and skips ~3 intermediate
        # (T, N, C) fp32 tensors worth of HBM traffic.
        #
        # Output dtype: allocate grad directly in the user's input dtype when
        # we can store it from the kernel (fp32 / fp16 are fine on all CUDA
        # gens; bf16 needs the `cvt.bf16.f32` PTX which is sm_80+).  For bf16
        # on older GPUs we keep grad fp32 and cast Python-side.
        if ctx.input_dtype == torch.bfloat16 and not _bf16_native(dev):
            grad_dtype = torch.float32
        else:
            grad_dtype = ctx.input_dtype
        grad = torch.empty((t_steps, batch, classes), dtype=grad_dtype, device=dev)

        block_c = min(triton.next_power_of_2(max(classes, 16)), 1024)
        if block_c >= 512:
            grad_warps = 8
        elif block_c >= 128:
            grad_warps = 4
        else:
            grad_warps = 2

        with torch_device_fn.device(dev):
            _ctc_grad_kernel[(t_steps, batch)](
                work_log_probs,
                posterior_sum,
                scale,
                input_lengths,
                grad,
                t_steps,
                batch,
                classes,
                grad.stride(0),
                grad.stride(1),
                grad.stride(2),
                BLOCK_C=block_c,
                num_warps=grad_warps,
                num_stages=2,
            )

        if grad.dtype != ctx.input_dtype:
            grad = grad.to(ctx.input_dtype)
        if unbatched:
            grad = grad.squeeze(1)

        return grad, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    logger.debug("GEMS CTC_LOSS")
    return _CtcLoss.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        int(blank),
        _normalize_reduction(reduction),
        bool(zero_infinity),
    )
