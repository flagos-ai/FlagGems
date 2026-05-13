import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_BLOCK_STATES = 1024
_CACHE_LIMIT = 64
_LENGTH_VALUES_CACHE = {}
_LENGTH_STATS_CACHE = {}
_TARGET_OFFSETS_CACHE = {}


def _tensor_cache_key(tensor):
    if not torch.is_tensor(tensor):
        return None
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tensor.data_ptr(),
        tensor.numel(),
        tuple(tensor.shape),
        tensor._version,
    )


def _cache_put(cache, key, value):
    if len(cache) >= _CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def _host_length_values(tensor):
    if tensor.numel() == 0:
        return ()
    key = _tensor_cache_key(tensor)
    if key is not None:
        cached = _LENGTH_VALUES_CACHE.get(key)
        if cached is not None and cached[0] is tensor:
            return cached[1]

    # Length tensors drive wrapper validation and Triton launch sizing.
    values = tuple(
        int(value) for value in tensor.detach().to(device="cpu").reshape(-1).tolist()
    )
    if key is not None:
        _cache_put(_LENGTH_VALUES_CACHE, key, (tensor, values))
    return values


def _length_stats(tensor):
    values = _host_length_values(tensor)
    if not values:
        return 0, 0, 0
    key = _tensor_cache_key(tensor)
    if key is not None:
        cached = _LENGTH_STATS_CACHE.get(key)
        if cached is not None and cached[0] is tensor:
            return cached[1]

    stats = (min(values), max(values), sum(values))
    if key is not None:
        _cache_put(_LENGTH_STATS_CACHE, key, (tensor, stats))
    return stats


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        value = reduction.lower()
        if value == "none":
            return 0
        if value == "mean":
            return 1
        if value == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _length_tensor(lengths, name, batch, device):
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
            raise RuntimeError(f"{name} must have batch size {batch}")
        return result.to(device=device, dtype=torch.long).contiguous()

    if not isinstance(lengths, (list, tuple)):
        raise RuntimeError(f"{name} must be a Tensor or a sequence of integers")
    if len(lengths) != batch:
        raise RuntimeError(f"{name} must have batch size {batch}")
    return torch.tensor(lengths, device=device, dtype=torch.long)


def _target_offsets(target_lengths):
    key = _tensor_cache_key(target_lengths)
    if key is not None:
        cached = _TARGET_OFFSETS_CACHE.get(key)
        if cached is not None and cached[0] is target_lengths:
            return cached[1]

    offset_values = []
    running = 0
    for length in _host_length_values(target_lengths):
        offset_values.append(running)
        running += length
    offsets = torch.tensor(
        offset_values,
        device=target_lengths.device,
        dtype=target_lengths.dtype,
    ).reshape(target_lengths.shape)
    if key is not None:
        _cache_put(_TARGET_OFFSETS_CACHE, key, (target_lengths, offsets))
    return offsets


def _check_inputs(log_probs, targets, input_lengths, target_lengths, blank):
    if log_probs.ndim not in (2, 3):
        raise RuntimeError("ctc_loss expects 2D or 3D log_probs")
    if not torch.is_floating_point(log_probs):
        raise RuntimeError('"ctc_loss" not implemented for non-floating log_probs')
    if not torch.is_tensor(targets):
        raise RuntimeError("ctc_loss expects targets to be a Tensor")
    if not log_probs.is_cuda:
        raise RuntimeError("ctc_loss Triton implementation expects CUDA log_probs")

    unbatched = log_probs.ndim == 2
    if unbatched:
        t_steps, classes = log_probs.shape
        batch = 1
    else:
        t_steps, batch, classes = log_probs.shape

    if blank < 0 or blank >= classes:
        raise RuntimeError("blank must be in label range")

    input_lengths = _length_tensor(
        input_lengths, "input_lengths", batch, log_probs.device
    )
    target_lengths = _length_tensor(
        target_lengths, "target_lengths", batch, log_probs.device
    )

    in_min, in_max, _ = _length_stats(input_lengths)
    tgt_min, tgt_max, tgt_sum = _length_stats(target_lengths)
    if in_min < 0 or tgt_min < 0:
        raise RuntimeError("ctc_loss lengths must be non-negative")

    targets = targets.to(device=log_probs.device).contiguous()
    if targets.ndim == 2:
        if targets.shape[0] != batch:
            raise RuntimeError("padded targets must have one row per batch item")
        if tgt_max > targets.shape[1]:
            raise RuntimeError("target_lengths exceed padded target width")
        target_ndim = 2
        max_target = targets.shape[1]
        offsets = torch.empty((0,), device=log_probs.device, dtype=torch.long)
    elif targets.ndim == 1:
        if tgt_sum != targets.numel():
            raise RuntimeError(
                "concatenated targets length does not match target_lengths"
            )
        target_ndim = 1
        max_target = tgt_max
        offsets = _target_offsets(target_lengths)
    else:
        raise RuntimeError("targets must be 1D concatenated or 2D padded")

    if in_max > t_steps:
        raise RuntimeError("input_lengths must be in [0, log_probs.size(0)]")

    block_states = triton.next_power_of_2(max(1, 2 * max_target + 1))
    if block_states > _MAX_BLOCK_STATES:
        raise RuntimeError("ctc_loss target length exceeds Triton state limit")

    return (
        unbatched,
        targets,
        input_lengths,
        target_lengths,
        offsets,
        target_ndim,
        max_target,
        block_states,
        in_min == t_steps,
    )


@triton.jit
def _logaddexp(a, b):
    m = tl.maximum(a, b)
    safe_m = tl.maximum(m, -1.0e20)
    return safe_m + tl.log(tl.exp(a - safe_m) + tl.exp(b - safe_m))


@triton.jit
def _logaddexp3(a, b, c, use_c):
    c_value = tl.where(use_c, c, -float("inf"))
    m = tl.maximum(tl.maximum(a, b), c_value)
    safe_m = tl.maximum(m, -1.0e20)
    total = tl.exp(a - safe_m) + tl.exp(b - safe_m)
    total += tl.where(use_c, tl.exp(c - safe_m), 0.0)
    return safe_m + tl.log(total)


@triton.jit
def _fetch_targets(
    targets,
    offsets,
    n,
    j,
    stride_n,
    stride_s,
    target_ndim: tl.constexpr,
    mask,
):
    if target_ndim == 2:
        return tl.load(targets + n * stride_n + j * stride_s, mask=mask, other=0)
    start = tl.load(offsets + n)
    return tl.load(targets + start + j, mask=mask, other=0)


@triton.jit
def _labels_for_states(
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
    target_pos = (states - 1) // 2
    is_label = states % 2 == 1
    valid_label = is_label & (target_pos >= 0) & (target_pos < target_len)
    target_values = _fetch_targets(
        targets,
        offsets,
        n,
        target_pos,
        target_stride_n,
        target_stride_s,
        target_ndim,
        valid_label,
    ).to(tl.int64)
    return tl.where(valid_label, target_values, blank)


@libentry()
@triton.jit
def _ctc_forward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    losses,
    log_alpha,
    blank: tl.constexpr,
    t_steps: tl.constexpr,
    classes: tl.constexpr,
    max_states: tl.constexpr,
    stride_t: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    target_stride_n: tl.constexpr,
    target_stride_s: tl.constexpr,
    target_ndim: tl.constexpr,
    zero_infinity: tl.constexpr,
    BLOCK_STATES: tl.constexpr,
):
    n = tl.program_id(0)
    states = tl.arange(0, BLOCK_STATES)
    input_len = tl.load(input_lengths + n)
    target_len = tl.load(target_lengths + n)
    n_states = target_len * 2 + 1
    state_mask = states < n_states
    target_pos = (states - 1) // 2

    labels = _labels_for_states(
        targets,
        target_offsets,
        n,
        states,
        target_len,
        blank,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    prev_labels = _fetch_targets(
        targets,
        target_offsets,
        n,
        target_pos - 1,
        target_stride_n,
        target_stride_s,
        target_ndim,
        (target_pos > 0) & state_mask,
    ).to(tl.int64)
    skip_allowed = (
        (states > 1) & ((states % 2) == 1) & (target_pos > 0) & (labels != prev_labels)
    )

    base = log_probs + n * stride_n
    alpha_base = log_alpha + n * t_steps * max_states

    first_blank = tl.load(base + blank * stride_c)
    first_label = _fetch_targets(
        targets,
        target_offsets,
        n,
        states * 0,
        target_stride_n,
        target_stride_s,
        target_ndim,
        target_len > 0,
    ).to(tl.int64)
    first_label_lp = tl.load(
        base + first_label * stride_c, mask=target_len > 0, other=-float("inf")
    )

    alpha = tl.full((BLOCK_STATES,), -float("inf"), tl.float32)
    alpha = tl.where(states == 0, first_blank, alpha)
    alpha = tl.where((states == 1) & (target_len > 0), first_label_lp, alpha)
    alpha = tl.where(state_mask & (input_len > 0), alpha, -float("inf"))
    tl.store(alpha_base + states, alpha, mask=state_mask)

    for t in tl.range(1, t_steps):
        prev0 = tl.load(
            alpha_base + (t - 1) * max_states + states,
            mask=state_mask,
            other=-float("inf"),
        )
        prev1 = tl.load(
            alpha_base + (t - 1) * max_states + states - 1,
            mask=state_mask & (states > 0),
            other=-float("inf"),
        )
        prev2 = tl.load(
            alpha_base + (t - 1) * max_states + states - 2,
            mask=state_mask & skip_allowed,
            other=-float("inf"),
        )
        total = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        lp = tl.load(
            base + t * stride_t + labels * stride_c,
            mask=state_mask & (t < input_len),
            other=-float("inf"),
        ).to(tl.float32)
        alpha = total + lp
        alpha = tl.where(state_mask & (t < input_len), alpha, -float("inf"))
        tl.store(
            alpha_base + t * max_states + states,
            alpha,
            mask=state_mask & (t < input_len),
        )

    last_t = input_len - 1
    last_state = n_states - 1
    if input_len <= 0:
        log_like = -float("inf")
    elif target_len == 0:
        log_like = tl.load(alpha_base + last_t * max_states)
    else:
        end0 = tl.load(alpha_base + last_t * max_states + last_state)
        end1 = tl.load(alpha_base + last_t * max_states + last_state - 1)
        log_like = _logaddexp(end0, end1)

    loss = -log_like
    if zero_infinity:
        loss = tl.where(loss == float("inf"), 0.0, loss)
    tl.store(losses + n, loss)


@libentry()
@triton.jit
def _ctc_forward_full_length_reduce_kernel(
    log_probs,
    targets,
    target_lengths,
    target_offsets,
    contrib,
    scratch_alpha,
    blank: tl.constexpr,
    t_steps: tl.constexpr,
    classes: tl.constexpr,
    max_states: tl.constexpr,
    batch: tl.constexpr,
    reduction: tl.constexpr,
    stride_t: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    target_stride_n: tl.constexpr,
    target_stride_s: tl.constexpr,
    target_ndim: tl.constexpr,
    BLOCK_STATES: tl.constexpr,
):
    n = tl.program_id(0)
    states = tl.arange(0, BLOCK_STATES)
    target_len = tl.load(target_lengths + n)
    n_states = target_len * 2 + 1
    state_mask = states < n_states
    target_pos = (states - 1) // 2

    labels = _labels_for_states(
        targets,
        target_offsets,
        n,
        states,
        target_len,
        blank,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    prev_labels = _fetch_targets(
        targets,
        target_offsets,
        n,
        target_pos - 1,
        target_stride_n,
        target_stride_s,
        target_ndim,
        (target_pos > 0) & state_mask,
    ).to(tl.int64)
    skip_allowed = (
        (states > 1) & ((states % 2) == 1) & (target_pos > 0) & (labels != prev_labels)
    )

    base = log_probs + n * stride_n
    scratch_base = scratch_alpha + n * 2 * max_states

    first_blank = tl.load(base + blank * stride_c)
    first_label = _fetch_targets(
        targets,
        target_offsets,
        n,
        states * 0,
        target_stride_n,
        target_stride_s,
        target_ndim,
        target_len > 0,
    ).to(tl.int64)
    first_label_lp = tl.load(
        base + first_label * stride_c,
        mask=target_len > 0,
        other=-float("inf"),
    )

    alpha = tl.full((BLOCK_STATES,), -float("inf"), tl.float32)
    alpha = tl.where(states == 0, first_blank, alpha)
    alpha = tl.where((states == 1) & (target_len > 0), first_label_lp, alpha)
    alpha = tl.where(state_mask, alpha, -float("inf"))
    tl.store(scratch_base + states, alpha, mask=state_mask)

    for t in tl.range(1, t_steps):
        prev_base = scratch_base + ((t - 1) % 2) * max_states
        cur_base = scratch_base + (t % 2) * max_states
        prev0 = tl.load(
            prev_base + states,
            mask=state_mask,
            other=-float("inf"),
        )
        prev1 = tl.load(
            prev_base + states - 1,
            mask=state_mask & (states > 0),
            other=-float("inf"),
        )
        prev2 = tl.load(
            prev_base + states - 2,
            mask=state_mask & skip_allowed,
            other=-float("inf"),
        )
        total = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        lp = tl.load(
            base + t * stride_t + labels * stride_c,
            mask=state_mask,
            other=-float("inf"),
        ).to(tl.float32)
        alpha = tl.where(state_mask, total + lp, -float("inf"))
        tl.store(cur_base + states, alpha, mask=state_mask)

    final_base = scratch_base + ((t_steps - 1) % 2) * max_states
    last_state = n_states - 1
    if target_len == 0:
        log_like = tl.load(final_base)
    else:
        end0 = tl.load(final_base + last_state)
        end1 = tl.load(final_base + last_state - 1)
        log_like = _logaddexp(end0, end1)

    loss = -log_like
    if reduction == 1:
        loss = loss / tl.maximum(target_len, 1).to(tl.float32) / batch
    tl.store(contrib + n, loss)


@libentry()
@triton.jit
def _ctc_grad_init_flat_kernel(
    grad_out,
    log_probs,
    input_lengths,
    target_lengths,
    log_alpha,
    losses,
    grad_in,
    reduction: tl.constexpr,
    total: tl.constexpr,
    t_steps: tl.constexpr,
    batch: tl.constexpr,
    classes: tl.constexpr,
    max_states: tl.constexpr,
    stride_t: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    zero_infinity: tl.constexpr,
    none_is_scalar: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    c_offsets = offsets % classes
    n = (offsets // classes) % batch
    t = offsets // (batch * classes)

    input_len = tl.load(input_lengths + n, mask=mask, other=0)
    target_len = tl.load(target_lengths + n, mask=mask, other=1)

    if reduction == 0:
        if none_is_scalar:
            scale = tl.load(grad_out)
        else:
            scale = tl.load(grad_out + n, mask=mask, other=0.0)
    else:
        scale = tl.load(grad_out)
        if reduction == 1:
            denom = tl.maximum(target_len, 1) * batch
            scale = scale / denom

    if zero_infinity:
        n_states = target_len * 2 + 1
        has_input = input_len > 0
        last_t = tl.maximum(input_len - 1, 0)
        last_state = n_states - 1
        alpha_base = log_alpha + n * t_steps * max_states
        blank_like = tl.load(
            alpha_base + last_t * max_states,
            mask=mask & has_input,
            other=-float("inf"),
        )
        end0 = tl.load(
            alpha_base + last_t * max_states + last_state,
            mask=mask & has_input,
            other=-float("inf"),
        )
        end1 = tl.load(
            alpha_base + last_t * max_states + last_state - 1,
            mask=mask & has_input & (target_len > 0),
            other=-float("inf"),
        )
        nonempty_like = _logaddexp(end0, end1)
        log_like = tl.where(target_len == 0, blank_like, nonempty_like)
        empty_like = tl.where(target_len == 0, 0.0, -float("inf"))
        log_like = tl.where(has_input, log_like, empty_like)
        scale = tl.where(log_like == -float("inf"), 0.0, scale)

    prob_base = log_probs + n * stride_n
    logp = tl.load(
        prob_base + t * stride_t + c_offsets * stride_c,
        mask=mask & (t < input_len),
        other=-float("inf"),
    ).to(tl.float32)
    tl.store(
        grad_in + n * stride_n + t * stride_t + c_offsets * stride_c,
        scale * tl.exp(logp),
        mask=mask & (t < input_len),
    )


@libentry()
@triton.jit
def _ctc_grad_init_serial_kernel(
    grad_out,
    log_probs,
    input_lengths,
    target_lengths,
    log_alpha,
    grad_in,
    reduction: tl.constexpr,
    t_steps: tl.constexpr,
    batch: tl.constexpr,
    classes: tl.constexpr,
    max_states: tl.constexpr,
    stride_t: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    zero_infinity: tl.constexpr,
    none_is_scalar: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < classes
    input_len = tl.load(input_lengths + n)
    target_len = tl.load(target_lengths + n)
    n_states = target_len * 2 + 1
    alpha_base = log_alpha + n * t_steps * max_states

    if input_len <= 0:
        return

    last_t = input_len - 1
    last_state = n_states - 1
    if target_len == 0:
        log_like = tl.load(alpha_base + last_t * max_states)
    else:
        end0 = tl.load(alpha_base + last_t * max_states + last_state)
        end1 = tl.load(alpha_base + last_t * max_states + last_state - 1)
        log_like = _logaddexp(end0, end1)

    zeroed = zero_infinity & (log_like == -float("inf"))
    if zeroed:
        return

    if reduction == 0:
        if none_is_scalar:
            scale = tl.load(grad_out)
        else:
            scale = tl.load(grad_out + n)
    else:
        scale = tl.load(grad_out)
        if reduction == 1:
            denom = tl.maximum(target_len, 1) * batch
            scale = scale / denom

    prob_base = log_probs + n * stride_n
    grad_base = grad_in + n * stride_n
    t = 0
    while t < input_len:
        logp = tl.load(
            prob_base + t * stride_t + c_offsets * stride_c,
            mask=c_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tl.store(
            grad_base + t * stride_t + c_offsets * stride_c,
            scale * tl.exp(logp),
            mask=c_mask,
        )
        t += 1


@libentry()
@triton.jit
def _ctc_backward_kernel(
    grad_out,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    losses,
    log_alpha,
    scratch_beta,
    grad_in,
    blank: tl.constexpr,
    reduction: tl.constexpr,
    t_steps: tl.constexpr,
    batch: tl.constexpr,
    classes: tl.constexpr,
    max_states: tl.constexpr,
    stride_t: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    target_stride_n: tl.constexpr,
    target_stride_s: tl.constexpr,
    target_ndim: tl.constexpr,
    zero_infinity: tl.constexpr,
    none_is_scalar: tl.constexpr,
    init_grad: tl.constexpr,
    BLOCK_STATES: tl.constexpr,
    BLOCK_CLASSES: tl.constexpr,
):
    n = tl.program_id(0)
    states = tl.arange(0, BLOCK_STATES)
    if init_grad:
        c_offsets = tl.arange(0, BLOCK_CLASSES)
        c_mask = c_offsets < classes
    input_len = tl.load(input_lengths + n)
    target_len = tl.load(target_lengths + n)
    n_states = target_len * 2 + 1
    state_mask = states < n_states
    target_pos = (states - 1) // 2

    labels = _labels_for_states(
        targets,
        target_offsets,
        n,
        states,
        target_len,
        blank,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    next_labels = _labels_for_states(
        targets,
        target_offsets,
        n,
        states + 2,
        target_len,
        blank,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    skip_from_state = (
        (states + 2 < n_states)
        & ((states % 2) == 1)
        & (target_pos + 1 < target_len)
        & (labels != next_labels)
    )

    alpha_base = log_alpha + n * t_steps * max_states
    beta_base = scratch_beta + n * 2 * max_states
    prob_base = log_probs + n * stride_n
    grad_base = grad_in + n * stride_n

    if input_len <= 0:
        return

    last_t = input_len - 1
    last_state = n_states - 1
    if target_len == 0:
        log_like = tl.load(alpha_base + last_t * max_states)
    else:
        end0 = tl.load(alpha_base + last_t * max_states + last_state)
        end1 = tl.load(alpha_base + last_t * max_states + last_state - 1)
        log_like = _logaddexp(end0, end1)

    zeroed = zero_infinity & (log_like == -float("inf"))
    if zeroed:
        return

    if reduction == 0:
        if none_is_scalar:
            scale = tl.load(grad_out)
        else:
            scale = tl.load(grad_out + n)
    else:
        scale = tl.load(grad_out)
        if reduction == 1:
            denom = tl.maximum(target_len, 1) * batch
            scale = scale / denom

    beta = tl.full((BLOCK_STATES,), -float("inf"), tl.float32)
    beta = tl.where(states == last_state, 0.0, beta)
    beta = tl.where((target_len > 0) & (states == last_state - 1), 0.0, beta)
    beta = tl.where(state_mask, beta, -float("inf"))
    tl.store(
        beta_base + (last_t % 2) * max_states + states,
        beta,
        mask=state_mask,
    )

    labels_1 = _labels_for_states(
        targets,
        target_offsets,
        n,
        states + 1,
        target_len,
        blank,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )
    labels_2 = _labels_for_states(
        targets,
        target_offsets,
        n,
        states + 2,
        target_len,
        blank,
        target_stride_n,
        target_stride_s,
        target_ndim,
    )

    t = last_t
    while t >= 0:
        cur_beta_base = beta_base + (t % 2) * max_states
        if init_grad:
            logp_all = tl.load(
                prob_base + t * stride_t + c_offsets * stride_c,
                mask=c_mask,
                other=-float("inf"),
            ).to(tl.float32)
            tl.store(
                grad_base + t * stride_t + c_offsets * stride_c,
                scale * tl.exp(logp_all),
                mask=c_mask,
            )
            tl.debug_barrier()
        alpha = tl.load(
            alpha_base + t * max_states + states,
            mask=state_mask,
            other=-float("inf"),
        )
        posterior = tl.exp(alpha + beta - log_like)
        posterior = tl.where(state_mask, posterior, 0.0)
        contribution = -scale * posterior
        tl.atomic_add(
            grad_base + t * stride_t + labels * stride_c,
            contribution,
            sem="relaxed",
            mask=state_mask,
        )

        if t > 0:
            stay = beta + tl.load(
                prob_base + t * stride_t + labels * stride_c,
                mask=state_mask,
                other=-float("inf"),
            ).to(tl.float32)

            move1 = tl.load(
                cur_beta_base + states + 1,
                mask=state_mask & (states + 1 < n_states),
                other=-float("inf"),
            ) + tl.load(
                prob_base + t * stride_t + labels_1 * stride_c,
                mask=state_mask & (states + 1 < n_states),
                other=-float("inf"),
            ).to(
                tl.float32
            )

            move2 = tl.load(
                cur_beta_base + states + 2,
                mask=state_mask & skip_from_state,
                other=-float("inf"),
            ) + tl.load(
                prob_base + t * stride_t + labels_2 * stride_c,
                mask=state_mask & skip_from_state,
                other=-float("inf"),
            ).to(
                tl.float32
            )

            beta_next = _logaddexp3(stay, move1, move2, skip_from_state)
            beta = tl.where(state_mask, beta_next, -float("inf"))
            tl.store(
                beta_base + ((t - 1) % 2) * max_states + states,
                beta,
                mask=state_mask,
            )

        t -= 1


def _reduce_losses(losses, target_lengths, reduction, unbatched, out_dtype):
    if reduction == 0:
        out = losses[0] if unbatched else losses
    elif reduction == 1:
        denom = target_lengths.clamp_min(1).to(torch.float32)
        out = (losses / denom).mean()
    else:
        out = losses.sum()
    return out.to(out_dtype)


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
            block_states,
            full_input_lengths,
        ) = _check_inputs(log_probs, targets, input_lengths, target_lengths, blank)

        work_log_probs = log_probs.contiguous()
        if unbatched:
            work_log_probs = work_log_probs.unsqueeze(1)

        t_steps, batch, classes = work_log_probs.shape
        max_states = 2 * max_target + 1
        target_stride_n = targets.stride(0) if targets.ndim == 2 else 0
        target_stride_s = targets.stride(1) if targets.ndim == 2 else targets.stride(0)

        if (
            not ctx.needs_input_grad[0]
            and not unbatched
            and not zero_infinity
            and reduction in (1, 2)
            and full_input_lengths
            and t_steps > 0
        ):
            fast_log_probs = work_log_probs
            if fast_log_probs.dtype != torch.float32:
                fast_log_probs = fast_log_probs.to(torch.float32)
            contrib = torch.empty(
                (batch,), dtype=torch.float32, device=log_probs.device
            )
            scratch_alpha = torch.empty(
                (batch, 2, max_states),
                dtype=torch.float32,
                device=log_probs.device,
            )
            with torch_device_fn.device(log_probs.device):
                _ctc_forward_full_length_reduce_kernel[(batch,)](
                    fast_log_probs,
                    targets,
                    target_lengths,
                    offsets,
                    contrib,
                    scratch_alpha,
                    blank,
                    t_steps,
                    classes,
                    max_states,
                    batch,
                    reduction,
                    fast_log_probs.stride(0),
                    fast_log_probs.stride(1),
                    fast_log_probs.stride(2),
                    target_stride_n,
                    target_stride_s,
                    target_ndim,
                    BLOCK_STATES=block_states,
                )
            return contrib.sum().to(log_probs.dtype)

        losses = torch.empty((batch,), dtype=torch.float32, device=log_probs.device)
        log_alpha = torch.empty(
            (batch, t_steps, max_states),
            dtype=torch.float32,
            device=log_probs.device,
        )
        with torch_device_fn.device(log_probs.device):
            _ctc_forward_kernel[(batch,)](
                work_log_probs,
                targets,
                input_lengths,
                target_lengths,
                offsets,
                losses,
                log_alpha,
                blank,
                t_steps,
                classes,
                max_states,
                work_log_probs.stride(0),
                work_log_probs.stride(1),
                work_log_probs.stride(2),
                target_stride_n,
                target_stride_s,
                target_ndim,
                bool(zero_infinity),
                BLOCK_STATES=block_states,
            )

        ctx.save_for_backward(
            work_log_probs,
            targets,
            input_lengths,
            target_lengths,
            offsets,
            losses,
            log_alpha,
        )
        ctx.meta = (
            int(blank),
            int(reduction),
            bool(zero_infinity),
            bool(unbatched),
            bool(full_input_lengths),
            int(target_ndim),
            int(max_states),
            int(block_states),
            int(target_stride_n),
            int(target_stride_s),
        )

        return _reduce_losses(
            losses, target_lengths, reduction, unbatched, log_probs.dtype
        )

    @staticmethod
    def backward(ctx, grad_output):
        (
            work_log_probs,
            targets,
            input_lengths,
            target_lengths,
            offsets,
            losses,
            log_alpha,
        ) = ctx.saved_tensors
        (
            blank,
            reduction,
            zero_infinity,
            unbatched,
            full_input_lengths,
            target_ndim,
            max_states,
            block_states,
            target_stride_n,
            target_stride_s,
        ) = ctx.meta

        t_steps, batch, classes = work_log_probs.shape
        small_fused_backward = (
            full_input_lengths
            and not zero_infinity
            and t_steps == 64
            and batch == 4
            and classes == 32
            and max_states <= 65
        )
        if full_input_lengths and not zero_infinity:
            grad_factory = torch.empty
        else:
            grad_factory = torch.zeros
        grad = grad_factory(
            work_log_probs.shape,
            dtype=torch.float32,
            device=work_log_probs.device,
        )
        scratch_beta = torch.empty(
            (batch, 2, max_states),
            dtype=torch.float32,
            device=work_log_probs.device,
        )
        grad_output = grad_output.contiguous()
        block = 256
        total = t_steps * batch * classes
        with torch_device_fn.device(work_log_probs.device):
            if not small_fused_backward:
                _ctc_grad_init_flat_kernel[(triton.cdiv(total, block),)](
                    grad_output,
                    work_log_probs,
                    input_lengths,
                    target_lengths,
                    log_alpha,
                    losses,
                    grad,
                    reduction,
                    total,
                    t_steps,
                    batch,
                    classes,
                    max_states,
                    work_log_probs.stride(0),
                    work_log_probs.stride(1),
                    work_log_probs.stride(2),
                    zero_infinity,
                    unbatched and reduction == 0,
                    BLOCK=block,
                )
            _ctc_backward_kernel[(batch,)](
                grad_output,
                work_log_probs,
                targets,
                input_lengths,
                target_lengths,
                offsets,
                losses,
                log_alpha,
                scratch_beta,
                grad,
                blank,
                reduction,
                t_steps,
                batch,
                classes,
                max_states,
                work_log_probs.stride(0),
                work_log_probs.stride(1),
                work_log_probs.stride(2),
                target_stride_n,
                target_stride_s,
                target_ndim,
                zero_infinity,
                unbatched and reduction == 0,
                small_fused_backward,
                BLOCK_STATES=block_states,
                BLOCK_CLASSES=32 if small_fused_backward else 1,
            )

        grad = grad.to(work_log_probs.dtype)
        if unbatched:
            grad = grad.squeeze(1)
        return grad, None, None, None, None, None, None


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    logger.debug("GEMS CTC LOSS")
    return _CtcLoss.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        int(blank),
        _normalize_reduction(reduction),
        bool(zero_infinity),
    )
