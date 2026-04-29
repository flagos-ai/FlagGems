import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2
_LENGTH_STATS_CACHE = {}
_LENGTH_STATS_CACHE_LIMIT = 256


@triton.jit
def _logaddexp(a, b):
    max_ab = tl.maximum(a, b)
    min_ab = tl.minimum(a, b)
    return tl.where(
        max_ab == -float("inf"),
        -float("inf"),
        max_ab + tl.log(1.0 + tl.exp(min_ab - max_ab)),
    )


@triton.jit
def _logaddexp3(a, b, c, use_c):
    c = tl.where(use_c, c, -float("inf"))
    max_abc = tl.maximum(tl.maximum(a, b), c)
    safe_max = tl.where(max_abc == -float("inf"), 0.0, max_abc)
    exp_sum = tl.exp(a - safe_max) + tl.exp(b - safe_max) + tl.exp(c - safe_max)
    return tl.where(
        max_abc == -float("inf"),
        -float("inf"),
        max_abc + tl.log(exp_sum),
    )


@libentry()
@triton.jit
def _ctc_loss_forward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX

    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)

    if TARGET_1D:
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += tl.load(
                target_lengths + prev_batch,
                mask=prev_batch < batch,
                other=0,
            )
        target_origin = target_base
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    t0_active = input_len > 0
    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    init_logp = tl.load(
        log_probs + batch * C + labels,
        mask=init_state & stored_state & t0_active,
        other=0.0,
    ).to(tl.float32)
    alpha = tl.where(init_state & valid_state & t0_active, init_logp, -float("inf"))
    tl.store(
        log_alpha + batch * T * STATE_COUNT_MAX + states,
        alpha,
        mask=stored_state,
    )

    for t in tl.range(1, T):
        prev_base = log_alpha + batch * T * STATE_COUNT_MAX + (t - 1) * STATE_COUNT_MAX
        prev0 = tl.load(prev_base + states, mask=stored_state, other=-float("inf")).to(
            tl.float32
        )
        prev1 = tl.load(
            prev_base + tl.where(states > 0, states - 1, 0),
            mask=(states > 0) & stored_state,
            other=-float("inf"),
        ).to(tl.float32)
        prev2 = tl.load(
            prev_base + tl.where(states > 1, states - 2, 0),
            mask=(states > 1) & stored_state,
            other=-float("inf"),
        ).to(tl.float32)

        prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
        prev_target_value = tl.load(
            targets + target_origin + prev_target_index,
            mask=target_mask & (target_index > 0),
            other=BLANK,
        )
        skip_allowed = (
            (~is_blank_state) & (target_index > 0) & (target_value != prev_target_value)
        )

        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)

        logp = tl.load(
            log_probs + t * N * C + batch * C + labels,
            mask=valid_state & (t < input_len),
            other=0.0,
        ).to(tl.float32)
        alpha = tl.where(valid_state & (t < input_len), acc + logp, -float("inf"))
        tl.store(
            log_alpha + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states,
            alpha,
            mask=stored_state,
        )

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        tl.debug_barrier()
        final_base = (
            log_alpha + batch * T * STATE_COUNT_MAX + (input_len - 1) * STATE_COUNT_MAX
        )
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = tl.load(
            final_base + tl.where(target_len > 0, state_count - 2, 0),
            mask=target_len > 0,
            other=-float("inf"),
        ).to(tl.float32)
        log_likelihood = _logaddexp(last, prev_last)
        loss = -log_likelihood

    tl.store(neg_log_likelihood + batch, loss)


@libentry()
@triton.jit
def _ctc_loss_forward_full_length_reduce_kernel(
    log_probs,
    targets,
    target_lengths,
    contrib,
    scratch_alpha,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)

    target_len = tl.load(target_lengths + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX

    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)

    if TARGET_1D:
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += tl.load(
                target_lengths + prev_batch,
                mask=prev_batch < batch,
                other=0,
            )
        target_origin = target_base
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index

    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    init_logp = tl.load(
        log_probs + batch * C + labels,
        mask=init_state & stored_state,
        other=0.0,
    ).to(tl.float32)
    alpha = tl.where(init_state & valid_state, init_logp, -float("inf"))
    scratch_batch = scratch_alpha + batch * 2 * STATE_COUNT_MAX
    tl.store(scratch_batch + states, alpha, mask=stored_state)

    for t in tl.range(1, T):
        prev_base = scratch_batch + ((t - 1) % 2) * STATE_COUNT_MAX
        cur_base = scratch_batch + (t % 2) * STATE_COUNT_MAX
        prev0 = tl.load(prev_base + states, mask=stored_state, other=-float("inf")).to(
            tl.float32
        )
        prev1 = tl.load(
            prev_base + tl.where(states > 0, states - 1, 0),
            mask=(states > 0) & stored_state,
            other=-float("inf"),
        ).to(tl.float32)
        prev2 = tl.load(
            prev_base + tl.where(states > 1, states - 2, 0),
            mask=(states > 1) & stored_state,
            other=-float("inf"),
        ).to(tl.float32)

        prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
        prev_target_value = tl.load(
            targets + target_origin + prev_target_index,
            mask=target_mask & (target_index > 0),
            other=BLANK,
        )
        skip_allowed = (
            (~is_blank_state) & (target_index > 0) & (target_value != prev_target_value)
        )

        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        logp = tl.load(
            log_probs + t * N * C + batch * C + labels,
            mask=valid_state,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.where(valid_state, acc + logp, -float("inf"))
        tl.store(cur_base + states, alpha, mask=stored_state)

    if T <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        final_base = scratch_batch + ((T - 1) % 2) * STATE_COUNT_MAX
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = tl.load(
            final_base + tl.where(target_len > 0, state_count - 2, 0),
            mask=target_len > 0,
            other=-float("inf"),
        ).to(tl.float32)
        loss = -_logaddexp(last, prev_last)

    if REDUCTION == 1:
        loss = loss / tl.maximum(target_len, 1).to(tl.float32) / N
    tl.store(contrib + batch, loss)


@libentry()
@triton.jit
def _ctc_loss_init_grad_kernel(
    log_probs,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    grad_output,
    grad_input,
    total: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    batch = (offsets // C) % N
    t = offsets // (N * C)

    input_len = tl.load(input_lengths + batch, mask=mask, other=0)
    target_len = tl.load(target_lengths + batch, mask=mask, other=1)
    nll = tl.load(neg_log_likelihood + batch, mask=mask, other=0.0).to(tl.float32)

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch, mask=mask, other=0.0).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            denom = tl.maximum(target_len, 1).to(tl.float32) * N
            scale = scale / denom

    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    logp = tl.load(log_probs + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    grad = tl.where((t < input_len) & mask, tl.exp(logp) * scale, 0.0)
    nan_grad = float("nan")
    grad = tl.where(
        (t < input_len) & mask & (scale != 0.0) & (logp == -float("inf")),
        nan_grad,
        grad,
    )
    if not ZERO_INFINITY:
        grad = tl.where((t < input_len) & mask & (nll == float("inf")), nan_grad, grad)
    tl.store(grad_input + offsets, grad, mask=mask)


@libentry()
@triton.jit
def _ctc_loss_backward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    grad_output,
    grad_input,
    scratch_beta,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
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
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += tl.load(
                target_lengths + prev_batch,
                mask=prev_batch < batch,
                other=0,
            )
        target_origin = target_base
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
    log_likelihood = tl.where(scale != 0.0, -nll, 0.0)

    for step in tl.range(0, T):
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
        tl.atomic_add(
            grad_input + safe_t * N * C + batch * C + labels,
            -scale * posterior,
            sem="relaxed",
            mask=active & valid_state & stored_state,
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


def _reduction_enum(reduction):
    if isinstance(reduction, str):
        if reduction == "none":
            return _REDUCTION_NONE
        if reduction == "mean":
            return _REDUCTION_MEAN
        if reduction == "sum":
            return _REDUCTION_SUM
        raise ValueError(
            "ctc_loss expected reduction to be one of 'none', 'mean', or 'sum', "
            f"but got {reduction!r}"
        )
    return int(reduction)


_INTEGRAL_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _is_integral_dtype(dtype):
    return dtype in _INTEGRAL_DTYPES


def _lengths_to_tensor(lengths, device, name):
    if torch.is_tensor(lengths):
        if not _is_integral_dtype(lengths.dtype):
            raise RuntimeError(f"{name} must be integral")
        out = lengths.to(device=device)
    else:
        out = torch.tensor(lengths, device=device)
        if not _is_integral_dtype(out.dtype):
            raise RuntimeError(f"{name} must be integral")
    if out.dtype != torch.long:
        out = out.to(dtype=torch.long)
    return out.reshape(1) if out.ndim == 0 else out.reshape(-1).contiguous()


def _length_stats(lengths):
    key = None
    if torch.is_tensor(lengths):
        key = (
            lengths.device.type,
            lengths.device.index,
            lengths.data_ptr(),
            lengths.numel(),
            lengths._version,
        )
        cached = _LENGTH_STATS_CACHE.get(key)
        if cached is not None:
            return cached[1]

    stats_tensor = torch.stack((lengths.min(), lengths.max(), lengths.sum())).cpu()
    stats = tuple(int(value) for value in stats_tensor.tolist())
    if key is not None:
        if len(_LENGTH_STATS_CACHE) >= _LENGTH_STATS_CACHE_LIMIT:
            _LENGTH_STATS_CACHE.clear()
        _LENGTH_STATS_CACHE[key] = (lengths, stats)
    return stats


def _compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _ctc_loss_forward_no_grad_torch_dp(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank,
    reduction,
    zero_infinity,
    target_1d,
    unbatched,
    original_dtype,
):
    device = log_probs.device
    input_lengths_list = input_lengths.detach().cpu().tolist()
    target_lengths_list = target_lengths.detach().cpu().tolist()

    losses = []
    target_offset = 0
    for batch, (input_len, target_len) in enumerate(
        zip(input_lengths_list, target_lengths_list)
    ):
        if target_1d:
            target = targets[target_offset : target_offset + target_len]
            target_offset += target_len
        else:
            target = targets[batch, :target_len]

        if input_len == 0:
            loss = torch.full((), float("inf"), dtype=torch.float32, device=device)
            if target_len == 0:
                loss = torch.zeros((), dtype=torch.float32, device=device)
            losses.append(loss)
            continue

        state_count = target_len * 2 + 1
        labels = torch.full((state_count,), blank, dtype=torch.long, device=device)
        if target_len > 0:
            labels[1::2] = target

        alpha = torch.full(
            (state_count,), -float("inf"), dtype=torch.float32, device=device
        )
        alpha[0] = log_probs[0, batch, blank]
        if target_len > 0:
            alpha[1] = log_probs[0, batch, target[0]]

        skip_allowed = torch.zeros((state_count,), dtype=torch.bool, device=device)
        if target_len > 1:
            skip_allowed[3::2] = target[1:] != target[:-1]

        for step in range(1, input_len):
            prev1 = torch.full_like(alpha, -float("inf"))
            prev1[1:] = alpha[:-1]
            prev2 = torch.full_like(alpha, -float("inf"))
            if state_count > 2:
                prev2[2:] = alpha[:-2]
            prev2 = torch.where(
                skip_allowed, prev2, torch.full_like(alpha, -float("inf"))
            )
            alpha = torch.logaddexp(torch.logaddexp(alpha, prev1), prev2)
            alpha = alpha + log_probs[step, batch, labels]

        if target_len > 0:
            loss = -torch.logaddexp(alpha[-1], alpha[-2])
        else:
            loss = -alpha[-1]
        losses.append(loss)

    neg_log_likelihood = torch.stack(losses)
    if zero_infinity:
        neg_log_likelihood = torch.where(
            torch.isinf(neg_log_likelihood),
            torch.zeros((), dtype=neg_log_likelihood.dtype, device=device),
            neg_log_likelihood,
        )

    if reduction == _REDUCTION_NONE:
        output = neg_log_likelihood.squeeze(0) if unbatched else neg_log_likelihood
    elif reduction == _REDUCTION_SUM:
        output = neg_log_likelihood.sum()
    else:
        output = (neg_log_likelihood / target_lengths.clamp_min(1)).mean()

    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    return output


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
        elif work_targets.ndim == 2:
            target_1d = False
            if max_target > work_targets.shape[1]:
                raise RuntimeError(
                    "ctc_loss target_lengths cannot exceed padded target width"
                )
            target_stride = work_targets.shape[1]
        else:
            raise RuntimeError(
                "ctc_loss expects targets to be a 1D concatenated tensor or a "
                f"2D padded tensor, but got {work_targets.ndim}D"
            )

        needs_log_probs_grad = ctx.needs_input_grad[0]
        block_s = triton.next_power_of_2(state_count_max)

        if (
            not needs_log_probs_grad
            and not unbatched
            and not zero_infinity
            and reduction in (_REDUCTION_MEAN, _REDUCTION_SUM)
            and min_input_length == work_log_probs.shape[0]
            and work_log_probs.shape[0] > 0
        ):
            contrib = torch.empty(
                (batch_size,), dtype=torch.float32, device=log_probs.device
            )
            scratch_alpha = torch.empty(
                (batch_size, 2, state_count_max),
                dtype=torch.float32,
                device=log_probs.device,
            )
            with torch_device_fn.device(log_probs.device):
                _ctc_loss_forward_full_length_reduce_kernel[(batch_size,)](
                    work_log_probs,
                    work_targets,
                    work_target_lengths,
                    contrib,
                    scratch_alpha,
                    work_log_probs.shape[0],
                    batch_size,
                    work_log_probs.shape[2],
                    target_stride,
                    state_count_max,
                    blank,
                    target_1d,
                    reduction,
                    block_s,
                )
            output = contrib.sum()
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
            return output

        if not needs_log_probs_grad:
            return _ctc_loss_forward_no_grad_torch_dp(
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                blank,
                reduction,
                zero_infinity,
                target_1d,
                unbatched,
                original_dtype,
            )

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
            neg_log_likelihood = torch.where(
                torch.isinf(neg_log_likelihood),
                torch.zeros(
                    (), dtype=neg_log_likelihood.dtype, device=log_probs.device
                ),
                neg_log_likelihood,
            )

        if reduction == _REDUCTION_NONE:
            output = neg_log_likelihood
            if unbatched:
                output = output.squeeze(0)
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
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
                (ctx.batch_size, 2, ctx.state_count_max),
                dtype=torch.float32,
                device=work_log_probs.device,
            )
            block_s = triton.next_power_of_2(ctx.state_count_max)
            _ctc_loss_backward_kernel[(ctx.batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
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
    logger.debug("GEMS CTC LOSS")
    return CtcLossFunction.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )
