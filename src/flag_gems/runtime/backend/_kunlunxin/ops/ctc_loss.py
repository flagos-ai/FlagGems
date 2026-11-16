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
"""Kunlunxin (XPU3) backend override for ``torch.nn.functional.ctc_loss``.

Why a backend override is required
----------------------------------
``aten::ctc_loss`` (the autograd-facing operator, both ``.Tensor`` and
``.IntList``) is implemented generically in ``src/flag_gems/ops/ctc_loss.py``.
All three of its DP kernels keep the previous lattice row in **global memory**
and let a lane read the values its neighbours wrote on the previous loop
iteration:

* ``_ctc_loss_forward_kernel``  (grad path, full ``log_alpha``),
* ``_ctc_loss_forward_no_grad_kernel`` / ``_ctc_loss_forward_full_length_reduce_kernel``
  (2-row ring scratch),
* ``_ctc_loss_backward_kernel`` (beta recurrence, shift-by-1/2).

On XPU3 an intra-kernel global-memory store -> cross-lane load is **not
coherent** (``tl.debug_barrier`` does not fence it, cache modifiers and
``volatile`` do not fix it, and an atomic fence costs milliseconds per
iteration), so every one of those kernels returns silently wrong values.
The forward kernel additionally unrolls ``T`` as a ``constexpr`` and hits
``OutOfResources (TritonXPUUnrollControl)`` for ``T >= ~256``.  See
``harness/KernelOpt_Knowledge/wiki/patterns/xpu3-intra-kernel-gm-crosslane-incoherent.md``.

Correct design used here
------------------------
A kernel-launch boundary *is* a full memory fence on this backend, so the
alpha/beta recurrences are executed as **one launch per timestep** over a
``(batch,)`` grid — exactly the design already validated for
``aten::_ctc_loss`` in ``_kunlunxin/ops/_ctc_loss.py`` (whose
``_ctc_init_kernel`` / ``_ctc_step_kernel`` / ``_ctc_final_kernel`` are reused
verbatim here, so the two operators share one DP implementation):

* forward: ``_ctc_init_kernel`` -> ``_ctc_step_kernel`` x (T-1) -> ``_ctc_final_kernel``,
* backward: ``_ctc_grad_init_kernel`` (elementwise ``exp(logp) * scale``) ->
  ``_ctc_beta_init_kernel`` -> ``_ctc_beta_step_kernel`` x max_input_len.

The shift loads clamp the pointer and apply the ``-inf`` bound *after* the
load: XPU3 masked loads return the in-bounds value on masked lanes instead of
``other`` (``wiki/patterns/xpu3-masked-load-other-inbounds-leak.md``).

The reduction (``none`` / ``sum`` / ``mean``) and ``zero_infinity`` are fused
into one extra tiny launch (``_ctc_loss_finalize_kernel``) instead of calling
torch reductions, so no part of the operator's numerical semantics is
delegated to aten.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._ctc_loss import _length_stats, _prepare
from flag_gems.ops.ctc_loss import (
    _REDUCTION_MEAN,
    _REDUCTION_NONE,
    _REDUCTION_SUM,
    _logaddexp3,
    _reduction_enum,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._ctc_loss import _ctc_final_kernel, _ctc_init_kernel, _ctc_step_kernel

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _ctc_loss_finalize_kernel(
    nll,  # (BLOCK_N,) fp32, zero padded past `N`
    target_lengths,  # (N,) int64
    out,  # (N,) or 0-d, dtype == original dtype
    N,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    # Unmasked load: `nll` is allocated zero-padded to BLOCK_N (>= 64 lanes) so
    # the reduction is neither a masked-memory path nor a sub-warp reduction
    # (both are miscompiled / silently wrong on XPU3).
    vals = tl.load(nll + offs)
    if ZERO_INFINITY:
        vals = tl.where(tl.abs(vals) == float("inf"), 0.0, vals)
    if REDUCTION == 0:  # _REDUCTION_NONE
        # `out` owns one extra dummy slot at index `N`: the padding lanes are
        # funnelled there (a masked store is not honoured on XPU3) while every
        # real lane writes its own element exactly once.
        addr = tl.minimum(offs, N)
        tl.store(out + addr, vals)
    else:
        contrib = vals
        if REDUCTION == 1:  # _REDUCTION_MEAN
            tlen = tl.load(target_lengths + tl.minimum(offs, N - 1)).to(tl.float32)
            contrib = vals / tl.maximum(tlen, 1.0)
            total = tl.sum(contrib) / N
        else:
            total = tl.sum(contrib)
        tl.store(out, total)


@libentry()
@triton.jit
def _ctc_pad_targets_kernel(
    targets,
    target_lengths,
    target_offsets,
    out,
    total,
    N,
    MAX_TARGET: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """1-D concatenated targets -> 2-D padded layout (0 filled tail)."""
    batch = tl.program_id(0)
    j = tl.arange(0, BLOCK_M)
    tlen = tl.load(target_lengths + batch)
    off = tl.load(target_offsets + batch)
    val = tl.load(targets + tl.minimum(off + j, total - 1))
    val = tl.where(j < tlen, val, 0)
    # Lanes j >= MAX_TARGET go to a dedicated dummy region `[N*MAX_TARGET, ...)`
    # (masked stores are not honoured on XPU3 and a shared clamped address would
    # race with the real lane MAX_TARGET-1).
    dst = tl.where(j < MAX_TARGET, batch * MAX_TARGET + j, N * MAX_TARGET + j)
    tl.store(out + dst, val)


@libentry()
@triton.jit
def _ctc_grad_init_kernel(
    log_probs,
    input_lengths,
    target_lengths,
    nll,
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
    # Clamped-offset / value-mask idiom: XPU3 does not honour masked stores, so
    # the tail lanes re-write the (identical) contribution of the last real
    # element instead of being masked away.
    safe = tl.minimum(offsets, total - 1)
    batch = (safe // C) % N
    t = safe // (N * C)

    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    nll_b = tl.load(nll + batch)

    if REDUCTION == 0:  # _REDUCTION_NONE
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:  # _REDUCTION_MEAN
            scale = scale / (tl.maximum(target_len, 1).to(tl.float32) * N)

    if ZERO_INFINITY:
        scale = tl.where(nll_b == float("inf"), 0.0, scale)

    logp = tl.load(log_probs + safe).to(tl.float32)
    active = t < input_len
    grad = tl.where(active, tl.exp(logp) * scale, 0.0)
    nan_grad = float("nan")
    grad = tl.where(active & (scale != 0.0) & (logp == -float("inf")), nan_grad, grad)
    if not ZERO_INFINITY:
        grad = tl.where(active & (nll_b == float("inf")), nan_grad, grad)
    tl.store(grad_input + safe, grad)


@libentry()
@triton.jit
def _ctc_beta_init_kernel(
    input_lengths,
    target_lengths,
    scratch_beta,
    N: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)
    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    beta_init = tl.where(
        ((states == state_count - 1) | ((states == state_count - 2) & (target_len > 0)))
        & valid_state
        & (input_len > 0),
        0.0,
        -float("inf"),
    )
    # The ring rows are BLOCK_S wide, so this store is fully in bounds.
    tl.store(scratch_beta + batch * 2 * BLOCK_S + states, beta_init)


@libentry()
@triton.jit
def _ctc_beta_step_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    nll,
    log_alpha,
    grad_output,
    grad_input,
    scratch_beta,
    step,
    cur_off,
    next_off,
    num_classes,
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
    nll_b = tl.load(nll + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    in_rows = states < STATE_COUNT_MAX

    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch)
    else:
        target_origin = batch * MAX_TARGET

    target_value_raw = tl.load(targets + target_origin + target_safe_index)
    target_value = tl.where(target_mask, target_value_raw, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)

    state1 = states + 1
    is_blank_state1 = (state1 % 2) == 0
    target_index1 = (state1 - 1) // 2
    target_mask1 = (target_index1 >= 0) & (target_index1 < target_len)
    target_safe_index1 = tl.where(target_mask1, target_index1, 0)
    target_value1_raw = tl.load(targets + target_origin + target_safe_index1)
    labels1 = tl.where(
        is_blank_state1, BLANK, tl.where(target_mask1, target_value1_raw, BLANK)
    )

    state2 = states + 2
    target_index2 = (state2 - 1) // 2
    target_mask2 = (target_index2 >= 0) & (target_index2 < target_len)
    target_safe_index2 = tl.where(target_mask2, target_index2, 0)
    target_value2_raw = tl.load(targets + target_origin + target_safe_index2)
    target_value2 = tl.where(target_mask2, target_value2_raw, BLANK)
    labels2 = target_value2

    if REDUCTION == 0:  # _REDUCTION_NONE
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:  # _REDUCTION_MEAN
            scale = scale / (tl.maximum(target_len, 1).to(tl.float32) * N)

    if ZERO_INFINITY:
        scale = tl.where(nll_b == float("inf"), 0.0, scale)

    t = input_len - 1 - step
    active = t >= 0
    safe_t = tl.where(active, t, 0)

    ring_cur = scratch_beta + batch * 2 * BLOCK_S + cur_off
    ring_next = scratch_beta + batch * 2 * BLOCK_S + next_off

    beta_raw = tl.load(ring_cur + states)
    beta = tl.where(in_rows, beta_raw, -float("inf"))

    # Clamp the shift-by-0 read; the row has STATE_COUNT_MAX slots only.
    alpha_raw = tl.load(
        log_alpha
        + batch * T * STATE_COUNT_MAX
        + safe_t * STATE_COUNT_MAX
        + tl.minimum(states, STATE_COUNT_MAX - 1)
    ).to(tl.float32)
    alpha_t = tl.where(active & in_rows, alpha_raw, -float("inf"))

    log_likelihood = tl.where(scale != 0.0, -nll_b, 0.0)
    log_post = alpha_t + beta - log_likelihood
    posterior = tl.where(
        active & valid_state & (scale != 0.0), tl.exp(log_post), 0.0
    )
    # No atomics here: ``tl.atomic_add`` mis-accumulates on this XPU3 build
    # (measured: 64 lanes adding arange(0,64) into one address produced 2420
    # instead of 2016, and 64 *distinct* addresses ended up with max 2.0, i.e.
    # duplicate / wrong-address writes).  Instead every (t, batch) program writes
    # its whole gradient row exactly once, folding the states into classes with a
    # per-class masked reduction.
    #
    # ``tt.reduce`` cannot be legalized inside ``scf.if`` on this backend, so the
    # loop is unconditional: batches whose ``t`` does not exist write into the
    # extra dummy row ``T`` of the gradient buffer instead of being skipped.
    dst_t = tl.where(active, safe_t, T)
    for c in tl.range(0, num_classes):
        p_sum = tl.sum(tl.where(labels == c, posterior, 0.0))
        logp_c = tl.load(log_probs + safe_t * N * C + batch * C + c).to(tl.float32)
        base = tl.exp(logp_c) * scale
        base = tl.where((scale != 0.0) & (logp_c == -float("inf")), float("nan"), base)
        if not ZERO_INFINITY:
            base = tl.where(nll_b == float("inf"), float("nan"), base)
        tl.store(grad_input + dst_t * N * C + batch * C + c, base - scale * p_sum)

    logp_raw = tl.load(log_probs + safe_t * N * C + batch * C + labels).to(tl.float32)
    stay = beta + tl.where(active & valid_state, logp_raw, -float("inf"))

    beta1_raw = tl.load(ring_cur + tl.minimum(states + 1, BLOCK_S - 1))
    cond1 = (states + 1 < state_count) & in_rows
    beta1 = tl.where(cond1, beta1_raw, -float("inf"))
    logp1_raw = tl.load(log_probs + safe_t * N * C + batch * C + labels1).to(
        tl.float32
    )
    next1 = beta1 + tl.where(active & cond1, logp1_raw, -float("inf"))

    skip_allowed = (
        (~is_blank_state) & (states + 2 < state_count) & (target_value != target_value2)
    )
    beta2_raw = tl.load(ring_cur + tl.minimum(states + 2, BLOCK_S - 1))
    beta2 = tl.where((states + 2 < state_count) & in_rows, beta2_raw, -float("inf"))
    logp2_raw = tl.load(log_probs + safe_t * N * C + batch * C + labels2).to(
        tl.float32
    )
    next2 = beta2 + tl.where(active & skip_allowed, logp2_raw, -float("inf"))

    beta_next = _logaddexp3(stay, next1, next2, skip_allowed)
    tl.store(ring_next + states, tl.where(active, beta_next, -float("inf")))


def _forward_dp(setup, T, nll, log_alpha, blank):
    """Alpha lattice, one launch per timestep (see module docstring)."""
    wp = setup.work_log_probs
    grid = (setup.batch_size,)
    with torch_device_fn.device(wp.device):
        _ctc_init_kernel[grid](
            wp,
            setup.work_targets,
            setup.work_input_lengths,
            setup.work_target_lengths,
            log_alpha,
            T,
            setup.batch_size,
            wp.shape[2],
            setup.target_stride,
            setup.state_count_max,
            blank,
            setup.target_1d,
            setup.block_s,
        )
        for t in range(1, T):
            _ctc_step_kernel[grid](
                wp,
                setup.work_targets,
                setup.work_input_lengths,
                setup.work_target_lengths,
                nll,
                log_alpha,
                t,
                T,
                setup.batch_size,
                wp.shape[2],
                setup.target_stride,
                setup.state_count_max,
                blank,
                setup.target_1d,
                setup.block_s,
            )
        _ctc_final_kernel[grid](
            setup.work_input_lengths,
            setup.work_target_lengths,
            nll,
            log_alpha,
            T,
            setup.state_count_max,
        )


def _pad_concatenated_targets(setup, device):
    """Rewrite 1-D concatenated targets into the padded 2-D layout.

    Needed because the single-batch + concatenated combination makes the DP
    kernels' target-origin loop
    (``for prev_batch in tl.range(0, N): target_base += tl.load(..., mask=False)``)
    run with an all-False mask on its only iteration, which on XPU3 produced a
    garbage target base and an out-of-bounds device read
    (``KL_XID_KERNEL_EXCEPTION`` + ``KL_XID64_AXI_ADDR_ERROR``; see
    ``harness/run_logs/ctc_loss_batch13/dmesg_dev7_xid.log``).  After this
    rewrite the DP runs with ``TARGET_1D=False``, which is the configuration
    validated on-device by the ``fwd_tiny`` / ``fwd_bwd_tiny`` probes.
    """
    max_target = setup.target_stride
    n = setup.batch_size
    block_m = max(64, triton.next_power_of_2(max(max_target, 1)))
    flat = torch.empty(n * max_target + block_m, dtype=torch.long, device=device)
    with torch_device_fn.device(device):
        _ctc_pad_targets_kernel[(n,)](
            setup.work_targets,
            setup.work_target_lengths,
            setup.work_target_offsets,
            flat,
            setup.work_targets.numel(),
            n,
            max_target,
            block_m,
        )
    setup.work_targets = flat[: n * max_target].view(n, max_target)
    setup.target_1d = False
    return setup


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
        red = _reduction_enum(reduction)
        if red not in (_REDUCTION_NONE, _REDUCTION_MEAN, _REDUCTION_SUM):
            raise ValueError(f"ctc_loss got invalid reduction enum {red}")

        setup = _prepare(log_probs, targets, input_lengths, target_lengths, blank)
        # XPU3 silently widens ``tl.arange(0, n)`` to ``round_up(n, 64)`` physical
        # lanes (KERNEL_OPT_EXPERIENCE #44), so a tile narrower than one warp
        # makes the (unmasked / non-honoured masked) stores of the DP kernels
        # run past the row and, at the end of the buffer, past the allocation
        # (measured: ``KL_XID_KERNEL_EXCEPTION`` + ``KL_XID64_AXI_ADDR_ERROR``).
        # Make the lattice row width == the physical tile width (>= 64) so every
        # store is exactly in bounds; pad lanes carry -inf.
        block_s = max(64, triton.next_power_of_2(setup.state_count_max))
        setup.block_s = block_s
        setup.state_count_max = block_s
        if setup.target_1d and setup.batch_size == 1:
            # Only the single-batch + concatenated combination faults; every
            # other concatenated case was verified to work as-is.
            setup = _pad_concatenated_targets(setup, setup.work_log_probs.device)

        wp = setup.work_log_probs
        T = wp.shape[0]
        batch_size = setup.batch_size
        device = wp.device

        # `nll` is zero padded to at least one warp so that the finalize kernel
        # can reduce it without masks (sub-warp / masked reductions are
        # miscompiled on XPU3).
        block_n = max(64, triton.next_power_of_2(max(batch_size, 1)))
        nll = torch.zeros((block_n,), dtype=torch.float32, device=device)
        # Allocation only; every row in [0, T) is written by the DP launches.
        log_alpha = torch.empty(
            (batch_size, T, setup.state_count_max), dtype=torch.float32, device=device
        )

        if T > 0:
            _forward_dp(setup, T, nll, log_alpha, blank)
        else:
            # Degenerate empty-time input: nothing can be launched.  Mirrors the
            # generic fallback for this shape (all input lengths are 0 here).
            for batch in range(batch_size):
                target_len = int(setup.work_target_lengths[batch].item())
                nll[batch] = 0.0 if target_len == 0 else float("inf")

        if red == _REDUCTION_NONE:
            # Per-batch losses (+1 dummy slot for the finalize padding lanes).
            out_len = batch_size + 1
        else:
            # "sum" / "mean" collapse the batch dimension to a 0-d scalar.
            out_len = 1
        out_buf = torch.empty(out_len, dtype=setup.original_dtype, device=device)
        with torch_device_fn.device(device):
            _ctc_loss_finalize_kernel[(1,)](
                nll,
                setup.work_target_lengths,
                out_buf,
                batch_size,
                red,
                zero_infinity,
                block_n,
            )
        if red == _REDUCTION_NONE:
            output = out_buf[:batch_size]
            if setup.unbatched:
                output = output.squeeze(0)
        else:
            output = out_buf[0]

        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(
                wp,
                setup.work_targets,
                setup.work_input_lengths,
                setup.work_target_lengths,
                setup.work_target_offsets,
                nll,
                log_alpha,
            )
            ctx.blank = blank
            ctx.reduction = red
            ctx.zero_infinity = zero_infinity
            ctx.unbatched = setup.unbatched
            ctx.batch_size = batch_size
            ctx.original_dtype = setup.original_dtype
            ctx.max_target = setup.target_stride
            ctx.state_count_max = setup.state_count_max
            ctx.target_1d = setup.target_1d
            ctx.block_s = setup.block_s
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            nll,
            log_alpha,
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        T = work_log_probs.shape[0]
        batch_size = ctx.batch_size
        num_classes = work_log_probs.shape[2]
        total = work_log_probs.numel()
        block = 256

        # One extra dummy row: the beta-step kernel writes rows that do not exist
        # for a given batch there instead of using a data-dependent store guard
        # (`tt.reduce` cannot be legalized inside `scf.if` on this backend).
        grad_work = torch.empty(
            (T + 1, batch_size, num_classes),
            dtype=torch.float32,
            device=work_log_probs.device,
        )
        grad_log_probs = grad_work[:T]
        scratch_beta = torch.empty(
            (batch_size, 2, ctx.block_s), dtype=torch.float32, device=work_log_probs.device
        )

        max_steps = _length_stats(work_input_lengths)[1]
        with torch_device_fn.device(work_log_probs.device):
            _ctc_grad_init_kernel[(triton.cdiv(total, block),)](
                work_log_probs,
                work_input_lengths,
                work_target_lengths,
                nll,
                grad_output,
                grad_work,
                total,
                T,
                batch_size,
                num_classes,
                ctx.reduction,
                ctx.zero_infinity,
                block,
            )
            if max_steps > 0:
                _ctc_beta_init_kernel[(batch_size,)](
                    work_input_lengths,
                    work_target_lengths,
                    scratch_beta,
                    batch_size,
                    ctx.state_count_max,
                    ctx.block_s,
                )
                for step in range(max_steps):
                    _ctc_beta_step_kernel[(batch_size,)](
                        work_log_probs,
                        work_targets,
                        work_input_lengths,
                        work_target_lengths,
                        work_target_offsets,
                        nll,
                        log_alpha,
                        grad_output,
                        grad_work,
                        scratch_beta,
                        step,
                        (step % 2) * ctx.block_s,
                        ((step + 1) % 2) * ctx.block_s,
                        num_classes,
                        T,
                        batch_size,
                        num_classes,
                        ctx.max_target,
                        ctx.state_count_max,
                        ctx.blank,
                        ctx.target_1d,
                        ctx.reduction,
                        ctx.zero_infinity,
                        ctx.block_s,
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
