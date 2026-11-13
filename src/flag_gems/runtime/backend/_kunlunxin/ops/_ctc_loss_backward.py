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

from flag_gems.ops._ctc_loss_backward import _prepare_backward
from flag_gems.ops.ctc_loss import _REDUCTION_NONE, _logaddexp3
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Kunlunxin XPU lowering notes (see harness KernelOpt_Knowledge):
#   * `tl.load(..., mask=..., other=...)` does NOT reliably honour the `other=`
#     fill on this backend: masked-out lanes read neighbouring memory instead of
#     `other`. A `-inf` "other" that leaks garbage makes `exp(alpha + beta -
#     log_likelihood)` explode, which is exactly the ~1e6 magnitude corruption
#     seen against the CPU reference.
#   * `tl.atomic_add` drops increments, so the posterior scatter-add is done
#     with an atomic-free gather-reduce instead.
#   * `tl.debug_barrier` is NOT a reliable cross-thread barrier on this backend
#     once the grid has more than one program: a beta recursion that reads the
#     neighbouring state lane (beta[s+1] / beta[s+2]) through a shared/scratch
#     buffer races and silently corrupts every batch except the first. The beta
#     pass is therefore rewritten as a *scalar* recursion over the state axis,
#     which has no cross-lane dependency and needs no barrier at all.
#
# Every gather below therefore uses a clamped address + an unmasked load + a
# `tl.where` value select, and never relies on `other=` or a masked load.


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
    # grad[t, n, c] = exp(log_probs[t, n, c]) * scale[n] for t < input_len[n],
    # else 0. One flat program over total elements; addresses are clamped and
    # the value is gated by `tl.where` so masked-out lanes never leak garbage.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total
    offs_safe = tl.minimum(offsets, total - 1)
    batch = (offs_safe // C) % N
    t = offs_safe // (N * C)

    input_len = tl.load(input_lengths + batch).to(tl.int32)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    nll = tl.load(neg_log_likelihood + batch).to(tl.float32)

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            denom = tl.maximum(target_len, 1).to(tl.float32) * N
            scale = scale / denom

    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    logp = tl.load(log_probs + offs_safe).to(tl.float32)
    grad = tl.where((t < input_len) & valid, tl.exp(logp) * scale, 0.0)
    grad = tl.where(
        (t < input_len) & valid & (scale != 0.0) & (logp == -float("inf")),
        float("nan"),
        grad,
    )
    if not ZERO_INFINITY:
        grad = tl.where(
            (t < input_len) & valid & (nll == float("inf")), float("nan"), grad
        )
    tl.store(grad_input + offsets, grad, mask=valid)


@triton.jit
def _ctc_loss_backward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
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

    input_len = tl.load(input_lengths + batch).to(tl.int32)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    nll = tl.load(neg_log_likelihood + batch).to(tl.float32)
    state_count = target_len * 2 + 1

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            denom = tl.maximum(target_len, 1).to(tl.float32) * N
            scale = scale / denom

    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    log_likelihood = tl.where(scale != 0.0, -nll, 0.0)

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch).to(tl.int32)
    else:
        target_origin = batch * MAX_TARGET

    # ------------------------------------------------------------------
    # Phase 1: beta recursion, scalar over the state axis (no cross-lane
    # reads, no barrier). scratch_beta is (N, T, STATE_COUNT_MAX); slot
    # (batch, t, s) holds beta[t][s] for t in [0, input_len).
    # ------------------------------------------------------------------
    beta_base = scratch_beta + batch * T * STATE_COUNT_MAX

    # init: beta[input_len-1][s] = 0 for the final blank / final label state.
    init_t = tl.maximum(input_len - 1, 0)
    for s in tl.range(0, STATE_COUNT_MAX):
        last = (s == state_count - 1) | ((s == state_count - 2) & (target_len > 0))
        valid = (s < state_count) & (input_len > 0)
        v = tl.where(last & valid, 0.0, -float("inf"))
        tl.store(beta_base + init_t * STATE_COUNT_MAX + s, v)

    # recurse: beta[t][s] = logaddexp3 over beta[t+1][s], beta[t+1][s+1],
    # beta[t+1][s+2] using logp at time t+1. `step` runs 1..input_len-1.
    for step in tl.range(1, input_len):
        t = input_len - 1 - step
        t_src = t + 1
        for s in tl.range(0, STATE_COUNT_MAX):
            if s < state_count:
                # label of state s (scalar)
                is_blank = (s % 2) == 0
                ti = (s - 1) // 2
                tmask = (ti >= 0) & (ti < target_len)
                tsafe = tl.where(tmask, ti, 0)
                tv = tl.load(targets + target_origin + tsafe)
                tv = tl.where(tmask, tv, BLANK)
                label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)

                beta_s = tl.load(beta_base + t_src * STATE_COUNT_MAX + s).to(tl.float32)
                lp_s = tl.load(log_probs + t_src * N * C + batch * C + label_s).to(
                    tl.float32
                )
                stay = beta_s + lp_s

                # next1: state s+1
                s1 = s + 1
                s1_valid = s1 < state_count
                is_blank1 = (s1 % 2) == 0
                ti1 = (s1 - 1) // 2
                tmask1 = (ti1 >= 0) & (ti1 < target_len)
                tsafe1 = tl.where(tmask1, ti1, 0)
                tv1 = tl.load(targets + target_origin + tsafe1)
                tv1 = tl.where(tmask1, tv1, BLANK)
                label_s1 = tl.where(is_blank1, BLANK, tv1).to(tl.int32)
                beta_s1 = tl.load(beta_base + t_src * STATE_COUNT_MAX + s1).to(
                    tl.float32
                )
                lp_s1 = tl.load(log_probs + t_src * N * C + batch * C + label_s1).to(
                    tl.float32
                )
                next1 = tl.where(s1_valid, beta_s1 + lp_s1, -float("inf"))

                # next2: state s+2 (skip)
                s2 = s + 2
                is_blank2 = (s2 % 2) == 0
                ti2 = (s2 - 1) // 2
                tmask2 = (ti2 >= 0) & (ti2 < target_len)
                tsafe2 = tl.where(tmask2, ti2, 0)
                tv2 = tl.load(targets + target_origin + tsafe2)
                tv2 = tl.where(tmask2, tv2, BLANK)
                label_s2 = tl.where(is_blank2, BLANK, tv2).to(tl.int32)
                skip_allowed = (~is_blank) & (s2 < state_count) & (tv != tv2)
                beta_s2 = tl.load(beta_base + t_src * STATE_COUNT_MAX + s2).to(
                    tl.float32
                )
                lp_s2 = tl.load(log_probs + t_src * N * C + batch * C + label_s2).to(
                    tl.float32
                )
                next2 = tl.where(skip_allowed, beta_s2 + lp_s2, -float("inf"))

                bnext = _logaddexp3(stay, next1, next2, skip_allowed)
                tl.store(beta_base + t * STATE_COUNT_MAX + s, bnext)

    # ------------------------------------------------------------------
    # Phase 2: posterior + scatter, vector over states. Each lane owns one
    # state; beta[t][:] is gathered from scratch with clamped indices, and
    # the class reduction uses a static per-class 1-D loop (atomic-free).
    # ------------------------------------------------------------------
    states = tl.arange(0, BLOCK_S)
    valid_state = states < state_count
    states_beta = tl.minimum(states, STATE_COUNT_MAX - 1)

    # labels of every lane (clamped + value select, never a masked load)
    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)
    target_value = tl.load(targets + target_origin + target_safe_index)
    target_value = tl.where(target_mask, target_value, BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)
    labels_i32 = labels.to(tl.int32)

    for t in tl.range(0, input_len):
        beta_t = tl.load(beta_base + t * STATE_COUNT_MAX + states_beta).to(tl.float32)
        beta_t = tl.where(valid_state, beta_t, -float("inf"))

        alpha_t = tl.load(
            log_alpha + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states_beta
        ).to(tl.float32)
        alpha_t = tl.where(valid_state, alpha_t, -float("inf"))

        log_post = alpha_t + beta_t - log_likelihood
        posterior = tl.where(valid_state & (scale != 0.0), tl.exp(log_post), 0.0)

        # atomic-free scatter-add: sum posterior over states whose label == c.
        for c in tl.static_range(C):
            per_class_c = tl.sum(tl.where(labels_i32 == c, posterior, 0.0))
            gi_c = t * N * C + batch * C + c
            gic = tl.load(grad_input + gi_c).to(tl.float32)
            tl.store(grad_input + gi_c, gic - scale * per_class_c)


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
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS_BACKWARD")

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

        # scratch_beta holds beta[t][s] for t in [0, T), s in [0, state_count_max).
        scratch_beta = torch.empty(
            (batch_size, work_log_probs.shape[0], state_count_max),
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
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS_BACKWARD_OUT")

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


def _install():
    """Wire the XPU implementation into the submodule entrypoint.

    ``tests/test__ctc_loss_backward.py`` calls the op through the
    ``flag_gems.ops`` namespace (``flag_gems.ops._ctc_loss_backward`` /
    ``flag_gems.ops._ctc_loss_backward_out``). ``SpecOpRegistrar`` only
    rewrites the top-level ``flag_gems`` globals, so it can not reach this
    direct-import entrypoint. Replace the attribute on the already-imported
    module, mirroring ``_kunlunxin.fused.mhc_pre._install``.
    """
    import sys

    mod = sys.modules.get("flag_gems.ops")
    if mod is not None:
        mod._ctc_loss_backward = _ctc_loss_backward
        mod._ctc_loss_backward_out = _ctc_loss_backward_out


_install()
