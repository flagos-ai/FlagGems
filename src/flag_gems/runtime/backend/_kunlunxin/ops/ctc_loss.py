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

"""Kunlunxin XPU ``ctc_loss`` (flag_gems.ctc_loss) override.

The generic ``flag_gems.ops.ctc_loss`` kernels (upstream Triton) lower badly on
this backend (see ``_kunlunxin/ops/_ctc_loss.py`` for the same diagnosis on the
``_ctc_loss`` primitive):

* ``tl.load(..., mask=..., other=...)`` does not reliably honour the ``other=``
  fill: masked-out lanes read neighbouring memory instead, which feeds a wrong
  value into the ``logaddexp`` recurrence (measured ~1e-2..1e0 loss error,
  ~6.5% relative on the seed-2026 core matrix case).
* ``tl.debug_barrier()`` cannot be trusted to make a *previous* iteration's
  vector store visible to the *next* iteration's vector load on this backend
  (staged gm2lm/lm2gm copies without an explicit fence), so the per-state
  vector recursion races and silently corrupts the alpha DP.

Both problems disappear if every state lane is *scalarised* (one scalar loop
over the state axis per timestep, guard by ``if s < state_count``): scalar
store -> scalar load on the same program address has in-order visibility and
needs no barrier at all.  That is the correctness baseline used here.

Performance: the DP is serial over T per batch; the benchmark only exercises
the no-grad full-length reduce path.  For that path the kernel still runs the
phases in a single program (one per batch element), so the achievable speedup
is limited by the serial depth; the scalarised design keeps every memory
access dense and 4-byte aligned (no gathers for logp: labels are devectorised
into a small per-program label cache), which is the fastest correct shape that
avoids the masked-load / barrier traps.
"""

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.ops.ctc_loss import (
    _compute_dtype,
    _logaddexp,
    _logaddexp3,
    _reduction_enum,
    _length_stats,
    _lengths_to_tensor,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---- tle.raw payload for the full-length no-grad fast path (xpu3 cluster) ----
# One core per batch element; the whole alpha DP runs with the alpha rows and
# the current log-prob row resident in local memory, so there is no per-node GM
# round trip.  The scalar Triton kernel stays as the fallback for every case the
# payload does not cover (or when the xpu payload toolchain is unavailable).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CTC_RAW_BROKEN = False
try:
    import triton.experimental.tle as tle

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "ctc_loss_raw.xpu"))
    def ctc_fwd_f32(
        log_probs, targets, target_lengths, target_offsets, contrib,
        T, N, C, STRIDE, blank, reduction, target_1d,
    ):
        ...

    @triton.jit(
        do_not_specialize=[
            "T", "N", "C", "STRIDE", "blank", "reduction", "target_1d",
            "target_offsets", "target_lengths",
        ]
    )
    def _ctc_loss_forward_full_length_raw_kernel(
        log_probs, targets, target_lengths, target_offsets, contrib,
        T, N, C, STRIDE, blank, reduction, target_1d,
    ):
        tle.raw.call(
            ctc_fwd_f32,
            (
                log_probs,
                targets,
                target_lengths,
                target_offsets,
                contrib,
                T,
                N,
                C,
                STRIDE,
                blank,
                reduction,
                target_1d,
            ),
        )

    _HAS_CTC_RAW = True
except Exception:  # pragma: no cover - xpu payload toolchain unavailable
    _HAS_CTC_RAW = False

# ---- with-grad payload (ctc_loss_backward benchmark path) ----
_HAS_CTC_RAW_BWD = False
try:
    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "ctc_loss_bwd_raw.xpu"))
    def ctc_fwd_grad_f32(
        log_probs, targets, target_lengths, target_offsets, nll_out, log_alpha,
        T, N, C, STRIDE, blank, target_1d,
    ):
        ...

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "ctc_loss_bwd_raw.xpu"))
    def ctc_bwd_f32(
        log_probs, targets, target_lengths, target_offsets, neg_log_likelihood,
        log_alpha, grad_output, grad_input, T, N, C, STRIDE, blank, reduction,
        target_1d,
    ):
        ...

    @triton.jit(
        do_not_specialize=[
            "T", "N", "C", "STRIDE", "blank", "target_1d",
            "target_offsets", "target_lengths",
        ]
    )
    def _ctc_loss_forward_grad_raw_kernel(
        log_probs, targets, target_lengths, target_offsets, nll_out, log_alpha,
        T, N, C, STRIDE, blank, target_1d,
    ):
        tle.raw.call(
            ctc_fwd_grad_f32,
            (
                log_probs, targets, target_lengths, target_offsets, nll_out,
                log_alpha, T, N, C, STRIDE, blank, target_1d,
            ),
        )

    @triton.jit(
        do_not_specialize=[
            "T", "N", "C", "STRIDE", "blank", "reduction", "target_1d",
            "target_offsets", "target_lengths",
        ]
    )
    def _ctc_loss_backward_raw_kernel(
        log_probs, targets, target_lengths, target_offsets, neg_log_likelihood,
        log_alpha, grad_output, grad_input, T, N, C, STRIDE, blank, reduction,
        target_1d,
    ):
        tle.raw.call(
            ctc_bwd_f32,
            (
                log_probs, targets, target_lengths, target_offsets,
                neg_log_likelihood, log_alpha, grad_output, grad_input, T, N, C,
                STRIDE, blank, reduction, target_1d,
            ),
        )

    _HAS_CTC_RAW_BWD = True
except Exception:  # pragma: no cover - xpu payload toolchain unavailable
    _HAS_CTC_RAW_BWD = False

_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2


@libentry()
@triton.jit
def _ctc_loss_forward_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
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
    """Forward alpha pass; log_alpha is [batch, T, BLOCK_S] (padded)."""
    batch = tl.program_id(0)
    input_len = tl.load(input_lengths + batch).to(tl.int32)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    state_count = target_len * 2 + 1

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch).to(tl.int32)
    else:
        target_origin = batch * MAX_TARGET

    alpha_base = log_alpha + batch * T * BLOCK_S

    # init: alpha[0][s]
    for s in tl.range(0, BLOCK_S):
        if s < state_count:
            is_blank = (s % 2) == 0
            ti = (s - 1) // 2
            tmask = (ti >= 0) & (ti < target_len)
            tsafe = tl.where(tmask, ti, 0)
            tv = tl.load(targets + target_origin + tsafe)
            tv = tl.where(tmask, tv, BLANK)
            label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)
            init = (s == 0) | ((s == 1) & (target_len > 0))
            logp = tl.load(log_probs + batch * C + label_s).to(tl.float32)
            alpha = tl.where(init & (input_len > 0), logp, -float("inf"))
            tl.store(alpha_base + s, alpha)

    # recurrence: alpha[t][s] = logaddexp3(stay, next1, next2) + logp[t][label(s)]
    for t in tl.range(1, input_len):
        for s in tl.range(0, BLOCK_S):
            if s < state_count:
                is_blank = (s % 2) == 0
                ti = (s - 1) // 2
                tmask = (ti >= 0) & (ti < target_len)
                tsafe = tl.where(tmask, ti, 0)
                tv = tl.load(targets + target_origin + tsafe)
                tv = tl.where(tmask, tv, BLANK)
                label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)

                a0 = tl.load(alpha_base + (t - 1) * BLOCK_S + s).to(tl.float32)
                s1 = s - 1
                a1 = tl.load(alpha_base + (t - 1) * BLOCK_S + tl.maximum(s1, 0)).to(
                    tl.float32
                )
                a1 = tl.where(s1 >= 0, a1, -float("inf"))
                s2 = s - 2
                ti2 = ti - 1
                tmask2 = (ti2 >= 0) & (ti2 < target_len)
                tsafe2 = tl.where(tmask2, ti2, 0)
                tv2 = tl.load(targets + target_origin + tsafe2)
                tv2 = tl.where(tmask2, tv2, BLANK)
                skip_allowed = (~is_blank) & (ti > 0) & (tv != tv2)
                a2 = tl.load(alpha_base + (t - 1) * BLOCK_S + tl.maximum(s2, 0)).to(
                    tl.float32
                )
                a2 = tl.where(s2 >= 0, a2, -float("inf"))

                acc = _logaddexp3(a0, a1, a2, skip_allowed)
                logp = tl.load(log_probs + t * N * C + batch * C + label_s).to(
                    tl.float32
                )
                tl.store(alpha_base + t * BLOCK_S + s, acc + logp)

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        last_t = input_len - 1
        last = tl.load(alpha_base + last_t * BLOCK_S + (state_count - 1)).to(
            tl.float32
        )
        prev_last = tl.load(alpha_base + last_t * BLOCK_S + tl.maximum(state_count - 2, 0)).to(
            tl.float32
        )
        prev_last = tl.where(target_len > 0, prev_last, -float("inf"))
        loss = -_logaddexp(last, prev_last)

    tl.store(neg_log_likelihood + batch, loss)


@libentry()
@triton.jit
def _ctc_loss_forward_no_grad_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    target_offsets,
    neg_log_likelihood,
    scratch_alpha,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    STATE_COUNT_MAX: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """No-grad forward for variable input_lengths; scratch is [batch, 2, BLOCK_S]."""
    batch = tl.program_id(0)
    input_len = tl.load(input_lengths + batch).to(tl.int32)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    state_count = target_len * 2 + 1

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch).to(tl.int32)
    else:
        target_origin = batch * MAX_TARGET

    # Full alpha history (row t = alpha[t]) so no memory slot is reused
    # across loop iterations (store->reload of the same slot is miscompiled).
    scratch_batch = scratch_alpha + batch * T * BLOCK_S

    for s in tl.range(0, BLOCK_S):
        if s < state_count:
            is_blank = (s % 2) == 0
            ti = (s - 1) // 2
            tmask = (ti >= 0) & (ti < target_len)
            tsafe = tl.where(tmask, ti, 0)
            tv = tl.load(targets + target_origin + tsafe)
            tv = tl.where(tmask, tv, BLANK)
            label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)
            init = (s == 0) | ((s == 1) & (target_len > 0))
            logp = tl.load(log_probs + batch * C + label_s).to(tl.float32)
            alpha = tl.where(init & (input_len > 0), logp, -float("inf"))
            tl.store(scratch_batch + s, alpha)

    for t in tl.range(1, input_len):
        cur_base = scratch_batch + t * BLOCK_S
        prev_base = scratch_batch + (t - 1) * BLOCK_S
        for s in tl.range(0, BLOCK_S):
            if s < state_count:
                is_blank = (s % 2) == 0
                ti = (s - 1) // 2
                tmask = (ti >= 0) & (ti < target_len)
                tsafe = tl.where(tmask, ti, 0)
                tv = tl.load(targets + target_origin + tsafe)
                tv = tl.where(tmask, tv, BLANK)
                label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)

                a0 = tl.load(prev_base + s).to(tl.float32)
                s1 = s - 1
                a1 = tl.load(prev_base + tl.maximum(s1, 0)).to(tl.float32)
                a1 = tl.where(s1 >= 0, a1, -float("inf"))
                s2 = s - 2
                ti2 = ti - 1
                tmask2 = (ti2 >= 0) & (ti2 < target_len)
                tsafe2 = tl.where(tmask2, ti2, 0)
                tv2 = tl.load(targets + target_origin + tsafe2)
                tv2 = tl.where(tmask2, tv2, BLANK)
                skip_allowed = (~is_blank) & (ti > 0) & (tv != tv2)
                a2 = tl.load(prev_base + tl.maximum(s2, 0)).to(tl.float32)
                a2 = tl.where(s2 >= 0, a2, -float("inf"))

                acc = _logaddexp3(a0, a1, a2, skip_allowed)
                logp = tl.load(log_probs + t * N * C + batch * C + label_s).to(
                    tl.float32
                )
                tl.store(cur_base + s, acc + logp)

    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        final_base = scratch_batch + (input_len - 1) * BLOCK_S
        last = tl.load(final_base + (state_count - 1)).to(tl.float32)
        prev_last = tl.load(final_base + tl.maximum(state_count - 2, 0)).to(tl.float32)
        prev_last = tl.where(target_len > 0, prev_last, -float("inf"))
        loss = -_logaddexp(last, prev_last)

    tl.store(neg_log_likelihood + batch, loss)


@libentry()
@triton.jit
def _ctc_loss_forward_full_length_reduce_kernel(
    log_probs,
    targets,
    target_lengths,
    target_offsets,
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
    """No-grad fast path: all input_lengths == T, reduction in {mean, sum}."""
    batch = tl.program_id(0)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    state_count = target_len * 2 + 1

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch).to(tl.int32)
    else:
        target_origin = batch * MAX_TARGET

    # Full alpha history (row t = alpha[t]) so no memory slot is reused
    # across loop iterations (store->reload of the same slot is miscompiled).
    scratch_batch = scratch_alpha + batch * T * BLOCK_S

    for s in tl.range(0, BLOCK_S):
        if s < state_count:
            is_blank = (s % 2) == 0
            ti = (s - 1) // 2
            tmask = (ti >= 0) & (ti < target_len)
            tsafe = tl.where(tmask, ti, 0)
            tv = tl.load(targets + target_origin + tsafe)
            tv = tl.where(tmask, tv, BLANK)
            label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)
            init = (s == 0) | ((s == 1) & (target_len > 0))
            logp = tl.load(log_probs + batch * C + label_s).to(tl.float32)
            alpha = tl.where(init, logp, -float("inf"))
            tl.store(scratch_batch + s, alpha)

    for t in tl.range(1, T):
        cur_base = scratch_batch + t * BLOCK_S
        prev_base = scratch_batch + (t - 1) * BLOCK_S
        for s in tl.range(0, BLOCK_S):
            if s < state_count:
                is_blank = (s % 2) == 0
                ti = (s - 1) // 2
                tmask = (ti >= 0) & (ti < target_len)
                tsafe = tl.where(tmask, ti, 0)
                tv = tl.load(targets + target_origin + tsafe)
                tv = tl.where(tmask, tv, BLANK)
                label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)

                a0 = tl.load(prev_base + s).to(tl.float32)
                s1 = s - 1
                a1 = tl.load(prev_base + tl.maximum(s1, 0)).to(tl.float32)
                a1 = tl.where(s1 >= 0, a1, -float("inf"))
                s2 = s - 2
                ti2 = ti - 1
                tmask2 = (ti2 >= 0) & (ti2 < target_len)
                tsafe2 = tl.where(tmask2, ti2, 0)
                tv2 = tl.load(targets + target_origin + tsafe2)
                tv2 = tl.where(tmask2, tv2, BLANK)
                skip_allowed = (~is_blank) & (ti > 0) & (tv != tv2)
                a2 = tl.load(prev_base + tl.maximum(s2, 0)).to(tl.float32)
                a2 = tl.where(s2 >= 0, a2, -float("inf"))

                acc = _logaddexp3(a0, a1, a2, skip_allowed)
                logp = tl.load(log_probs + t * N * C + batch * C + label_s).to(
                    tl.float32
                )
                tl.store(cur_base + s, acc + logp)

    if T <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        final_base = scratch_batch + (T - 1) * BLOCK_S
        last = tl.load(final_base + (state_count - 1)).to(tl.float32)
        prev_last = tl.load(final_base + tl.maximum(state_count - 2, 0)).to(tl.float32)
        prev_last = tl.where(target_len > 0, prev_last, -float("inf"))
        loss = -_logaddexp(last, prev_last)

    if REDUCTION == 1:
        loss = loss / tl.maximum(target_len, 1).to(tl.float32) / N
    tl.store(contrib + batch, loss)


@libentry()
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
    scratch_beta,
    post_buf,
    label_buf,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    MAX_TARGET: tl.constexpr,
    BLANK: tl.constexpr,
    TARGET_1D: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Scalar backward beta pass.

    Emits the per-(t, s) posterior ``exp(alpha[t, s] + beta_gen[t+1, s] - ll)``
    and the per-state label into scratch buffers; it never does a
    read-modify-write on a memory address (that pattern is miscompiled by the
    XPU backend and duplicates the write, see solution note).  The class
    accumulation / grad_input update is done by the separate apply kernel.
    """
    batch = tl.program_id(0)
    input_len = tl.load(input_lengths + batch).to(tl.int32)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    nll = tl.load(neg_log_likelihood + batch).to(tl.float32)
    state_count = target_len * 2 + 1

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch).to(tl.int32)
    else:
        target_origin = batch * MAX_TARGET

    if REDUCTION == 0:
        scale = tl.load(grad_output + batch).to(tl.float32)
    else:
        scale = tl.load(grad_output).to(tl.float32)
        if REDUCTION == 1:
            denom = tl.maximum(target_len, 1).to(tl.float32) * N
            scale = scale / denom

    if ZERO_INFINITY:
        scale = tl.where(nll == float("inf"), 0.0, scale)

    # Full beta history (row t = beta_gen[t]) so no memory slot is reused
    # across loop iterations; a store->reload of the same slot is
    # miscompiled by this backend (see solution note).
    scratch_batch = scratch_beta + batch * (T + 1) * BLOCK_S
    init_row = scratch_batch + input_len * BLOCK_S
    for s in tl.range(0, BLOCK_S):
        b0 = tl.where(
            ((s == state_count - 1) | ((s == state_count - 2) & (target_len > 0)))
            & (input_len > 0),
            0.0,
            -float("inf"),
        )
        tl.store(init_row + s, b0)

    log_likelihood = tl.where(scale != 0.0, -nll, 0.0)

    for step in tl.range(0, input_len):
        t = input_len - 1 - step
        beta_base = scratch_batch + (t + 1) * BLOCK_S
        next_beta_base = scratch_batch + t * BLOCK_S
        for s in tl.range(0, BLOCK_S):
            if s < state_count:
                is_blank = (s % 2) == 0
                ti = (s - 1) // 2
                tmask = (ti >= 0) & (ti < target_len)
                tsafe = tl.where(tmask, ti, 0)
                tv = tl.load(targets + target_origin + tsafe)
                tv = tl.where(tmask, tv, BLANK)
                label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)
                tl.store(label_buf + batch * BLOCK_S + s, label_s)

                beta_s = tl.load(beta_base + s).to(tl.float32)
                alpha_ts = tl.load(
                    log_alpha + batch * T * BLOCK_S + t * BLOCK_S + s
                ).to(tl.float32)
                log_post = alpha_ts + beta_s - log_likelihood
                post = tl.where(
                    scale != 0.0,
                    tl.exp(log_post),
                    0.0,
                )
                tl.store(post_buf + batch * T * BLOCK_S + t * BLOCK_S + s, post)

                stay = beta_s + tl.load(
                    log_probs + t * N * C + batch * C + label_s
                ).to(tl.float32)

                s1 = s + 1
                b1 = -float("inf")
                if s1 < state_count:
                    is_blank1 = (s1 % 2) == 0
                    ti1 = (s1 - 1) // 2
                    tmask1 = (ti1 >= 0) & (ti1 < target_len)
                    tsafe1 = tl.where(tmask1, ti1, 0)
                    tv1 = tl.load(targets + target_origin + tsafe1)
                    tv1 = tl.where(tmask1, tv1, BLANK)
                    label_s1 = tl.where(is_blank1, BLANK, tv1).to(tl.int32)
                    b1 = tl.load(beta_base + s1).to(tl.float32) + tl.load(
                        log_probs + t * N * C + batch * C + label_s1
                    ).to(tl.float32)

                s2 = s + 2
                skip_allowed = False
                b2 = -float("inf")
                if s2 < state_count:
                    ti2 = (s2 - 1) // 2
                    tmask2 = (ti2 >= 0) & (ti2 < target_len)
                    tsafe2 = tl.where(tmask2, ti2, 0)
                    tv2 = tl.load(targets + target_origin + tsafe2)
                    tv2 = tl.where(tmask2, tv2, BLANK)
                    # s2 is implicitly non-blank (s odd => s2 odd)
                    skip_allowed = (~is_blank) & (ti2 > 0) & (tv != tv2)
                    b2 = tl.load(beta_base + s2).to(tl.float32) + tl.load(
                        log_probs + t * N * C + batch * C + tl.where(
                            is_blank, BLANK, tv2
                        ).to(tl.int32)
                    ).to(tl.float32)

                beta_next = _logaddexp3(stay, b1, b2, skip_allowed)
                tl.store(next_beta_base + s, beta_next)


@libentry()
@triton.jit
def _ctc_loss_backward_apply_kernel(
    log_probs,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    grad_output,
    grad_input,
    post_buf,
    label_buf,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INFINITY: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """One program per (batch, t): computes the final grad slice with a single
    store per class (no read-modify-write on the same address)."""
    batch = tl.program_id(0)
    t = tl.program_id(1)
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

    state_count = target_len * 2 + 1

    if t < input_len:
        for c in tl.range(0, C):
            logp = tl.load(log_probs + t * N * C + batch * C + c).to(tl.float32)
            base = tl.exp(logp) * scale
            base = tl.where(
                (scale != 0.0) & (logp == -float("inf")), float("nan"), base
            )
            if not ZERO_INFINITY:
                base = tl.where(nll == float("inf"), float("nan"), base)
            acc = 0.0
            for s in tl.range(0, BLOCK_S):
                if s < state_count:
                    label_s = tl.load(label_buf + batch * BLOCK_S + s).to(tl.int32)
                    post = tl.load(
                        post_buf + batch * T * BLOCK_S + t * BLOCK_S + s
                    ).to(tl.float32)
                    acc = tl.where(label_s == c, acc + post, acc)
            tl.store(grad_input + t * N * C + batch * C + c, base - scale * acc)
    else:
        for c in tl.range(0, C):
            tl.store(grad_input + t * N * C + batch * C + c, 0.0)


def _ctc_loss_forward_impl(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank,
    reduction,
    zero_infinity,
    needs_log_probs_grad,
    unbatched,
):
    """Common setup + launch; returns (output, saved tuple or None)."""
    global _CTC_RAW_BROKEN
    original_dtype = log_probs.dtype
    compute_dtype = _compute_dtype(original_dtype)

    work_log_probs = log_probs.unsqueeze(1) if unbatched else log_probs
    work_log_probs = work_log_probs.contiguous()
    if work_log_probs.dtype != compute_dtype:
        work_log_probs = work_log_probs.to(compute_dtype)

    if torch.is_floating_point(targets):
        work_targets = targets.to(dtype=torch.long).contiguous()
    else:
        work_targets = targets.contiguous()
    work_input_lengths = _lengths_to_tensor(
        input_lengths, log_probs.device, "input_lengths"
    )
    work_target_lengths = _lengths_to_tensor(
        target_lengths, log_probs.device, "target_lengths"
    )
    batch_size = work_log_probs.shape[1]

    min_input_length, max_input_length, _ = _length_stats(work_input_lengths)
    min_target_length, max_target, total_target_length = _length_stats(
        work_target_lengths
    )

    state_count_max = 2 * max_target + 1
    target_stride = max_target
    if work_targets.ndim == 1:
        target_1d = True
        work_target_offsets = (
            work_target_lengths.cumsum(0) - work_target_lengths
        ).contiguous()
    else:
        target_1d = False
        target_stride = work_targets.shape[1]
        work_target_offsets = work_target_lengths

    # Empty concatenated targets: keep a valid address for the masked-out label
    # gathers (their values are never used when every target length is 0).
    if work_targets.numel() == 0:
        work_targets = work_targets.new_zeros(1)

    block_s = triton.next_power_of_2(state_count_max)

    if not needs_log_probs_grad:
        if (
            not unbatched
            and not zero_infinity
            and reduction in (_REDUCTION_MEAN, _REDUCTION_SUM)
            and min_input_length == work_log_probs.shape[0]
            and work_log_probs.shape[0] > 0
        ):
            contrib = torch.empty(
                (batch_size,), dtype=torch.float32, device=log_probs.device
            )
            scratch_alpha = torch.empty(
                (batch_size, work_log_probs.shape[0], block_s),
                dtype=torch.float32,
                device=log_probs.device,
            )
            with torch_device_fn.device(log_probs.device):
                if (
                    _HAS_CTC_RAW
                    and not _CTC_RAW_BROKEN
                    and state_count_max <= 256
                    and work_log_probs.shape[2] <= 160
                    and work_targets.numel() >= 8
                ):
                    try:
                        _ctc_loss_forward_full_length_raw_kernel[(1,)](
                            work_log_probs,
                            work_targets,
                            work_target_lengths,
                            work_target_offsets,
                            contrib,
                            work_log_probs.shape[0],
                            batch_size,
                            work_log_probs.shape[2],
                            target_stride,
                            blank,
                            reduction,
                            1 if target_1d else 0,
                        )
                    except Exception:
                        _CTC_RAW_BROKEN = True
                        _ctc_loss_forward_full_length_reduce_kernel[(batch_size,)](
                            work_log_probs,
                            work_targets,
                            work_target_lengths,
                            work_target_offsets,
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
                else:
                    _ctc_loss_forward_full_length_reduce_kernel[(batch_size,)](
                        work_log_probs,
                        work_targets,
                        work_target_lengths,
                        work_target_offsets,
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
            return output, None, False

        raw_neg_log_likelihood = torch.empty(
            (batch_size,), dtype=torch.float32, device=log_probs.device
        )
        scratch_alpha = torch.empty(
            (batch_size, work_log_probs.shape[0], block_s),
            dtype=torch.float32,
            device=log_probs.device,
        )
        with torch_device_fn.device(log_probs.device):
            _ctc_loss_forward_no_grad_kernel[(batch_size,)](
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                raw_neg_log_likelihood,
                scratch_alpha,
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
        output = _reduce_nll(
            neg_log_likelihood, reduction, work_target_lengths, unbatched
        )
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output, None, False

    raw_neg_log_likelihood = torch.empty(
        (batch_size,), dtype=torch.float32, device=log_probs.device
    )
    use_payload_grad = (
        _HAS_CTC_RAW_BWD
        and not _CTC_RAW_BROKEN
        and not zero_infinity
        and min_input_length == work_log_probs.shape[0]
        and work_log_probs.shape[0] > 0
        and state_count_max <= 256
        and work_log_probs.shape[2] <= 128
        and work_targets.numel() >= 8
    )
    if use_payload_grad:
        log_alpha = torch.empty(
            (batch_size, work_log_probs.shape[0], 256),
            dtype=torch.float32,
            device=log_probs.device,
        )
        with torch_device_fn.device(log_probs.device):
            _ctc_loss_forward_grad_raw_kernel[(1,)](
                work_log_probs,
                work_targets,
                work_target_lengths,
                work_target_offsets,
                raw_neg_log_likelihood,
                log_alpha,
                work_log_probs.shape[0],
                batch_size,
                work_log_probs.shape[2],
                target_stride,
                blank,
                1 if target_1d else 0,
            )
    else:
        log_alpha = torch.empty(
            (batch_size, work_log_probs.shape[0], block_s),
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
    neg_log_likelihood = raw_neg_log_likelihood
    if zero_infinity:
        neg_log_likelihood = torch.where(
            torch.isinf(neg_log_likelihood),
            torch.zeros(
                (), dtype=neg_log_likelihood.dtype, device=log_probs.device
            ),
            neg_log_likelihood,
        )
    output = _reduce_nll(
        neg_log_likelihood, reduction, work_target_lengths, unbatched
    )
    if output.dtype != original_dtype:
        output = output.to(original_dtype)

    saved = (
        work_log_probs,
        work_targets,
        work_input_lengths,
        work_target_lengths,
        work_target_offsets,
        raw_neg_log_likelihood,
        log_alpha,
    )
    return output, saved, use_payload_grad


def _reduce_nll(neg_log_likelihood, reduction, work_target_lengths, unbatched):
    if reduction == _REDUCTION_NONE:
        output = neg_log_likelihood
        if unbatched:
            output = output.squeeze(0)
    elif reduction == _REDUCTION_SUM:
        output = neg_log_likelihood.sum()
    else:
        denom = work_target_lengths.clamp_min(1)
        output = (neg_log_likelihood / denom).mean()
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

        unbatched = log_probs.ndim == 2
        batch_size = 1 if unbatched else log_probs.shape[1]

        work_targets = targets
        if torch.is_floating_point(work_targets):
            work_targets = work_targets.to(dtype=torch.long)
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
        if min_input_length < 0 or max_input_length > log_probs.shape[0]:
            raise RuntimeError("ctc_loss input_lengths must be in [0, T]")
        if min_target_length < 0:
            raise RuntimeError("ctc_loss target_lengths must be non-negative")

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
        else:
            raise RuntimeError(
                "ctc_loss expects targets to be a 1D concatenated tensor or a "
                f"2D padded tensor, but got {work_targets.ndim}D"
            )

        output, saved, paygrad = _ctc_loss_forward_impl(
            log_probs,
            work_targets,
            input_lengths,
            target_lengths,
            blank,
            reduction,
            zero_infinity,
            ctx.needs_input_grad[0],
            unbatched,
        )

        if saved is not None:
            (
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                raw_neg_log_likelihood,
                log_alpha,
            ) = saved
            ctx.save_for_backward(
                work_log_probs,
                work_targets,
                work_input_lengths,
                work_target_lengths,
                work_target_offsets,
                raw_neg_log_likelihood,
                log_alpha,
            )
            ctx.blank = blank
            ctx.reduction = reduction
            ctx.zero_infinity = zero_infinity
            ctx.unbatched = unbatched
            ctx.batch_size = batch_size
            ctx.original_dtype = log_probs.dtype
            ctx.max_target = work_targets.shape[1] if not target_1d else 0
            ctx.target_1d = target_1d
            ctx.block_s = log_alpha.shape[2]
            ctx.paygrad = paygrad

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            work_log_probs,
            work_targets,
            work_input_lengths,
            work_target_lengths,
            work_target_offsets,
            neg_log_likelihood,
            log_alpha,
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_log_probs = torch.empty_like(work_log_probs)
        t_steps = work_log_probs.shape[0]
        if getattr(ctx, "paygrad", False):
            with torch_device_fn.device(work_log_probs.device):
                _ctc_loss_backward_raw_kernel[(1,)](
                    work_log_probs,
                    work_targets,
                    work_target_lengths,
                    work_target_offsets,
                    neg_log_likelihood,
                    log_alpha,
                    grad_output.to(torch.float32),
                    grad_log_probs,
                    t_steps,
                    ctx.batch_size,
                    work_log_probs.shape[2],
                    ctx.max_target,
                    ctx.blank,
                    ctx.reduction,
                    1 if ctx.target_1d else 0,
                )
        else:
            with torch_device_fn.device(work_log_probs.device):
                scratch_beta = torch.empty(
                    (ctx.batch_size, t_steps + 1, ctx.block_s),
                    dtype=torch.float32,
                    device=work_log_probs.device,
                )
                post_buf = torch.empty(
                    (ctx.batch_size, t_steps, ctx.block_s),
                    dtype=torch.float32,
                    device=work_log_probs.device,
                )
                label_buf = torch.empty(
                    (ctx.batch_size, ctx.block_s),
                    dtype=torch.int32,
                    device=work_log_probs.device,
                )
                _ctc_loss_backward_kernel[(ctx.batch_size,)](
                    work_log_probs,
                    work_targets,
                    work_input_lengths,
                    work_target_lengths,
                    work_target_offsets,
                    neg_log_likelihood,
                    log_alpha,
                    grad_output,
                    scratch_beta,
                    post_buf,
                    label_buf,
                    t_steps,
                    ctx.batch_size,
                    work_log_probs.shape[2],
                    ctx.max_target,
                    ctx.blank,
                    ctx.target_1d,
                    ctx.reduction,
                    ctx.zero_infinity,
                    ctx.block_s,
                )
                _ctc_loss_backward_apply_kernel[(ctx.batch_size, t_steps)](
                    work_log_probs,
                    work_input_lengths,
                    work_target_lengths,
                    neg_log_likelihood,
                    grad_output,
                    grad_log_probs,
                    post_buf,
                    label_buf,
                    t_steps,
                    ctx.batch_size,
                    work_log_probs.shape[2],
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


def _install():
    """Wire the XPU implementation into the top-level and ops namespaces."""
    import sys

    fg_mod = sys.modules.get("flag_gems")
    ops_mod = sys.modules.get("flag_gems.ops")
    if fg_mod is not None:
        fg_mod.ctc_loss = ctc_loss
    if ops_mod is not None:
        ops_mod.ctc_loss = ctc_loss


_install()
