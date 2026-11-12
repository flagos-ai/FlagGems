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

from flag_gems.ops._ctc_loss import _prepare
from flag_gems.ops.ctc_loss import _logaddexp, _logaddexp3
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Kunlunxin XPU lowering notes (see harness KernelOpt_Knowledge):
#   * `tl.load(..., mask=..., other=...)` does NOT reliably honour the `other=`
#     fill on this backend: masked-out lanes read neighbouring memory instead of
#     `other`. A `-inf` "other" that leaks garbage makes `exp(alpha + beta -
#     log_likelihood)` explode, which is exactly the ~1e6 magnitude corruption
#     seen against the CPU reference.
#   * `tl.debug_barrier` is NOT a reliable cross-thread barrier on this backend
#     once the grid has more than one program: a vector alpha recursion that
#     reads the neighbouring state lane (alpha[s] / alpha[s-1] / alpha[s-2])
#     through a shared/scratch buffer races and silently corrupts every batch
#     except the first. The alpha pass is therefore rewritten as a *scalar*
#     recursion over the state axis, which has no cross-lane dependency and
#     needs no barrier at all.
#
# Every gather below therefore uses a clamped address + an unmasked load + a
# `tl.where` value select, and never relies on `other=` or a masked load.


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
    batch = tl.program_id(0)

    input_len = tl.load(input_lengths + batch).to(tl.int32)
    target_len = tl.load(target_lengths + batch).to(tl.int32)
    state_count = target_len * 2 + 1

    if TARGET_1D:
        target_origin = tl.load(target_offsets + batch).to(tl.int32)
    else:
        target_origin = batch * MAX_TARGET

    alpha_base = log_alpha + batch * T * STATE_COUNT_MAX

    # init: alpha[0][s] = logp[0][label(s)] for the start states, else -inf.
    for s in tl.range(0, STATE_COUNT_MAX):
        if s < state_count:
            is_blank = (s % 2) == 0
            ti = (s - 1) // 2
            tmask = (ti >= 0) & (ti < target_len)
            tsafe = tl.where(tmask, ti, 0)
            tv = tl.load(targets + target_origin + tsafe)
            tv = tl.where(tmask, tv, BLANK)
            label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)

            init = (s == 0) | ((s == 1) & (target_len > 0))
            logp0 = tl.load(log_probs + batch * C + label_s).to(tl.float32)
            alpha0 = tl.where(init & (input_len > 0), logp0, -float("inf"))
            tl.store(alpha_base + s, alpha0)

    # recurse: alpha[t][s] = logaddexp3(alpha[t-1][s], alpha[t-1][s-1],
    # alpha[t-1][s-2], skip) + logp[t][label(s)].
    for t in tl.range(1, input_len):
        for s in tl.range(0, STATE_COUNT_MAX):
            if s < state_count:
                is_blank = (s % 2) == 0
                ti = (s - 1) // 2
                tmask = (ti >= 0) & (ti < target_len)
                tsafe = tl.where(tmask, ti, 0)
                tv = tl.load(targets + target_origin + tsafe)
                tv = tl.where(tmask, tv, BLANK)
                label_s = tl.where(is_blank, BLANK, tv).to(tl.int32)

                # stay: alpha[t-1][s]
                a0 = tl.load(alpha_base + (t - 1) * STATE_COUNT_MAX + s).to(tl.float32)

                # next1: alpha[t-1][s-1]
                s1 = s - 1
                a1 = tl.load(
                    alpha_base + (t - 1) * STATE_COUNT_MAX + tl.maximum(s1, 0)
                ).to(tl.float32)
                a1 = tl.where(s1 >= 0, a1, -float("inf"))

                # next2 (skip): alpha[t-1][s-2], allowed only between two
                # distinct labels (target[ti] != target[ti-1]).
                s2 = s - 2
                ti2 = ti - 1
                tmask2 = (ti2 >= 0) & (ti2 < target_len)
                tsafe2 = tl.where(tmask2, ti2, 0)
                tv2 = tl.load(targets + target_origin + tsafe2)
                tv2 = tl.where(tmask2, tv2, BLANK)
                skip_allowed = (~is_blank) & (ti > 0) & (tv != tv2)
                a2 = tl.load(
                    alpha_base + (t - 1) * STATE_COUNT_MAX + tl.maximum(s2, 0)
                ).to(tl.float32)
                a2 = tl.where(s2 >= 0, a2, -float("inf"))

                acc = _logaddexp3(a0, a1, a2, skip_allowed)
                logp = tl.load(log_probs + t * N * C + batch * C + label_s).to(
                    tl.float32
                )
                tl.store(alpha_base + t * STATE_COUNT_MAX + s, acc + logp)

    # loss = -logaddexp(alpha[input_len-1][last], alpha[input_len-1][prev_last]).
    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float("inf"))
    else:
        last_t = input_len - 1
        last_s = state_count - 1
        last = tl.load(alpha_base + last_t * STATE_COUNT_MAX + last_s).to(tl.float32)
        prev_s = state_count - 2
        prev_last = tl.load(alpha_base + last_t * STATE_COUNT_MAX + prev_s).to(
            tl.float32
        )
        prev_last = tl.where(target_len > 0, prev_last, -float("inf"))
        loss = -_logaddexp(last, prev_last)

    tl.store(neg_log_likelihood + batch, loss)


def _ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS")

    setup = _prepare(log_probs, targets, input_lengths, target_lengths, blank)
    batch_size = setup.batch_size
    original_dtype = setup.original_dtype
    unbatched = setup.unbatched
    state_count_max = setup.state_count_max
    work_log_probs = setup.work_log_probs
    work_targets = setup.work_targets
    work_input_lengths = setup.work_input_lengths
    work_target_lengths = setup.work_target_lengths
    work_target_offsets = setup.work_target_offsets
    target_stride = setup.target_stride
    target_1d = setup.target_1d
    block_s = setup.block_s

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

    if unbatched:
        neg_log_likelihood = neg_log_likelihood.squeeze(0)
        log_alpha = log_alpha.squeeze(0)

    if neg_log_likelihood.dtype != original_dtype:
        neg_log_likelihood = neg_log_likelihood.to(original_dtype)
    if log_alpha.dtype != original_dtype:
        log_alpha = log_alpha.to(original_dtype)

    return neg_log_likelihood, log_alpha


def _ctc_loss_out(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
    *,
    out0: Optional[torch.Tensor] = None,
    out1: Optional[torch.Tensor] = None,
):
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS_OUT")

    neg_log_likelihood, log_alpha = _ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank, zero_infinity
    )

    if out0 is not None:
        if out0.shape != neg_log_likelihood.shape:
            out0.resize_(neg_log_likelihood.shape)
        out0.copy_(neg_log_likelihood)
    else:
        out0 = neg_log_likelihood

    if out1 is not None:
        if out1.shape != log_alpha.shape:
            out1.resize_(log_alpha.shape)
        out1.copy_(log_alpha)
    else:
        out1 = log_alpha

    return out0, out1


def _install():
    """Wire the XPU implementation into the submodule entrypoint.

    ``tests/test__ctc_loss.py`` calls the op through the ``flag_gems.ops``
    namespace (``flag_gems.ops._ctc_loss`` / ``flag_gems.ops._ctc_loss_out``),
    which ``SpecOpRegistrar`` does not rewrite (it only patches the top-level
    ``flag_gems`` globals). Replace the attribute on the already-imported
    module, mirroring ``_kunlunxin.fused.mhc_pre._install``.
    """
    import sys

    mod = sys.modules.get("flag_gems.ops")
    if mod is not None:
        mod._ctc_loss = _ctc_loss
        mod._ctc_loss_out = _ctc_loss_out


_install()
