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
"""Kunlunxin (XPU3) backend override for ``aten::_ctc_loss``.

Why a backend override is required
----------------------------------
The generic implementation (``src/flag_gems/ops/ctc_loss.py``) computes the
alpha lattice with *one program per batch* and a ``t``-loop that keeps the
previous row in **global memory** (``log_alpha``).  A lane reads the values
written by its neighbours on the previous loop iteration.  On XPU3 an
intra-kernel GM store -> GM load across lanes is **not coherent**:
``tl.debug_barrier`` does not fence it, cache modifiers (``.cv`` / ``.wt``)
do not fix it, and an atomic-based fence costs milliseconds per iteration.
Measured symptoms of the generic kernel on this backend:

* wrong forward values as soon as the tile reaches 64 physical lanes
  (masked ``tl.load(..., other=)`` also leaks the in-bounds read on masked
  lanes, which the shift-by-1/2 loads rely on), and
* ``OutOfResources`` (``TritonXPUUnrollControl``) once ``T`` is unrolled
  (``T >= ~256``), i.e. the largest benchmark shape cannot even compile.

Correct design used here
------------------------
Kernel launches are ordered and fully coherent across launches, so the DP is
decomposed into **one launch per timestep** over a ``(batch,)`` grid:

* ``_ctc_init_kernel``  -> writes ``log_alpha[:, 0, :]`` (time 0),
* ``_ctc_step_kernel``  -> for each ``t in [1, T)`` reads row ``t-1`` from GM
  and writes row ``t`` (each row is a single coherent unit because it was
  produced by the previous launch),
* ``_ctc_final_kernel`` -> reduces the last valid row into the loss.

This keeps the exact numerics of the generic kernel (``logaddexp`` DP with
skip transitions gated by ``_logaddexp3``) while never relying on
intra-kernel cross-lane memory.  The shift loads clamp the pointer and mask
*after* the load (``tl.where``) instead of using the broken masked-``other``.

Note: the shared-memory (``tle.gpu.smem``) single-kernel variant and the
``tl.associative_scan`` shift variants were evaluated and rejected — see
``harness/solution/_ctc_loss/README.md`` for the measured evidence.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.ops._ctc_loss import _prepare
from flag_gems.ops.ctc_loss import _logaddexp, _logaddexp3
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _ctc_init_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    log_alpha,
    T,
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
    valid_state = states < (target_len * 2 + 1)
    stored_state = states < STATE_COUNT_MAX
    is_blank_state = (states % 2) == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)
    if TARGET_1D:
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += tl.load(
                target_lengths + prev_batch, mask=prev_batch < batch, other=0
            )
        target_origin = target_base
    else:
        target_origin = batch * MAX_TARGET
    target_value = tl.load(
        targets + target_origin + target_safe_index, mask=target_mask, other=BLANK
    )
    labels = tl.where(is_blank_state, BLANK, target_value)
    init_state = (states == 0) | ((states == 1) & (target_len > 0))
    init_logp = tl.load(
        log_probs + batch * C + labels,
        mask=init_state & stored_state & (input_len > 0),
        other=0.0,
    ).to(tl.float32)
    alpha = tl.where(init_state & valid_state & (input_len > 0), init_logp, -float("inf"))
    tl.store(log_alpha + batch * T * STATE_COUNT_MAX + states, alpha, mask=stored_state)


@libentry()
@triton.jit
def _ctc_step_kernel(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    t,
    T,
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
                target_lengths + prev_batch, mask=prev_batch < batch, other=0
            )
        target_origin = target_base
    else:
        target_origin = batch * MAX_TARGET

    target_value = tl.load(
        targets + target_origin + target_safe_index, mask=target_mask, other=BLANK
    )
    labels = tl.where(is_blank_state, BLANK, target_value)

    prev_base = log_alpha + batch * T * STATE_COUNT_MAX + (t - 1) * STATE_COUNT_MAX
    prev0 = tl.load(prev_base + states, mask=stored_state, other=-float("inf")).to(
        tl.float32
    )
    # Clamp the pointer and apply the -inf bound *after* the load: XPU3 masked
    # loads return the in-bounds value on masked lanes instead of `other`.
    prev1_raw = tl.load(
        prev_base + tl.where(states > 0, states - 1, 0), mask=stored_state, other=0.0
    ).to(tl.float32)
    prev1 = tl.where(states > 0, prev1_raw, -float("inf"))
    prev2_raw = tl.load(
        prev_base + tl.where(states > 1, states - 2, 0), mask=stored_state, other=0.0
    ).to(tl.float32)
    prev2 = tl.where(states > 1, prev2_raw, -float("inf"))

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


@libentry()
@triton.jit
def _ctc_final_kernel(
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    T,
    STATE_COUNT_MAX: tl.constexpr,
):
    batch = tl.program_id(0)
    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    if input_len > 0:
        state_count = target_len * 2 + 1
        base = (
            log_alpha + batch * T * STATE_COUNT_MAX + (input_len - 1) * STATE_COUNT_MAX
        )
        last = tl.load(base + state_count - 1)
        prev_last_raw = tl.load(
            base + tl.where(target_len > 0, state_count - 2, 0),
            mask=target_len > 0,
            other=0.0,
        )
        prev_last = tl.where(target_len > 0, prev_last_raw, -float("inf"))
        tl.store(neg_log_likelihood + batch, -_logaddexp(last, prev_last))
    else:
        tl.store(
            neg_log_likelihood + batch,
            tl.where(target_len == 0, 0.0, float("inf")),
        )


def _ctc_loss_impl(log_probs, targets, input_lengths, target_lengths, blank):
    setup = _prepare(log_probs, targets, input_lengths, target_lengths, blank)
    wp = setup.work_log_probs
    T = wp.shape[0]
    batch_size = setup.batch_size
    num_classes = wp.shape[2]

    raw_nll = torch.empty((batch_size,), dtype=torch.float32, device=log_probs.device)
    log_alpha = torch.empty(
        (batch_size, T, setup.state_count_max),
        dtype=torch.float32,
        device=log_probs.device,
    )

    if T > 0:
        with torch_device_fn.device(log_probs.device):
            grid = (batch_size,)
            _ctc_init_kernel[grid](
                wp,
                setup.work_targets,
                setup.work_input_lengths,
                setup.work_target_lengths,
                log_alpha,
                T,
                batch_size,
                num_classes,
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
                    raw_nll,
                    log_alpha,
                    t,
                    T,
                    batch_size,
                    num_classes,
                    setup.target_stride,
                    setup.state_count_max,
                    blank,
                    setup.target_1d,
                    setup.block_s,
                )
            _ctc_final_kernel[grid](
                setup.work_input_lengths,
                setup.work_target_lengths,
                raw_nll,
                log_alpha,
                T,
                setup.state_count_max,
            )
    else:
        for batch in range(batch_size):
            input_len = int(setup.work_input_lengths[batch].item())
            target_len = int(setup.work_target_lengths[batch].item())
            if input_len <= 0:
                raw_nll[batch] = 0.0 if target_len == 0 else float("inf")
            else:
                raise RuntimeError("ctc_loss encountered an empty log_probs tensor")

    return setup, raw_nll, log_alpha


def _ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
    blank: int = 0,
    zero_infinity: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS")
    setup, raw_nll, log_alpha = _ctc_loss_impl(
        log_probs, targets, input_lengths, target_lengths, blank
    )
    neg_log_likelihood = raw_nll
    if setup.unbatched:
        neg_log_likelihood = neg_log_likelihood.squeeze(0)
        log_alpha = log_alpha.squeeze(0)
    if neg_log_likelihood.dtype != setup.original_dtype:
        neg_log_likelihood = neg_log_likelihood.to(setup.original_dtype)
    if log_alpha.dtype != setup.original_dtype:
        log_alpha = log_alpha.to(setup.original_dtype)
    return neg_log_likelihood, log_alpha


def _ctc_loss_out(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
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


# The dedicated tests / benchmark for this operator call ``flag_gems.ops._ctc_loss``
# directly (not the aten-dispatch entry), so publish the override there too —
# the same pattern the Kunlunxin ``special_chebyshev_polynomial_t`` override uses.
import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    for _name, _fn in (
        ("_ctc_loss", _ctc_loss),
        ("_ctc_loss_out", _ctc_loss_out),
    ):
        setattr(_generic_ops_module, _name, _fn)
