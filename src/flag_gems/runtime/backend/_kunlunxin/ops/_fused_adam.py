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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops._fused_adam import _fused_adam as _generic_fused_adam

logger = logging.getLogger(__name__)

# Tile widths (in elements) allowed on the unmasked fast path.  Every entry is a
# power of two -- non power-of-two constexpr widths are mis-lowered by TritonXPU.
# 1024 is deliberately absent: the generic kernel miscomputes with BLOCK_SIZE ==
# 1024 (its `other=0.0` loads silently read valid lanes as zero, index 0 is off
# by 3.0e-1) and the width is kept out of the whitelist to stay clear of it.
_FAST_BLOCKS = (8192, 4096, 2048, 512)
# Prefer a tile width that still leaves this many programs to spread over cores.
_MIN_GRID = 4
# `tl.program_id(0)` is int32 here; keep the flat offset inside that range.
_INT32_MAX = 2**31 - 1
_STEP_DTYPES = (torch.int64, torch.int32, torch.float32, torch.float64)


@triton.jit
def _one_minus_exp(x):
    """Accurate ``1 - exp(x)`` for ``x <= 0``.

    ``exp`` is fp32 here and ``tl_extra_shim.expm1`` is only ``exp(x) - 1`` in
    disguise, so the direct form loses four decimal digits exactly where Adam
    needs them: for ``beta2 = 0.999, step = 1`` the true value is ``1e-3`` while
    ``1 - exp(-1.0005e-3)`` carries 1.3e-5 relative error.  Near zero we use the
    Taylor series of ``-expm1`` instead (12 terms, |x| <= 1 => <1e-9 relative),
    and fall back to the direct form where there is no cancellation left.
    These are scalar ops, so the extra `where` costs nothing per element.
    """
    xs = tl.maximum(x, -1.0)
    t = 1.0 + xs * (1.0 / 13.0)
    t = 1.0 + xs * (1.0 / 12.0) * t
    t = 1.0 + xs * (1.0 / 11.0) * t
    t = 1.0 + xs * (1.0 / 10.0) * t
    t = 1.0 + xs * (1.0 / 9.0) * t
    t = 1.0 + xs * (1.0 / 8.0) * t
    t = 1.0 + xs * (1.0 / 7.0) * t
    t = 1.0 + xs * (1.0 / 6.0) * t
    t = 1.0 + xs * (1.0 / 5.0) * t
    t = 1.0 + xs * (1.0 / 4.0) * t
    t = 1.0 + xs * (1.0 / 3.0) * t
    t = 1.0 + xs * 0.5 * t
    return tl.where(x > -1.0, -(xs * t), 1.0 - tl.exp(x))


@triton.jit
def _fused_adam_flat_kernel(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    step_ptr,
    BLOCK: tl.constexpr,
    LR: tl.constexpr,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    EPS: tl.constexpr,
    LOG_BETA1: tl.constexpr,
    LOG_BETA2: tl.constexpr,
    AMSGRAD: tl.constexpr,
    MAXIMIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # Fully unmasked, stride-1 block DMA.  No `mask=`/`other=` anywhere: on this
    # backend `other=` silently zeroes valid lanes even when the mask is all-true.
    p = tl.load(param + offsets)
    g = tl.load(grad + offsets)
    m = tl.load(exp_avg + offsets)
    v = tl.load(exp_avg_sq + offsets)

    if MAXIMIZE:
        g = -g

    m = BETA1 * m + (1.0 - BETA1) * g
    v = BETA2 * v + (1.0 - BETA2) * g * g
    tl.store(exp_avg + offsets, m)
    tl.store(exp_avg_sq + offsets, v)

    # The step lives on the device.  Reading it here instead of on the host keeps
    # the whole operator asynchronous; a host-side `.item()` costs 0.03-0.12 ms
    # of synchronisation, which dominates every shape in the benchmark matrix.
    # The load is issued *after* the vector loads so its latency overlaps them.
    step = tl.load(step_ptr).to(tl.float32)
    inv_bc1 = 1.0 / _one_minus_exp(step * LOG_BETA1)
    inv_bc2 = 1.0 / _one_minus_exp(step * LOG_BETA2)

    corrected_m = m * inv_bc1
    if AMSGRAD:
        corrected_v = v * inv_bc2
        prev_max = tl.load(max_exp_avg_sq + offsets)
        new_max = tl.maximum(prev_max, corrected_v)
        tl.store(max_exp_avg_sq + offsets, new_max)
        denom = tl.sqrt(new_max) + EPS
    else:
        denom = tl.sqrt(v * inv_bc2) + EPS

    if WEIGHT_DECAY > 0:
        update = corrected_m / denom + WEIGHT_DECAY * p
    else:
        update = corrected_m / denom

    tl.store(param + offsets, p - LR * update)


def _choose_block(n):
    """Largest whitelisted power-of-two tile that exactly divides ``n``."""
    divisors = [b for b in _FAST_BLOCKS if n % b == 0]
    if not divisors:
        return None
    wide = [b for b in divisors if n // b >= _MIN_GRID]
    return wide[0] if wide else divisors[0]


def _fast_plan(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    amsgrad,
    grad_scale,
    found_inf,
):
    """Pure-metadata eligibility gate for the unmasked fast path.

    Returns a list of ``(tensors..., block)`` launch descriptors or ``None``.
    Only shape/stride/dtype/device metadata is inspected: touching any FlagGems
    operator here would add 0.1-0.35 ms of dispatch to every call.
    """
    if grad_scale is not None or found_inf is not None:
        return None
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        return None
    if not isinstance(beta1, (int, float)) or not isinstance(beta2, (int, float)):
        return None
    # log(beta) must be finite; a host-side inf/nan is garbage inside the kernel.
    if not (0.0 < beta1 < 1.0) or not (0.0 < beta2 < 1.0):
        return None

    n_params = len(params)
    if n_params == 0:
        return None
    if not (
        len(grads) == len(exp_avgs) == len(exp_avg_sqs) == len(state_steps) == n_params
    ):
        return None
    if amsgrad and len(max_exp_avg_sqs) != n_params:
        return None

    plan = []
    for i in range(n_params):
        param = params[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        n = param.numel()
        if n == 0 or n > _INT32_MAX:
            return None
        block = _choose_block(n)
        if block is None:
            return None

        state = (param, grad, exp_avg, exp_avg_sq)
        if amsgrad:
            state = state + (max_exp_avg_sqs[i],)
        for t in state:
            if t.dtype is not torch.float32:
                return None
            if t.numel() != n or not t.is_contiguous():
                return None
            if t.device != param.device:
                return None

        if step.numel() < 1 or step.dtype not in _STEP_DTYPES:
            return None
        if not step.is_contiguous() or step.device != param.device:
            return None

        plan.append(
            (
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                max_exp_avg_sqs[i] if amsgrad else exp_avg_sq,
                step,
                block,
            )
        )
    return plan


def _legacy_fused_adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale,
    found_inf,
):
    """Original Kunlunxin wrapper: 512-element chunks for tiny tensors.

    Kept verbatim for inputs the fast path refuses (non-contiguous, non-fp32,
    odd element counts, grad scaling, tensor ``lr``, ...).
    """
    kernel_params = []
    kernel_grads = []
    kernel_exp_avgs = []
    kernel_exp_avg_sqs = []
    kernel_max_exp_avg_sqs = []
    kernel_state_steps = []

    for index, param in enumerate(params):
        # A single-program 1024-element launch miscomputes index zero on XPU.
        # Preserve the known-good 512-element launches only for this small case.
        chunks = param.reshape(-1).split(512) if param.numel() <= 1024 else [param]
        kernel_params.extend(chunks)
        kernel_grads.extend(
            grads[index].reshape(-1).split(512)
            if param.numel() <= 1024
            else [grads[index]]
        )
        kernel_exp_avgs.extend(
            exp_avgs[index].reshape(-1).split(512)
            if param.numel() <= 1024
            else [exp_avgs[index]]
        )
        kernel_exp_avg_sqs.extend(
            exp_avg_sqs[index].reshape(-1).split(512)
            if param.numel() <= 1024
            else [exp_avg_sqs[index]]
        )
        if max_exp_avg_sqs:
            kernel_max_exp_avg_sqs.extend(
                max_exp_avg_sqs[index].reshape(-1).split(512)
                if param.numel() <= 1024
                else [max_exp_avg_sqs[index]]
            )
        kernel_state_steps.extend([state_steps[index]] * len(chunks))

    _generic_fused_adam(
        kernel_params,
        kernel_grads,
        kernel_exp_avgs,
        kernel_exp_avg_sqs,
        kernel_max_exp_avg_sqs,
        kernel_state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _fused_adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS_KUNLUNXIN FUSED_ADAM")

    plan = _fast_plan(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        amsgrad,
        grad_scale,
        found_inf,
    )
    if plan is None:
        _legacy_fused_adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize,
            grad_scale,
            found_inf,
        )
        return params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs

    log_beta1 = math.log(beta1)
    log_beta2 = math.log(beta2)
    for param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, step, block in plan:
        _fused_adam_flat_kernel[(param.numel() // block,)](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            step,
            block,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            log_beta1,
            log_beta2,
            amsgrad,
            maximize,
        )
    return params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs


def _fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS_KUNLUNXIN FUSED_ADAM_")
    _fused_adam(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return None
