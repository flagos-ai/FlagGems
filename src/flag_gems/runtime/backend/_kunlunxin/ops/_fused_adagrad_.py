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

import torch
import triton
import triton.language as tl

from flag_gems.ops._fused_adagrad_ import _fused_adagrad_ as _generic_fused_adagrad_

logger = logging.getLogger(__name__)

# Tile widths (in elements) allowed on the unmasked fast path.  Every entry is a
# power of two -- non power-of-two constexpr widths are mis-lowered by TritonXPU.
# 1024 is deliberately absent: the generic kernel miscomputes with BLOCK_SIZE ==
# 1024 (its ``other=0.0`` loads silently read valid lanes as zero) and the width
# is kept out of the whitelist to stay clear of it.
_FAST_BLOCKS = (8192, 4096, 2048, 512)
# Prefer a tile width that still leaves this many programs to spread over cores.
_MIN_GRID = 4
# `tl.program_id(0)` is int32 here; keep the flat offset inside that range.
_INT32_MAX = 2**31 - 1


@triton.jit
def _fused_adagrad_flat_kernel(
    param,
    grad,
    state_sum,
    step_ptr,
    BLOCK: tl.constexpr,
    LR: tl.constexpr,
    LR_DECAY: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    EPS: tl.constexpr,
    MAXIMIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # Fully unmasked, stride-1 block DMA.  No `mask=`/`other=` anywhere: on this
    # backend `other=` silently zeroes valid lanes even when the mask is all-true,
    # so the generic masked kernel miscomputes.
    p = tl.load(param + offsets)
    g = tl.load(grad + offsets)
    s = tl.load(state_sum + offsets)

    if MAXIMIZE:
        g = -g

    if WEIGHT_DECAY > 0:
        g = g + p * WEIGHT_DECAY

    s = s + g * g

    # The step lives on the device.  Reading it here instead of on the host keeps
    # the whole operator asynchronous; a host-side `.item()` costs a synchronisation
    # that dominates every shape in the benchmark matrix.  Issued after the vector
    # loads so its latency overlaps them.
    step = tl.load(step_ptr).to(tl.float32)
    corrected_lr = LR / (1.0 + (step - 1.0) * LR_DECAY)

    p = p - corrected_lr * g / (tl.sqrt(s) + EPS)

    tl.store(param + offsets, p)
    tl.store(state_sum + offsets, s)


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
    state_sums,
    state_steps,
    lr,
    lr_decay,
    weight_decay,
    grad_scale,
):
    """Pure-metadata eligibility gate for the unmasked fast path.

    Returns a list of ``(param, grad, state_sum, step, block)`` launch
    descriptors or ``None``.  Only shape/stride/dtype/device metadata is
    inspected: touching any FlagGems operator here would add dispatch overhead
    to every call.  Anything the gate refuses is handed to the generic
    implementation, which performs the full validation and (masked) update.
    """
    if grad_scale is not None:
        return None
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        return None
    if not isinstance(lr_decay, (int, float)) or isinstance(lr_decay, bool):
        return None
    if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool):
        return None

    n_params = len(params)
    if n_params == 0:
        return None
    if not (len(grads) == len(state_sums) == len(state_steps) == n_params):
        return None

    plan = []
    for i in range(n_params):
        param = params[i]
        grad = grads[i]
        state_sum = state_sums[i]
        step = state_steps[i]

        n = param.numel()
        if n == 0 or n > _INT32_MAX:
            return None
        block = _choose_block(n)
        if block is None:
            return None

        for t in (param, grad, state_sum):
            if t.dtype is not torch.float32:
                return None
            if t.numel() != n or not t.is_contiguous():
                return None
            if t.device != param.device:
                return None

        if step.numel() != 1 or step.dtype is not torch.float32:
            return None
        if not step.is_contiguous() or step.device != param.device:
            return None

        plan.append((param, grad, state_sum, step, block))

    return plan


def _fused_adagrad_(
    params,
    grads,
    state_sums,
    state_steps,
    *,
    lr: float = 1e-2,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    eps: float = 1e-10,
    maximize: bool = False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS_KUNLUNXIN _FUSED_ADAGRAD_")

    # ATen compares ``found_inf`` against 1 exactly (any other value -- including
    # 2.0 -- still applies the update).  Resolve it on the host so the kernel no
    # longer carries a data-dependent early-return branch; that branch lowers to a
    # ``cf.cond_br`` with an early ``tt.return`` that crashes the ``TritonXPUMask``
    # pass on this backend.
    if found_inf is not None:
        if found_inf.item() == 1.0:
            return None
        found_inf = None

    plan = _fast_plan(
        params,
        grads,
        state_sums,
        state_steps,
        lr,
        lr_decay,
        weight_decay,
        grad_scale,
    )
    if plan is None:
        _generic_fused_adagrad_(
            params,
            grads,
            state_sums,
            state_steps,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=found_inf,
        )
        return None

    for param, grad, state_sum, step, block in plan:
        _fused_adagrad_flat_kernel[(param.numel() // block,)](
            param,
            grad,
            state_sum,
            step,
            block,
            lr,
            lr_decay,
            weight_decay,
            eps,
            maximize,
        )

    return None
