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
"""Kunlunxin (XPU3) backend override for ``aten::_ctc_loss_backward``.

Why a backend override is required
----------------------------------
The generic implementation (``src/flag_gems/ops/_ctc_loss_backward.py``) runs
the beta recurrence in a single kernel that keeps the previous beta row in
**global memory** and lets a lane read the values its neighbours wrote on the
previous loop iteration.  On XPU3 an intra-kernel GM store -> cross-lane load
is **not coherent** (``tl.debug_barrier`` does not fence it, cache modifiers
and ``volatile`` do not help), so the returned gradient is silently wrong; the
SMA scattering step additionally relies on ``tl.atomic_add``, which
mis-accumulates on this build.  See
``harness/KernelOpt_Knowledge/wiki/patterns/xpu3-intra-kernel-gm-crosslane-incoherent.md``
and ``wiki/patterns/xpu3-atomic-add-misaccumulates.md``.

Correct design used here
------------------------
Kernel launches are ordered and fully coherent across launches, so the beta
recurrence is executed as **one launch per timestep** over a ``(batch,)``
grid.  The three kernels are the ones already validated for the autograd
``ctc_loss`` override (``_kunlunxin/ops/ctc_loss.py``); this module only
supplies the standalone ``aten::_ctc_loss_backward`` wrapper, which consumes
the forward outputs (``neg_log_likelihood``, ``log_alpha``) supplied by the
caller instead of invoking the forward DP:

* ``_ctc_grad_init_kernel``  -> ``grad = exp(logp) * grad_out`` for
  ``t < input_length`` (0 elsewhere, NaN on the documented infinite-loss
  corners),
* ``_ctc_beta_init_kernel``  -> last-row beta seeds,
* ``_ctc_beta_step_kernel`` x ``max(input_lengths)`` -> beta recurrence plus
  the ``-grad_out * posterior`` class fold (one exact store per (t, batch,
  class), no atomics).

``log_alpha`` keeps the caller's row width (``2 * max(target_lengths) + 1``)
while the beta tile always spans at least one warp of physical lanes: XPU3
silently widens ``tl.arange(0, n)`` to ``round_up(n, 64)``, so a tile narrower
than a warp would store past the scratch row (measured as
``KL_XID_KERNEL_EXCEPTION`` for the sibling ``ctc_loss`` operator).  All shift
loads clamp the index and bound the value with ``tl.where`` *after* the load:
XPU3 masked loads return the in-bounds value on masked lanes instead of
``other`` (``wiki/patterns/xpu3-masked-load-other-inbounds-leak.md``).
"""

import logging
from typing import Optional

import torch
import triton

from flag_gems.ops._ctc_loss_backward import _prepare_backward
from flag_gems.ops.ctc_loss import _REDUCTION_NONE, _length_stats
from flag_gems.runtime import torch_device_fn

from .ctc_loss import (
    _ctc_beta_init_kernel,
    _ctc_beta_step_kernel,
    _ctc_grad_init_kernel,
)

logger = logging.getLogger(__name__)

_GRAD_INIT_BLOCK = 256


def _ctc_loss_backward_impl(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity,
):
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

    wp = setup.work_log_probs
    T = wp.shape[0]
    N = setup.batch_size
    C = wp.shape[2]
    total = wp.numel()
    device = wp.device

    # The alpha rows handed in by the caller are ``2 * max_target + 1`` wide.
    # The beta tile needs >= 64 physical lanes; see the module docstring.
    state_count_max = setup.state_count_max
    block_s = max(64, triton.next_power_of_2(max(state_count_max, 1)))

    # One extra dummy row: the beta-step kernel funnels (t < 0) work there
    # instead of using a data-dependent store guard (``tt.reduce`` cannot be
    # legalized inside ``scf.if`` on this backend).
    grad_work = torch.empty((T + 1, N, C), dtype=torch.float32, device=device)
    scratch_beta = torch.empty((N, 2, block_s), dtype=torch.float32, device=device)

    work_grad = setup.work_grad
    if work_grad.ndim == 0:
        # Unbatched call: the 0-d upstream gradient broadcasts to batch 0.
        work_grad = work_grad.reshape(1)

    max_steps = _length_stats(setup.work_input_lengths)[1] if N > 0 else 0

    if T > 0 and total > 0:
        with torch_device_fn.device(device):
            _ctc_grad_init_kernel[(triton.cdiv(total, _GRAD_INIT_BLOCK),)](
                wp,
                setup.work_input_lengths,
                setup.work_target_lengths,
                setup.work_neg_log_likelihood,
                work_grad,
                grad_work,
                total,
                T,
                N,
                C,
                _REDUCTION_NONE,
                zero_infinity,
                _GRAD_INIT_BLOCK,
            )
            if max_steps > 0:
                _ctc_beta_init_kernel[(N,)](
                    setup.work_input_lengths,
                    setup.work_target_lengths,
                    scratch_beta,
                    N,
                    state_count_max,
                    block_s,
                )
                for step in range(max_steps):
                    _ctc_beta_step_kernel[(N,)](
                        wp,
                        setup.work_targets,
                        setup.work_input_lengths,
                        setup.work_target_lengths,
                        setup.work_target_offsets,
                        setup.work_neg_log_likelihood,
                        setup.work_log_alpha,
                        work_grad,
                        grad_work,
                        scratch_beta,
                        step,
                        (step % 2) * block_s,
                        ((step + 1) % 2) * block_s,
                        C,
                        T,
                        N,
                        C,
                        setup.target_stride,
                        state_count_max,
                        blank,
                        setup.target_1d,
                        _REDUCTION_NONE,
                        zero_infinity,
                        block_s,
                    )

    grad_log_probs = grad_work[:T]
    if setup.unbatched:
        grad_log_probs = grad_log_probs.squeeze(1)
    if grad_log_probs.dtype != setup.original_dtype:
        grad_log_probs = grad_log_probs.to(setup.original_dtype)

    return grad_log_probs


def _ctc_loss_backward(
    grad: torch.Tensor,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
    neg_log_likelihood: torch.Tensor,
    log_alpha: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN _CTC_LOSS_BACKWARD")
    return _ctc_loss_backward_impl(
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


def _ctc_loss_backward_out(
    grad: torch.Tensor,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths,
    target_lengths,
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


# The dedicated tests / benchmark for this operator call
# ``flag_gems.ops._ctc_loss_backward`` directly (not the aten-dispatch entry),
# so publish the override there too — the same pattern used by the Kunlunxin
# ``_ctc_loss`` and ``special_chebyshev_polynomial_t`` overrides.
import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    for _name, _fn in (
        ("_ctc_loss_backward", _ctc_loss_backward),
        ("_ctc_loss_backward_out", _ctc_loss_backward_out),
    ):
        setattr(_generic_ops_module, _name, _fn)
