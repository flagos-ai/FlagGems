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

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _batch_norm_with_update_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    """Kunlunxin/XPU override of ``aten::_batch_norm_with_update_functional``.

    The generic KernelGen implementation
    (``src/flag_gems/ops/_batch_norm_with_update_functional.py``) does not lower
    on the XPU compiler: its ``_bn_fused_kernel`` trips ``TritonXPUUnrollControl``
    (``arith.addi`` operand-type mismatch at the ``offs = rowbase + cols`` line)
    for most shapes, and the handful that do lower produce wrong values. This
    override delegates to the already-tuned vendor ``batch_norm`` wrapper
    (single-launch fused-train / 2-launch / 3-launch plans) with the ATen
    running-stat contract opted in explicitly.

    ATen semantics:
      * returns ``(out, save_mean, save_invstd, reserve, running_mean_out,
        running_var_out)``;
      * ``save_invstd`` is ``1/sqrt(var + eps)``;
      * ``running_mean`` is updated in *every* input dtype and ``running_var``
        uses the unbiased estimate ``var * n/(n-1)``;
      * the caller's ``running_mean`` / ``running_var`` tensors are **not**
        mutated: the updated statistics live in fresh output buffers.

    The vendor wrapper updates the running statistics in place, so we hand it
    device-local clones of the caller's buffers and return those clones.
    """
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_WITH_UPDATE_FUNCTIONAL")

    C = int(input.shape[1]) if input.dim() >= 2 else 0
    dev = input.device
    dtype = input.dtype

    # Fresh output buffers (ATen never mutates the caller's running stats).
    if running_mean is not None:
        rmean_out = running_mean.clone()
    else:
        rmean_out = torch.zeros(C, dtype=dtype, device=dev)
    if running_var is not None:
        rvar_out = running_var.clone()
    else:
        rvar_out = torch.ones(C, dtype=dtype, device=dev)

    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        rmean_out,
        rvar_out,
        True,
        momentum,
        eps,
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )

    reserve = torch.empty((0,), dtype=torch.uint8, device=dev)
    return output, save_mean, save_invstd, reserve, rmean_out, rvar_out


import sys as _sys  # noqa: E402

# benchmark/test_batch_norm_with_update_functional.py binds its candidate from
# the *generic* module path
# (`from flag_gems.ops._batch_norm_with_update_functional import ...`) instead of
# `flag_gems._batch_norm_with_update_functional`. On XPU3 that generic kernel
# cannot compile at all (TritonXPUUnrollControl / uni_sram pass failure), so the
# benchmark can never be measured even though the registrar already routes
# `flag_gems._batch_norm_with_update_functional` to this module. Rebind the
# generic module attribute to this vendor implementation so both call paths hit
# the kernel that actually runs on XPU (same pattern as
# `_kunlunxin/ops/_batch_norm_impl_index.py`, which rebinds
# `flag_gems.ops._batch_norm_impl_index`).
_generic_module = _sys.modules.get("flag_gems.ops._batch_norm_with_update_functional")
if _generic_module is not None:
    _generic_module._batch_norm_with_update_functional = _batch_norm_with_update_functional
