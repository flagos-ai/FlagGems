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

"""Kunlunxin backend override for ``_native_batch_norm_legit``.

The generic module imports ``flag_gems.ops.batch_norm.batch_norm`` directly, so
it never reaches the Kunlunxin ``batch_norm``, whose forward kernel uses a
different reduction layout and a three-stage fallback. On this backend the
generic kernels do not lower:

    'tt.addptr' op failed to verify that result type matches ptr type
    'triton_xpu.convert_layout' op requires the same shape for all operands
    RuntimeError: PassManager::run failed: out of resource: uni_sram

Everything here is the generic module with one change: the batch-norm call is
resolved to the vendor implementation.
"""

import logging

import torch
from torch import Tensor

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _native_batch_norm_legit(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    """Batch normalization with in-place running statistics updates.

    Semantics match ``aten::_native_batch_norm_legit``: ``running_mean`` and
    ``running_var`` are updated in place and the function returns the tuple
    ``(output, save_mean, save_invstd)`` where ``save_invstd`` is the inverse
    standard deviation used for normalization (i.e. ``1 / sqrt(var + eps)``).
    """
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT")

    output, save_mean, save_invstd = batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
        # aten::_native_batch_norm_legit updates the running statistics for
        # every dtype (only the `_no_training` variant leaves them alone), and
        # its running variance is the unbiased 1/(N-1) estimate -- measured
        # against this backend's own reference, which reproduces the unbiased
        # value for fp16, bf16 and fp32.
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )
    return output, save_mean, save_invstd


def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
):
    """Batch normalization without persistent running statistics."""
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    if not training:
        raise RuntimeError("Expected has_running_mean to be true, but got false.")

    channels = input.shape[1]
    running_mean = torch.zeros(channels, dtype=input.dtype, device=input.device)
    running_var = torch.ones(channels, dtype=input.dtype, device=input.device)
    return batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=True,
        momentum=momentum,
        eps=eps,
        update_running_all_dtypes=True,
    )


def _copy_outputs(result, out, save_mean, save_invstd):
    result_out, result_mean, result_invstd = result
    out.resize_as_(result_out).copy_(result_out)
    save_mean.resize_as_(result_mean).copy_(result_mean)
    save_invstd.resize_as_(result_invstd).copy_(result_invstd)
    return out, save_mean, save_invstd


def _native_batch_norm_legit_out(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_OUT")
    result = _native_batch_norm_legit(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_no_stats_out(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    result = _native_batch_norm_legit_no_stats(
        input,
        weight,
        bias,
        training,
        momentum,
        eps,
    )
    return _copy_outputs(result, out, save_mean, save_invstd)
