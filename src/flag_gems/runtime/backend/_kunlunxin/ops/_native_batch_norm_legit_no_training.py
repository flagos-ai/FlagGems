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

from flag_gems.runtime import torch_device_fn

from ._batch_norm_no_update import _batch_norm_no_update_kernel_flat, _flat_block

logger = logging.getLogger(__name__)

_EMPTY_SAVE = {}


def _empty_save(dtype, device):
    """Cached empty (0,)-shaped tensor; inference mode computes no batch stats."""
    key = (dtype, device)
    t = _EMPTY_SAVE.get(key)
    if t is None:
        t = torch.empty((0,), dtype=dtype, device=device)
        _EMPTY_SAVE[key] = t
    return t


def _native_batch_norm_legit_no_training(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    """Kunlunxin/XPU inference-only batch normalization using running stats.

    Mirrors ``torch.ops.aten._native_batch_norm_legit_no_training``. Each program
    covers one contiguous run of ``BLOCK`` elements and re-derives the per-lane
    channel from the flat index, so the data movement is one contiguous pass and
    the channel statistics come from an address that repeats every ``SPATIAL``
    lanes (see ``_batch_norm_no_update`` for the tuning notes). Returns
    (output, save_mean, save_var)
    where save_mean/save_var are EMPTY, since no batch statistics are computed
    in this mode.
    """
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")

    if running_mean is None or running_var is None:
        raise RuntimeError(
            "running_mean and running_var are required for "
            "_native_batch_norm_legit_no_training"
        )

    channels = input.shape[1]
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise RuntimeError("running statistics must contain one value per channel")

    x = input if input.is_contiguous() else input.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()

    if n_elements > 0:
        inner = n_elements // (x.shape[0] * channels)
        block = _flat_block(n_elements)
        has_weight = weight is not None
        has_bias = bias is not None
        with torch_device_fn.device(x.device):
            _batch_norm_no_update_kernel_flat[(triton.cdiv(n_elements, block),)](
                x,
                weight if has_weight else x,
                bias if has_bias else x,
                running_mean,
                running_var,
                output,
                channels,
                n_elements,
                eps,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                BLOCK=block,
                SPATIAL=inner,
                NEED_MASK=(n_elements % block) != 0,
                num_warps=4,
            )

    save_mean = _empty_save(input.dtype, input.device)
    save_var = _empty_save(input.dtype, input.device)
    return output, save_mean, save_var
