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

from flag_gems.runtime import torch_device_fn

from ._batch_norm_no_update import (
    BNNU_MAX_PROGRAMS,
    BNNU_TILE_S,
    _batch_norm_no_update_kernel,
)

logger = logging.getLogger(__name__)


def make_3d_for_bn(input):
    """View the input as [N, C, S] for batch normalization."""
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


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

    Mirrors ``torch.ops.aten._native_batch_norm_legit_no_training``. The generic
    implementation's 2D-tile kernel (grid = feat_dim, [BLOCK_M, BLOCK_N] 2D loops)
    does not lower on the XPU compiler for most shapes (SramCode/``TritonXPULegalize``
    pass failures). This override maps one program to each contiguous (n, c) slice
    (grid = N*C, the same pattern as the batch_norm inference path), so each slice
    is a contiguous block-DMA run sharing ONE channel's stats/affine scalars.
    Returns (output, save_mean, save_var) where save_mean/save_var are EMPTY
    (shape (0,)) since no batch statistics are computed in this mode.
    """
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")

    if running_mean is None or running_var is None:
        raise RuntimeError(
            "running_mean and running_var are required for "
            "_native_batch_norm_legit_no_training"
        )

    input_3d = make_3d_for_bn(input)
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    _, feat_dim, spatial_dim = input_3d.shape
    n_slices = input_3d.shape[0] * feat_dim

    if running_mean.numel() != feat_dim or running_var.numel() != feat_dim:
        raise RuntimeError("running statistics must contain one value per channel")

    output = torch.empty_like(input_3d)
    if n_slices > 0:
        input_flat = input_3d.reshape(-1)
        output_flat = output.reshape(-1)
        has_weight = weight is not None
        has_bias = bias is not None
        with torch_device_fn.device(input.device):
            for slice_offset in range(0, n_slices, BNNU_MAX_PROGRAMS):
                slice_count = min(BNNU_MAX_PROGRAMS, n_slices - slice_offset)
                _batch_norm_no_update_kernel[(slice_count,)](
                    input_flat[slice_offset * spatial_dim :],
                    weight if has_weight else input_flat,
                    bias if has_bias else input_flat,
                    running_mean,
                    running_var,
                    output_flat[slice_offset * spatial_dim :],
                    feat_dim,
                    spatial_dim,
                    eps,
                    HAS_WEIGHT=has_weight,
                    HAS_BIAS=has_bias,
                    TILE_S=BNNU_TILE_S,
                    NEED_MASK=(spatial_dim % BNNU_TILE_S) != 0,
                    num_warps=4,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    return output.view_as(input), save_mean, save_var