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
from torch import Tensor

from .batch_norm import (
    BNB_FUSED_MAX_ELEMS,
    batch_norm_backward_combine_kernel,
    batch_norm_backward_fused_kernel,
    batch_norm_backward_grad_kernel,
    batch_norm_backward_stats_kernel,
    batch_norm_reduce_partials_kernel,
    make_3d_for_bn,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _miopen_train_tile_s(spatial_dim):
    """Exact-fit XPU tile policy for the miopen backward 3-stage path (2026-09-04).

    The vendor ``_bn_train_tile_s`` uses a fixed 2048-lane (masked) tile for every
    ``S <= 2048``. Measured on P800 (micro A/B, 2026-09-04): the XPU backend vectorizes
    each program's loads only when at least 4 elements/thread are used (128-bit
    accesses), and a 512/1024-lane *unmasked-or-lightly-masked* tile is 2-10x faster
    than 128/256-lane tiles while being ~15% faster than the 2048-masked tile for the
    S=384/704/1024 shapes that dominate the miopen backward matrix:
      S=384:  187.5us (T=2048) -> 160.3us (T=512)   stats+combine+grad
      S=704:  191.1us (T=2048) -> 172.8us (T=1024)
      S=1024: 185.6us (T=2048) -> 159.7us (T=1024)
    For S > 2048 the pow2-4096 tile remains optimal (unchanged from the vendor policy).
    """
    if spatial_dim <= 0:
        return 1, False
    if spatial_dim <= 512:
        # 256+ S-runs are fully masked at 512 (validity fraction >= 0.5); 128/256-lane
        # tiles force 32-bit (scalar) accesses and cost 2-10x more per element.
        return 512, (spatial_dim % 512) != 0
    if spatial_dim <= 1024:
        return 1024, (spatial_dim % 1024) != 0
    tile = min(triton.next_power_of_2(spatial_dim), 4096)
    return tile, (spatial_dim % tile) != 0


def miopen_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    epsilon: float = 1e-05,
) -> tuple:
    """Backward pass for batch normalization (MIOpen variant) on Kunlunxin XPU.

    The MIOpen schema calls the saved inverse standard deviation argument
    ``save_var``. This override re-uses the Kunlunxin ``batch_norm_backward`` kernel
    family (see ``_kunlunxin/ops/batch_norm.py``; the generic 2D-tile
    ``batch_norm_backward_kernel`` does not lower on XPU), but with a miopen-local
    exact-fit tile policy for the contiguous 3-stage path (see
    ``_miopen_train_tile_s`` above) that measurably beats the vendor 2048-masked
    policy on the miopen benchmark matrix. Small per-channel counts keep the single
    fused (grid=C, TILE=128) launch.

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias).
    """
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_BACKWARD")
    # Natural [N, C, S] layout: NO transpose (the old vendor path paid two permuted
    # copies + stride-C discrete gathers). Each (n, c) slice is S contiguous elements.
    input_3d = make_3d_for_bn(input)  # [N, C, S]
    grad_3d = make_3d_for_bn(grad_output)  # [N, C, S]
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    if not grad_3d.is_contiguous():
        grad_3d = grad_3d.contiguous()

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim

    input_grad = torch.empty_like(input_3d)
    weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)

    if n_slices == 0 or count == 0:
        # Match torch: empty reductions yield 0-filled weight/bias grads.
        weight_grad.zero_()
        bias_grad.zero_()
        return input_grad.view_as(input), weight_grad, bias_grad

    input_flat = input_3d.reshape(-1)
    grad_flat = grad_3d.reshape(-1)
    has_weight = weight is not None

    if count <= BNB_FUSED_MAX_ELEMS:
        # Small per-channel count: single launch, grid=(C,), two streaming passes.
        with torch_device_fn.device(input.device):
            batch_norm_backward_fused_kernel[(feat_dim,)](
                grad_flat,
                input_flat,
                save_mean,
                save_var,
                weight if has_weight else input_flat,
                input_grad.reshape(-1),
                weight_grad,
                bias_grad,
                batch_dim,
                feat_dim,
                spatial_dim,
                HAS_WEIGHT=has_weight,
                IG_MASK=True,
                WG_MASK=True,
                BG_MASK=True,
                TILE_S=128,
                NEED_MASK=(spatial_dim % 128) != 0,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    else:
        # Stage 1: per-(n, c) partial (term1, term2) over the contiguous spatial run.
        tile_s, need_mask = _miopen_train_tile_s(spatial_dim)
        partial_batch_dim = (
            triton.cdiv(batch_dim, 32) * 32 if batch_dim > 32 else batch_dim
        )
        if batch_dim > 32:
            part_t1 = torch.zeros(
                partial_batch_dim * feat_dim, device=input.device, dtype=torch.float32
            )
            part_t2 = torch.zeros_like(part_t1)
        else:
            part_t1 = torch.empty(
                partial_batch_dim * feat_dim, device=input.device, dtype=torch.float32
            )
            part_t2 = torch.empty_like(part_t1)
        term1 = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        term2 = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        with torch_device_fn.device(input.device):
            max_programs = 4096
            for slice_offset in range(0, n_slices, max_programs):
                slice_count = min(max_programs, n_slices - slice_offset)
                batch_norm_backward_stats_kernel[(slice_count,)](
                    grad_flat[slice_offset * spatial_dim :],
                    input_flat[slice_offset * spatial_dim :],
                    save_mean,
                    save_var,
                    part_t1[slice_offset:],
                    part_t2[slice_offset:],
                    feat_dim,
                    spatial_dim,
                    TILE_S=tile_s,
                    NEED_MASK=need_mask,
                    num_warps=4,
                    buffer_size_limit=2048,
                    isCloseVectorization=True,
                )
            # Stage 2: reduce the batch partials -> per-channel term1 / term2.
            combine_t1 = part_t1
            combine_t2 = part_t2
            combine_batch_dim = partial_batch_dim
            while combine_batch_dim > 32:
                reduced_batch_dim = triton.cdiv(combine_batch_dim, 32)
                if reduced_batch_dim > 32:
                    storage_batch_dim = triton.cdiv(reduced_batch_dim, 32) * 32
                else:
                    storage_batch_dim = triton.next_power_of_2(reduced_batch_dim)
                reduced_t1 = torch.zeros(
                    storage_batch_dim * feat_dim,
                    device=input.device,
                    dtype=torch.float32,
                )
                reduced_t2 = torch.zeros_like(reduced_t1)
                batch_norm_reduce_partials_kernel[(reduced_batch_dim * feat_dim,)](
                    combine_t1,
                    combine_t2,
                    reduced_t1,
                    reduced_t2,
                    combine_batch_dim,
                    feat_dim,
                    TILE_N=32,
                    num_warps=4,
                    buffer_size_limit=2048,
                    isCloseVectorization=True,
                )
                combine_t1 = reduced_t1
                combine_t2 = reduced_t2
                combine_batch_dim = storage_batch_dim
            batch_norm_backward_combine_kernel[(feat_dim,)](
                combine_t1,
                combine_t2,
                term1,
                term2,
                weight_grad,
                bias_grad,
                combine_batch_dim,
                feat_dim,
                WG_MASK=True,
                BG_MASK=True,
                TILE_N=triton.next_power_of_2(combine_batch_dim),
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
            # Stage 3: per-(n, c) slice grad computation.
            input_grad_flat = input_grad.reshape(-1)
            for slice_offset in range(0, n_slices, max_programs):
                slice_count = min(max_programs, n_slices - slice_offset)
                batch_norm_backward_grad_kernel[(slice_count,)](
                    grad_flat[slice_offset * spatial_dim :],
                    input_flat[slice_offset * spatial_dim :],
                    save_mean,
                    save_var,
                    term1,
                    term2,
                    weight if has_weight else input_flat,
                    input_grad_flat[slice_offset * spatial_dim :],
                    feat_dim,
                    spatial_dim,
                    count,
                    HAS_WEIGHT=has_weight,
                    TILE_S=tile_s,
                    NEED_MASK=need_mask,
                    num_warps=4,
                    buffer_size_limit=2048,
                    isCloseVectorization=True,
                )

    return input_grad.view_as(input), weight_grad, bias_grad