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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


# NOTE (kunlunxin / XPU perf rewrite, 2026-08-17):
# The previous flat kernel indexed per-lane channel stats as
# `channels = (offsets // INNER) % C` and loaded running_mean/running_var/weight/bias
# through that per-lane gather, which the XPU compiler lowered to slow discrete
# accesses (measured ~6-12 ms/call for the small benchmark shapes, ~0.009x speedup).
#
# In the natural [N, C, S] contiguous layout each (n, c) slice is a run of S
# CONTIGUOUS elements sharing ONE channel. So we map one program to each (n, c)
# slice (grid = N*C, same pattern as the batch_norm 3-stage normalize kernel):
# stats/affine are loaded ONCE per program as scalars, and the data tiles are
# contiguous block-DMA (masked only when S % TILE_S != 0). Measured: all benchmark
# cases drop from ~6-12 ms to ~0.06-0.15 ms. TILE_S < 64 is deliberately avoided:
# the XPU compiler miscompiles scalar+small-tile broadcast math for TILE<=32
# (wrong results, verified); TILE_S=4096 with num_warps=4 is the latency optimum.

BNNU_TILE_S = 4096
BNNU_MAX_PROGRAMS = 4096


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _batch_norm_no_update_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    running_mean_pointer,  # [C]
    running_var_pointer,  # [C]
    output_pointer,
    feat_dim,
    spatial_dim,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim

    mean = tl.load(running_mean_pointer + c).to(tl.float32)
    inv_std = rsqrt(tl.load(running_var_pointer + c).to(tl.float32) + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0

    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx,
                y.to(output_pointer.dtype.element_ty),
                mask=mask,
            )
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(
                output_pointer + base + idx, y.to(output_pointer.dtype.element_ty)
            )


def _batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-5,
):
    logger.debug("GEMS_KUNLUNXIN _BATCH_NORM_NO_UPDATE")
    if input.ndim < 2:
        raise RuntimeError("batch_norm expects input with at least 2 dimensions")
    if running_mean is None or running_var is None:
        raise RuntimeError(
            "running_mean and running_var are required for no-update batch_norm"
        )

    channels = input.shape[1]
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise RuntimeError("running statistics must contain one value per channel")

    input_contiguous = input.contiguous()
    output = torch.empty_like(input_contiguous)
    n_elements = input_contiguous.numel()
    batch_dim = input.shape[0]
    n_slices = batch_dim * channels
    inner = n_elements // n_slices if n_slices > 0 else 0

    if n_elements > 0:
        input_flat = input_contiguous.reshape(-1)
        output_flat = output.reshape(-1)
        need_mask = (inner % BNNU_TILE_S) != 0
        weight_pointer = input_flat if weight is None else weight
        bias_pointer = input_flat if bias is None else bias
        with torch_device_fn.device(input.device):
            for slice_offset in range(0, n_slices, BNNU_MAX_PROGRAMS):
                slice_count = min(BNNU_MAX_PROGRAMS, n_slices - slice_offset)
                _batch_norm_no_update_kernel[(slice_count,)](
                    input_flat[slice_offset * inner :],
                    weight_pointer,
                    bias_pointer,
                    running_mean,
                    running_var,
                    output_flat[slice_offset * inner :],
                    channels,
                    inner,
                    eps,
                    HAS_WEIGHT=weight is not None,
                    HAS_BIAS=bias is not None,
                    TILE_S=BNNU_TILE_S,
                    NEED_MASK=need_mask,
                    num_warps=4,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_var, reserved
