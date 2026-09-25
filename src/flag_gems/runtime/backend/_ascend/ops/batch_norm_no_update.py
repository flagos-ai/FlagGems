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

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_norm_channel_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    batch_dim,
    channels,
    spatial_dim,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    channel = tl.program_id(0)
    mean = tl.load(mean_ptr + channel).to(tl.float32)
    var = tl.load(var_ptr + channel).to(tl.float32)
    inv_std = tl.rsqrt(var + eps)

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + channel).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel).to(tl.float32)
    else:
        bias = 0.0

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        batch = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
        batch_mask = batch < batch_dim
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            spatial = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial < spatial_dim
            offsets = (
                batch[:, None] * channels * spatial_dim
                + channel * spatial_dim
                + spatial[None, :]
            )
            mask = batch_mask[:, None] & spatial_mask[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            y = weight * (x - mean) * inv_std + bias
            tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _batch_norm_batch_channel_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    channels,
    spatial_dim,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    batch = task // channels
    channel = task % channels

    mean = tl.load(mean_ptr + channel).to(tl.float32)
    var = tl.load(var_ptr + channel).to(tl.float32)
    inv_std = tl.rsqrt(var + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + channel).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel).to(tl.float32)
    else:
        bias = 0.0

    base = (batch * channels + channel) * spatial_dim
    for step in range(0, tl.cdiv(spatial_dim, BLOCK)):
        spatial = step * BLOCK + tl.arange(0, BLOCK)
        mask = spatial < spatial_dim
        x = tl.load(x_ptr + base + spatial, mask=mask, other=0.0).to(tl.float32)
        y = weight * (x - mean) * inv_std + bias
        tl.store(out_ptr + base + spatial, y, mask=mask)


def batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    output = torch.empty_like(input)
    batch_dim = input.shape[0]
    channels = input.shape[1]
    spatial_dim = input.numel() // (batch_dim * channels)
    weight_arg = weight if weight is not None else input
    bias_arg = bias if bias is not None else input
    mean_arg = running_mean if running_mean is not None else input
    var_arg = running_var if running_var is not None else input

    if spatial_dim >= 65536:
        _batch_norm_batch_channel_kernel[(batch_dim * channels,)](
            input,
            weight_arg,
            bias_arg,
            mean_arg,
            var_arg,
            output,
            channels,
            spatial_dim,
            eps=eps,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            BLOCK=2048,
            num_warps=8,
        )
    else:
        block_m = min(64, triton.next_power_of_2(batch_dim))
        block_n = min(triton.next_power_of_2(spatial_dim), max(1, 1024 // block_m))
        _batch_norm_channel_kernel[(channels,)](
            input,
            weight_arg,
            bias_arg,
            mean_arg,
            var_arg,
            output,
            batch_dim,
            channels,
            spatial_dim,
            eps=eps,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=2,
        )

    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output, save_mean, save_var, reserved
