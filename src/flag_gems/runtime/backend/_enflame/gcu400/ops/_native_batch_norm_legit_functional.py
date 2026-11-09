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
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 的向量寄存器有限，单个 tile 控制在 8K 元素；公版的 16K tile 在这里
# 会显著增加寄存器压力，也让 autotune 反复改写 running stats。
MAX_TILE_ELEMS = 16384
# t8 批 Step 4 A02：8192 让 (16,8,128,128) 的 BLOCK_N=512、每 CTA 循环 32 次；
# 16384 把循环减半到 16 次，实测 0.2935ms -> 0.0721ms（4.07x）。
# 32768 及以上会触发 dsm memory 硬件上限（1050624 > 917504），已排除。
# Welford 在线累加顺序随之改变，差异为 ULP 级（float32 <=4.77e-7）。


def make_3d_for_bn(input: Tensor) -> Tensor:
    """把输入折叠成 (N, C, HxW) 视图。"""
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


@libentry()
@triton.jit(do_not_specialize=["momentum", "eps", "unbiased_scale"])
def _batch_norm_forward_kernel(
    Input,
    Weight,
    Bias,
    Mean,
    InvStd,
    Output,
    RunningMean,
    RunningVar,
    momentum,
    eps,
    unbiased_scale,
    # 维度和 stride 都做成 constexpr：GCU 在开启 int64 支持时会把运行时 int
    # 标量提升为 i64，与 make_range 产出的 i32 混合比较会让编译器崩溃。
    batch_dim: tl.constexpr,
    spatial_dim: tl.constexpr,
    input_batch_stride: tl.constexpr,
    input_feat_stride: tl.constexpr,
    input_spatial_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    output_feat_stride: tl.constexpr,
    output_spatial_stride: tl.constexpr,
    is_train: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    has_running_stats: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(0)
    num_samples = batch_dim * spatial_dim
    m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
    n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

    if is_train:
        mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = (batch_offset[:, None] < batch_dim) & (
                    spatial_offset[None, :] < spatial_dim
                )
                curr_input_pointer = (
                    Input
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )
                curr_input = tl.load(
                    curr_input_pointer, mask=mask, other=0.0
                ).to(tl.float32)

                step = m_step * n_num_steps + n_step + 1
                new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
                new_var = tl.where(
                    mask,
                    var + (curr_input - new_mean) * (curr_input - mean),
                    var,
                )
                cnt += mask.to(tl.int32)
                mean = new_mean
                var = new_var

        count = cnt.to(tl.float32)
        final_mean = tl.sum(mean * count) / num_samples
        variance = tl.sum(var + count * (mean - final_mean) * (mean - final_mean)) / (
            num_samples
        )
        inv_std = tl.math.rsqrt(variance + eps)
        mean_value = final_mean

        tl.store(Mean + feat_pid, mean_value)
        tl.store(InvStd + feat_pid, inv_std)

        if has_running_stats:
            running_mean = tl.load(RunningMean + feat_pid).to(tl.float32)
            running_var = tl.load(RunningVar + feat_pid).to(tl.float32)
            tl.store(
                RunningMean + feat_pid,
                (1 - momentum) * running_mean + momentum * mean_value,
            )
            tl.store(
                RunningVar + feat_pid,
                (1 - momentum) * running_var
                + momentum * variance * unbiased_scale,
            )
    else:
        mean_value = tl.load(RunningMean + feat_pid).to(tl.float32)
        inv_std = tl.math.rsqrt(
            tl.load(RunningVar + feat_pid).to(tl.float32) + eps
        )

    if has_weight:
        weight = tl.load(Weight + feat_pid).to(tl.float32)
    else:
        weight = 1.0
    if has_bias:
        bias = tl.load(Bias + feat_pid).to(tl.float32)
    else:
        bias = 0.0

    for m_step in range(0, m_num_steps):
        for n_step in range(0, n_num_steps):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (batch_offset[:, None] < batch_dim) & (
                spatial_offset[None, :] < spatial_dim
            )
            curr_input_pointer = (
                Input
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                Output
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )
            curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                tl.float32
            )
            output = weight * (curr_input - mean_value) * inv_std + bias
            tl.store(curr_output_pointer, output, mask=mask)


def _native_batch_norm_legit_functional(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    """GCU400 的 functional batch norm，返回 output/save_mean/save_var 等五元组。"""
    logger.debug("GEMS_ENFLAME _NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")

    input_3d = make_3d_for_bn(input)
    batch_dim, feat_dim, spatial_dim = input_3d.shape

    output = torch.empty_like(input_3d)
    mean = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    inv_std = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)

    has_weight = weight is not None
    has_bias = bias is not None
    has_running_stats = running_mean is not None and running_var is not None

    # 可选指针用编译期 flag 隔离，这里只给 kernel 一个合法的占位 tensor。
    weight_pointer = weight if has_weight else mean
    bias_pointer = bias if has_bias else inv_std
    if not has_running_stats:
        running_mean = mean
        running_var = inv_std

    num_samples = batch_dim * spatial_dim
    unbiased_scale = (
        num_samples / (num_samples - 1) if num_samples > 1 else 1.0
    )

    block_m = min(triton.next_power_of_2(batch_dim), MAX_TILE_ELEMS)
    block_n = min(
        triton.next_power_of_2(spatial_dim),
        max(1, MAX_TILE_ELEMS // block_m),
    )

    with torch_device_fn.device(input.device):
        _batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            weight_pointer,
            bias_pointer,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            momentum,
            eps,
            unbiased_scale,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            training,
            has_weight,
            has_bias,
            has_running_stats,
            block_m,
            block_n,
            num_warps=4,
        )

    # PyTorch 的 save_var 其实是 inv_std = 1 / sqrt(var + eps)，不是方差。
    save_var = inv_std
    return (
        output.view_as(input),
        mean,
        save_var,
        running_mean,
        running_var,
    )
