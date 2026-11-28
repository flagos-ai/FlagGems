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

# GCU400 has limited vector registers, so a single tile is capped at 8K elements; the upstream 16K tile here
# significantly increases register pressure and also makes autotune repeatedly rewrite running stats.
MAX_TILE_ELEMS = 16384
# t8 batch Step 4 A02: 8192 gives BLOCK_N=512 for (16,8,128,128), with 32 iterations per CTA;
# 16384 halves the loop to 16 iterations; measured 0.2935ms -> 0.0721ms (4.07x).
# 32768 and above trigger the dsm memory hardware limit (1050624 > 917504) and are excluded.
# The Welford online accumulation order changes accordingly; the difference is at the ULP level (float32 <=4.77e-7).


def _pick_tile(batch_dim: int, spatial_dim: int):
    """Picks (BLOCK_M, BLOCK_N): BLOCK_M = next_pow2(batch), BLOCK_N fills the tile budget.

    R02 tried a mask-waste-minimizing enumeration (BLOCK_M=1/4 at batch=20) and measured
    fp16/bf16 on (20,6,65536) degrading from 0.84ms to 1.30ms (a large BLOCK_M 2D tile
    vectorizes better for 2-byte types), so the original policy is kept.
    """
    block_m = min(triton.next_power_of_2(batch_dim), MAX_TILE_ELEMS)
    block_n = min(
        triton.next_power_of_2(spatial_dim),
        max(1, MAX_TILE_ELEMS // block_m),
    )
    return block_m, block_n


def make_3d_for_bn(input: Tensor) -> Tensor:
    """Folds the input into an (N, C, HxW) view."""
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
    # Dimensions and strides are constexpr: when int64 support is enabled, GCU promotes runtime int
    # scalars to i64, and comparing them with the i32 produced by make_range crashes the compiler.
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
                curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                    tl.float32
                )

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
                (1 - momentum) * running_var + momentum * variance * unbiased_scale,
            )
    else:
        mean_value = tl.load(RunningMean + feat_pid).to(tl.float32)
        inv_std = tl.math.rsqrt(tl.load(RunningVar + feat_pid).to(tl.float32) + eps)

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


# ---------------------------------------------------------------------------
# Two-stage spatial-split path for large shapes
# ---------------------------------------------------------------------------
# The single-kernel version has a grid of only (feat_dim,): at feat=6/8, only
# 12~17% of the 48 CTA slots are used; measured effective bandwidth for (20,6,65536) fp32 is only ~96 GB/s. Here the
# spatial dim is split into
# n_splits segments:
#   K1: each (feat, split) computes the local Welford (mean, M2) for its segment and writes it to scratch;
#   K2: each (feat, split) first uses the Chan parallel merge formula to combine the n_splits local values for that feat
#       into the overall mean/variance (only n_splits scalars, so redundant reads are negligible), then normalizes
#       the segment it owns.
# Total traffic is still 3N (read N + read N + write N), same as the single-kernel version, but parallelism rises to ~48
# CTAs.
SPLIT_MIN_ELEMS = 4_000_000  # Below 4M elements, the cost of an extra launch plus two reads does not pay off
SPLIT_TARGET_CTAS = 48  # Number of CTA slots on GCU400
SPLIT_MAX_FEAT_DIM = (
    8  # Once feat_dim exceeds 8, the single-kernel grid already fills all CTA slots
)


@libentry()
@triton.jit
def _bn_split_partial_stats_kernel(
    Input,
    PartMean,
    PartM2,
    # All dims/strides are constexpr: mixing i32/i64 under GCU's int64 support crashes the compiler
    batch_dim: tl.constexpr,
    spatial_dim: tl.constexpr,
    input_batch_stride: tl.constexpr,
    input_feat_stride: tl.constexpr,
    input_spatial_stride: tl.constexpr,
    chunk: tl.constexpr,
    n_splits: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(0)
    split_pid = tl.program_id(1)
    # The window for this segment is [split*chunk, split*chunk + chunk); out-of-bounds parts are clipped by the absolute
    # spatial mask
    n_base = split_pid * chunk
    m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
    n_num_steps = tl.cdiv(chunk, BLOCK_N)

    mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for m_step in range(0, m_num_steps):
        for n_step in range(0, n_num_steps):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            spatial_offset = n_base + n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (batch_offset[:, None] < batch_dim) & (
                spatial_offset[None, :] < spatial_dim
            )
            curr_input_pointer = (
                Input
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                tl.float32
            )
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
    segment_count = tl.sum(count)
    local_mean = tl.sum(mean * count) / segment_count
    local_m2 = tl.sum(var + count * (mean - local_mean) * (mean - local_mean))
    scratch_index = feat_pid * n_splits + split_pid
    tl.store(PartMean + scratch_index, local_mean)
    tl.store(PartM2 + scratch_index, local_m2)


@libentry()
@triton.jit(do_not_specialize=["momentum", "eps", "unbiased_scale"])
def _bn_split_finalize_norm_kernel(
    Input,
    Weight,
    Bias,
    Mean,
    InvStd,
    Output,
    RunningMean,
    RunningVar,
    PartMean,
    PartM2,
    momentum,
    eps,
    unbiased_scale,
    batch_dim: tl.constexpr,
    spatial_dim: tl.constexpr,
    input_batch_stride: tl.constexpr,
    input_feat_stride: tl.constexpr,
    input_spatial_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    output_feat_stride: tl.constexpr,
    output_spatial_stride: tl.constexpr,
    chunk: tl.constexpr,
    n_splits: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    has_running_stats: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    feat_pid = tl.program_id(0)
    split_pid = tl.program_id(1)

    # --- Chan parallel merge: combine the (mean, M2) of n_splits segments into global statistics ---
    s_offs = tl.arange(0, BLOCK_S)
    s_valid = s_offs < n_splits
    part_mean = tl.load(
        PartMean + feat_pid * n_splits + s_offs, mask=s_valid, other=0.0
    )
    part_m2 = tl.load(PartM2 + feat_pid * n_splits + s_offs, mask=s_valid, other=0.0)
    # Actual sample count for segment s = spatial length within the segment (possibly truncated by spatial_dim) × batch
    segment_len = tl.minimum(spatial_dim - s_offs * chunk, chunk).to(tl.float32)
    segment_count = segment_len * batch_dim
    segment_count = tl.where(s_valid & (segment_len > 0), segment_count, 0.0)
    total_count = tl.sum(segment_count)
    mean_value = tl.sum(part_mean * segment_count) / total_count
    total_m2 = tl.sum(
        part_m2 + segment_count * (part_mean - mean_value) * (part_mean - mean_value)
    )
    variance = total_m2 / total_count
    inv_std = tl.math.rsqrt(variance + eps)

    # Statistics computed by each split depend only on feat and are identical, so unconditional Mean/InvStd writes are
    # safe.
    tl.store(Mean + feat_pid, mean_value)
    tl.store(InvStd + feat_pid, inv_std)
    if has_running_stats:
        # running stats follow "read old value → write new value"; if every split did this, a later-writing CTA would
        # read
        # an already-updated value, equivalent to applying momentum one extra time (measured relative error 0.9).
        # Therefore only
        # split 0 performs it; implemented with a 1-element vector + mask to avoid a runtime if branch.
        head_slot = tl.arange(0, 1)
        head_mask = (split_pid == 0) & (head_slot < 1)
        running_mean = tl.load(
            RunningMean + feat_pid + head_slot, mask=head_mask, other=0.0
        ).to(tl.float32)
        running_var = tl.load(
            RunningVar + feat_pid + head_slot, mask=head_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            RunningMean + feat_pid + head_slot,
            (1 - momentum) * running_mean + momentum * mean_value,
            mask=head_mask,
        )
        tl.store(
            RunningVar + feat_pid + head_slot,
            (1 - momentum) * running_var + momentum * variance * unbiased_scale,
            mask=head_mask,
        )

    if has_weight:
        weight = tl.load(Weight + feat_pid).to(tl.float32)
    else:
        weight = 1.0
    if has_bias:
        bias = tl.load(Bias + feat_pid).to(tl.float32)
    else:
        bias = 0.0

    n_base = split_pid * chunk
    m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
    n_num_steps = tl.cdiv(chunk, BLOCK_N)
    for m_step in range(0, m_num_steps):
        for n_step in range(0, n_num_steps):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            spatial_offset = n_base + n_step * BLOCK_N + tl.arange(0, BLOCK_N)
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


def _plan_split(feat_dim: int, spatial_dim: int, block_n: int):
    """Plans spatial chunks from the CTA slot budget, returning (chunk, n_splits)."""
    n_blocks = triton.cdiv(spatial_dim, block_n)
    budget = max(1, SPLIT_TARGET_CTAS // feat_dim)
    # Take the largest power of 2 that stays within budget and still gives each segment at least one BLOCK_N
    n_splits = 1
    while n_splits * 2 <= min(budget, n_blocks):
        n_splits *= 2
    n_splits = min(n_splits, n_blocks)
    # Round chunk up to a multiple of BLOCK_N so the loop trip count inside the kernel is a compile-time constant
    chunk = triton.cdiv(n_blocks, n_splits) * block_n
    return chunk, n_splits


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
    """GCU400 functional batch norm, returning the output / save_mean / save_var and other five-tuple values."""
    logger.debug("GEMS_ENFLAME _NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")

    input_3d = make_3d_for_bn(input)
    batch_dim, feat_dim, spatial_dim = input_3d.shape

    output = torch.empty_like(input_3d)
    mean = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    inv_std = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)

    has_weight = weight is not None
    has_bias = bias is not None
    has_running_stats = running_mean is not None and running_var is not None

    # Optional pointers are isolated by compile-time flags; here the kernel only receives one valid placeholder tensor.
    weight_pointer = weight if has_weight else mean
    bias_pointer = bias if has_bias else inv_std
    if not has_running_stats:
        running_mean = mean
        running_var = inv_std

    num_samples = batch_dim * spatial_dim
    unbiased_scale = num_samples / (num_samples - 1) if num_samples > 1 else 1.0

    block_m, block_n = _pick_tile(batch_dim, spatial_dim)

    # The two-stage path is worthwhile only when there are few features (CTAs are severely starved) and the element
    # count is large enough:
    #   * feat_dim <= SPLIT_MAX_FEAT_DIM: with many feats the single-kernel grid already fills 48 CTAs;
    #   * numel >= SPLIT_MIN_ELEMS: otherwise the extra launch does not pay off;
    #   * spatial must be divisible by chunk: otherwise the last segment has a partially masked tile, so lanes have
    # unequal sample counts,
    #     introducing a ~1.4e-4 variance bias (fp32 rtol=1.3e-6 would be violated);
    #   * n_splits >= 2: with only one segment there is no parallelism gain, just a wasted launch.
    use_split_path = (
        training
        and feat_dim <= SPLIT_MAX_FEAT_DIM
        and input_3d.numel() >= SPLIT_MIN_ELEMS
    )
    if use_split_path:
        chunk, n_splits = _plan_split(feat_dim, spatial_dim, block_n)
        use_split_path = n_splits >= 2 and spatial_dim % chunk == 0

    with torch_device_fn.device(input.device):
        if use_split_path:
            part_mean = torch.empty(
                (feat_dim * n_splits,), dtype=torch.float32, device=input.device
            )
            part_m2 = torch.empty(
                (feat_dim * n_splits,), dtype=torch.float32, device=input.device
            )
            _bn_split_partial_stats_kernel[(feat_dim, n_splits)](
                input_3d,
                part_mean,
                part_m2,
                batch_dim,
                spatial_dim,
                *input_3d.stride(),
                chunk,
                n_splits,
                block_m,
                block_n,
                num_warps=4,
            )
            _bn_split_finalize_norm_kernel[(feat_dim, n_splits)](
                input_3d,
                weight_pointer,
                bias_pointer,
                mean,
                inv_std,
                output,
                running_mean,
                running_var,
                part_mean,
                part_m2,
                momentum,
                eps,
                unbiased_scale,
                batch_dim,
                spatial_dim,
                *input_3d.stride(),
                *output.stride(),
                chunk,
                n_splits,
                has_weight,
                has_bias,
                has_running_stats,
                block_m,
                block_n,
                triton.next_power_of_2(n_splits),
                num_warps=4,
            )
        else:
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

    # PyTorch's save_var is actually inv_std = 1 / sqrt(var + eps), not the variance.
    save_var = inv_std
    return (
        output.view_as(input),
        mean,
        save_var,
        running_mean,
        running_var,
    )
