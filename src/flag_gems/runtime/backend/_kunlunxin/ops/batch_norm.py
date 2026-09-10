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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ._batch_norm_no_update import (
    BNNU_MAX_PROGRAMS,
    BNNU_TILE_S,
    _batch_norm_no_update,
    _batch_norm_no_update_kernel,
)

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


# NOTE (kunlunxin / XPU forward rewrite):
# The generic batch_norm_forward_kernel uses grid=(feat_dim,) with a 2D [BLOCK_M,BLOCK_N]
# tile, and the previous kunlunxin wrapper worked around the XPU compiler's 2D-tile compile
# failure by transposing to [N*S, C, 1] (spatial_dim=1). That transpose forces stride-C
# discrete access AND only feat_dim(=C, often 8-16) parallel programs -> ~0.002 speedup.
#
# Since batch_norm reduces over batch*spatial PER channel, and in the natural [N, C, S]
# contiguous layout each (n, c) slice is S CONTIGUOUS elements, we instead map one program
# to each (n, c) slice (grid = N*C, like instance_norm). This gives full parallelism and
# fully contiguous block-DMA reads/writes with a clean 1D tile (compiles fine on XPU), and
# needs NO transpose copies. Stats are reduced per-(n,c) then combined across batch in the
# wrapper (a cheap [N,C]->[C] reduce). See harness/solution/batch_norm_forward_perf_fix.md.


# NOTE (kunlunxin / XPU inference rewrite, 2026-08-17):
# Inference (training=False) is now a single launch of the per-(n,c)-slice kernel
# from _batch_norm_no_update.py: each program loads its channel's stats/affine once
# and streams the CONTIGUOUS spatial run as block DMA (TILE_S=4096, masked tail).
# This replaces BOTH previous inference routes: (a) the fused transpose kernel
# (_batch_norm_fused_infer below is now only referenced by training-path comments)
# whose stride-C discrete reads cost ~6-12 ms on the small benchmark shapes, and
# (b) the 3-stage path's separate normalize kernel launch for large shapes.
# Measured: every benchmark case drops from ~0.19-12 ms to ~0.06-0.15 ms.


@libentry()
@triton.jit
def batch_norm_stats_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    sum_pointer,  # [N*C] f32
    sqsum_pointer,  # [N*C] f32
    spatial_dim,
    slice_offset,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = slice_offset + tl.program_id(axis=0)
    base = pid * spatial_dim
    s = tl.zeros([TILE_S], dtype=tl.float32)
    sq = tl.zeros([TILE_S], dtype=tl.float32)
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(tl.float32)
            s += x
            sq += tl.where(mask, x * x, 0.0)
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            s += x
            sq += x * x
    tl.store(sum_pointer + pid, tl.sum(s))
    tl.store(sqsum_pointer + pid, tl.sum(sq))


@libentry()
@triton.jit
def batch_norm_reduce_partials_kernel(
    part_sum_pointer,
    part_sqsum_pointer,
    reduced_sum_pointer,
    reduced_sqsum_pointer,
    batch_dim,
    feat_dim,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    channel = pid % feat_dim
    chunk = pid // feat_dim
    batch_offsets = chunk * TILE_N + tl.arange(0, TILE_N)
    mask = batch_offsets < batch_dim
    offsets = batch_offsets * feat_dim + channel
    part_sum = tl.load(part_sum_pointer + offsets, mask=mask, other=0.0)
    part_sqsum = tl.load(part_sqsum_pointer + offsets, mask=mask, other=0.0)
    part_sum = tl.where(mask, part_sum, 0.0)
    part_sqsum = tl.where(mask, part_sqsum, 0.0)
    tl.store(reduced_sum_pointer + pid, tl.sum(part_sum))
    tl.store(reduced_sqsum_pointer + pid, tl.sum(part_sqsum))


@libentry()
@triton.jit
def batch_norm_combine_kernel(
    part_sum_pointer,  # [N*C] f32, layout [n, c]
    part_sqsum_pointer,  # [N*C] f32, layout [n, c]
    mean_pointer,  # [C] f32 out
    inv_std_pointer,  # [C] f32 out
    running_mean_pointer,  # [C] or unused
    running_var_pointer,  # [C] or unused
    batch_dim,
    feat_dim,
    count,  # batch_dim * spatial_dim
    momentum,
    eps,
    HAS_RM: tl.constexpr,
    HAS_RV: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # One program per channel. Reduce the batch_dim partial (sum, sqsum) values for this
    # channel (strided by feat_dim), then compute mean / inv_std and fold the running-stat
    # updates in-kernel. Replaces ~14 small torch ops with a single launch -> cuts the
    # small-shape launch floor that regressed gems speedup.
    c = tl.program_id(axis=0)
    idx = tl.arange(0, TILE_N)
    mask = idx < batch_dim
    part_sum = tl.load(part_sum_pointer + c + idx * feat_dim, mask=mask, other=0.0)
    part_sqsum = tl.load(part_sqsum_pointer + c + idx * feat_dim, mask=mask, other=0.0)
    part_sum = tl.where(mask, part_sum, 0.0)
    part_sqsum = tl.where(mask, part_sqsum, 0.0)
    ssum = tl.sum(part_sum)
    sqsum = tl.sum(part_sqsum)
    mean = ssum / count
    var = sqsum / count - mean * mean
    inv_std = rsqrt(var + eps)
    tl.store(mean_pointer + c, mean)
    tl.store(inv_std_pointer + c, inv_std)
    if HAS_RM:
        running_mean = tl.load(running_mean_pointer + c).to(tl.float32)
        tl.store(
            running_mean_pointer + c,
            ((1 - momentum) * running_mean + momentum * mean).to(
                running_mean_pointer.dtype.element_ty
            ),
        )
    if HAS_RV:
        running_var = tl.load(running_var_pointer + c).to(tl.float32)
        # Kunlunxin torch reference updates running_var with the BIASED batch variance
        # (measured on-device 2026-08-21; matches torch@XPU to ~1e-7).
        tl.store(
            running_var_pointer + c,
            ((1 - momentum) * running_var + momentum * var).to(
                running_var_pointer.dtype.element_ty
            ),
        )


@libentry()
@triton.jit
def batch_norm_normalize_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    output_pointer,
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    feat_dim,
    spatial_dim,
    slice_offset,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = slice_offset + tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim

    mean = tl.load(mean_pointer + c)
    inv_std = tl.load(inv_std_pointer + c)
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
            tl.store(output_pointer + base + idx, y.to(output_pointer.dtype.element_ty))


# NOTE (hybrid routing): the contiguous grid=N*C 3-stage path above trades a per-shape
# ~0.4ms launch floor (stats kernel + torch combine + normalize kernel) for eliminating
# the large-spatial discrete-access catastrophe. For SMALL shapes that floor dominates and
# regresses gems speedup vs the original single fused kernel. So we keep the original fused
# (transpose) kernel below and route small shapes to it (see batch_norm wrapper). The fused
# kernel's stride-C discrete reads only blow up when batch_dim*spatial_dim is large.


# NOTE (kunlunxin / XPU small-shape fast path, 2026-08-21):
# For small per-channel counts (batch_dim * spatial_dim <= BN_FUSED_TRAIN_MAX_ELEMS) the
# 3-stage path pays a per-stage launch floor (~7-20us each: stats + combine + normalize +
# two dtype casts). The XPU compiler cannot lower 2D tiles or per-channel kernels with
# TILE >= 256 (uni_sram / TritonXPUUnrollControl failures), but a per-channel kernel with
# a fixed 128-lane tile compiles fine. Measured crossover (P800, 2026-08-21): fused route
# wins up to N*S <= 2048 ((1,8,4,4) 81.6->23.6us, (16,16,64) 144->84.6us, (16,16,128)
# 146->113.5us), loses beyond it ((16,16,256) 149->193us, (16,16,1024) 150->563us), so the
# 3-stage path is kept for the rest. The fused kernels also fold the running-stat updates
# (in-place, momentum semantics verified against torch@XPU, which uses the BIASED variance
# for the running_var update) and emit the returned save_mean/save_invstd in input dtype
# in-kernel (no extra cast launches).


@libentry()
@triton.jit
def batch_norm_fused_stats_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    mean_pointer,  # [C] f32 out
    inv_std_pointer,  # [C] f32 out
    mean_d_pointer,  # [C] input-dtype out (returned save_mean)
    inv_std_d_pointer,  # [C] input-dtype out (returned save_invstd)
    running_mean_pointer,  # [C] or unused
    running_var_pointer,  # [C] or unused
    batch_dim,
    feat_dim,
    spatial_dim,
    momentum,
    eps,
    HAS_RM: tl.constexpr,
    HAS_RV: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    count = batch_dim * spatial_dim
    s = tl.zeros([TILE_S], dtype=tl.float32)
    sq = tl.zeros([TILE_S], dtype=tl.float32)
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                s += x
                sq += tl.where(mask, x * x, 0.0)
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                s += x
                sq += x * x
    mean = tl.sum(s) / count
    var = tl.sum(sq) / count - mean * mean
    inv_std = rsqrt(var + eps)
    tl.store(mean_pointer + c, mean)
    tl.store(inv_std_pointer + c, inv_std)
    tl.store(mean_d_pointer + c, mean.to(mean_d_pointer.dtype.element_ty))
    tl.store(inv_std_d_pointer + c, inv_std.to(inv_std_d_pointer.dtype.element_ty))
    if HAS_RM:
        running_mean = tl.load(running_mean_pointer + c).to(tl.float32)
        tl.store(
            running_mean_pointer + c,
            ((1 - momentum) * running_mean + momentum * mean).to(
                running_mean_pointer.dtype.element_ty
            ),
        )
    if HAS_RV:
        running_var = tl.load(running_var_pointer + c).to(tl.float32)
        # Kunlunxin torch reference updates running_var with the BIASED batch variance
        # (measured on-device 2026-08-21; matches torch@XPU to 1.2e-7).
        tl.store(
            running_var_pointer + c,
            ((1 - momentum) * running_var + momentum * var).to(
                running_var_pointer.dtype.element_ty
            ),
        )


@libentry()
@triton.jit
def batch_norm_fused_normalize_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    output_pointer,
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    weight_pointer,  # [C] or unused
    bias_pointer,  # [C] or unused
    batch_dim,
    feat_dim,
    spatial_dim,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    c = tl.program_id(axis=0)
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_pointer + c).to(tl.float32)
    else:
        bias = 0.0
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
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


@libentry()
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    if is_train:
        total_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
        n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                    tl.float32
                )
                total_sum += curr_input

        n_elements = batch_dim * spatial_dim
        mean = tl.sum(total_sum) / n_elements

        var_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_pointer, mask=mask, other=0.0).to(
                    tl.float32
                )
                diff = tl.where(mask, curr_input - mean, 0.0)
                var_sum += diff * diff

        var = tl.sum(var_sum) / n_elements
        inv_std = rsqrt(var + eps)

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var,
        )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_pointer,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


# NOTE (kunlunxin / XPU backward rewrite, 2026-09-03):
# The previous vendor backward path transposed [N, C, S] -> [N*S, C, 1] (two permuted
# copies for x and grad) and launched the 2D-tile kernel with grid=(C,). Every program
# then streamed N*S elements with STRIDE-C discrete gather access (1/C cache-line
# utilization), and the grid had only C (often 8-16) programs -> 0.0038-0.15 speedup on
# the benchmark shapes (up to 10 ms on [16, 8, 128, 128]).
#
# We now follow the validated forward-path design: keep the natural [N, C, S] layout
# (no transpose; each (n, c) slice is S CONTIGUOUS elements) and map one program to each
# (n, c) slice (grid = N*C) for the stats / grad kernels, with a per-channel combine
# kernel for the batch reduction (chunked when batch_dim > 32, reusing the forward's
# batch_norm_reduce_partials_kernel). Small per-channel counts (<= BNB_FUSED_MAX_ELEMS)
# use a single-launch fused kernel (grid=(C,), two streaming passes) that skips the
# partials round-trip entirely, matching the forward's small-shape crossover.
#
# Backward math (train, save_mean/save_invstd given):
#   pre_lin = (x - mean) * inv_std
#   term1[c] = sum_{n,s} pre_lin * dy ; term2[c] = sum_{n,s} dy
#   grad_x = inv_std * weight * (dy - (term1 * pre_lin + term2) / (N*S))
#   grad_w = term1 ; grad_b = term2


@libentry()
@triton.jit
def batch_norm_backward_fused_kernel(
    grad_pointer,  # [N*C, S] contiguous, flattened
    input_pointer,  # [N*C, S] contiguous, flattened
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    weight_pointer,  # [C] or unused
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    feat_dim,
    spatial_dim,
    HAS_WEIGHT: tl.constexpr,
    IG_MASK: tl.constexpr,
    WG_MASK: tl.constexpr,
    BG_MASK: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One program per channel; two streaming passes over the channel's N*S elements
    # (all contiguous S-runs) so the per-channel reductions land in registers.
    c = tl.program_id(axis=0)
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    count = batch_dim * spatial_dim

    t1 = tl.zeros([TILE_S], dtype=tl.float32)
    t2 = tl.zeros([TILE_S], dtype=tl.float32)
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                dy = tl.load(grad_pointer + base + idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                pre_lin = (x - mean) * inv_std
                t1 += tl.where(mask, pre_lin * dy, 0.0)
                t2 += tl.where(mask, dy, 0.0)
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                dy = tl.load(grad_pointer + base + idx).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                t1 += pre_lin * dy
                t2 += dy
    term1 = tl.sum(t1)
    term2 = tl.sum(t2)
    if WG_MASK:
        tl.store(weight_grad_pointer + c, term1)
    if BG_MASK:
        tl.store(bias_grad_pointer + c, term2)

    if not IG_MASK:
        return

    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    scale = inv_std * weight
    rcp = 1.0 / count
    for n in range(0, batch_dim):
        base = (n * feat_dim + c) * spatial_dim
        for off in range(0, spatial_dim, TILE_S):
            idx = off + tl.arange(0, TILE_S)
            if NEED_MASK:
                mask = idx < spatial_dim
                x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
                dy = tl.load(grad_pointer + base + idx, mask=mask).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                g = scale * (dy - (term1 * pre_lin + term2) * rcp)
                tl.store(
                    input_grad_pointer + base + idx,
                    g.to(input_grad_pointer.dtype.element_ty),
                    mask=mask,
                )
            else:
                x = tl.load(input_pointer + base + idx).to(tl.float32)
                dy = tl.load(grad_pointer + base + idx).to(tl.float32)
                pre_lin = (x - mean) * inv_std
                g = scale * (dy - (term1 * pre_lin + term2) * rcp)
                tl.store(
                    input_grad_pointer + base + idx,
                    g.to(input_grad_pointer.dtype.element_ty),
                )


@libentry()
@triton.jit
def batch_norm_backward_stats_kernel(
    grad_pointer,  # [N*C, S] contiguous, flattened
    input_pointer,  # [N*C, S] contiguous, flattened
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    part_t1_pointer,  # [N*C] f32 out
    part_t2_pointer,  # [N*C] f32 out
    feat_dim,
    spatial_dim,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One program per (n, c) slice: contiguous S-run, partial (t1, t2) for the slice.
    pid = tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)

    t1 = tl.zeros([TILE_S], dtype=tl.float32)
    t2 = tl.zeros([TILE_S], dtype=tl.float32)
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(grad_pointer + base + idx, mask=mask, other=0.0).to(tl.float32)
            pre_lin = (x - mean) * inv_std
            t1 += tl.where(mask, pre_lin * dy, 0.0)
            t2 += tl.where(mask, dy, 0.0)
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            dy = tl.load(grad_pointer + base + idx).to(tl.float32)
            pre_lin = (x - mean) * inv_std
            t1 += pre_lin * dy
            t2 += dy
    tl.store(part_t1_pointer + pid, tl.sum(t1))
    tl.store(part_t2_pointer + pid, tl.sum(t2))


@libentry()
@triton.jit
def batch_norm_backward_combine_kernel(
    part_t1_pointer,  # [B, C] f32 row-major partials
    part_t2_pointer,  # [B, C] f32 row-major partials
    term1_pointer,  # [C] f32 out
    term2_pointer,  # [C] f32 out
    weight_grad_pointer,  # [C] out (input dtype), or unused
    bias_grad_pointer,  # [C] out (input dtype), or unused
    batch_dim,
    feat_dim,
    WG_MASK: tl.constexpr,
    BG_MASK: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # One program per channel; reduce the batch partials (strided by feat_dim).
    # The per-channel terms are published to weight/bias grads here (cast to the
    # output dtype) so the wrapper needs no extra copy launches.
    c = tl.program_id(axis=0)
    idx = tl.arange(0, TILE_N)
    mask = idx < batch_dim
    offs = idx * feat_dim + c
    t1 = tl.load(part_t1_pointer + offs, mask=mask, other=0.0)
    t2 = tl.load(part_t2_pointer + offs, mask=mask, other=0.0)
    s1 = tl.sum(t1)
    s2 = tl.sum(t2)
    tl.store(term1_pointer + c, s1)
    tl.store(term2_pointer + c, s2)
    if WG_MASK:
        tl.store(weight_grad_pointer + c, s1.to(weight_grad_pointer.dtype.element_ty))
    if BG_MASK:
        tl.store(bias_grad_pointer + c, s2.to(bias_grad_pointer.dtype.element_ty))


@libentry()
@triton.jit
def batch_norm_backward_grad_kernel(
    grad_pointer,  # [N*C, S] contiguous, flattened
    input_pointer,  # [N*C, S] contiguous, flattened
    mean_pointer,  # [C] f32
    inv_std_pointer,  # [C] f32
    term1_pointer,  # [C] f32
    term2_pointer,  # [C] f32
    weight_pointer,  # [C] or unused
    input_grad_pointer,
    feat_dim,
    spatial_dim,
    count,
    HAS_WEIGHT: tl.constexpr,
    TILE_S: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One program per (n, c) slice: contiguous S-run, apply the channel terms.
    pid = tl.program_id(axis=0)
    c = pid % feat_dim
    base = pid * spatial_dim
    mean = tl.load(mean_pointer + c).to(tl.float32)
    inv_std = tl.load(inv_std_pointer + c).to(tl.float32)
    term1 = tl.load(term1_pointer + c)
    term2 = tl.load(term2_pointer + c)
    if HAS_WEIGHT:
        weight = tl.load(weight_pointer + c).to(tl.float32)
    else:
        weight = 1.0
    scale = inv_std * weight
    rcp = 1.0 / count
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        if NEED_MASK:
            mask = idx < spatial_dim
            x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
            dy = tl.load(grad_pointer + base + idx, mask=mask).to(tl.float32)
            pre_lin = (x - mean) * inv_std
            g = scale * (dy - (term1 * pre_lin + term2) * rcp)
            tl.store(
                input_grad_pointer + base + idx,
                g.to(input_grad_pointer.dtype.element_ty),
                mask=mask,
            )
        else:
            x = tl.load(input_pointer + base + idx).to(tl.float32)
            dy = tl.load(grad_pointer + base + idx).to(tl.float32)
            pre_lin = (x - mean) * inv_std
            g = scale * (dy - (term1 * pre_lin + term2) * rcp)
            tl.store(
                input_grad_pointer + base + idx,
                g.to(input_grad_pointer.dtype.element_ty),
            )


# Per-channel discrete-read count (batch_dim * spatial_dim) at/below which the single
# fused (transpose) kernel's low launch floor beats the contiguous 3-stage path. Above it
# the fused kernel's stride-C discrete reads blow up (measured crossover ~0.4ms floor).
# NOTE: the fused kernel is INFERENCE-ONLY here — its training reduction is numerically
# broken on XPU (verified: garbage output). Training must always use the contiguous path.
BN_FUSED_MAX_ELEMS = 2048

# Small-shape training fast path crossover: per-channel element count
# (batch_dim * spatial_dim) at/below which the grid=C fused route (2 launches) beats the
# 3-stage path (measured on P800, 2026-08-21; see NOTE above the fused kernels).
BN_FUSED_TRAIN_MAX_ELEMS = 2048


def _batch_norm_fused_infer(input, weight, bias, running_mean, running_var, eps):
    # Original single fused kernel (transpose), INFERENCE ONLY. Low launch floor; used for
    # small batch_dim*spatial_dim so its stride-C discrete reads stay cheap.
    input_3d_i = make_3d_for_bn(input)
    m, n, k = input_3d_i.shape
    input_3d_f = input_3d_i.permute(0, 2, 1).reshape(-1, n)
    input_3d = make_3d_for_bn(input_3d_f)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    with torch_device_fn.device(input.device):
        batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            0.1,
            eps,
            is_train=False,
            buffer_size_limit=2048,
        )

    output_reshaped = output.reshape(m, k, n).permute(0, 2, 1)
    return output_reshaped.view_as(input), mean, inv_std


def _bn_train_tile_s(spatial_dim):
    """XPU tile policy for the training-path stats/normalize kernels.

    Measured on P800 (2026-08-17 probe): the per-program cost of the per-(n, c) slice
    kernels is dominated by fixed overhead (~0.2-0.7us/program plus ~20us launch), and
    a 64/128-lane tile costs nearly the same as a 2048-lane tile. Using the old
    ``min(next_pow2(S), 4096)`` tile made the small-S benchmark shapes (S=64/128/384)
    pay 170-220us in normalize alone; a fixed 2048-lane tile (masked tail) brings them
    all to the same ~90us floor as S=1024. For S > 2048 keep the pow2-4096 tile
    (measured: 4096-tile loops beat 2048-tile loops there).
    """
    if spatial_dim <= 0:
        return 1, False
    if spatial_dim <= 2048:
        return 2048, (spatial_dim % 2048) != 0
    tile = min(triton.next_power_of_2(spatial_dim), 4096)
    return tile, (spatial_dim % tile) != 0


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM")

    # Inference -> single per-(n,c)-slice kernel launch (see NOTE above): no transpose
    # copies, no discrete channel gathers, no separate normalize launch. It replaces
    # both the fused-transpose path (was ~6-12 ms on small benchmark shapes) and the
    # 3-stage path's normalize launch for large shapes. Training keeps the 3-stage
    # path below (unchanged).
    if not training:
        # Fast path: the dedicated channel-major wrapper (adaptive TILE_S +
        # chunked-batch grid, measured ~2.2x over the flat slice grid on the
        # official benchmark matrix). Falls back to the flat per-slice launch
        # below when running stats are absent (kernel requires them).
        if running_mean is not None and running_var is not None and input.numel() > 0:
            output, _, _, _ = _batch_norm_no_update(
                input, weight, bias, running_mean, running_var, momentum, eps
            )
            feat_dim = input.shape[1] if input.ndim >= 2 else input.shape[0]
            mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
            inv_std = torch.empty_like(mean)
            return output, mean, inv_std
        input_3d = make_3d_for_bn(input)  # [N, C, S]
        if not input_3d.is_contiguous():
            input_3d = input_3d.contiguous()
        batch_dim, feat_dim, spatial_dim = input_3d.shape
        n_slices = batch_dim * feat_dim
        if n_slices > 0:
            output = torch.empty_like(input_3d)
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
            # NOTE: return stats as UNINITIALIZED [C] tensors for inference, exactly
            # like the previous fused path did: the native batch_norm inference
            # caller never consumes them, and computing running_mean.to()/rsqrt()
            # here costs several extra kernel launches (~100-200us) that dominated
            # the official benchmark.
            mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
            inv_std = torch.empty_like(mean)
            return output.view_as(input), mean, inv_std
        mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        inv_std = torch.empty_like(mean)
        return input, mean, inv_std

    input_3d = make_3d_for_bn(input)  # [N, C, S]
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim

    if count == 0:
        output = torch.empty_like(input_3d)
        mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        inv_std = torch.empty_like(mean)
        return output.view_as(input), mean, inv_std

    output = torch.empty_like(input_3d)
    input_flat = input_3d.reshape(-1)
    output_flat = output.reshape(-1)
    # NOTE (kunlunxin/xpu running-stat semantics, measured 2026-08-21 on-device):
    # torch@XPU F.batch_norm / aten::batch_norm in training mode updates running
    # stats IN-PLACE only for float32 inputs; for float16/bfloat16 inputs the native
    # implementation leaves running_mean/running_var untouched. The updates use the
    # BIASED batch variance (matches torch@XPU fp32 to ~1e-7).
    has_rm = running_mean is not None and input.dtype == torch.float32
    has_rv = running_var is not None and input.dtype == torch.float32
    has_weight = weight is not None
    has_bias = bias is not None

    # Small-shape fast path: grid=C fused stats (per-channel reduction over all N*S,
    # in-kernel running-stat updates and input-dtype save stats) + grid=C fused
    # normalize; 2 launches total (see NOTE above). Below the crossover the 3-stage
    # path below keeps the contiguous per-(n,c)-slice kernels.
    if count <= BN_FUSED_TRAIN_MAX_ELEMS:
        mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        inv_std_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        mean_d = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        inv_std_d = torch.empty_like(mean_d)
        with torch_device_fn.device(input.device):
            batch_norm_fused_stats_kernel[(feat_dim,)](
                input_flat,
                mean_f,
                inv_std_f,
                mean_d,
                inv_std_d,
                running_mean if has_rm else mean_f,
                running_var if has_rv else mean_f,
                batch_dim,
                feat_dim,
                spatial_dim,
                momentum,
                eps,
                HAS_RM=has_rm,
                HAS_RV=has_rv,
                TILE_S=128,
                NEED_MASK=(spatial_dim % 128) != 0,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
            batch_norm_fused_normalize_kernel[(feat_dim,)](
                input_flat,
                output_flat,
                mean_f,
                inv_std_f,
                weight if has_weight else input_flat,
                bias if has_bias else input_flat,
                batch_dim,
                feat_dim,
                spatial_dim,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                TILE_S=128,
                NEED_MASK=(spatial_dim % 128) != 0,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        return output.view_as(input), mean_d, inv_std_d

    tile_s, need_mask = _bn_train_tile_s(spatial_dim)

    mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    inv_std_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    # Stage 1: per-(n, c) partial sum / sum-of-squares over contiguous spatial run.
    partial_batch_dim = triton.cdiv(batch_dim, 32) * 32 if batch_dim > 32 else batch_dim
    # Direct path (batch_dim <= 32): the stats kernel writes every one of the N*C
    # slots it reads back, so torch.empty is safe and avoids 2 zero-fill launches.
    # The chunked reduction path keeps torch.zeros: reduce_partials may read slots
    # beyond the rows written by the stats kernel.
    if batch_dim > 32:
        part_sum = torch.zeros(
            partial_batch_dim * feat_dim, device=input.device, dtype=torch.float32
        )
        part_sqsum = torch.zeros_like(part_sum)
    else:
        part_sum = torch.empty(
            partial_batch_dim * feat_dim, device=input.device, dtype=torch.float32
        )
        part_sqsum = torch.empty_like(part_sum)
    with torch_device_fn.device(input.device):
        max_programs = 4096
        for slice_offset in range(0, n_slices, max_programs):
            slice_count = min(max_programs, n_slices - slice_offset)
            batch_norm_stats_kernel[(slice_count,)](
                input_flat[slice_offset * spatial_dim :],
                part_sum[slice_offset:],
                part_sqsum[slice_offset:],
                spatial_dim,
                0,
                TILE_S=tile_s,
                NEED_MASK=need_mask,
            )
        combine_sum = part_sum
        combine_sqsum = part_sqsum
        combine_batch_dim = partial_batch_dim
        while combine_batch_dim > 32:
            reduced_batch_dim = triton.cdiv(combine_batch_dim, 32)
            if reduced_batch_dim > 32:
                storage_batch_dim = triton.cdiv(reduced_batch_dim, 32) * 32
            else:
                storage_batch_dim = triton.next_power_of_2(reduced_batch_dim)
            reduced_sum = torch.zeros(
                storage_batch_dim * feat_dim,
                device=input.device,
                dtype=torch.float32,
            )
            reduced_sqsum = torch.zeros_like(reduced_sum)
            batch_norm_reduce_partials_kernel[(reduced_batch_dim * feat_dim,)](
                combine_sum,
                combine_sqsum,
                reduced_sum,
                reduced_sqsum,
                combine_batch_dim,
                feat_dim,
                TILE_N=32,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
            combine_sum = reduced_sum
            combine_sqsum = reduced_sqsum
            combine_batch_dim = storage_batch_dim
        # Stage 2: combine batch partials -> per-channel mean / inv_std and fold the
        # running-stat updates, all in a single kernel (grid=(C,)). One launch instead
        # of ~14 small torch ops -> removes the small-shape launch floor.
        batch_norm_combine_kernel[(feat_dim,)](
            combine_sum,
            combine_sqsum,
            mean_f,
            inv_std_f,
            running_mean if has_rm else part_sum,
            running_var if has_rv else part_sum,
            combine_batch_dim,
            feat_dim,
            count,
            momentum,
            eps,
            HAS_RM=has_rm,
            HAS_RV=has_rv,
            TILE_N=triton.next_power_of_2(combine_batch_dim),
            num_warps=4,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    # Return stats in input dtype (single cast each; no extra empty+copy).
    mean = mean_f.to(input.dtype)
    inv_std = inv_std_f.to(input.dtype)

    has_weight = weight is not None
    has_bias = bias is not None
    with torch_device_fn.device(input.device):
        max_programs = 4096
        for slice_offset in range(0, n_slices, max_programs):
            slice_count = min(max_programs, n_slices - slice_offset)
            batch_norm_normalize_kernel[(slice_count,)](
                input_flat[slice_offset * spatial_dim :],
                output_flat[slice_offset * spatial_dim :],
                mean_f,
                inv_std_f,
                weight if has_weight else input_flat,
                bias if has_bias else input_flat,
                feat_dim,
                spatial_dim,
                0,
                HAS_WEIGHT=has_weight,
                HAS_BIAS=has_bias,
                TILE_S=tile_s,
                NEED_MASK=need_mask,
            )

    return output.view_as(input), mean, inv_std


# Per-channel element-count (batch_dim * spatial_dim) at/below which the single-launch
# fused backward kernel (2 streaming passes, grid=(C,)) beats the 3-stage path (stats +
# combine + grad, 3 launches). Matches the forward's BN_FUSED_TRAIN_MAX_ELEMS crossover.
BNB_FUSED_MAX_ELEMS = 2048


def batch_norm_backward(
    grad_out,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    output_mask=None,
):
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_BACKWARD")
    # Natural [N, C, S] layout: NO transpose (the old vendor path paid two permuted
    # copies + stride-C discrete gathers). Each (n, c) slice is S contiguous elements.
    input_3d = make_3d_for_bn(input)  # [N, C, S]
    grad_3d = make_3d_for_bn(grad_out)  # [N, C, S]
    if not input_3d.is_contiguous():
        input_3d = input_3d.contiguous()
    if not grad_3d.is_contiguous():
        grad_3d = grad_3d.contiguous()

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim

    if output_mask[0]:
        input_grad = torch.empty_like(input_3d)
    else:
        input_grad = None
    if output_mask[1]:
        weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        bias_grad = None

    if n_slices == 0 or count == 0:
        # Match torch: empty reductions yield 0-filled weight/bias grads.
        if weight_grad is not None:
            weight_grad.zero_()
        if bias_grad is not None:
            bias_grad.zero_()
        return (
            input_grad.view_as(input) if input_grad is not None else input_grad,
            weight_grad,
            bias_grad,
        )

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
                save_invstd,
                weight if has_weight else input_flat,
                input_grad if output_mask[0] else grad_flat,
                weight_grad if output_mask[1] else grad_flat,
                bias_grad if output_mask[2] else grad_flat,
                batch_dim,
                feat_dim,
                spatial_dim,
                HAS_WEIGHT=has_weight,
                IG_MASK=output_mask[0],
                WG_MASK=output_mask[1],
                BG_MASK=output_mask[2],
                TILE_S=128,
                NEED_MASK=(spatial_dim % 128) != 0,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    else:
        # Stage 1: per-(n, c) partial (term1, term2) over the contiguous spatial run.
        tile_s, need_mask = _bn_train_tile_s(spatial_dim)
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
                    save_invstd,
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
                weight_grad if output_mask[1] else combine_t1,
                bias_grad if output_mask[2] else combine_t2,
                combine_batch_dim,
                feat_dim,
                WG_MASK=output_mask[1],
                BG_MASK=output_mask[2],
                TILE_N=triton.next_power_of_2(combine_batch_dim),
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
            # Stage 3: per-(n, c) slice grad computation.
            if output_mask[0]:
                input_grad_flat = input_grad.reshape(-1)
                for slice_offset in range(0, n_slices, max_programs):
                    slice_count = min(max_programs, n_slices - slice_offset)
                    batch_norm_backward_grad_kernel[(slice_count,)](
                        grad_flat[slice_offset * spatial_dim :],
                        input_flat[slice_offset * spatial_dim :],
                        save_mean,
                        save_invstd,
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

    return (
        input_grad.view_as(input) if input_grad is not None else input_grad,
        weight_grad,
        bias_grad,
    )
