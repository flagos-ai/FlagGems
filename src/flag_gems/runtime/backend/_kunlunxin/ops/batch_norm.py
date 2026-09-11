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


@libentry()
@triton.jit
def batch_norm_stats_kernel(
    input_pointer,  # [N*C, S] contiguous, flattened
    sum_pointer,  # [N*C] f32
    sqsum_pointer,  # [N*C] f32
    spatial_dim,
    TILE_S: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (n, c) slice
    base = pid * spatial_dim
    s = tl.zeros([TILE_S], dtype=tl.float32)
    sq = tl.zeros([TILE_S], dtype=tl.float32)
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        mask = idx < spatial_dim
        x = tl.load(input_pointer + base + idx, mask=mask, other=0.0).to(tl.float32)
        s += x
        sq += tl.where(mask, x * x, 0.0)
    tl.store(sum_pointer + pid, tl.sum(s))
    tl.store(sqsum_pointer + pid, tl.sum(sq))


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
        unbiased_var = var * count / (count - 1)
        tl.store(
            running_var_pointer + c,
            ((1 - momentum) * running_var + momentum * unbiased_var).to(
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
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE_S: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (n, c) slice
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
        mask = idx < spatial_dim
        x = tl.load(input_pointer + base + idx, mask=mask).to(tl.float32)
        y = weight * (x - mean) * inv_std + bias
        tl.store(
            output_pointer + base + idx,
            y.to(output_pointer.dtype.element_ty),
            mask=mask,
        )


# NOTE (hybrid routing): the contiguous grid=N*C 3-stage path above trades a per-shape
# ~0.4ms launch floor (stats kernel + torch combine + normalize kernel) for eliminating
# the large-spatial discrete-access catastrophe. For SMALL shapes that floor dominates and
# regresses gems speedup vs the original single fused kernel. So we keep the original fused
# (transpose) kernel below and route small shapes to it (see batch_norm wrapper). The fused
# kernel's stride-C discrete reads only blow up when batch_dim*spatial_dim is large.


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
            (1 - momentum) * running_var
            + momentum * var * n_elements / (n_elements - 1),
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


def batch_norm_heur_block_m(args):
    return min(64, triton.next_power_of_2(args.get("batch_dim", 0)))


def batch_norm_heur_block_n(args):
    BLOCK_M = batch_norm_heur_block_m(args)
    BLOCK_N = triton.next_power_of_2(args.get("spatial_dim", 0))
    return min(BLOCK_N, max(1, 2**14 // BLOCK_M))


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": batch_norm_heur_block_m,
        "BLOCK_N": batch_norm_heur_block_n,
    },
)
@triton.jit
def batch_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    input_grad_mask: tl.constexpr,
    weight_grad_mask: tl.constexpr,
    bias_grad_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    mean = tl.load(feat_pid + mean_pointer).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
        batch_mask = batch_offset < batch_dim

        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_pointer, mask=mask, other=0).to(tl.float32)

            curr_pre_lin = ((curr_input - mean) * inv_std).to(tl.float32)
            curr_output_grad = tl.load(
                curr_output_grad_pointer, mask=mask, other=0.0
            ).to(tl.float32)

            term1 += curr_pre_lin * curr_output_grad
            term2 += curr_output_grad

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if weight_grad_mask:
        tl.store(feat_pid + weight_grad_pointer, term1)
    if bias_grad_mask:
        tl.store(feat_pid + bias_grad_pointer, term2)

    if not input_grad_mask:
        return

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
        weight = weight.to(tl.float32)

    count = batch_dim * spatial_dim

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input_grad_pointer = (
                input_grad_pointer
                + input_grad_feat_stride * feat_pid
                + input_grad_batch_stride * batch_offset[:, None]
                + input_grad_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(
                curr_output_grad_pointer,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            ).to(tl.float32)
            curr_input_grad = (
                inv_std
                * weight
                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
            )
            tl.store(
                curr_input_grad_pointer,
                curr_input_grad,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


# Per-channel discrete-read count (batch_dim * spatial_dim) at/below which the single
# fused (transpose) kernel's low launch floor beats the contiguous 3-stage path. Above it
# the fused kernel's stride-C discrete reads blow up (measured crossover ~0.4ms floor).
# NOTE: the fused kernel is INFERENCE-ONLY here — its training reduction is numerically
# broken on XPU (verified: garbage output). Training must always use the contiguous path.
BN_FUSED_MAX_ELEMS = 2048


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

    # Inference on small shapes -> low-floor fused kernel (correct in inference). All other
    # cases (any training, or large inference) -> contiguous 3-stage path, which is correct
    # everywhere and avoids the fused kernel's large-spatial discrete-access catastrophe.
    x_nat = make_3d_for_bn(input)  # [N, C, S]
    batch_dim, _, spatial_dim = x_nat.shape
    if (
        not training
        and running_mean is not None
        and running_var is not None
        and batch_dim * spatial_dim <= BN_FUSED_MAX_ELEMS
    ):
        return _batch_norm_fused_infer(
            input, weight, bias, running_mean, running_var, eps
        )

    input_3d = make_3d_for_bn(input).contiguous()  # [N, C, S] contiguous
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim

    output = torch.empty_like(input_3d)
    input_flat = input_3d.reshape(-1)
    output_flat = output.reshape(-1)

    tile_s = min(triton.next_power_of_2(spatial_dim), 4096) if spatial_dim > 0 else 1

    mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    inv_std_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    if training:
        # Stage 1: per-(n, c) partial sum / sum-of-squares over contiguous spatial run.
        part_sum = torch.empty(n_slices, device=input.device, dtype=torch.float32)
        part_sqsum = torch.empty(n_slices, device=input.device, dtype=torch.float32)
        has_rm = running_mean is not None
        has_rv = running_var is not None
        with torch_device_fn.device(input.device):
            batch_norm_stats_kernel[(n_slices,)](
                input_flat, part_sum, part_sqsum, spatial_dim, TILE_S=tile_s
            )
            # Stage 2: combine batch partials -> per-channel mean / inv_std and fold the
            # running-stat updates, all in a single kernel (grid=(C,)). One launch instead
            # of ~14 small torch ops -> removes the small-shape launch floor.
            batch_norm_combine_kernel[(feat_dim,)](
                part_sum,
                part_sqsum,
                mean_f,
                inv_std_f,
                running_mean if has_rm else part_sum,
                running_var if has_rv else part_sum,
                batch_dim,
                feat_dim,
                count,
                momentum,
                eps,
                HAS_RM=has_rm,
                HAS_RV=has_rv,
                TILE_N=triton.next_power_of_2(batch_dim),
            )
    else:
        mean_f = running_mean.to(torch.float32)
        inv_std_f = torch.rsqrt(running_var.to(torch.float32) + eps)

    # Return stats in input dtype (single cast each; no extra empty+copy).
    mean = mean_f.to(input.dtype)
    inv_std = inv_std_f.to(input.dtype)

    has_weight = weight is not None
    has_bias = bias is not None
    with torch_device_fn.device(input.device):
        batch_norm_normalize_kernel[(n_slices,)](
            input_flat,
            output_flat,
            mean_f.contiguous(),
            inv_std_f.contiguous(),
            weight if has_weight else input_flat,
            bias if has_bias else input_flat,
            feat_dim,
            spatial_dim,
            HAS_WEIGHT=has_weight,
            HAS_BIAS=has_bias,
            TILE_S=tile_s,
        )

    return output.view_as(input), mean, inv_std


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
    input_3d_i = make_3d_for_bn(input)
    m, n, k = input_3d_i.shape
    input_3d_f = input_3d_i.permute(0, 2, 1).reshape(-1, n)
    input_3d = make_3d_for_bn(input_3d_f)

    output_grad_3d_i = make_3d_for_bn(grad_out)
    output_grad_3d_f = output_grad_3d_i.permute(0, 2, 1).reshape(-1, n)
    output_grad_3d = make_3d_for_bn(output_grad_3d_f)

    batch_dim, feat_dim, spatial_dim = input_3d.shape

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

    with torch_device_fn.device(input.device):
        batch_norm_backward_kernel[(feat_dim, 1, 1)](
            output_grad_3d,
            input_3d,
            save_mean,
            save_invstd,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *output_grad_3d.stride(),
            *input_3d.stride(),
            *input_grad.stride(),
            *output_mask,
            buffer_size_limit=2048,
        )

    return (
        input_grad.reshape(m, k, n).permute(0, 2, 1).view_as(input),
        weight_grad,
        bias_grad,
    )
