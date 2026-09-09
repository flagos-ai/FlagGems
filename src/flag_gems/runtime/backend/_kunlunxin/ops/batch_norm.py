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
    # XPU masked loads with runtime masks can leak whole lanes (see the
    # nll_loss / vector_norm fixes): clamp the index to a valid one and
    # zero invalid lanes with tl.where instead of relying on the mask.
    for off in range(0, spatial_dim, TILE_S):
        idx = off + tl.arange(0, TILE_S)
        mask = idx < spatial_dim
        idx_safe = tl.minimum(idx, spatial_dim - 1)
        x = tl.load(input_pointer + base + idx_safe).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        s += x
        sq += x * x
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
    # Same XPU masked-load leak, and worse here: TILE_N is
    # next_pow2(batch_dim), so e.g. batch 2050 -> tile 4096 lets the
    # masked tail read up to ~2x past the partials buffer. Clamp to a
    # valid index and zero invalid lanes with tl.where.
    idx_safe = tl.minimum(idx, batch_dim - 1)
    part_sum = tl.load(part_sum_pointer + c + idx_safe * feat_dim)
    part_sqsum = tl.load(part_sqsum_pointer + c + idx_safe * feat_dim)
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

    # Always use the contiguous 3-stage path: the fused infer kernel's
    # permute(0, 2, 1).reshape forces a strided copy_ that fails to launch
    # on XPU ("CUDA error: invalid device function") for small eval shapes.
    x_nat = make_3d_for_bn(input)  # [N, C, S]
    batch_dim, _, spatial_dim = x_nat.shape
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
    # The triton backward kernel below was only ever launched with
    # spatial_dim == 1: the old wrapper transposed inputs into [N*S, C, 1]
    # via permute(0, 2, 1).reshape, whose strided copy_ fails on XPU
    # ("invalid device function"), and feeding the kernel the natural
    # [N, C, S] layout (spatial > 1) trips an XPU compiler crash
    # (RuntimeError: PassManager::run failed). Compute the backward with
    # eager fp32 torch ops on the natural contiguous layout instead.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_BACKWARD")
    if output_mask is None:
        output_mask = [True, True, True]
    input_3d = make_3d_for_bn(input).contiguous()
    grad_3d = make_3d_for_bn(grad_out).contiguous()
    n, c, sp = input_3d.shape
    count = n * sp
    # CPU aten returns EMPTY save tensors in eval mode; fall back to the
    # running stats whenever the saves are absent or empty.
    if save_mean is not None and save_mean.numel() > 0:
        mean = save_mean.to(torch.float32)
    else:
        mean = running_mean.to(torch.float32)
    if save_invstd is not None and save_invstd.numel() > 0:
        inv_std = save_invstd.to(torch.float32)
    else:
        inv_std = torch.rsqrt(running_var.to(torch.float32) + eps)
    x = input_3d.to(torch.float32)
    dy = grad_3d.to(torch.float32)
    xhat = (x - mean.view(1, c, 1)) * inv_std.view(1, c, 1)
    # flag_gems sum_dim dim_compress permutes non-last reduction dims onto
    # a strided layout whose copy_ fails to launch on XPU ("invalid device
    # function"). Reduce the contiguous last dim only (identity permute,
    # no copy), then fold the batch dim with a ones-vector GEMM.
    if n > 0:
        ones_n = torch.ones((1, n), dtype=torch.float32, device=input.device)
        sum_dy = torch.mm(ones_n, dy.sum(dim=2)).view(c)
        sum_dy_xhat = torch.mm(ones_n, (dy * xhat).sum(dim=2)).view(c)
    else:
        # zero-batch input: gems mm heuristic divides by K == 0
        sum_dy = torch.zeros(c, dtype=torch.float32, device=input.device)
        sum_dy_xhat = torch.zeros(c, dtype=torch.float32, device=input.device)
    if weight is not None:
        w = weight.to(torch.float32)
    else:
        w = torch.ones_like(mean)
    grads = []
    if output_mask[0]:
        if train:
            gi = (
                inv_std.view(1, c, 1)
                * w.view(1, c, 1)
                * (
                    dy
                    - sum_dy.view(1, c, 1) / count
                    - xhat * sum_dy_xhat.view(1, c, 1) / count
                )
            )
        else:
            # eval: running stats are constants, no centering correction
            gi = inv_std.view(1, c, 1) * w.view(1, c, 1) * dy
        grads.append(gi.to(input.dtype).view_as(input))
    else:
        grads.append(None)
    if output_mask[1]:
        if weight is not None:
            grads.append(sum_dy_xhat.to(input.dtype))
        else:
            grads.append(torch.zeros(c, dtype=input.dtype, device=input.device))
    else:
        grads.append(None)
    if output_mask[2]:
        grads.append(sum_dy.to(input.dtype))
    else:
        grads.append(None)
    return tuple(grads)


# ---------------------------------------------------------------------------
# _batch_norm_impl_index family (aten::batch_norm_impl_index and variants)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def batch_norm_impl_index(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    cudnn_enabled=True,
):
    # aten::batch_norm (functional) is a CompositeImplicitAutograd op that
    # internally redispatches to aten::_batch_norm_impl_index, so every
    # batch_norm-family entry point (F.batch_norm, native_batch_norm_legit,
    # instance_norm, ...) funnels here. The generic impl_index kernel's 2-D
    # strided pointer arithmetic fails XPU tt.addptr verification ("result
    # type matches ptr type", _batch_norm_impl_index.py:172) and surfaces as
    # OutOfResources/uni_sram on both training and eval paths. Delegate to
    # the kunlunxin 3-stage batch_norm implementation instead.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_IMPL_INDEX")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), save_mean, save_invstd, reserve, 0


def batch_norm_impl_index_backward(
    impl_index,
    input,
    grad_out,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    train=False,
    eps=1e-05,
    output_mask=None,
    reservedSpace=None,
):
    # aten::_batch_norm_impl_index_backward(int impl_index, Tensor input,
    # Tensor grad_output, ... bool[3] output_mask, Tensor reservedSpace):
    # impl_index comes FIRST; keep the formal order in schema order.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_IMPL_INDEX_BACKWARD")
    if output_mask is None:
        output_mask = [True, True, True]
    return batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,  # PyTorch's save_var is in fact inv_std
        train,
        eps,
        output_mask,
    )


def batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    # aten::_batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias,
    # Tensor? running_mean, Tensor? running_var, float momentum, float eps)
    # -> (Tensor, Tensor, Tensor, Tensor). Eval-only: normalize with the
    # running stats, never update them. CPU returns empty fp32 save tensors
    # and an empty uint8 reserve; the generic kernel trips the XPU compiler
    # (RuntimeError: PassManager::run failed) on every shape.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_NO_UPDATE")
    output, _, _ = batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )
    empty_f32 = torch.empty((0,), dtype=torch.float32, device=input.device)
    empty_u8 = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return output.view_as(input), empty_f32, empty_f32, empty_u8


def batch_norm_with_update_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    # aten::_batch_norm_with_update_functional is called directly by
    # torch.ops.aten; the generic kernel shares the impl_index addptr
    # failure. Delegate to the kunlunxin 3-stage path, whose training
    # branch folds the running-stat updates in place (combine kernel),
    # so the mutated running_mean/running_var tensors are the new stats.
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_WITH_UPDATE_FUNCTIONAL")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        True,
        momentum,
        eps,
    )
    # aten schema returns SIX tensors: (output, save_mean, save_var,
    # reserve, running_mean_out, running_var_out).
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return (
        output.view_as(input),
        save_mean,
        save_invstd,
        reserve,
        running_mean,
        running_var,
    )


def miopen_batch_norm_backward(
    input,
    grad_output,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_var,
    epsilon,
):
    # aten::miopen_batch_norm_backward mirrors the cudnn variant. Its
    # save_var argument holds the saved inverse standard deviation (rstd),
    # which is exactly what batch_norm_backward expects as save_invstd.
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_BACKWARD")
    input_grad, weight_grad, bias_grad = batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )
    return input_grad, weight_grad, bias_grad


# ---------------------------------------------------------------------------
# _native_batch_norm_legit family
# ---------------------------------------------------------------------------

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
    # aten::_native_batch_norm_legit mutates running stats in place and
    # returns (output, save_mean, save_invstd). The kunlunxin batch_norm
    # training branch folds the running-stat update into the combine
    # kernel, so the in-place mutation is covered.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT")
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    if not training:
        # aten CPU returns EMPTY save tensors in eval mode.
        empty = torch.empty(0, dtype=torch.float32, device=input.device)
        return output, empty, empty
    return output, save_mean, save_invstd


def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight,
    bias,
    training: bool,
    momentum: float,
    eps: float,
):
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    if not training:
        raise RuntimeError("Expected training to be true, but got false.")
    channels = input.shape[1]
    running_mean = torch.zeros(channels, dtype=input.dtype, device=input.device)
    running_var = torch.ones(channels, dtype=input.dtype, device=input.device)
    return batch_norm(
        input, weight, bias, running_mean, running_var, True, momentum, eps
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
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_OUT")
    result = _native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, training, momentum, eps
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
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    result = _native_batch_norm_legit_no_stats(
        input, weight, bias, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_functional(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    # Functional variant: the caller's running stats must NOT be mutated;
    # updated stats are returned as new tensors. Clone before delegating
    # so the in-place update inside batch_norm lands on the clones.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")
    rm = running_mean.clone()
    rv = running_var.clone()
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, rm, rv, training, momentum, eps
    )
    return output, save_mean, save_invstd, rm, rv


def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight,
    bias,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
):
    # Eval-only variant: normalize with the running stats, no update.
    logger.debug("GEMS_KUNLUNXIN NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")
    output, save_mean, save_invstd = batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )
    return output, save_mean, save_invstd
