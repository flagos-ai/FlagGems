import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim >= 4:
        input = input.flatten(2, -1)

    return input


# NOTE: This part of the kernel code is copied and modified
# from the https://github.com/BobMcDear/attorch codebase.


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("batch_norm"),
#     key=["batch_dim", "spatial_dim"],
#     restore_value=["running_mean_pointer", "running_var_pointer"],
# )
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

    # traning mode default track_running_stat
    if is_train:
        mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

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
                curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

                step = m_step * n_num_steps + n_step + 1
                new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
                new_var = tl.where(
                    mask, var + (curr_input - new_mean) * (curr_input - mean), var
                )
                cnt += mask.to(tl.int32)
                mean = new_mean
                var = new_var

        final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
        var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (
            batch_dim * spatial_dim
        )
        inv_std = rsqrt(var + eps)
        mean = final_mean

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        n = batch_dim * spatial_dim
        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
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
    return min(64, triton.next_power_of_2(args["batch_dim"]))


def batch_norm_heur_block_n(args):
    # A maximum of 16384 elements are loaded at once.
    BLOCK_M = batch_norm_heur_block_m(args)
    BLOCK_N = triton.next_power_of_2(args["spatial_dim"])
    return min(BLOCK_N, max(1, 2**14 // BLOCK_M))


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("batch_norm"),
#     key=["batch_dim", "spatial_dim"],
# )
@triton.heuristics(
    values={
        "BLOCK_M": batch_norm_heur_block_m,
        "BLOCK_N": batch_norm_heur_block_n,
    },
)
# @triton.heuristics(runtime.get_heuristic_config("batch_norm"))
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


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,  # self.running_mean if not self.training or self.track_running_state else None
    running_var=None,
    training=False,  # (self.running_mean is None) and (self.running_var is None)
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS BATCHNORM FORWARD")

    input_3d_i = make_3d_for_bn(input)
    m, n, k = input_3d_i.shape
    input_3d_f = input_3d_i.permute(0, 2, 1).reshape(-1, n)
    input_3d = make_3d_for_bn(input_3d_f)
    # input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    # Launches 1D grid where each program operates over one feature.
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
            momentum,
            eps,
            is_train=training,
            buffer_size_limit=2048,
        )

    output_reshaped = output.reshape(m, k, n).permute(0, 2, 1)
    return output_reshaped.view_as(input), mean, inv_std


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
    logger.debug("GEMS BATCHNORM BACKWARD")
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

    # Launches 1D grid where each program operates over one feature.
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

    # Pads output with None because a gradient is necessary for
    # all input arguments.
    return (
        input_grad.reshape(m, k, n).permute(0, 2, 1).view_as(input),
        weight_grad,
        bias_grad,
    )
