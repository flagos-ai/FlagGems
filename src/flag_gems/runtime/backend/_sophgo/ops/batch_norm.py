import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.ops import rsqrt as rsqrt_op
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


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
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
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
        # Use tl.rsqrt to avoid generating math.sqrt which is not supported by PPL
        inv_std = tl.rsqrt((var + eps).to(tl.float32))
        mean = final_mean

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)

        # Use 1-element tensor load pattern to load running_mean/running_var
        # Avoid scalar pointer dereference (does not work on TPU)
        train_param_idx = tl.arange(0, 1)
        train_param_mask = train_param_idx < 1

        running_mean_addr = running_mean_pointer + feat_pid + train_param_idx
        running_mean = tl.sum(
            tl.load(running_mean_addr, mask=train_param_mask, other=0.0)
        )

        running_var_addr = running_var_pointer + feat_pid + train_param_idx
        running_var = tl.sum(
            tl.load(running_var_addr, mask=train_param_mask, other=0.0)
        )

        n = batch_dim * spatial_dim
        # Store also requires correct address
        store_addr_mean = running_mean_pointer + feat_pid
        store_addr_var = running_var_pointer + feat_pid
        tl.store(store_addr_mean, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            store_addr_var,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        # During inference, directly use host-side pre-filled inv_std to avoid generating sqrt/rsqrt on device
        # Use 1-element tensor load pattern to force DMA load, avoiding scalar pointer dereference
        # (scalar pointer dereference does not work on TPU)
        param_idx = tl.arange(0, 1)  # Create 1-element tensor [0]
        param_mask = param_idx < 1  # Mask that is always true

        mean_addr = running_mean_pointer + feat_pid + param_idx
        mean = tl.sum(tl.load(mean_addr, mask=param_mask, other=0.0))

        inv_std_addr = inv_std_pointer + feat_pid + param_idx
        inv_std = tl.sum(tl.load(inv_std_addr, mask=param_mask, other=1.0)).to(
            tl.float32
        )

    # Use 1-element tensor load pattern to load weight and bias
    # Avoid scalar pointer dereference (does not work on TPU)
    param_idx = tl.arange(0, 1)
    param_mask = param_idx < 1

    if weight_pointer is None:
        weight = 1.0
    else:
        weight_addr = weight_pointer + feat_pid + param_idx
        weight = tl.sum(tl.load(weight_addr, mask=param_mask, other=1.0)).to(tl.float32)
    if bias_pointer is None:
        bias = 0.0
    else:
        bias_addr = bias_pointer + feat_pid + param_idx
        bias = tl.sum(tl.load(bias_addr, mask=param_mask, other=0.0)).to(tl.float32)

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


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
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

    # Use 1-element tensor load pattern to load mean and inv_std
    # Avoid scalar pointer dereference (does not work on TPU)
    param_idx = tl.arange(0, 1)
    param_mask = param_idx < 1

    mean_addr = mean_pointer + feat_pid + param_idx
    mean = tl.sum(tl.load(mean_addr, mask=param_mask, other=0.0)).to(tl.float32)

    inv_std_addr = inv_std_pointer + feat_pid + param_idx
    inv_std = tl.sum(tl.load(inv_std_addr, mask=param_mask, other=1.0)).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

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

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_pointer, mask=mask).to(
                tl.float32
            )

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

    # Use 1-element tensor load pattern to load weight
    # Avoid scalar pointer dereference (does not work on TPU)
    bwd_param_idx = tl.arange(0, 1)
    bwd_param_mask = bwd_param_idx < 1

    if weight_pointer:
        weight_addr = weight_pointer + feat_pid + bwd_param_idx
        weight = tl.sum(tl.load(weight_addr, mask=bwd_param_mask, other=1.0)).to(
            tl.float32
        )
    else:
        weight = 1.0

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
    logging.debug("GEMS BATCHNORM FORWARD (SOPHGO)")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    # Option B: Pre-fill mean and inv_std on host side during inference (reusing verified pointwise rsqrt operator)
    if not training:
        mean.copy_(running_mean)
        inv_std.copy_(rsqrt_op(running_var + eps))

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
    logging.debug("GEMS BATCHNORM BACKWARD (SOPHGO)")
    input_3d = make_3d_for_bn(input)
    output_grad_3d = make_3d_for_bn(grad_out)

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
        batch_norm_backward_kernel[(feat_dim,)](
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
        )

    # Pads output with None because a gradient is necessary for
    # all input arguments.
    return (
        input_grad.view_as(input),
        weight_grad,
        bias_grad,
    )
