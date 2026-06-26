import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._metax.heuristics_config_utils import (
    batch_norm_heur_block_m,
    batch_norm_heur_block_n,
)
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt

logger = logging.getLogger("flag_gems." + __name__)


def make_3d_for_bn(input: torch.Tensor) -> torch.Tensor:
    """
    Converts the input to a 3D view for batch normalization.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


@libentry()
@triton.heuristics(
    {
        "BLOCK_M": batch_norm_heur_block_m,
        "BLOCK_N": batch_norm_heur_block_n,
    }
)
@triton.jit
def batch_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    inv_std_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
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
    feat_pid = tle.program_id(0)

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

                curr_input_ptr = (
                    input_ptr
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(curr_input_ptr, mask=mask).to(tl.float32)

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
        inv_std_val = rsqrt(var + eps)
        mean = final_mean

        tl.store(feat_pid + mean_ptr, mean)
        tl.store(feat_pid + inv_std_ptr, inv_std_val)

        running_mean_ptr = running_mean_ptr + feat_pid
        running_var_ptr = running_var_ptr + feat_pid

        running_mean = tl.load(running_mean_ptr)
        running_var = tl.load(running_var_ptr)

        n = batch_dim * spatial_dim
        tl.store(running_mean_ptr, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_ptr,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        mean = tl.load(feat_pid + running_mean_ptr)
        inv_std_val = rsqrt(tl.load(feat_pid + running_var_ptr) + eps)

    if weight_ptr:
        weight = tl.load(feat_pid + weight_ptr).to(tl.float32)
    else:
        weight = 1.0
    if bias_ptr:
        bias = tl.load(feat_pid + bias_ptr).to(tl.float32)
    else:
        bias = 0.0

    mean_for_use = mean if is_train else mean
    inv_std_for_use = inv_std_val if is_train else inv_std_val

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_ptr = (
                output_ptr
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_ptr, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean_for_use) * inv_std_for_use + bias

            tl.store(
                curr_output_ptr,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


@libentry()
@triton.heuristics(
    {
        "BLOCK_M": batch_norm_heur_block_m,
        "BLOCK_N": batch_norm_heur_block_n,
    }
)
@triton.jit
def batch_norm_backward_kernel(
    output_grad_ptr,
    input_ptr,
    mean_ptr,
    inv_std_ptr,
    weight_ptr,
    input_grad_ptr,
    weight_grad_ptr,
    bias_grad_ptr,
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
    feat_pid = tle.program_id(0)

    mean = tl.load(feat_pid + mean_ptr).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_ptr).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_ptr = (
                output_grad_ptr
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_ptr, mask=mask).to(tl.float32)

            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_ptr, mask=mask).to(tl.float32)

            term1 += curr_pre_lin * curr_output_grad
            term2 += curr_output_grad

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if weight_grad_mask:
        tl.store(feat_pid + weight_grad_ptr, term1)
    if bias_grad_mask:
        tl.store(feat_pid + bias_grad_ptr, term2)

    if not input_grad_mask:
        return

    if weight_ptr:
        weight = tl.load(feat_pid + weight_ptr).to(tl.float32)
    else:
        weight = 1.0

    count = batch_dim * spatial_dim

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_ptr = (
                output_grad_ptr
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input_grad_ptr = (
                input_grad_ptr
                + input_grad_feat_stride * feat_pid
                + input_grad_batch_stride * batch_offset[:, None]
                + input_grad_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_ptr, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(
                curr_output_grad_ptr,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            ).to(tl.float32)
            curr_input_grad = (
                inv_std
                * weight
                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
            )
            tl.store(
                curr_input_grad_ptr,
                curr_input_grad,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight=None,
        bias=None,
        running_mean=None,
        running_var=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
    ):
        logger.debug("METAX GEMS BATCHNORM FORWARD")

        input_3d = make_3d_for_bn(input)

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)

        mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

        running_mean = input if running_mean is None else running_mean
        running_var = input if running_var is None else running_var

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

        if input.requires_grad:
            ctx.save_for_backward(input, weight, bias, mean, inv_std)
            ctx.batch_dim = batch_dim
            ctx.spatial_dim = spatial_dim
            ctx.training = training

        return output.view_as(input), mean, inv_std

    @staticmethod
    def backward(ctx, grad_out, save_mean, save_invstd):
        logger.debug("METAX GEMS BATCHNORM BACKWARD")

        (input, weight, bias, mean, inv_std) = ctx.saved_tensors
        batch_dim = ctx.batch_dim
        spatial_dim = ctx.spatial_dim

        input_3d = make_3d_for_bn(input)
        grad_out_3d = make_3d_for_bn(grad_out)

        input_grad = torch.empty_like(input_3d)

        if weight is not None:
            weight_grad = torch.empty(
                (input.shape[1],), dtype=input.dtype, device=input.device
            )
        else:
            weight_grad = None
        if bias is not None:
            bias_grad = torch.empty(
                (input.shape[1],), dtype=input.dtype, device=input.device
            )
        else:
            bias_grad = None

        with torch_device_fn.device(input.device):
            batch_norm_backward_kernel[(input.shape[1],)](
                grad_out_3d,
                input_3d,
                mean,
                inv_std,
                weight,
                input_grad,
                weight_grad,
                bias_grad,
                batch_dim,
                spatial_dim,
                *grad_out_3d.stride(),
                *input_3d.stride(),
                *input_grad.stride(),
                input_grad is not None,
                weight_grad is not None,
                bias_grad is not None,
            )

        return (
            input_grad.view_as(input),
            weight_grad,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
        )


def batch_norm(
    input: torch.Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    return BatchNorm.apply(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )


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
    logger.debug("METAX GEMS BATCHNORM BACKWARD")

    input_3d = make_3d_for_bn(input)
    grad_out_3d = make_3d_for_bn(grad_out)

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
        batch_norm_backward_kernel[(feat_dim,)](
            grad_out_3d,
            input_3d,
            save_mean,
            save_invstd,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *grad_out_3d.stride(),
            *input_3d.stride(),
            *input_grad.stride(),
            output_mask[0],
            output_mask[1],
            output_mask[2],
        )

    return (
        input_grad.view_as(input),
        weight_grad,
        bias_grad,
    )
