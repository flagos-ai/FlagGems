import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt
logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr = 1024,
):
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)

    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    mean_num = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)

    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW

        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]

        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        mean_num += X_val

    mean_sum = tl.sum(mean_num)
    mean = mean_sum / num_elements

    var_num = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)

    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]

        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.where(xy_mask, X_val - mean, 0.0)

        var_num += x * x

    var_sum = tl.sum(var_num)
    var = var_sum / num_elements
    rstd = rsqrt(var + eps)

    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0)[:, None]
    if B is None:
        bias = 0
    else:
        bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0)[:, None]
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]

        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.where(xy_mask, X_val - mean, 0.0)
        x_hat = x * rstd

        Y_val = x_hat * weight + bias

        tl.store(Y + xy_offset, Y_val, mask=xy_mask)
    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid
    tl.store(Mean_ptr, mean)
    tl.store(Rstd_ptr, rstd)


@libentry()
@triton.jit
def group_norm_backward_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr = 128,
):
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    wb_offset = group * group_size + group_offset

    wb_mask = wb_offset < C

    rstd = tl.load(Rstd + pid).to(tl.float32)
    mean = tl.load(Mean + pid).to(tl.float32)
    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

    dx_part2 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]

        dY_val = tl.load(grad_y + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)

        x_hat = tl.where(xy_mask, rstd * (X_val - mean), 0.0)
        dx_hat = weight * dY_val
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2)
    dx_3 = tl.sum(dx_part3)

    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        hw_mask = hw_offset < HW
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_mask[:, None] & hw_mask[None, :]

        dY_val = tl.load(grad_y + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)

        x_hat = tl.where(xy_mask, rstd * (X_val - mean), 0.0)
        dx_hat = weight * dY_val
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / num_elements)

        tl.store(grad_x + xy_offset, dx, xy_mask)


@libentry()
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    num_groups,
    group_size,
    N,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    hw_offset = tl.arange(0, BLOCK_HW)
    xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW
    mr_mask = n_offset < N

    mean_ptr = Mean + group + n_offset * num_groups
    rstd_ptr = Rstd + group + n_offset * num_groups

    dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
    x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]

    grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=xy_mask, other=0.0)
    x_f32 = x.to(tl.float32)
    mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    if dW is not None:
        dw = tl.sum((x_f32 - mean) * rstd * grad_y)
        tl.store(dW + pid, dw)
    if dB is not None:
        db = tl.sum(grad_y)
        tl.store(dB + pid, db)


def group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    logger.debug("GEMS GROUPNORM FORWARD")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    grid = (N * group,)
    with torch_device_fn.device(input.device):
        group_norm_kernel[grid](
            input,
            y,
            weight,
            bias,
            mean,
            rstd,
            group_size,
            C,
            HxW,
            group,
            eps,
            BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
            BLOCK_HW_SIZE=1024,
        )
    return y, mean, rstd


def group_norm_backward(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    logger.debug("GEMS GROUPNORM BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    group_size = triton.cdiv(C, group)

    if output_mask[0]:
        grad_inp = torch.empty_like(input)
        grid = (N * group,)
        with torch_device_fn.device(input.device):
            group_norm_backward_kernel[grid](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                group,
                group_size,
                grad_inp,
                C,
                HxW,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
            )
    else:
        grad_inp = None

    if output_mask[1] is False and output_mask[2] is False:
        return grad_inp, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(weight) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        weight_bias_backward_kernel[(C, 1, 1)](
            grad_out,
            input,
            mean,
            rstd,
            weight_grad,
            bias_grad,
            group,
            group_size,
            N,
            C,
            HxW,
            BLOCK_N=triton.next_power_of_2(N),
            BLOCK_HW=triton.next_power_of_2(HxW),
        )
    return grad_inp, weight_grad, bias_grad
