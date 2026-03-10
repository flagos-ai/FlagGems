import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

rsqrt = tl_extra_shim.rsqrt
logger = logging.getLogger(__name__)

MAX_BLOCK_HW = 1024


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
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)

    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid

    # Pass 1: compute sum to get mean
    _sum = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        _sum += X_val
    mean = tl.sum(_sum) / num_elements

    # Pass 2: compute variance
    _var = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.where(xy_mask, X_val - mean, 0.0)
        _var += x * x
    var = tl.sum(_var) / num_elements
    rstd = rsqrt(var + eps)

    # Load weight and bias (shape: [BLOCK_GROUP_SIZE])
    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0)[:, None]
    if B is None:
        bias = 0
    else:
        bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0)[:, None]

    # Pass 3: normalize and store output
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
        xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW
        X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.where(xy_mask, X_val - mean, 0.0)
        x_hat = x * rstd
        Y_val = x_hat * weight + bias
        tl.store(Y + xy_offset, Y_val, mask=xy_mask)

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
    pid = tl.program_id(0)
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
    pid = tl.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    mr_mask = n_offset < N

    mean = tl.load(Mean + group + n_offset * num_groups, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(Rstd + group + n_offset * num_groups, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    dw_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_N, BLOCK_HW], dtype=tl.float32)

    for off in range(0, HW, BLOCK_HW):
        hw_offset = off + tl.arange(0, BLOCK_HW)
        xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW

        dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
        x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]

        grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x_f32 = tl.load(x_ptr, mask=xy_mask, other=0.0).to(tl.float32)

        dw_acc += (x_f32 - mean) * rstd * grad_y
        db_acc += grad_y

    if dW is not None:
        dw = tl.sum(dw_acc)
        tl.store(dW + pid, dw)
    if dB is not None:
        db = tl.sum(db_acc)
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

    BLOCK_HW_SIZE = min(triton.next_power_of_2(HxW), MAX_BLOCK_HW)

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
            BLOCK_HW_SIZE=BLOCK_HW_SIZE,
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

    BLOCK_HW = min(triton.next_power_of_2(HxW), MAX_BLOCK_HW)

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
            BLOCK_HW=BLOCK_HW,
        )
    return grad_inp, weight_grad, bias_grad
