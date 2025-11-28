import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

rsqrt = tl_extra_shim.rsqrt


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
    mr_mask = n_offset < N
    mean_ptr = Mean + group + n_offset * num_groups
    rstd_ptr = Rstd + group + n_offset * num_groups
    mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    SUB_BLOCK_HW: tl.constexpr = 64

    dw_sum = 0.0
    db_sum = 0.0

    for hw_off in range(0, BLOCK_HW, SUB_BLOCK_HW):
        hw_offset = hw_off + tl.arange(0, SUB_BLOCK_HW)
        xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW
        dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
        x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
        grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr, mask=xy_mask, other=0.0)
        x_f32 = x.to(tl.float32)
        if dW is not None:
            dw_sum = dw_sum + tl.sum((x_f32 - mean) * rstd * grad_y)
        if dB is not None:
            db_sum = db_sum + tl.sum(grad_y)

    if dW is not None:
        dw = dw_sum
        tl.store(dW + pid, dw.to(mean.dtype))
    if dB is not None:
        db = db_sum
        tl.store(dB + pid, db.to(mean.dtype))


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SUB_HW_SIZE": 32}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 64}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 128}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 256}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 512}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 1024}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 2048}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 4096}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 8192}),
        triton.Config({"BLOCK_SUB_HW_SIZE": 16384}),
    ],
    key=["HW", "group_size"],
)
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
    BLOCK_SUB_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups

    # 计算当前group在整个tensor中的起始位置
    batch_offset = batch_idx * C * HW
    group_start_channel = group_idx * group_size

    num_elements = group_size * HW

    # 第一次遍历：计算均值
    X_sum = 0.0
    for hw_start in range(0, HW, BLOCK_SUB_HW_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SUB_HW_SIZE)
        hw_mask = hw_offsets < HW

        # 先按HW维度连续，再按channel维度
        for c_idx in range(BLOCK_GROUP_SIZE):
            if c_idx < group_size and (group_start_channel + c_idx) < C:
                channel_offset = group_start_channel + c_idx
                # 连续访问HW维度的数据
                base_offset = batch_offset + channel_offset * HW + hw_offsets
                X_vals = tl.load(X + base_offset, mask=hw_mask, other=0.0).to(
                    tl.float32
                )
                X_sum += tl.sum(X_vals)

    mean = X_sum / num_elements

    # 第二次遍历：计算方差
    X_var_sum = 0.0
    for hw_start in range(0, HW, BLOCK_SUB_HW_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SUB_HW_SIZE)
        hw_mask = hw_offsets < HW

        for c_idx in range(BLOCK_GROUP_SIZE):
            if c_idx < group_size and (group_start_channel + c_idx) < C:
                channel_offset = group_start_channel + c_idx
                base_offset = batch_offset + channel_offset * HW + hw_offsets
                X_vals = tl.load(X + base_offset, mask=hw_mask, other=mean).to(
                    tl.float32
                )
                x_centered = X_vals - mean
                X_var_sum += tl.sum(x_centered * x_centered)

    var = X_var_sum / num_elements
    rstd = rsqrt(var + eps)

    # 第三次遍历：归一化并写回
    for hw_start in range(0, HW, BLOCK_SUB_HW_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SUB_HW_SIZE)
        hw_mask = hw_offsets < HW

        for c_idx in range(BLOCK_GROUP_SIZE):
            if c_idx < group_size and (group_start_channel + c_idx) < C:
                channel_offset = group_start_channel + c_idx
                base_offset = batch_offset + channel_offset * HW + hw_offsets

                # 加载数据
                X_vals = tl.load(X + base_offset, mask=hw_mask, other=0.0).to(
                    tl.float32
                )

                # 归一化并应用仿射变换
                x_normalized = (X_vals - mean) * rstd
                if W is not None:
                    w_val = tl.load(W + channel_offset)
                    x_normalized = x_normalized * w_val
                if B is not None:
                    b_val = tl.load(B + channel_offset)
                    x_normalized = x_normalized + b_val

                # 存储结果
                tl.store(Y + base_offset, x_normalized, mask=hw_mask)

    # 存储均值和标准差
    mean_rstd_offset = batch_idx * num_groups + group_idx
    tl.store(Mean + mean_rstd_offset, mean)
    tl.store(Rstd + mean_rstd_offset, rstd)


def group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    logging.debug("ASCEND GEMS GROUPNORM FORWARD")
    logger.debug("ASCEND GEMS GROUPNORM FORWARD")
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
            BLOCK_HW_SIZE=triton.next_power_of_2(HxW),
        )
    return y, mean, rstd


def group_norm_backward(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    logging.debug("ASCEND GEMS GROUPNORM BACKWARD")
    logger.debug("ASCEND GEMS GROUPNORM BACKWARD")
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
