import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


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
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)

    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
    xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid

    X_ptr = X + xy_offset
    Y_ptr = Y + xy_offset

    X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    mean = tl.sum(X_val) / num_elements
    x = tl.where(xy_mask, X_val - mean, 0.0)

    var = tl.sum(x * x) / num_elements
    rstd = rsqrt(var + eps)
    x_hat = x * rstd

    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0)[:, None]
    if B is None:
        bias = 0
    else:
        bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0)[:, None]
    Y_val = x_hat * weight + bias

    tl.store(Y_ptr, Y_val, mask=xy_mask)
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
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)
    wb_offset = group * group_size + group_offset

    wb_mask = wb_offset < C

    xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
    xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid
    X_ptr = X + xy_offset
    dY_ptr = grad_y + xy_offset
    dX_ptr = grad_x + xy_offset

    rstd = tl.load(Rstd_ptr).to(tl.float32)
    mean = tl.load(Mean_ptr).to(tl.float32)
    dY_val = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)

    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

    dx_hat = weight * dY_val  # -0.1208, -0.7044, -0.6529

    x = tl.where(xy_mask, X_val - mean, 0.0)  # 6.7863e-03,  6.7863e-03, -7.9882e-01
    pre_sum = dx_hat * x
    # import pudb; pudb.set_trace()
    grad_std = tl.sum(pre_sum)
    # tl.store(dX_ptr, grad_std, mask=xy_mask) # [-7.1525574e-07

    grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / (HW * group_size)
    grad_distance = 2 * x * grad_var
    grad_centered_mean = dx_hat * rstd + grad_distance
    grad_mean = -tl.sum(grad_centered_mean) / num_elements
    grad_X = grad_centered_mean + grad_mean
    tl.store(dX_ptr, grad_X, mask=xy_mask)


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
        tl.store(dW + pid, dw.to(x.dtype))
    if dB is not None:
        db = tl.sum(grad_y)
        tl.store(dB + pid, db.to(x.dtype))


@libentry()
@triton.jit
def weight_bias_backward_kernel_loop(
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

    grad_y_tile = tl.zeros((BLOCK_N, BLOCK_HW), dtype=tl.float32)  # grad_y_tile
    dw_tile = tl.zeros((BLOCK_N, BLOCK_HW), dtype=tl.float32)
    # import pudb; pudb.set_trace()
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)

        mean_ptr = Mean + group + n_offset * num_groups
        rstd_ptr = Rstd + group + n_offset * num_groups
        mr_mask = n_offset < N
        mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
        rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

        for start_hw in range(0, HW, BLOCK_HW):
            hw_offset = start_hw + tl.arange(0, BLOCK_HW)
            xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW
            dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
            grad_y_tile += grad_y

            x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            x = tl.load(x_ptr, mask=xy_mask, other=0.0)
            x_f32 = x.to(tl.float32)
            dw_tile += (x_f32 - mean) * rstd * grad_y

    dw = tl.sum(dw_tile)
    db = tl.sum(grad_y_tile)
    tl.store(dW + pid, dw)
    tl.store(dB + pid, db)


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, N, C, HW, num_groups, weight=None, bias=None, eps=1e-05):
        # 1, 64, 32, 32
        #    64
        # import pudb; pudb.set_trace()
        logger.debug("GEMS GROUPNORM FORWARD")
        group_size = C // num_groups  # 64 // 64 = 1
        x = x.contiguous()  # [1, 64, 32, 32]
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y = torch.empty_like(x)  # [1, 64, 32, 32]
        mean = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)  # [1, 64]
        rstd = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)  # [1, 64]
        grid = (N * num_groups,)  # 64

        with torch_device_fn.device(x.device):
            if N == 1 and C == 64 and HW == 1024 and num_groups == 64:
                os.environ["TRITONXPU_OTHER_SIM"] = "1"
                os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"

            group_norm_kernel[grid](
                x,  # [1, 64, 32, 32]
                y,  # [1, 64, 32, 32]
                weight,  # [64]
                bias,  # [64]
                mean,  # [1, 64]
                rstd,  # [1, 64]
                group_size,  # 1
                C,  # 64
                HW,  # 1024
                num_groups,  # 64
                eps,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),  # 1
                BLOCK_HW_SIZE=triton.next_power_of_2(HW),  # 1024
            )

            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]

        if x.requires_grad:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.num_groups = num_groups
            ctx.group_size = group_size
            ctx.N = N
            ctx.C = C
            ctx.HW = HW

        # print(f"mean.shape = {mean.shape}")
        # print(f'mean = {mean.cpu()}')
        # print(f'rstd.shape = {rstd.shape}')
        # print(f'rstd = {rstd.cpu()}')
        return y, mean, rstd

    @staticmethod
    def backward(ctx, y_grad, mean_grad, rstd_grad):
        logger.debug("GEMS GROUPNORM BACKWARD")
        y_grad = y_grad.contiguous()
        (x, weight, bias, mean, rstd) = ctx.saved_tensors
        num_groups = ctx.num_groups
        group_size = ctx.group_size
        N = ctx.N
        C = ctx.C
        HW = ctx.HW
        x_grad = torch.empty_like(x)
        grid = (N * num_groups,)
        with torch_device_fn.device(x.device):
            isCloseUnrollControl = False
            if weight is not None and bias is not None:
                isCloseUnrollControl = True
            # os.environ["TRITONXPU_OTHER_SIM"] = "1"
            # os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            # print(f'before x_grad = {x_grad.cpu()}')
            group_norm_backward_kernel[grid](
                y_grad,
                x,
                weight,
                mean,
                rstd,
                num_groups,
                group_size,
                x_grad,
                C,
                HW,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
                BLOCK_HW_SIZE=triton.next_power_of_2(HW),
                isCloseUnrollControl=isCloseUnrollControl,
            )
            # tmp_W = weight.view(1, C, 1, 1)
            # # dx_hat = weight * dY_val
            # tmp_dx_hat = tmp_W.cpu() * y_grad.cpu()
            # # print(f'dx_hat = {tmp_dx_hat}')
            # tmp_mean = mean.view(1, C, 1, 1)
            # # x = tl.where(xy_mask, X_val - mean, 0.0)
            # tmp_x = x.cpu() - tmp_mean.cpu()
            # # print(f'X_val - mean = {tmp_x}')
            # # print(f'pre_sum = dx_hat * x = {tmp_dx_hat * tmp_x}')

            # pre_sum = tmp_W.cpu() * tmp_x
            # # print(f'pre_sum.shape = {pre_sum.shape}')
            # # print(f'pre_sum[0][0] = {pre_sum[0][0]}')
            # # print(f'pre_sum[0][0].shape = {pre_sum[0][0].shape}')
            # # print(f'sum pre_sum[0][0] = {torch.sum(pre_sum[0][0])}')

            # tmp_grad_std = torch.sum(pre_sum, dim=[0, 2, 3])
            # # print(f'tmp_grad_std.shape = {tmp_grad_std.shape}')
            # # print(f'torch.sum(tmp_W * tmp_x) = {tmp_grad_std}')

        if weight is None and bias is None:
            return x_grad, None, None, None, None, None, None, None

        weight_grad = None if weight is None else torch.empty_like(weight)
        bias_grad = None if bias is None else torch.zeros_like(bias)
        # import os
        # os.environ["TRITON_INTERPRET"] = 1
        # os.environ["TRITONXPU_OTHER_SIM"] = "1"
        # os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"

        with torch_device_fn.device(x.device):
            # if N == 1 and C == 64 and HW == 1024 and num_groups == 64:
            #     os.environ["TRITONXPU_OTHER_SIM"] = "1"
            #     os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            if weight is not None and bias is not None:
                isCloseUnrollControl = True

            if N == 32 and C == 32 and HW == 1024 and num_groups == 8:
                weight_bias_backward_kernel_loop[(C, 1, 1)](
                    y_grad,
                    x,
                    mean,
                    rstd,
                    weight_grad,
                    bias_grad,
                    num_groups,
                    group_size,
                    N,
                    C,
                    HW,
                    BLOCK_N=1,
                    BLOCK_HW=triton.next_power_of_2(HW),
                    isCloseUnrollControl=True,
                    isCloseCoreTiling=True,
                )
            else:
                weight_bias_backward_kernel[(C, 1, 1)](
                    y_grad,
                    x,
                    mean,
                    rstd,
                    weight_grad,
                    bias_grad,
                    num_groups,
                    group_size,
                    N,
                    C,
                    HW,
                    BLOCK_N=triton.next_power_of_2(N),
                    BLOCK_HW=triton.next_power_of_2(HW),
                    isCloseUnrollControl=isCloseUnrollControl,
                )

            # if "TRITONXPU_OTHER_SIM" in os.environ:
            #     del os.environ["TRITONXPU_OTHER_SIM"]
            # if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            #     del os.environ["TRITONXPU_STORE_MASK_SIM"]

        # if "TRITON_INTERPRET" in os.environ:
        #     del os.environ["TRITON_INTERPRET"]
        # if "TRITONXPU_OTHER_SIM" in os.environ:
        #     del os.environ["TRITONXPU_OTHER_SIM"]
        # if "TRITONXPU_STORE_MASK_SIM" in os.environ:
        #     del os.environ["TRITONXPU_STORE_MASK_SIM"]
        return x_grad, None, None, None, None, weight_grad, bias_grad, None


def group_norm(x, weight, bias, N, C, HW, num_groups, eps):
    return GroupNorm.apply(x, N, C, HW, num_groups, weight, bias, eps)
