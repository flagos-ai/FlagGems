import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


def group_norm_kernel_opt_prune(configs, named_args, **kwargs):
    pruned_configs = []
    hw = kwargs["HW"]
    num_groups = named_args["num_groups"]
    all_sizes = []
    for config in configs:
        BLOCK_HW_SIZE = config.kwargs["BLOCK_HW_SIZE"]
        if BLOCK_HW_SIZE not in all_sizes:
            all_sizes.append(BLOCK_HW_SIZE)

    for config in configs:
        BLOCK_HW_SIZE = config.kwargs["BLOCK_HW_SIZE"]
        SPLIT = config.kwargs["SPLIT"]
        if (hw > 4096) and (BLOCK_HW_SIZE >= 4096) and (SPLIT <= 1):
            pruned_configs.append(config)
        elif (BLOCK_HW_SIZE >= hw) and (SPLIT <= num_groups):
            not_step_bigger = False
            for size in all_sizes:
                if (size < BLOCK_HW_SIZE) and (size > hw):
                    not_step_bigger = True
            if not not_step_bigger:
                pruned_configs.append(config)
    return pruned_configs


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"SPLIT": s, "BLOCK_HW_SIZE": size}, num_stages=3, num_warps=1)
        for size in [64, 256, 512, 1024, 2048, 4096, 5120]
        for s in [1, 4, 6, 8, 16]
    ],
    key=["X", "group_size", "C", "HW", "num_groups"],
    prune_configs_by={"early_config_prune": group_norm_kernel_opt_prune},
)
@triton.jit(do_not_specialize=["eps"])
def group_norm_kernel_opt(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    num_groups,
    eps,
    HW: tl.constexpr,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    div_v = tl.cdiv(num_groups, SPLIT)
    div_mod = num_groups % SPLIT
    split_group = pid % div_v
    split_n = pid // div_v
    real_num_elements = group_size * HW

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)
    if BLOCK_HW_SIZE >= HW:
        hw_offset = tl.arange(0, HW)
    hw_iter = tl.cdiv(HW, BLOCK_HW_SIZE)

    if W is None:
        W_ptr = None
    else:
        W_ptr = W + split_group * SPLIT * group_size
    if B is None:
        B_ptr = None
    else:
        B_ptr = B + split_group * SPLIT * group_size

    Mean_ptr = Mean + split_n * num_groups + split_group * SPLIT
    Rstd_ptr = Rstd + split_n * num_groups + split_group * SPLIT

    xy_offset = (
        split_n * C * HW
        + split_group * SPLIT * real_num_elements
        + group_offset[:, None] * HW
        + hw_offset[None, :]
    )

    ub = SPLIT
    if (div_mod != 0) and ((split_group + 1) == div_v):
        ub = div_mod
    for idx in range(0, ub):
        if BLOCK_HW_SIZE >= HW:
            tmp = tl.load(X + xy_offset, cache_modifier=".cg").to(tl.float32)
            mean = tl.sum(tmp) / real_num_elements
            x = tmp - mean
            var = tl.sum(x * x) / real_num_elements
            var = tl.rsqrt(var + eps)

            tl.store(Mean_ptr + idx, mean)
            tl.store(Rstd_ptr + idx, var)

            if W_ptr is None:
                weight = 1
            else:
                weight = tl.load(W_ptr + group_offset, cache_modifier=".cg")[:, None]
            if B_ptr is None:
                bias = 0
            else:
                bias = tl.load(B_ptr + group_offset, cache_modifier=".cg")[:, None]
            tmp = (tmp - mean) * var
            tmp = tmp * weight + bias
            tl.store(Y + xy_offset, tmp)
        else:
            mean = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], tl.float32)
            var = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], tl.float32)
            for idy in range(0, hw_iter):
                xy_mask = (
                    group_offset[:, None] < group_size
                    and (idy * BLOCK_HW_SIZE + hw_offset[None, :]) < HW
                )
                tmp = tl.load(
                    X + idy * BLOCK_HW_SIZE + xy_offset,
                    mask=xy_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                mean += tmp
                var += tmp * tmp
            mean = tl.sum(mean) / real_num_elements
            var = tl.sum(var) / real_num_elements - (mean * mean)
            var = tl.rsqrt(var + eps)
            tl.store(Mean_ptr + idx, mean)
            tl.store(Rstd_ptr + idx, var)

            if W_ptr is None:
                weight = 1
            else:
                weight = tl.load(W_ptr + group_offset, cache_modifier=".cg")[:, None]
            if B_ptr is None:
                bias = 0
            else:
                bias = tl.load(B_ptr + group_offset, cache_modifier=".cg")[:, None]

            for idy in range(0, hw_iter):
                xy_mask = (
                    group_offset[:, None] < group_size
                    and (idy * BLOCK_HW_SIZE + hw_offset[None, :]) < HW
                )
                tmp = tl.load(
                    X + idy * BLOCK_HW_SIZE + xy_offset,
                    mask=xy_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                tmp = (tmp - mean) * var
                tmp = tmp * weight + bias
                tl.store(Y + idy * BLOCK_HW_SIZE + xy_offset, tmp, mask=xy_mask)

        xy_offset += real_num_elements
        group_offset += group_size


def group_norm_backward_kernel_opt_prune(configs, named_args, **kwargs):
    pruned_configs = []
    hw = kwargs["HW"]
    all_sizes = []
    for config in configs:
        BLOCK_HW_SIZE = config.kwargs["BLOCK_HW_SIZE"]
        if BLOCK_HW_SIZE not in all_sizes:
            all_sizes.append(BLOCK_HW_SIZE)
    for config in configs:
        BLOCK_HW_SIZE = config.kwargs["BLOCK_HW_SIZE"]
        SPLIT = config.kwargs["SPLIT"]
        if (hw > 2048) and (BLOCK_HW_SIZE >= 2048) and (SPLIT <= 1):
            pruned_configs.append(config)
        elif BLOCK_HW_SIZE > hw:
            not_step_bigger = False
            for size in all_sizes:
                if (size < BLOCK_HW_SIZE) and (size > hw):
                    not_step_bigger = True
            if not not_step_bigger:
                pruned_configs.append(config)
    return pruned_configs


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"SPLIT": s, "BLOCK_HW_SIZE": size}, num_stages=3, num_warps=1)
        for s in [1, 4, 6, 8]
        for size in [64, 256, 512, 1024, 2048]
    ],
    prune_configs_by={"early_config_prune": group_norm_backward_kernel_opt_prune},
    key=["X", "group_size", "C", "HW", "num_groups"],
)
@triton.jit()
def group_norm_backward_kernel_opt(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW: tl.constexpr,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    div_v = tl.cdiv(num_groups, SPLIT)
    div_mod = num_groups % SPLIT
    split_group = pid % div_v
    split_n = pid // div_v
    real_num_elements = group_size * HW
    hw_iter = tl.cdiv(HW, BLOCK_HW_SIZE)

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)
    if BLOCK_HW_SIZE >= HW:
        hw_offset = tl.arange(0, HW)

    if W is None:
        W_ptr = None
    else:
        W_ptr = W + split_group * SPLIT * group_size

    Mean_ptr = Mean + split_n * num_groups + split_group * SPLIT
    Rstd_ptr = Rstd + split_n * num_groups + split_group * SPLIT

    xy_offset = (
        split_n * real_num_elements * num_groups
        + split_group * SPLIT * real_num_elements
        + group_offset[:, None] * HW
        + hw_offset[None, :]
    )

    ub = SPLIT
    if (div_mod != 0) and ((split_group + 1) == div_v):
        ub = div_mod
    for idx in range(0, ub):
        wb_mask = group_offset < C

        if W_ptr is None:
            weight = 1
        else:
            weight = tl.load(
                W_ptr + group_offset, mask=wb_mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)[:, None]
        rstd = tl.load(Rstd_ptr + idx).to(tl.float32)
        mean = tl.load(Mean_ptr + idx).to(tl.float32)

        if BLOCK_HW_SIZE >= HW:
            dY_val = tl.load(grad_y + xy_offset, cache_modifier=".cg").to(tl.float32)
            X_val = tl.load(X + xy_offset, cache_modifier=".cg").to(tl.float32)
            dx_hat = weight * dY_val

            x = X_val - mean

            grad_std = tl.sum(dx_hat * x)
            grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / real_num_elements
            grad_distance = 2 * x * grad_var
            grad_centered_mean = dx_hat * rstd + grad_distance
            grad_mean = -tl.sum(grad_centered_mean) / real_num_elements
            grad_X = grad_centered_mean + grad_mean

            tl.store(grad_x + xy_offset, grad_X)
        else:
            dx_hat = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], tl.float32)
            x = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], tl.float32)
            dx_hat_x = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], tl.float32)

            for idy in range(0, hw_iter):
                xy_mask = (
                    group_offset[:, None] < C
                    and (idy * BLOCK_HW_SIZE + hw_offset[None, :]) < HW
                )
                dY_val = tl.load(
                    grad_y + idy * BLOCK_HW_SIZE + xy_offset,
                    mask=xy_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                X_val = tl.load(
                    X + idy * BLOCK_HW_SIZE + xy_offset,
                    mask=xy_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                dx_hat_tmp = weight * dY_val
                dx_hat += dx_hat_tmp
                x_tmp = tl.where(xy_mask, X_val - mean, 0.0)
                x += x_tmp
                dx_hat_x += dx_hat_tmp * x_tmp

            grad_std = tl.sum(dx_hat_x)
            grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / real_num_elements

            grad_distance = 2 * x * grad_var
            grad_centered_mean = dx_hat * rstd + grad_distance
            grad_mean = -tl.sum(grad_centered_mean) / real_num_elements

            for idy in range(0, hw_iter):
                xy_mask = (
                    group_offset[:, None] < C
                    and (idy * BLOCK_HW_SIZE + hw_offset[None, :]) < HW
                )
                dY_val = tl.load(
                    grad_y + idy * BLOCK_HW_SIZE + xy_offset,
                    mask=xy_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                X_val = tl.load(
                    X + idy * BLOCK_HW_SIZE + xy_offset,
                    mask=xy_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
                dx_hat = weight * dY_val
                x = tl.where(xy_mask, X_val - mean, 0.0)
                grad_distance = 2 * x * grad_var
                grad_centered_mean = dx_hat * rstd + grad_distance
                grad_X = grad_centered_mean + grad_mean
                tl.store(grad_x + idy * BLOCK_HW_SIZE + xy_offset, grad_X, mask=xy_mask)

        xy_offset += real_num_elements
        group_offset += group_size


def weight_bias_backward_kernel_opt_prune(configs, named_args, **kwargs):
    pruned_configs = []
    pruned_configs_cached = []
    n = named_args["N"]
    hw = kwargs["HW"]
    all_sizes = []
    for config in configs:
        BLOCK_HW_SIZE = config.kwargs["BLOCK_HW_SIZE"]
        if BLOCK_HW_SIZE not in all_sizes:
            all_sizes.append(BLOCK_HW_SIZE)
    for config in configs:
        BLOCK_HW_SIZE = config.kwargs["BLOCK_HW_SIZE"]
        BLOCK_N = config.kwargs["BLOCK_N"]
        if (hw > 2048) and (BLOCK_HW_SIZE >= 2048) and (BLOCK_N <= 4):
            pruned_configs_cached.append(config)
        elif BLOCK_HW_SIZE > hw:
            not_step_bigger = False
            for size in all_sizes:
                if (size < BLOCK_HW_SIZE) and (size > hw):
                    not_step_bigger = True
            if not not_step_bigger:
                pruned_configs_cached.append(config)
    # remove some block n
    for config in pruned_configs_cached:
        block_n = config.kwargs["BLOCK_N"]
        if n % block_n == 0:
            pruned_configs.append(config)
    return pruned_configs


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn, "BLOCK_HW_SIZE": size}, num_stages=3, num_warps=1)
        for bn in [1, 4, 8, 16]
        for size in [512, 1024, 2048]
    ],
    prune_configs_by={"early_config_prune": weight_bias_backward_kernel_opt_prune},
    key=["X", "N", "C", "HW", "num_groups"],
)
@triton.jit
def weight_bias_backward_kernel_opt(
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
    HW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    C_SPLIT = tl.cdiv(C, pnum)
    N_SPLIT = tl.cdiv(N, BLOCK_N)
    hw_iter = tl.cdiv(HW, BLOCK_HW_SIZE)

    n_offset = tl.arange(0, BLOCK_N)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)
    if BLOCK_HW_SIZE >= HW:
        hw_offset = tl.arange(0, HW)

    lb = pid * C_SPLIT
    ub = tl.minimum((pid + 1) * C_SPLIT, C)
    for c_start in range(lb, ub):
        if BLOCK_HW_SIZE >= HW:
            dY_ptr = dY + c_start * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            x_ptr = X + c_start * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            grad_y = tl.load(dY_ptr, cache_modifier=".cg").to(tl.float32)

            x = tl.load(x_ptr, cache_modifier=".cg")
            x_f32 = x.to(tl.float32)

            mean_ptr = Mean + c_start // group_size + n_offset * num_groups
            rstd_ptr = Rstd + c_start // group_size + n_offset * num_groups

            mean = tl.load(mean_ptr, cache_modifier=".cg").to(tl.float32)[:, None]
            rstd = tl.load(rstd_ptr, cache_modifier=".cg").to(tl.float32)[:, None]

            dB_val = tl.sum(grad_y)
            dW_val = tl.sum((x_f32 - mean) * rstd * grad_y)

            for n_start in range(1, N_SPLIT):
                new_n_offset = n_start * BLOCK_N + n_offset

                dY_ptr = (
                    dY
                    + c_start * HW
                    + new_n_offset[:, None] * C * HW
                    + hw_offset[None, :]
                )
                x_ptr = (
                    X
                    + c_start * HW
                    + new_n_offset[:, None] * C * HW
                    + hw_offset[None, :]
                )
                grad_y = tl.load(dY_ptr, cache_modifier=".cg").to(tl.float32)

                x = tl.load(x_ptr, cache_modifier=".cg")
                x_f32 = x.to(tl.float32)

                mean_ptr = Mean + c_start // group_size + new_n_offset * num_groups
                rstd_ptr = Rstd + c_start // group_size + new_n_offset * num_groups

                mean = tl.load(mean_ptr, cache_modifier=".cg").to(tl.float32)[:, None]
                rstd = tl.load(rstd_ptr, cache_modifier=".cg").to(tl.float32)[:, None]

                dB_val += tl.sum(grad_y)
                dW_val += tl.sum((x_f32 - mean) * rstd * grad_y)

            if dW is not None:
                tl.store(dW + c_start, dW_val.to(x.dtype))
            if dB is not None:
                tl.store(dB + c_start, dB_val.to(x.dtype))
        else:
            xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW

            dY_ptr = dY + c_start * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            x_ptr = X + c_start * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
            grad_y = tl.load(dY_ptr, cache_modifier=".cg").to(tl.float32)

            x = tl.load(x_ptr, cache_modifier=".cg")
            x_f32 = x.to(tl.float32)

            mean_ptr = Mean + c_start // group_size + n_offset * num_groups
            rstd_ptr = Rstd + c_start // group_size + n_offset * num_groups

            mean = tl.load(mean_ptr, cache_modifier=".cg").to(tl.float32)[:, None]
            rstd = tl.load(rstd_ptr, cache_modifier=".cg").to(tl.float32)[:, None]

            dB_val = tl.sum(grad_y)
            dW_val = tl.sum((x_f32 - mean) * rstd * grad_y)

            for idx in range(1, hw_iter):
                xy_mask = (
                    n_offset[:, None] < N
                    and (idx * BLOCK_HW_SIZE + hw_offset[None, :]) < HW
                )
                dY_ptr = (
                    dY
                    + c_start * HW
                    + n_offset[:, None] * C * HW
                    + hw_offset[None, :]
                    + idx * BLOCK_HW_SIZE
                )
                x_ptr = (
                    X
                    + c_start * HW
                    + n_offset[:, None] * C * HW
                    + hw_offset[None, :]
                    + idx * BLOCK_HW_SIZE
                )

                grad_y = tl.load(
                    dY_ptr, mask=xy_mask, other=0.0, cache_modifier=".cg"
                ).to(tl.float32)
                x = tl.load(x_ptr, mask=xy_mask, other=0.0, cache_modifier=".cg")
                x_f32 = x.to(tl.float32)
                dB_val += tl.sum(grad_y)
                x_f32 = tl.where(xy_mask, x_f32 - mean, 0.0)
                dW_val += tl.sum(x_f32 * rstd * grad_y)

            for n_start in range(1, N_SPLIT):
                new_n_offset = n_start * BLOCK_N + n_offset
                xy_mask = new_n_offset[:, None] < N and hw_offset[None, :] < HW

                dY_ptr = (
                    dY
                    + c_start * HW
                    + new_n_offset[:, None] * C * HW
                    + hw_offset[None, :]
                )
                x_ptr = (
                    X
                    + c_start * HW
                    + new_n_offset[:, None] * C * HW
                    + hw_offset[None, :]
                )
                grad_y = tl.load(
                    dY_ptr, mask=xy_mask, other=0.0, cache_modifier=".cg"
                ).to(tl.float32)

                x = tl.load(x_ptr, mask=xy_mask, other=0.0, cache_modifier=".cg")
                x_f32 = x.to(tl.float32)

                mean_ptr = Mean + c_start // group_size + new_n_offset * num_groups
                rstd_ptr = Rstd + c_start // group_size + new_n_offset * num_groups

                mean = tl.load(mean_ptr, cache_modifier=".cg").to(tl.float32)[:, None]
                rstd = tl.load(rstd_ptr, cache_modifier=".cg").to(tl.float32)[:, None]

                dB_val += tl.sum(grad_y)
                dW_val += tl.sum((x_f32 - mean) * rstd * grad_y)

                for idx in range(1, hw_iter):
                    xy_mask = (
                        new_n_offset[:, None] < N
                        and (idx * BLOCK_HW_SIZE + hw_offset[None, :]) < HW
                    )
                    dY_ptr = (
                        dY
                        + c_start * HW
                        + new_n_offset[:, None] * C * HW
                        + hw_offset[None, :]
                        + idx * BLOCK_HW_SIZE
                    )
                    x_ptr = (
                        X
                        + c_start * HW
                        + new_n_offset[:, None] * C * HW
                        + hw_offset[None, :]
                        + idx * BLOCK_HW_SIZE
                    )

                    grad_y = tl.load(
                        dY_ptr, mask=xy_mask, other=0.0, cache_modifier=".cg"
                    ).to(tl.float32)
                    x = tl.load(x_ptr, mask=xy_mask, other=0.0, cache_modifier=".cg")
                    x_f32 = x.to(tl.float32)
                    dB_val += tl.sum(grad_y)
                    x_f32 = tl.where(xy_mask, x_f32 - mean, 0.0)
                    dW_val += tl.sum(x_f32 * rstd * grad_y)
                if dW is not None:
                    tl.store(dW + c_start, dW_val.to(x.dtype))
                if dB is not None:
                    tl.store(dB + c_start, dB_val.to(x.dtype))


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, N, C, HW, num_groups, weight=None, bias=None, eps=1e-05):
        logger.debug("GEMS_CAMBRICON GROUPNORM FORWARD")
        group_size = C // num_groups
        x = x.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        grid = lambda meta: (N * triton.cdiv(num_groups, meta["SPLIT"]),)

        with torch_device_fn.device(x.device):
            group_norm_kernel_opt[grid](
                x,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                C,
                num_groups,
                eps,
                HW=HW,
                BLOCK_GROUP_SIZE=group_size,
            )
        if x.requires_grad:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.num_groups = num_groups
            ctx.group_size = group_size
            ctx.N = N
            ctx.C = C
            ctx.HW = HW
        return y, mean, rstd

    @staticmethod
    def backward(ctx, y_grad, mean_grad, rstd_grad):
        logger.debug("GEMS_CAMBRICON GROUPNORM BACKWARD")
        y_grad = y_grad.contiguous()
        (x, weight, bias, mean, rstd) = ctx.saved_tensors
        num_groups = ctx.num_groups
        group_size = ctx.group_size
        N = ctx.N
        C = ctx.C
        HW = ctx.HW
        x_grad = torch.empty_like(x)
        grid = lambda meta: (N * triton.cdiv(num_groups, meta["SPLIT"]),)
        with torch_device_fn.device(x.device):
            group_norm_backward_kernel_opt[grid](
                y_grad,
                x,
                weight,
                mean,
                rstd,
                num_groups,
                group_size,
                x_grad,
                C,
                HW=HW,
                BLOCK_GROUP_SIZE=group_size,
            )
        if weight is None and bias is None:
            return x_grad, None, None, None, None, None, None, None

        weight_grad = None if weight is None else torch.empty_like(weight)
        bias_grad = None if bias is None else torch.empty_like(bias)
        with torch_device_fn.device(x.device):
            weight_bias_backward_kernel_opt[(TOTAL_CORE_NUM, 1, 1)](
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
                HW=HW,
            )
        return x_grad, None, None, None, None, weight_grad, bias_grad, None


def group_norm(x, weight, bias, N, C, HW, num_groups, eps):
    return GroupNorm.apply(x, N, C, HW, num_groups, weight, bias, eps)
