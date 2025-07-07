import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.type_utils import get_accumulator_dtype

logger = logging.getLogger(__name__)
Tensor = torch.Tensor


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("instancenorm"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def instance_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,  # M = B * C
    N,
    C,
    eps,
    TILE_N: tl.constexpr,
    HAS_WEIGHT_BIAS: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    m_mask = pid < M
    c_offsets = pid % C

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if HAS_WEIGHT_BIAS:
        w = tl.load(weight_ptr + c_offsets, mask=m_mask)
        b = tl.load(bias_ptr + c_offsets, mask=m_mask)
        out = (x - m) * rstd * w + b
    else:
        out = (x - m) * rstd

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("instancenorm"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def instance_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,  # M = B * C
    N,
    C,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    HAS_WEIGHT_BIAS: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M
    c_offsets = m_offsets % C

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if HAS_WEIGHT_BIAS:
        w = tl.load(weight_ptr + c_offsets, mask=m_mask)
        b = tl.load(bias_ptr + c_offsets, mask=m_mask)
        out = (x - m[:, None]) * rstd[:, None] * w[:, None] + b[:, None]
    else:
        out = (x - m[:, None]) * rstd[:, None]

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("instance_norm_loop"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def instance_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,  # M = B * C
    N,
    C,
    eps,
    TILE_N: tl.constexpr,
    HAS_WEIGHT_BIAS: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    m_mask = pid < M
    c_offsets = pid % C

    # Compute mean
    m = tl.zeros((TILE_N,), dtype=tl.float32)  # mean
    s = tl.zeros((TILE_N,), dtype=tl.float32)  # sum((x - m)^2)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # the last step
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = tl.math.rsqrt(var + eps)
    m = final_m
    # Write mean / rstd
    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if HAS_WEIGHT_BIAS:
        w = tl.load(weight_ptr + c_offsets, mask=m_mask)
        b = tl.load(bias_ptr + c_offsets, mask=m_mask)
    else:
        w = 1
        b = 0

    # reverse the order of the second sweep
    # Normalize and apply linear transformation
    prev_multiple = prev_multiple_of(N, TILE_N)
    # the first step, masking is needed
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("instancenorm"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def instance_norm_use_running_stats_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,  # pointer to the mean
    running_var_ptr,  # pointer to the var
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,  # M = B * C
    N,
    C,
    eps,
    TILE_N: tl.constexpr,
    HAS_WEIGHT_BIAS: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    m_mask = pid < M
    c_offsets = pid % C

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.load(running_mean_ptr + c_offsets, mask=m_mask)
    var = tl.load(running_var_ptr + c_offsets, mask=m_mask)
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if HAS_WEIGHT_BIAS:
        w = tl.load(weight_ptr + c_offsets, mask=m_mask)
        b = tl.load(bias_ptr + c_offsets, mask=m_mask)
        out = (x - m) * rstd * w + b
    else:
        out = (x - m) * rstd

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@triton.jit
def update_running_stats_kernel(
    mean_ptr,  # pointer to the mean
    rstd_ptr,  # pointer to the 1/std
    running_mean_ptr,
    running_var_ptr,
    momentum,
    B,
    C,
    N,
    eps,
    BLOCK_BATCH_SIZE: tl.constexpr = 1,
    BLOCK_CHANNEL_SIZE: tl.constexpr = 2048,
):
    cid = tl.program_id(0) * BLOCK_CHANNEL_SIZE + tl.arange(0, BLOCK_CHANNEL_SIZE)
    col_mask = cid < C
    running_mean = tl.load(running_mean_ptr + cid, mask=col_mask).to(tl.float32)
    running_var = tl.load(running_var_ptr + cid, mask=col_mask).to(tl.float32)

    new_mean = tl.zeros((BLOCK_CHANNEL_SIZE,), dtype=tl.float32)
    new_var = tl.zeros((BLOCK_CHANNEL_SIZE,), dtype=tl.float32)
    for b in range(0, B, BLOCK_BATCH_SIZE):
        bid = b * BLOCK_BATCH_SIZE + tl.arange(0, BLOCK_BATCH_SIZE)[:, None]
        row_mask = bid < B
        mask = row_mask and col_mask[None, :]
        mean = tl.load(mean_ptr + bid * C + cid[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        rstd = tl.load(rstd_ptr + bid * C + cid[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        var = (
            (1 / (rstd * rstd) + eps) * N / (N - 1)
        )  # NOTE: use unbiased var to update running_var

        new_mean += tl.sum(mean, axis=0)
        new_var += tl.sum(var, axis=0)

    new_running_mean = (1 - momentum) * running_mean + momentum * new_mean / B
    new_running_var = (1 - momentum) * running_var + momentum * new_var / B

    tl.store(running_mean_ptr + cid, new_running_mean, mask=col_mask)
    tl.store(running_var_ptr + cid, new_running_var, mask=col_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("instance_norm_backward"),
    key=["M", "N", "C"],
)
@triton.jit
def instance_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,  # [B, C]
    Rstd,  # [B, C]
    dX,
    M,  # M = B * C
    N,
    C,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    HAS_WEIGHT_BIAS: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    c_offsets = pid % C
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean, mask=row_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask, other=1.0).to(tl.float32)
    if HAS_WEIGHT_BIAS:
        w = tl.load(W + c_offsets, mask=row_mask).to(tl.float32)
    else:
        w = 1

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("instance_norm_weight_bias_backward"),
    key=["N", "B", "C"],
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,  # [B, C]
    Rstd,  # [B, C]
    dW,
    dB,
    M,
    N,
    B,
    C,
    BLOCK_BATCH_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    cid = tl.program_id(0)[None]
    cid = cid[:, None]
    dW += cid
    dB += cid
    c_mask = cid < C

    accW = tl.zeros([BLOCK_BATCH_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_BATCH_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for b_off in range(0, B, BLOCK_BATCH_SIZE):
        bid = b_off + tl.arange(0, BLOCK_BATCH_SIZE)[:, None]
        mid = bid * C + cid
        row_mask = bid < B
        mean = tl.load(Mean + mid, mask=row_mask).to(tl.float32)
        rstd = tl.load(Rstd + mid, mask=row_mask).to(tl.float32)
        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(dY + mid * N + cols[None, :], mask).to(tl.float32)
            x = tl.load(X + mid * N + cols[None, :], mask).to(tl.float32)
            x = tl.where(mask, x - mean, 0.0)
            x_hat = x * rstd
            accW += dy * x_hat
            accB += dy
    dw = tl.sum(accW)
    db = tl.sum(accB)
    tl.store(dW, dw, mask=c_mask)
    tl.store(dB, db, mask=c_mask)


class InstanceNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight=None,
        bias=None,
        running_mean=None,
        running_var=None,
        use_input_stats=False,
        momentum=0.1,
        eps=1e-05,
        cudnn_enable=False,
    ):
        logger.debug("GEMS INSTANCENORM FORWARD")
        assert len(x.shape) in [
            3,
            4,
            5,
        ], f"x.shape should be [B, C, N] or [B, C, H, W] or [B, C, H, W, L], but got {x.shape}"
        B, C = x.shape[:2]
        N = math.prod(x.shape[2:])
        M = x.numel() // N

        x = x.contiguous()
        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        y = torch.empty_like(x)

        has_weight_bias = weight is not None
        if has_weight_bias:
            assert weight is not None and bias is not None

        has_running_stats = running_mean is not None
        if has_running_stats:
            assert (
                N > 1
            ), f"Expected more than 1 spatial element when training, got input size {x.shape}"
            assert (
                running_mean is not None and running_var is not None
            ), "running_mean and running_var should not both be None"
            assert (
                running_mean.shape == running_var.shape and running_mean.shape[0] == C
            ), f"running_mean and running_var should have shape as {[C,]}"
            assert (
                running_mean.dtype == running_var.dtype
            ), "running_mean and running_var should have the same dtype"
        if not use_input_stats:
            assert (
                has_running_stats
            ), "Expected running_mean and running_var to be defined when use_input_stats is False"

        # NOTE: when the input is half-precision(either float16 or bfloat16)
        # these statistical data saved for backward is in single precision
        acc_type = get_accumulator_dtype(x.dtype)
        mean = torch.empty(size=(B, C), dtype=acc_type, device=x.device)
        rstd = torch.empty(size=(B, C), dtype=acc_type, device=x.device)

        with torch_device_fn.device(x.device):
            if use_input_stats:
                if N <= 128:
                    TILE_N = triton.next_power_of_2(N)
                    TILE_M = triton.cdiv(1024, TILE_N)
                    grid = (triton.cdiv(M, TILE_M), 1, 1)
                    instance_norm_persistent_kernel_multiline[grid](
                        x,
                        y,
                        weight,
                        bias,
                        mean,
                        rstd,
                        M,
                        N,
                        C,
                        eps,
                        TILE_M,
                        TILE_N,
                        HAS_WEIGHT_BIAS=has_weight_bias,
                    )
                elif N <= 4096:
                    TILE_N = triton.next_power_of_2(N)
                    grid = (M, 1, 1)
                    instance_norm_persistent_kernel[grid](
                        x,
                        y,
                        weight,
                        bias,
                        mean,
                        rstd,
                        M,
                        N,
                        C,
                        eps,
                        TILE_N,
                        HAS_WEIGHT_BIAS=has_weight_bias,
                    )
                else:
                    grid = (M, 1, 1)
                    instance_norm_loop_kernel[grid](
                        x,
                        y,
                        weight,
                        bias,
                        mean,
                        rstd,
                        M,
                        N,
                        C,
                        eps,
                        HAS_WEIGHT_BIAS=has_weight_bias,
                    )
                if has_running_stats and use_input_stats:  # update running stats
                    grid = lambda meta: (
                        triton.cdiv(C, meta["BLOCK_CHANNEL_SIZE"]),
                        1,
                        1,
                    )
                    update_running_stats_kernel[grid](
                        mean,
                        rstd,
                        running_mean,
                        running_var,
                        momentum,
                        B,
                        C,
                        N,
                        eps,
                    )
            else:  # use running stats instead of input stats
                TILE_N = triton.next_power_of_2(N)
                grid = (M, 1, 1)
                instance_norm_use_running_stats_kernel[grid](
                    x,
                    y,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    mean,
                    rstd,
                    M,
                    N,
                    C,
                    eps,
                    TILE_N,
                    HAS_WEIGHT_BIAS=has_weight_bias,
                )

        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.M = M
        ctx.N = N
        ctx.C = C
        ctx.has_weight_bias = has_weight_bias
        return y

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS INSTANCENORM BACKWARD")
        out_grad = out_grad.contiguous()
        (x, weight, mean, rstd) = ctx.saved_tensors
        M = ctx.M
        N = ctx.N
        C = ctx.C
        B = M // C

        with torch_device_fn.device(x.device):
            in_grad = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
            instance_norm_backward_kernel[grid](
                out_grad,
                x,
                weight,
                mean,
                rstd,
                in_grad,
                M,
                N,
                C,
                HAS_WEIGHT_BIAS=ctx.has_weight_bias,
            )

            if ctx.has_weight_bias:
                grid = lambda meta: (C, 1, 1)
                weight_grad = torch.empty_like(weight)
                bias_grad = torch.empty_like(weight)
                weight_bias_backward_kernel[grid](
                    out_grad, x, mean, rstd, weight_grad, bias_grad, M, N, B, C
                )
            else:
                weight_grad = None
                bias_grad = None
        return in_grad, weight_grad, bias_grad, None, None, None, None, None, None


def instance_norm(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
    cudnn_enable: bool = False,
) -> Tensor:
    r"""Applies Instance Normalization for each channel in each data sample in a
    batch.
    Inputs:
        input: input tensor of shape :math:`(N, C, *)`
        weight: weight tensor of shape :math:`(C)`
        bias: bias tensor of shape :math:`(C)`
        running_mean: running mean tensor of shape :math:`(C)`
        running_var: running variance tensor of shape :math:`(C)`
        use_input_stats: whether to use the mean and variance of the input tensor
        momentum: momentum value for the running mean and variance
        eps: epsilon value for numerical stability
        cudnn_enable: whether to use cudnn for normalization
    Returns:
        output tensor of shape :math:`(N, C, *)`
    """

    return InstanceNorm.apply(
        input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps
    )
