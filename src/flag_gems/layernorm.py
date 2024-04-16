import torch
import triton
import triton.language as tl
from .__libentry__ import libentry
import math


def cfggen():
    warps = [1, 2, 4, 8, 16, 32]
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": 1, "BLOCK_COL_SIZE": 2048}, num_warps=w)
        for w in warps
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def layer_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = row < M
    X += row * N
    Y += row * N

    # Compute mean
    _mean = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    _mean /= N
    mean = tl.sum(_mean, axis=1)[:, None]

    # Compute variance
    _var = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        x = tl.where(col_mask, x - mean, 0.0)
        _var += x * x
    _var /= N
    var = tl.sum(_var, axis=1)[:, None]
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        w = tl.load(W + cols, col_mask)
        b = tl.load(B + cols, col_mask)
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": 2048}, num_warps=w)
        for m in [1, 2, 4, 8]
        for w in [4, 8, 16]
    ],
    key=["M", "N"],
)
@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean)
    rstd = tl.load(Rstd)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask)
        x = tl.load(X + cols[None, :], mask)
        x_hat = (x - mean) * rstd
        w = tl.load(W + cols, mask=cols < N)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2)
    dx_3 = tl.sum(dx_part3)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask)
        x = tl.load(X + cols[None, :], mask)
        w = tl.load(W + cols, mask=cols < N)
        x_hat = (x - mean) * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx.to(dy.dtype), mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": 1024, "BLOCK_COL_SIZE": n}, num_warps=w)
        for n in [1, 2, 4, 8]
        for w in [4, 8]
    ],
    key=["N"],
)
@triton.jit
def wb_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)[None, :]
    col_mask = pid < N
    dY += pid
    X += pid
    dW += pid
    dB += pid
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)
        row_mask = rows[:, None] < M
        mask = row_mask and col_mask
        dy = tl.load(dY + rows[:, None] * N, mask).to(tl.float32)
        x = tl.load(X + rows[:, None] * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M)[:, None].to(tl.float32)
        x_hat = (x - mean) * rstd
        accW += dy * x_hat
        accB += dy
    dw = tl.sum(accW, axis=0)
    db = tl.sum(accB, axis=0)
    tl.store(dW, dw.to(dW.dtype.element_ty)[None, :], mask=col_mask)
    tl.store(dB, db.to(dB.dtype.element_ty)[None, :], mask=col_mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
        if __debug__:
            print("GEMS LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty(M, dtype=x.dtype, device=x.device)
        rstd = torch.empty(M, dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)

        layer_norm_kernel[grid](x, y, weight, bias, mean, rstd, M, N, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.M = M
        ctx.N = N
        return y, mean, rstd

    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        if __debug__:
            print("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        (x, weight, mean, rstd) = ctx.saved_tensors
        M = ctx.M
        N = ctx.N
        in_grad = torch.empty_like(x)
        grid = (M, 1, 1)
        layer_norm_backward_kernel[grid](out_grad, x, weight, mean, rstd, in_grad, M, N)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
        weight_grad = torch.empty_like(weight)
        bias_grad = torch.empty_like(weight)
        wb_backward_kernel[grid](out_grad, x, mean, rstd, weight_grad, bias_grad, M, N)
        return in_grad, None, weight_grad, bias_grad, None, None


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    return LayerNorm.apply(x, normalized_shape, weight, bias, eps, cudnn_enable)
