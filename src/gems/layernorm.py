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


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    if __debug__:
        print("GEMS LAYERNORM")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    x = x.contiguous()
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=x.dtype, device=x.device)
    rstd = torch.empty(M, dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)

    layer_norm_kernel[grid](x, y, weight, bias, mean, rstd, M, N, eps)
    return y, mean, rstd
