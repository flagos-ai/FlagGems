import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)
# When the reduced dimension is greater than MAX_C_MLU_SKIP_LAYERNORM_FORWARD,
# it is necessary to split the reduced dimension.
MAX_C_MLU_SKIP_LAYERNORM_FORWARD = 8192


def cfggen_middle_n():
    block_m = [1, 2, 4, 6, 8, 10]

    warps = [1]
    num_stages = [1, 3]
    configs = [
        triton.Config(
            {
                "BLOCK_ROW_SIZE": m,
            },
            num_warps=w,
            num_stages=s,
        )
        for m in block_m
        for w in warps
        for s in num_stages
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen_middle_n(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_middle_n_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    M,  # number of rows in X
    eps,  # epsilon to avoid division by zero
    N: tl.constexpr,  # number of columns in X
    BLOCK_ROW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROW_SIZE
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_ROW_SIZE

    cols_n = tl.arange(0, N)
    X += cols_n[None, :]
    R += cols_n[None, :]
    Y += cols_n[None, :]
    cols_off = tl.arange(0, N)[None, :]
    w = tl.load(W + cols_off)
    b = tl.load(B + cols_off)
    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_ROW_SIZE)
        mask = row_off[:, None] < M
        off = row_off[:, None] * N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        r = tl.load(R + off, mask, other=0.0).to(tl.float32)
        x += r

        # TODO: Use the following code as a fallback once the optimization for trans is complete.
        # mean = tl.sum(x_v, axis=1) / N
        # var = tl.sum(x_v * x_v, axis=1) / N - (mean * mean)
        # mean_bc = mean[:, None]

        x_v = tl.view(x, (BLOCK_ROW_SIZE, N))
        x_trans = tl.trans(x_v)
        mean = tl.sum(x_trans, axis=0) / N
        mean_bc = mean[:, None]
        var = tl.sum(x_trans * x_trans, axis=0) / N - (mean * mean)
        var = var[:, None]
        rstd = 1 / tl.sqrt(var + eps)
        x = x - mean_bc
        x_hat = x * rstd
        y = x_hat * w + b
        tl.store(Y + off, y, mask=mask)


def cfggen():
    block_m = [i for i in range(1, 36, 4)]  # [1, 2, 4]
    block_n = [i for i in range(64, 193, 64)]
    warps = [1]
    num_stages = [1, 3]
    configs = [
        triton.Config(
            {"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n}, num_warps=w, num_stages=s
        )
        for m in block_m
        for n in block_n
        for w in warps
        for s in num_stages
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    M,  # number of rows in X
    eps,  # epsilon to avoid division by zero
    N: tl.constexpr,  # number of columns in X
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = row < M
    Y += row * N
    X += row * N
    R += row * N

    # Compute mean
    _mean = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    # Compute variance
    _var = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _mean += x
        _var += x * x
    trans_mean = tl.trans(_mean)
    mean = tl.sum(trans_mean, axis=0) / N
    mean_bc = mean[:, None]
    trans_var = tl.trans(_var)
    var = tl.sum(trans_var, axis=0) / N - (mean * mean)
    var = var[:, None]
    rstd = 1 / tl.sqrt(var + eps)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        w = tl.load(W + cols, col_mask)
        b = tl.load(B + cols, col_mask)
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        x = tl.where(col_mask, x - mean_bc, 0.0)
        x_hat = x * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS_CAMBRICON SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        if N < MAX_C_MLU_SKIP_LAYERNORM_FORWARD:
            grid = lambda META: (
                min(triton.cdiv(M, META["BLOCK_ROW_SIZE"]), TOTAL_CORE_NUM),
            )
            with torch.cuda.device(x.device):
                skip_layer_norm_middle_n_kernel[grid](
                    y, x, residual, weight, bias, M, eps, N
                )
        else:
            grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
            with torch_device_fn.device(x.device):
                skip_layer_norm_kernel[grid](y, x, residual, weight, bias, M, eps, N)
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
