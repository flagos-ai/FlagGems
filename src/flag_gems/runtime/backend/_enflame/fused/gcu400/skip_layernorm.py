import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ...gcu400.utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

NUM_SIPS = 24
MAX_BLOCK_N = 32768


@libentry()
@triton.jit(do_not_specialize=["M", "N", "eps"])
def skip_layer_norm_kernel_2d(
    Y, X, R, W, B, stride, M, N, eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += (pid * BLOCK_SIZE_M - 1) * y_stride_r
    X += (pid * BLOCK_SIZE_M - 1) * x_stride_r
    R += (pid * BLOCK_SIZE_M - 1) * r_stride_r
    for i in range(BLOCK_SIZE_M):
        if pid * BLOCK_SIZE_M + i < M:
            Y += y_stride_r
            X += x_stride_r
            R += r_stride_r

    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    for row_start in tl.range(pid * BLOCK_M, M, num_pids * BLOCK_M):
        x_blk = tl.make_block_ptr(
            base=X, shape=(M, N), strides=(stride, 1),
            offsets=(row_start, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
        )
        r_blk = tl.make_block_ptr(
            base=R, shape=(M, N), strides=(stride, 1),
            offsets=(row_start, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
        )
        y_blk = tl.make_block_ptr(
            base=Y, shape=(M, N), strides=(stride, 1),
            offsets=(row_start, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
        )

        x = tl.load(x_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        r = tl.load(r_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        x = x + r

        mean = tl.sum(x, axis=1) / N
        diff = x - mean[:, None]
        var = tl.sum(diff * diff, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        y = w[None, :] * diff * rstd[:, None] + b[None, :]
        tl.store(y_blk, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.jit(do_not_specialize=["M", "N", "eps", "stride"])
def skip_layer_norm_kernel_large_n(
    Y, X, R, W, B, stride, M, N, eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    for row in tl.range(pid, M, num_pids):
        base = row * stride

        total_sum = 0.0
        total_sum2 = 0.0
        for start in tl.range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            tile_mask = cols < N
            x = tl.load(X + base + cols, mask=tile_mask, other=0.0).to(tl.float32)
            r = tl.load(R + base + cols, mask=tile_mask, other=0.0).to(tl.float32)
            xr = x + r
            total_sum += tl.sum(xr, axis=0)
            total_sum2 += tl.sum(xr * xr, axis=0)
        mean = total_sum / N
        var = total_sum2 / N - mean * mean
        var = tl.maximum(var, 0.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        for start in tl.range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            tile_mask = cols < N
            x = tl.load(X + base + cols, mask=tile_mask, other=0.0).to(tl.float32)
            r = tl.load(R + base + cols, mask=tile_mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=tile_mask, other=0.0).to(tl.float32)
            b_val = tl.load(B + cols, mask=tile_mask, other=0.0).to(tl.float32)
            val = w * ((x + r) - mean) * rstd + b_val
            tl.store(Y + base + cols, val.to(Y.dtype.element_ty), mask=tile_mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        if N > MAX_BLOCK_N:
            grid_size = min(M, NUM_SIPS * 2)
            with torch_device_fn.device(x.device):
                skip_layer_norm_kernel_large_n[(grid_size,)](
                    y, x, residual, weight, bias,
                    N, M, N, eps,
                    MAX_BLOCK_N,
                    num_warps=1,
                )
        else:
            BLOCK_N = triton.next_power_of_2(N)
            BLOCK_M = max(1, min(64, 32768 // BLOCK_N))
            num_row_blocks = (M + BLOCK_M - 1) // BLOCK_M
            grid_size = min(num_row_blocks, NUM_SIPS * 2)
            with torch_device_fn.device(x.device):
                skip_layer_norm_kernel_2d[(grid_size,)](
                    y, x, residual, weight, bias,
                    N, M, N, eps,
                    BLOCK_M, BLOCK_N,
                    num_warps=1,
                )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
