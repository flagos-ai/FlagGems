import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.type_utils import get_accumulator_dtype

pow = tl_extra_shim.pow


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_common_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)

    X = X + row * N
    Y = Y + row * N

    # Compute mean
    mean = 0.0
    var = 0.0
    num_pid_n = tl.cdiv(N, TILE_N)
    x_ptr_desc = tl.make_block_ptr(
        base=X,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )
    for off_n in range(0, num_pid_n):
        a = tl.load(
            x_ptr_desc,
            boundary_check=[0],
        )
        mean += tl.sum(a)
        var += tl.sum(pow(a, (2).to(X.type.element_ty)))

        x_ptr_desc = tl.advance(x_ptr_desc, [TILE_N])

    mean = mean / N
    var = var / N - (mean * mean)
    rstd = tl.math.rsqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    x_ptr_desc = tl.make_block_ptr(
        base=X,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )

    weight_ptr_desc = tl.make_block_ptr(
        base=W,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )

    bias_ptr_desc = tl.make_block_ptr(
        base=B,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )
    y_ptr_desc = tl.make_block_ptr(
        base=Y,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )

    for off_n in range(0, num_pid_n):
        a = tl.load(
            x_ptr_desc,
            boundary_check=[0],
        )
        x_hat = (a - mean) * rstd

        x_ptr_desc = tl.advance(x_ptr_desc, [TILE_N])

        if W is None:
            w = 1
        else:
            w = tl.load(
                weight_ptr_desc,
                boundary_check=[0],
            )
            weight_ptr_desc = tl.advance(weight_ptr_desc, [TILE_N])

        if B is None:
            b = 0
        else:
            b = tl.load(
                bias_ptr_desc,
                boundary_check=[0],
            )
            bias_ptr_desc = tl.advance(bias_ptr_desc, [TILE_N])

        y = x_hat * w + b
        tl.store(
            y_ptr_desc,
            y,
            boundary_check=[0],
        )
        y_ptr_desc = tl.advance(y_ptr_desc, [TILE_N])


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_backward"),
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

    mean = tl.load(Mean, mask=row_mask).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask).to(tl.float32)

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
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
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
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_bias_backward_kernel(
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
        x = tl.where(col_mask, x - mean, 0.0)
        x_hat = x * rstd
        accW += dy * x_hat
        accB += dy
    if dW is not None:
        dw = tl.sum(accW, axis=0)
        tl.store(dW + pid, dw[None, :], mask=col_mask)
    if dB is not None:
        db = tl.sum(accB, axis=0)
        tl.store(dB + pid, db[None, :], mask=col_mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
        logging.debug("GEMS LAYERNORM FORWARD")
        # dim = x.ndim - len(normalized_shape)
        # M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        M = x.numel() // N

        x = x.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y = torch.empty_like(x)

        # NOTE: when the input is half-precision(either float16 or bfloat16)
        # these statistical data saved for backward is in single precision
        acc_type = get_accumulator_dtype(x.dtype)
        mean = torch.empty(M, dtype=acc_type, device=x.device)
        rstd = torch.empty(M, dtype=acc_type, device=x.device)

        TILE_N = 512
        with torch_device_fn.device(x.device):
            layer_norm_common_kernel[(M,)](
                x, y, weight, bias, mean, rstd, M, N, eps, TILE_N=TILE_N
            )

        if x.requires_grad:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.M = M
            ctx.N = N
        return y, mean, rstd

    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        logging.debug("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        (x, weight, bias, mean, rstd) = ctx.saved_tensors
        M = ctx.M
        N = ctx.N

        with torch_device_fn.device(x.device):
            in_grad = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
            layer_norm_backward_kernel[grid](
                out_grad, x, weight, mean, rstd, in_grad, M, N
            )

        if weight is None and bias is None:
            return in_grad, None, None, None, None, None

        with torch_device_fn.device(x.device):
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
            weight_grad = None if weight is None else torch.empty_like(weight)
            bias_grad = None if bias is None else torch.empty_like(bias)
            weight_bias_backward_kernel[grid](
                out_grad, x, mean, rstd, weight_grad, bias_grad, M, N
            )
        return in_grad, None, weight_grad, bias_grad, None, None


def layer_norm(
    x, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True
):
    return LayerNorm.apply(x, normalized_shape, weight, bias, eps, cudnn_enable)
