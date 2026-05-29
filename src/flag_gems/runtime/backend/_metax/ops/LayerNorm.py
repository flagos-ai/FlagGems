import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt

logger = logging.getLogger("flag_gems." + __name__)


# Fixed BLOCK_N to limit register usage on Metax
BLOCK_SIZE = 4096


@libentry()
@triton.jit(do_not_specialize=["eps"])
def layer_norm_kernel_loop(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Loop-based kernel for large N values to reduce private memory usage."""
    pid = tle.program_id(0)

    # Compute mean and variance using Welford's online algorithm
    m = tl.zeros((BLOCK_N,), dtype=tl.float32)
    s = tl.zeros((BLOCK_N,), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, BLOCK_N)

    for step in range(0, num_steps - 1, 1):
        start_n = step * BLOCK_N
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # Last step with masking
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * BLOCK_N
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd_val = rsqrt(var + eps)
    m = final_m

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd_val)

    # Normalize and apply linear transformation
    # Process in reverse order for better cache utilization
    num_steps = tl.cdiv(N, BLOCK_N)
    prev_multiple = (num_steps - 1) * BLOCK_N

    # First block (may need masking)
    for start_n in range(0, BLOCK_N, BLOCK_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd_val + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    # Remaining blocks
    for start_n in range(BLOCK_N, N, BLOCK_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, BLOCK_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd_val + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@libentry()
@triton.jit
def layer_norm_backward_kernel_loop(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """Loop-based backward kernel for large N values."""
    pid = tle.program_id(0)

    mean = tl.load(Mean + pid).to(tl.float32)
    rstd = tl.load(Rstd + pid).to(tl.float32)

    dx_part2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dx_part3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = col_mask

        dy = tl.load(dY + pid * N + cols, mask=col_mask, other=0.0).to(tl.float32)
        x = tl.load(X + pid * N + cols, mask=col_mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 = dx_part2 + tl.where(col_mask, dx_hat, 0.0)
        dx_part3 = dx_part3 + tl.where(col_mask, dx_hat * x_hat, 0.0)

    dx_2 = tl.sum(dx_part2, axis=0)
    dx_3 = tl.sum(dx_part3, axis=0)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = col_mask

        dy = tl.load(dY + pid * N + cols, mask=col_mask, other=0.0).to(tl.float32)
        x = tl.load(X + pid * N + cols, mask=col_mask, other=0.0).to(tl.float32)
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + pid * N + cols, dx, mask=col_mask)


@libentry()
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
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = pid < N

    accW = tl.zeros((BLOCK_N,), dtype=tl.float32)
    accB = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row in range(0, M, 1):
        dy = tl.load(dY + row * N + pid, mask=col_mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * N + pid, mask=col_mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row).to(tl.float32)
        rstd_val = tl.load(Rstd + row).to(tl.float32)
        x = tl.where(col_mask, x - mean, 0.0)
        accW = accW + dy * x * rstd_val
        accB = accB + dy

    if dW is not None:
        dw = accW
        tl.store(dW + pid, dw, mask=col_mask)
    if dB is not None:
        db = accB
        tl.store(dB + pid, db, mask=col_mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        logger.debug("METAX GEMS LAYERNORM FORWARD")

        N = math.prod(normalized_shape)
        M = input.numel() // N

        input = input.contiguous()
        weight = None if weight is None else weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        y = torch.empty_like(input)

        mean = torch.empty(M, dtype=input.dtype, device=input.device)
        rstd = torch.empty(M, dtype=input.dtype, device=input.device)

        grid = (M,)

        # Use fixed BLOCK_N to limit register usage
        BLOCK_N = BLOCK_SIZE

        with torch_device_fn.device(input.device):
            layer_norm_kernel_loop[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                BLOCK_N,
            )

        if input.requires_grad:
            ctx.save_for_backward(input, weight, bias, mean, rstd)
            ctx.normalized_shape = normalized_shape
            ctx.M = M
            ctx.N = N
        return y, mean, rstd

    @staticmethod
    def backward(ctx, y_grad, mean_grad, rstd_grad):
        logger.debug("METAX GEMS LAYERNORM BACKWARD")

        y_grad = y_grad.contiguous()
        (input, weight, bias, mean, rstd) = ctx.saved_tensors
        M = ctx.M
        N = ctx.N

        BLOCK_N = BLOCK_SIZE

        if y_grad.requires_grad:
            in_grad = torch.empty_like(input)
            grid = (M,)
            with torch_device_fn.device(input.device):
                layer_norm_backward_kernel_loop[grid](
                    y_grad, input, weight, mean, rstd, in_grad, M, N, BLOCK_N
                )
        else:
            in_grad = None

        if weight is None and bias is None:
            return in_grad, None, None, None, None, None

        weight_grad = None if weight is None else torch.empty_like(weight)
        bias_grad = None if bias is None else torch.empty_like(bias)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        with torch_device_fn.device(input.device):
            weight_bias_backward_kernel[grid](
                y_grad, input, mean, rstd, weight_grad, bias_grad, M, N, BLOCK_N
            )
        return in_grad, None, weight_grad, bias_grad, None, None


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return LayerNorm.apply(input, normalized_shape, weight, bias, eps)
