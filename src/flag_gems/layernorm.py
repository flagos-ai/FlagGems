import torch
import triton
import triton.language as tl
from .libentry import libentry


def cfggen(all_args):
    x = all_args["X"]
    N = all_args["N"]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    return (BLOCK_SIZE,)


@libentry(cfggen=cfggen)
@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    if __debug__:
        print("FLAG LAYER NORM")
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_fused[(M,)](
        x_arg,
        y,
        weight,
        bias,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_ctas=1,
    )
    return y
