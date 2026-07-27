# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of renorm / renorm_.
#
# Root cause: the generic kernels (flag_gems/ops/renorm.py, renorm_.py) compute
# the per-row p-norm via `tl_extra_shim.pow(...)` / `tl_extra_shim.sqrt(...)`.
# On XPU these lower to `ld.lld: error: undefined symbol: Unsupported` at link
# time (same failure family as tl_extra_shim.lgamma) -> all 270 cases crash.
#
# Fix: express the power with `exp`/`log` intrinsics that do lower on XPU:
#   |x|^p        = exp(p * log(|x|))     (0 handled explicitly)
#   sum^(1/p)    = exp(log(sum) / p)
# One program per row: pass 1 accumulates sum(|x|^p), pass 2 rescales.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_row_kernel(X, Y, N, p, maxnorm, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        ax = tl.abs(x)
        powered = tl.where(ax == 0.0, 0.0, tl.exp(p * tl.log(ax)))
        acc += tl.where(mask, powered, 0.0)

    s = tl.sum(acc)
    norm = tl.where(s == 0.0, 0.0, tl.exp(tl.log(s) / p))
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * scale
        tl.store(Y + base + cols, y.to(X.dtype.element_ty), mask=mask)


def _launch(x_flat, out_flat, M, N, p, maxnorm):
    BLOCK = min(triton.next_power_of_2(N), 1024)
    grid = (M,)
    with torch_device_fn.device(x_flat.device):
        renorm_row_kernel[grid](
            x_flat,
            out_flat,
            N,
            float(p),
            float(maxnorm),
            BLOCK_SIZE=BLOCK,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )


def renorm(input, p, dim, maxnorm):
    logger.debug("GEMS_KUNLUNXIN RENORM")
    dim = dim % input.ndim

    perm = [dim] + [i for i in range(input.ndim) if i != dim]
    inv_perm = [0] * input.ndim
    for i, pi in enumerate(perm):
        inv_perm[pi] = i

    x_perm = input.permute(perm).contiguous()
    M = x_perm.shape[0]
    N = x_perm[0].numel()
    x_flat = x_perm.reshape(M, N)

    out_flat = torch.empty_like(x_flat)
    _launch(x_flat, out_flat, M, N, p, maxnorm)

    return out_flat.reshape(x_perm.shape).permute(inv_perm).contiguous()


def renorm_(input, p, dim, maxnorm):
    logger.debug("GEMS_KUNLUNXIN RENORM_")
    dim = dim % input.ndim

    perm = [dim] + [i for i in range(input.ndim) if i != dim]
    inv_perm = [0] * input.ndim
    for i, pi in enumerate(perm):
        inv_perm[pi] = i

    x_perm = input.permute(perm).contiguous()
    M = x_perm.shape[0]
    N = x_perm[0].numel()
    x_flat = x_perm.reshape(M, N)

    _launch(x_flat, x_flat, M, N, p, maxnorm)

    input.copy_(x_flat.reshape(x_perm.shape).permute(inv_perm))
    return input
