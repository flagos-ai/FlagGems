import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var_mean"))
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_M
    rows = pid + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    X_ptr = X + rows[:, None] * N

    # Two accumulators with different init values to prevent PPL register merging
    _sum_sq = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _sum = _sum_sq + 2

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask

        x = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
        mask_f = tl.where(mask, 1.0, 0.0)
        _sum += x * mask_f
        _sum_sq += x * x * mask_f

    total_sum = tl.sum(_sum, axis=1)
    actual_sum = total_sum - 2 * BLOCK_N
    mean = actual_sum / N
    tl.store(Mean + rows, mean, row_mask)

    total_sum_sq = tl.sum(_sum_sq, axis=1)
    var = (total_sum_sq - actual_sum * mean) / (N - correction)
    tl.store(Var + rows, var, row_mask)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logging.debug("GEMS VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        M = 1
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N

    var = torch.empty(shape, dtype=x.dtype, device=x.device)
    mean = torch.empty(shape, dtype=x.dtype, device=x.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(x.device):
        var_mean_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean