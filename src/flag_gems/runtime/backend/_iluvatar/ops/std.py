import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _std_map_kernel(X, Tmp_sum, Tmp_sum_sq, N, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(x, axis=0)
    sum_sq_val = tl.sum(x * x, axis=0)
    tl.store(Tmp_sum + pid, sum_val)
    tl.store(Tmp_sum_sq + pid, sum_sq_val)


@libentry()
@triton.jit
def _std_reduce_kernel(
    Tmp_sum, Tmp_sum_sq, Out, N, correction, BLOCK_NUM, BLOCK_SIZE: tl.constexpr
):
    total_sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total_sum_sq_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, BLOCK_NUM, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < BLOCK_NUM
        tmp_sum_vals = tl.load(Tmp_sum + offset, mask=mask, other=0.0).to(tl.float32)
        tmp_sum_sq_vals = tl.load(Tmp_sum_sq + offset, mask=mask, other=0.0).to(
            tl.float32
        )
        total_sum_acc += tmp_sum_vals
        total_sum_sq_acc += tmp_sum_sq_vals
    total_sum = tl.sum(total_sum_acc, axis=0)
    total_sum_sq = tl.sum(total_sum_sq_acc, axis=0)
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


@triton.jit(do_not_specialize=["N", "correction"])
def _std_dim_kernel(
    X,
    Out,
    N,
    correction,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row
    row_idx = ext.program_id(0)
    row_ptr = X + row_idx * N
    out_ptr = Out + row_idx

    # Two-pass approach in float32: first compute mean, then variance
    # Pass 1: mean
    _sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum, axis=0) / N

    # Pass 2: variance
    _var_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        _var_acc += tl.where(mask, diff * diff, 0.0)
    var = tl.sum(_var_acc, axis=0)

    denom = tl.maximum(N - correction, 1.0).to(tl.float32)
    var = var / denom
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    tl.store(out_ptr, std_dev.to(Out.dtype.element_ty))


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ILUVATAR STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)

        BLOCK_N_MAP = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N_MAP)
        tmp_sum = torch.empty((BLOCK_NUM,), dtype=torch.float32, device=x.device)
        tmp_sum_sq = torch.empty((BLOCK_NUM,), dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            _std_map_kernel[(BLOCK_NUM,)](
                x.contiguous(), tmp_sum, tmp_sum_sq, N, BLOCK_N_MAP
            )
        out = torch.empty([], device=x.device, dtype=x.dtype)
        BLOCK_SIZE_REDUCE = 1024
        with torch_device_fn.device(x.device):
            _std_reduce_kernel[(1,)](
                tmp_sum,
                tmp_sum_sq,
                out,
                N,
                effective_correction,
                BLOCK_NUM,
                BLOCK_SIZE_REDUCE,
            )
        return out.view([1] * input_ndim) if keepdim else out

    else:
        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
        dim_list_normalized = [d % input_ndim for d in dim_list]

        x_view = dim_compress(x, dim_list_normalized)

        N = 1
        for d in dim_list_normalized:
            N *= original_shape[d]
        M = x.numel() // N

        output_shape_kept = list(original_shape)
        for d in dim_list_normalized:
            output_shape_kept[d] = 1

        if M * N > 0 and (N - effective_correction <= 0):
            final_shape = [
                s for i, s in enumerate(original_shape) if i not in dim_list_normalized
            ]
            return torch.full(
                final_shape if not keepdim else output_shape_kept,
                float("nan"),
                device=x.device,
                dtype=x.dtype,
            )

        out = torch.empty(output_shape_kept, device=x.device, dtype=x.dtype)
        if M * N == 0:
            return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out

        BLOCK_N = 1024
        grid = (M,)

        with torch_device_fn.device(x.device):
            _std_dim_kernel[grid](
                x_view,
                out.view(M),
                N,
                effective_correction,
                BLOCK_N=BLOCK_N,
            )

        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out
