import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress

logger = logging.getLogger(__name__)


@triton.jit
def _std_map_kernel(X, Tmp_sum, Tmp_sum_sq, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(x, axis=0)
    sum_sq_val = tl.sum(x * x, axis=0)
    tl.store(Tmp_sum + pid, sum_val)
    tl.store(Tmp_sum_sq + pid, sum_sq_val)


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


@triton.autotune(configs=runtime.get_tuned_config("naive_reduction"), key=["M", "N"])
@triton.jit
def _std_fused_dim_kernel(
    X,
    Out,
    stride_x_row,
    stride_x_col,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_group = tl.program_id(axis=0)
    start_row = pid_group * BLOCK_M
    row_offsets = start_row + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    mean_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_row_ptrs = X + row_offsets[:, None] * stride_x_row

    for off in range(0, N, BLOCK_N):
        col_offsets = off + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        x_ptrs = x_row_ptrs + col_offsets[None, :] * stride_x_col
        final_mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptrs, mask=final_mask, other=0.0)
        mean_acc += x.to(tl.float32)

    mean = tl.sum(mean_acc, axis=1) / N

    var_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        col_offsets = off + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        x_ptrs = x_row_ptrs + col_offsets[None, :] * stride_x_col
        final_mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptrs, mask=final_mask, other=0.0)
        diff = x.to(tl.float32) - mean[:, None]
        var_acc += tl.where(final_mask, diff * diff, 0.0)

    var = tl.sum(var_acc, axis=1)

    denom = N - correction
    var = var / tl.maximum(denom, 1e-12)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)

    out_ptrs = Out + row_offsets
    tl.store(out_ptrs, std_dev.to(Out.dtype.element_ty), mask=row_mask)


def std(x, dim=None, *, correction=None, keepdim=False):
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        logger.debug("GEMS STD (Global Simple Map-Reduce Path)")
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)

        BLOCK_N_MAP = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N_MAP)
        tmp_sum = torch.empty((BLOCK_NUM,), dtype=torch.float32, device=x.device)
        tmp_sum_sq = torch.empty((BLOCK_NUM,), dtype=torch.float32, device=x.device)
        _std_map_kernel[(BLOCK_NUM,)](
            x.contiguous(), tmp_sum, tmp_sum_sq, N, BLOCK_N_MAP
        )
        out = torch.empty([], device=x.device, dtype=x.dtype)
        BLOCK_SIZE_REDUCE = 1024
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
        logger.warning(
            f"GEMS std: Using compatible but non-optimal path for dim={dim} (dim_compress)."
        )

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

        stride_x_row, stride_x_col = N, 1

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

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

        _std_fused_dim_kernel[grid](
            x_view, out.view(M), stride_x_row, stride_x_col, M, N, effective_correction
        )

        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out
