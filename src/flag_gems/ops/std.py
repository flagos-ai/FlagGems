# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

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


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit(do_not_specialize=["correction"])
def _std_dim_kernel_inner(
    Out,
    X,
    M,
    N,
    correction,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Pass 1: compute mean
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
    else:
        sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0) / N

    # Pass 2: compute sum of squared deviations
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
    else:
        sq_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            sq_acc += tl.where(mask, diff * diff, 0.0)
        sq_sum = tl.sum(sq_acc, axis=0)

    denom = N - correction
    var = sq_sum / tl.maximum(denom, 1e-12)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out + pid_m, std_dev.to(Out.dtype.element_ty), mask=pid_m < M)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit(do_not_specialize=["correction"])
def _std_dim_kernel_non_inner(
    Out,
    X,
    M,
    N,
    K,
    correction,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    # Pass 1: compute mean
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0, keep_dims=True) / N
    else:
        sum_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0, keep_dims=True) / N

    # Pass 2: compute sum of squared deviations
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0, keep_dims=True)
    else:
        sq_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            sq_acc += tl.where(mask, diff * diff, 0.0)
        sq_sum = tl.sum(sq_acc, axis=0, keep_dims=True)

    denom = N - correction
    var = sq_sum / tl.maximum(denom, 1e-12)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    out_offsets = pid_m * K + k_offsets
    tl.store(Out + out_offsets, std_dev.to(Out.dtype.element_ty), mask=k_offsets < K)


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)
        if N == 1 and effective_correction == 0.0:
            out = torch.zeros([], device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out

        BLOCK_N_MAP = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N_MAP)
        tmp_sum = torch.empty((BLOCK_NUM,), dtype=torch.float32, device=x.device)
        tmp_sum_sq = torch.empty((BLOCK_NUM,), dtype=torch.float32, device=x.device)
        out = torch.empty([], device=x.device, dtype=x.dtype)
        BLOCK_SIZE_REDUCE = 1024
        with torch_device_fn.device(x.device):
            _std_map_kernel[(BLOCK_NUM,)](
                x.contiguous(), tmp_sum, tmp_sum_sq, N, BLOCK_N_MAP
            )
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

        if len(dim_list_normalized) == 1:
            dim0 = dim_list_normalized[0]
            shape = list(original_shape)
            N = shape[dim0]
            M = 1
            for size in shape[:dim0]:
                M *= size
            K = 1
            for size in shape[dim0 + 1 :]:
                K *= size
            shape[dim0] = 1

            if M * N * K > 0 and (N - effective_correction <= 0):
                final_shape = shape if keepdim else shape[:dim0] + shape[dim0 + 1 :]
                return torch.full(
                    final_shape,
                    float("nan"),
                    device=x.device,
                    dtype=x.dtype,
                )

            if N == 1 and effective_correction == 0.0:
                final_shape = shape if keepdim else shape[:dim0] + shape[dim0 + 1 :]
                return torch.zeros(final_shape, device=x.device, dtype=x.dtype)

            out = torch.empty(shape, device=x.device, dtype=x.dtype)
            if M * N * K == 0:
                return out.squeeze(dim=dim0) if not keepdim else out

            x_contiguous = x.contiguous()
            with torch_device_fn.device(x.device):
                if K > 1:
                    grid = lambda META: (M, triton.cdiv(K, META["TILE_K"]), 1)
                    _std_dim_kernel_non_inner[grid](
                        out,
                        x_contiguous,
                        M,
                        N,
                        K,
                        effective_correction,
                    )
                else:
                    grid = (M, 1, 1)
                    _std_dim_kernel_inner[grid](
                        out,
                        x_contiguous,
                        M,
                        N,
                        effective_correction,
                    )

            return out.squeeze(dim=dim0) if not keepdim else out

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
