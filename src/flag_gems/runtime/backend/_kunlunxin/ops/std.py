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


def _std_dim_dispatch(out, x_contiguous, M, N, K, effective_correction):
    # Every dim reduction is routed through dim_compress => K is always 1 and we
    # only use the verified-correct inner kernel.
    with torch_device_fn.device(x_contiguous.device):
        grid = (M, 1, 1)
        _std_dim_kernel_inner[grid](out, x_contiguous, M, N, effective_correction)


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN STD")
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

    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    dim_list_normalized = [d % input_ndim for d in dim_list]

    # Route EVERY dim reduction (single-dim AND multi-dim) through dim_compress so
    # the reduced dims land on the trailing axis => it is always a contiguous
    # (M, N) inner reduction (K == 1). We only ever launch the @libentry-cached
    # _std_dim_kernel_inner. This (a) avoids the giant 2D tile + heuristic-supplied
    # launch param IR explosion of the old _std_fused_dim_kernel path
    # (ir-std-dev5.log = 7.7M lines) and (b) avoids the non_inner (K>1) softmax
    # kernel, which was numerically wrong on XPU (std ~sqrt(K)x too small).
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
    if N == 1 and effective_correction == 0.0:
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.zeros(
            final_shape if not keepdim else output_shape_kept,
            device=x.device,
            dtype=x.dtype,
        )

    out = torch.empty(output_shape_kept, device=x.device, dtype=x.dtype)
    if M * N == 0:
        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out

    _std_dim_dispatch(out.view(-1), x_view, M, N, 1, effective_correction)
    return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out
