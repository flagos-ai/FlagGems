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

"""GCU300 optimized std.

Design (三板斧):
  1. Fixed Grid (第一板斧): dim=None path uses GRID_DIM=6 with block_ptr DMA
     strided loop (from sum.py proven pattern).
  2. DSM / single-pass (第二板斧): sum/sum_sq single-pass eliminates the 2x
     data read of the generic 2-pass implementation. var = sum_sq/N - mean^2.
  3. Math (第三板斧): tl.sqrt is irreplaceable (no rsqrt variant needed).

Bound type: memory-bound reduction.

Paths:
  - dim=None (full reduction): 2 launches (map Fixed-Grid + reduce grid=1).
  - single dim, K==1 (inner-most): 1 launch, single-pass sum/sum_sq.
  - single dim, K>1 (non-inner): 1 launch, single-pass with K-tiling.
  - multiple dims: dim_compress + inner kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner

logger = logging.getLogger(__name__)

# GCU300 Fixed Grid: 24 SPs / 4 warps = 6 blocks
_GRID_DIM = 6
_BLOCK_SIZE_MAP = 32768
_DUAL_THRESHOLD = 4 * 1024 * 1024  # 4M elements


# ---------------------------------------------------------------------------
# dim=None path: Fixed Grid map + reduce (2 launches, unavoidable)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N"])
def _std_map_kernel(
    X,
    MidSum,
    MidSumSq,
    N,
    GRID_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    acc_sum = tl.zeros((), dtype=cdtype)
    acc_sq = tl.zeros((), dtype=cdtype)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    in_block = tl.make_block_ptr(
        base=X,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in tl.range(pid, num_blocks, GRID_DIM):
        x = tl.load(in_block, boundary_check=(0,), padding_option="zero").to(cdtype)
        acc_sum += tl.sum(x)
        acc_sq += tl.sum(x * x)
        in_block = tl.advance(in_block, (GRID_DIM * BLOCK_SIZE,))
    tl.store(MidSum + pid, acc_sum)
    tl.store(MidSumSq + pid, acc_sq)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _std_map_kernel_dual(
    X,
    MidSum,
    MidSumSq,
    N,
    GRID_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dual-accumulator variant for large N: break serial dependency."""
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    s0 = tl.zeros((), dtype=cdtype)
    s1 = tl.zeros((), dtype=cdtype)
    q0 = tl.zeros((), dtype=cdtype)
    q1 = tl.zeros((), dtype=cdtype)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    blk0 = tl.make_block_ptr(
        base=X,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    blk1 = tl.make_block_ptr(
        base=X,
        shape=(N,),
        strides=(1,),
        offsets=((pid + GRID_DIM) * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    stride = GRID_DIM * 2 * BLOCK_SIZE
    for _ in tl.range(pid, num_blocks, GRID_DIM * 2):
        x0 = tl.load(blk0, boundary_check=(0,), padding_option="zero").to(cdtype)
        x1 = tl.load(blk1, boundary_check=(0,), padding_option="zero").to(cdtype)
        s0 += tl.sum(x0)
        s1 += tl.sum(x1)
        q0 += tl.sum(x0 * x0)
        q1 += tl.sum(x1 * x1)
        blk0 = tl.advance(blk0, (stride,))
        blk1 = tl.advance(blk1, (stride,))
    tl.store(MidSum + pid, s0 + s1)
    tl.store(MidSumSq + pid, q0 + q1)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def _std_reduce_kernel(
    MidSum,
    MidSumSq,
    Out,
    N,
    correction,
    mid_size,
    MID_SIZE: tl.constexpr,
):
    offset = tl.arange(0, MID_SIZE)
    mask = offset < mid_size
    s = tl.load(MidSum + offset, mask=mask, other=0.0).to(tl.float32)
    sq = tl.load(MidSumSq + offset, mask=mask, other=0.0).to(tl.float32)
    total_sum = tl.sum(s)
    total_sq = tl.sum(sq)
    mean = total_sum / N
    var = (total_sq / N - mean * mean) * N / tl.maximum(N - correction, 1.0)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# dim inner (K==1): single-pass sum/sum_sq, BLOCK_M rows per block
# ---------------------------------------------------------------------------


def _keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    if BLOCK_M < 32:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(_keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["correction"])
def _std_dim_inner_kernel(
    X,
    Out,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    _sum_sq = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        x = tl.load(X + cols, mask, other=0).to(cdtype)
        _sum += x
        _sum_sq += x * x

    total_sum = tl.sum(_sum, axis=1)
    total_sq = tl.sum(_sum_sq, axis=1)
    mean = total_sum / N
    var = (total_sq / N - mean * mean) * N / tl.maximum(N - correction, 1.0)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    tl.store(Out, std_dev[:, None].to(Out.dtype.element_ty), row_mask)


# ---------------------------------------------------------------------------
# dim non-inner (K>1): single-pass, 1 row × BLOCK_K k-tile per block
# ---------------------------------------------------------------------------

_NON_INNER_CONFIGS = [
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=4, num_stages=3),
]


@libentry()
@triton.autotune(configs=_NON_INNER_CONFIGS, key=["N", "K"])
@triton.jit(do_not_specialize=["correction"])
def _std_dim_non_inner_kernel(
    X,
    Out,
    M,
    N,
    K,
    correction,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    k_mask = k_offsets < K

    sum_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=cdtype)
    sq_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        n_offsets = off + tl.arange(0, BLOCK_N)[:, None]
        mask = (n_offsets < N) & k_mask
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0).to(cdtype)
        sum_acc += x
        sq_acc += x * x

    total_sum = tl.sum(sum_acc, axis=0)  # [1, BLOCK_K]
    total_sq = tl.sum(sq_acc, axis=0)
    mean = total_sum / N
    var = (total_sq / N - mean * mean) * N / tl.maximum(N - correction, 1.0)
    safe_var = tl.maximum(var, 0.0)
    std_dev = tl.sqrt(safe_var)
    out_offsets = pid_m * K + k_offsets
    tl.store(Out + out_offsets, std_dev.to(Out.dtype.element_ty), mask=k_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ENFLAME STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        dim_list = list(range(input_ndim))
    else:
        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
        dim_list = [d % input_ndim for d in dim_list]

    # === Full reduction (dim is None or all dims) ===
    if len(dim_list) == input_ndim:
        x = x.contiguous()
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            out = torch.full([], float("nan"), device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out
        if N == 1 and effective_correction == 0.0:
            out = torch.zeros([], device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out

        out = torch.empty([], device=x.device, dtype=x.dtype)
        grid_dim = _GRID_DIM
        block_size = _BLOCK_SIZE_MAP
        kernel = _std_map_kernel_dual if N > _DUAL_THRESHOLD else _std_map_kernel
        mid_sum = torch.empty((grid_dim,), dtype=torch.float32, device=x.device)
        mid_sum_sq = torch.empty((grid_dim,), dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            kernel[(grid_dim,)](
                x,
                mid_sum,
                mid_sum_sq,
                N,
                GRID_DIM=grid_dim,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
            _std_reduce_kernel[(1,)](
                mid_sum,
                mid_sum_sq,
                out,
                N,
                effective_correction,
                grid_dim,
                MID_SIZE=triton.next_power_of_2(grid_dim),
            )
        return out.view([1] * input_ndim) if keepdim else out

    # === Partial reduction ===
    shape = list(original_shape)
    N = 1
    for d in dim_list:
        N *= shape[d]
        shape[d] = 1

    if N - effective_correction <= 0:
        final_shape = (
            shape
            if keepdim
            else [s for i, s in enumerate(original_shape) if i not in dim_list]
        )
        return torch.full(final_shape, float("nan"), device=x.device, dtype=x.dtype)

    if N == 1 and effective_correction == 0.0:
        final_shape = (
            shape
            if keepdim
            else [s for i, s in enumerate(original_shape) if i not in dim_list]
        )
        return torch.zeros(final_shape, device=x.device, dtype=x.dtype)

    x_contiguous = x.contiguous()

    # Single dim: use inner or non-inner kernel directly (no dim_compress)
    if len(dim_list) == 1:
        dim0 = dim_list[0]
        M = 1
        for s in original_shape[:dim0]:
            M *= s
        K = 1
        for s in original_shape[dim0 + 1 :]:
            K *= s

        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        if M * N * K == 0:
            return out.squeeze(dim=dim0) if not keepdim else out

        with torch_device_fn.device(x.device):
            if K > 1:
                grid = lambda meta: (
                    M,
                    triton.cdiv(K, meta["BLOCK_K"]),
                )
                _std_dim_non_inner_kernel[grid](
                    x_contiguous, out, M, N, K, effective_correction
                )
            else:
                grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
                _std_dim_inner_kernel[grid](
                    x_contiguous, out.view(M), M, N, effective_correction
                )
        return out.squeeze(dim=dim0) if not keepdim else out

    # Multiple dims: dim_compress + inner kernel
    x_view = dim_compress(x_contiguous, dim_list)
    M = x_view.numel() // N

    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    if M * N == 0:
        return out.squeeze(dim=tuple(dim_list)) if not keepdim else out

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(x.device):
        _std_dim_inner_kernel[grid](x_view, out.view(M), M, N, effective_correction)
    return out.squeeze(dim=tuple(dim_list)) if not keepdim else out
