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

from flag_gems.ops.addmv_ import addmv_ as default_addmv_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _addmv_acc2d_kernel(
    self_ptr,
    mat_ptr,
    vec_ptr,
    M,
    K,
    beta,
    alpha,
    stride_self,
    stride_m,
    stride_k,
    stride_v,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc2d = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        mask_k = kk < K
        vec_vals = tl.load(vec_ptr + kk * stride_v, mask=mask_k, other=0.0).to(
            tl.float32
        )
        mat_vals = tl.load(
            mat_ptr + offs_m[:, None] * stride_m + kk[None, :] * stride_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc2d += mat_vals * vec_vals[None, :]

    acc = tl.sum(acc2d, axis=1)
    self_vals = tl.load(self_ptr + offs_m * stride_self, mask=mask_m, other=0.0)
    if beta != 0.0:
        acc = beta * self_vals + alpha * acc
    else:
        acc = alpha * acc
    tl.store(self_ptr + offs_m * stride_self, acc, mask=mask_m)


@triton.jit
def _addmv_partial_kernel(
    mat_ptr,
    vec_ptr,
    partials_ptr,
    M,
    K,
    stride_m,
    stride_k,
    stride_v,
    NUM_K_SPLITS,
    KSPLIT_K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    k0 = pid_k * KSPLIT_K
    offs_k = tl.arange(0, BLOCK_K)
    for kk in range(0, KSPLIT_K, BLOCK_K):
        k = k0 + kk + offs_k
        mask_k = k < K
        vec_vals = tl.load(vec_ptr + k * stride_v, mask=mask_k, other=0.0).to(
            tl.float32
        )
        mat_vals = tl.load(
            mat_ptr + offs_m[:, None] * stride_m + k[None, :] * stride_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(mat_vals * vec_vals[None, :], axis=1)

    tl.store(
        partials_ptr
        + pid_m * (NUM_K_SPLITS * BLOCK_M)
        + pid_k * BLOCK_M
        + tl.arange(0, BLOCK_M),
        acc,
        mask=mask_m,
    )


@triton.jit
def _addmv_reduce_kernel(
    self_ptr,
    partials_ptr,
    M,
    beta,
    alpha,
    stride_self,
    NUM_K_SPLITS,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    base = partials_ptr + pid * (NUM_K_SPLITS * BLOCK_M) + offs_m
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for s in range(NUM_K_SPLITS):
        acc += tl.load(base + s * BLOCK_M, mask=mask_m, other=0.0)

    self_vals = tl.load(self_ptr + offs_m * stride_self, mask=mask_m, other=0.0)
    if beta != 0.0:
        acc = beta * self_vals + alpha * acc
    else:
        acc = alpha * acc
    tl.store(self_ptr + offs_m * stride_self, acc, mask=mask_m)


def _specialized_addmv_(self, mat, vec, *, beta=1, alpha=1):
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    beta = float(beta)
    alpha = float(alpha)

    M, K = mat.shape
    assert vec.numel() == K

    mat = mat.contiguous()
    vec = vec.contiguous()

    stride_self = self.stride(0)
    stride_m = mat.stride(0)
    stride_k = mat.stride(1)
    stride_v = vec.stride(0)

    itemsize = mat.element_size()
    total_bytes = M * K * itemsize

    if total_bytes >= 16 * 1024 * 1024:
        # Split-K: partial dots per (row block, K slice) + deterministic reduce.
        BLOCK_M = 64
        BLOCK_K = 128
        row_blocks = triton.cdiv(M, BLOCK_M)
        # Sweep-verified: f32 saturates HBM at ~256 CTAs, half dtypes at ~512.
        target_ctas = 256 if itemsize == 4 else 512
        num_splits = max(1, min(target_ctas // row_blocks, triton.cdiv(K, 64)))
        ksplits = triton.cdiv(K, num_splits)

        partials = torch.empty(
            (row_blocks * num_splits * BLOCK_M,),
            dtype=torch.float32,
            device=self.device,
        )

        # Sweep/eval-verified: f32 benefits from 8 warps (reduction ILP),
        # half dtypes keep 4 warps (occupancy at 512 CTAs).
        split_warps = 8 if itemsize == 4 else 4
        _addmv_partial_kernel[(row_blocks, num_splits)](
            mat,
            vec,
            partials,
            M,
            K,
            stride_m,
            stride_k,
            stride_v,
            num_splits,
            ksplits,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=split_warps,
            num_stages=3,
        )
        _addmv_reduce_kernel[(row_blocks,)](
            self,
            partials,
            M,
            beta,
            alpha,
            stride_self,
            num_splits,
            BLOCK_M=BLOCK_M,
            num_warps=4,
        )
    else:
        # Single-pass register-accumulator kernel.
        if K <= 128:
            BLOCK_M, BLOCK_K, warps = 32, 64, 4
        elif total_bytes < 2 * 1024 * 1024:
            # Single K-tile for (256,256): BK=256 covers K in one iteration.
            BLOCK_M, BLOCK_K, warps = 32, 256, 8
        else:
            # Half dtypes are compute-bound here: smaller BM=8 doubles CTAs to
            # 128 for better latency hiding; f32 stays at BM=16 (parity proven).
            if itemsize == 4:
                BLOCK_M, BLOCK_K, warps = 16, 256, 4
            else:
                BLOCK_M, BLOCK_K, warps = 8, 256, 4
        grid = (triton.cdiv(M, BLOCK_M),)
        _addmv_acc2d_kernel[grid](
            self,
            mat,
            vec,
            M,
            K,
            beta,
            alpha,
            stride_self,
            stride_m,
            stride_k,
            stride_v,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=warps,
            num_stages=2,
        )
    return self


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMV_")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
        and isinstance(mat, torch.Tensor)
        and mat.device.type == "musa"
        and mat.dtype in _SUPPORTED_DTYPES
        and isinstance(vec, torch.Tensor)
        and vec.device.type == "musa"
        and vec.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_addmv_(self, mat, vec, beta=beta, alpha=alpha)
    return default_addmv_(self, mat, vec, beta=beta, alpha=alpha)
