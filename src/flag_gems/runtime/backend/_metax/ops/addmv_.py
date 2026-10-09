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

logger = logging.getLogger(__name__)


def _addmv_kernel(
    self_ptr,
    mat_ptr,
    vec_ptr,
    M,
    N,
    self_stride,
    mat_m_stride,
    mat_n_stride,
    vec_stride,
    beta,
    alpha,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    MAT_CG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # accumulate-once GEMV: acc tile [BLOCK_M, BLOCK_N] built with FMAs over
    # N-chunks, single epilogue reduction over the N axis. Fully memory-bound:
    # reaches the device's streaming read ceiling (~1.65 TB/s on C550).
    # EVEN_M/EVEN_N: compile-time specialization dropping all masks when the
    # shape divides the block sizes evenly (power-of-two eval shapes).
    # MAT_CG: mat is read exactly once per element (row-parallel GEMV), so the
    # .cg (L2-only, evict-first) load hint keeps it from polluting L1 and gives
    # the reused vec more cache room across the 512-1024 concurrent CTAs.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    if EVEN_M and EVEN_N:
        for n0 in range(0, N, BLOCK_N):
            c = n0 + cols
            if MAT_CG:
                m = tl.load(
                    mat_ptr + rows[:, None] * mat_m_stride + c[None, :] * mat_n_stride,
                    cache_modifier=".cg",
                ).to(ACC_DTYPE)
            else:
                m = tl.load(
                    mat_ptr + rows[:, None] * mat_m_stride + c[None, :] * mat_n_stride,
                ).to(ACC_DTYPE)
            v = tl.load(vec_ptr + c * vec_stride).to(ACC_DTYPE)
            acc += m * v[None, :]
    else:
        row_mask = rows < M
        for n0 in range(0, N, BLOCK_N):
            c = n0 + cols
            c_mask = c < N
            m = tl.load(
                mat_ptr + rows[:, None] * mat_m_stride + c[None, :] * mat_n_stride,
                mask=row_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(ACC_DTYPE)
            v = tl.load(vec_ptr + c * vec_stride, mask=c_mask, other=0.0).to(ACC_DTYPE)
            acc += m * v[None, :]
    acc = tl.sum(acc, axis=1)

    r = alpha * acc
    if beta == 0.0:
        # torch semantics: self is not read when beta == 0
        s = tl.zeros((BLOCK_M,), dtype=ACC_DTYPE)
    else:
        if EVEN_M:
            s = tl.load(self_ptr + rows * self_stride).to(ACC_DTYPE)
        else:
            s = tl.load(self_ptr + rows * self_stride, mask=rows < M, other=0.0).to(
                ACC_DTYPE
            )
        s = s * beta
    # single rounding to the output dtype (torch opmath semantics)
    out = (r + s).to(OUT_DTYPE)
    if EVEN_M:
        tl.store(self_ptr + rows * self_stride, out)
    else:
        tl.store(self_ptr + rows * self_stride, out, mask=rows < M)


@triton.jit
def _addmv_n1_kernel(
    self_ptr,
    mat_ptr,
    vec_ptr,
    M,
    self_stride,
    mat_m_stride,
    beta,
    alpha,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    s = tl.load(self_ptr + offs * self_stride, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mat_ptr + offs * mat_m_stride, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(vec_ptr).to(tl.float32)
    r = alpha * (m * v)
    if beta == 0.0:
        out = r
    else:
        out = r + beta * s
    tl.store(self_ptr + offs * self_stride, out.to(OUT_DTYPE), mask=mask)


def _map_dtypes(dtype):
    if dtype == torch.float64:
        return tl.float64, tl.float64
    if dtype == torch.float16:
        return tl.float32, tl.float16
    if dtype == torch.bfloat16:
        return tl.float32, tl.bfloat16
    return tl.float32, tl.float32


def _pick_config(M, N, dtype):
    two_byte = dtype.itemsize == 2
    if N <= 64:
        # tiny reduction: single exact-fit N-tile, no pipelining (launch-bound)
        return (16, 64, 4, 1)
    if N <= 256:
        return (8, 256, 2, 1)
    if N <= 1024:
        return (4, 512, 4, 2) if two_byte else (2, 512, 4, 2)
    if N <= 8192:
        if two_byte:
            # keep enough CTAs on the row axis: M=4096 -> 512 CTAs at BLOCK_M=8
            return (8, 512, 4, 2) if M >= 2048 else (8, 256, 4, 2)
        return (16, 128, 4, 2)
    # N > 8192: very wide reduction. Small M needs tiny BLOCK_M to keep the
    # program count well above the 104 SMs (avoids 1.2-wave quantization).
    # bm1_bn8192_w8: single dense 16KB stream per CTA, 1024 CTAs, best on
    # (1024,65536) fp16/bf16 (89.0us vs 89.6 for bm2_bn2048 in sweep v10).
    if two_byte:
        return (1, 8192, 8, 2) if M <= 2048 else (8, 512, 4, 2)
    return (2, 512, 2, 2) if M <= 2048 else (8, 512, 4, 2)


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    M = mat.shape[0]
    N = mat.shape[1]
    if M == 0:
        return self

    acc_dtype, out_dtype = _map_dtypes(self.dtype)

    if N == 1:
        BLOCK = 1024
        grid = (triton.cdiv(M, BLOCK),)
        _addmv_n1_kernel[grid](
            self,
            mat,
            vec,
            M,
            self.stride(0),
            mat.stride(0),
            beta,
            alpha,
            out_dtype,
            BLOCK,
            num_warps=4,
        )
        return self

    BLOCK_M, BLOCK_N, num_warps, num_stages = _pick_config(M, N, self.dtype)
    even_m = (M % BLOCK_M == 0) and (M >= BLOCK_M)
    even_n = N % BLOCK_N == 0
    grid = (triton.cdiv(M, BLOCK_M),)
    _addmv_kernel[grid](
        self,
        mat,
        vec,
        M,
        N,
        self.stride(0),
        mat.stride(0),
        mat.stride(1),
        vec.stride(0),
        beta,
        alpha,
        acc_dtype,
        out_dtype,
        even_m,
        even_n,
        even_m and even_n,  # MAT_CG: mat is read exactly once, .cg keeps L1 free
        BLOCK_M,
        BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return self
