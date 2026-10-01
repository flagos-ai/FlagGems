# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.runtime import torch_device_fn

from .mm import mm as _gems_mm

logger = logging.getLogger(__name__)


@triton.jit
def _sparse_gate_kernel(
    A,
    Meta,
    O,
    M,
    K4,
    stride_am,
    stride_ak,
    stride_mm,
    stride_mn,
    stride_om,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Produce masked_mat1[m, 4k+p] = A[m, 4k+p] if kept else 0.

    kept = meta[m, k]      for p in {0, 1}
    kept = not meta[m, k]  for p in {2, 3}

    Purely 2D tiles: loads/stores over (BLOCK_M, BLOCK_K) groups; no rank-3
    broadcast and no ``tl.sum`` so it lowers cleanly on TritonXPU.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_ok = offs_m[:, None] < M
    k_ok = offs_k[None, :] < K4
    mk = m_ok & k_ok

    meta = tl.load(
        Meta + offs_m[:, None] * stride_mm + offs_k[None, :] * stride_mn,
        mask=mk,
        other=0,
    )
    keep_lo = meta != 0
    keep_hi = meta == 0

    for p in tl.static_range(4):
        col = offs_k * 4 + p
        a_ptrs = A + offs_m[:, None] * stride_am + col[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mk, other=0.0)
        keep = keep_lo if p < 2 else keep_hi
        val = tl.where(keep, a, tl.zeros_like(a))
        o_ptrs = O + offs_m[:, None] * stride_om + col[None, :] * stride_ok
        tl.store(o_ptrs, val, mask=mk)


def _sparse_semi_structured_mm(mat1, mat1_meta, mat2, *, out_dtype=None):
    """Sparse (2:4) semi-structured matmul, XPU-safe.

    mat1:      (M, 4*K4) dense with 2:4 pattern
    mat1_meta: (M, K4) bool mask (True -> keep first pair of each group of 4)
    mat2:      (4*K4, N)
    returns:   (M, N)
    """
    logger.debug("GEMS_KUNLUNXIN SPARSE_SEMI_STRUCTURED_MM")

    M = mat1.shape[0]
    K4 = mat1_meta.shape[1]
    N = mat2.shape[1]

    assert mat1.shape == (
        M,
        4 * K4,
    ), f"Expected mat1 shape ({M}, {4 * K4}), got {mat1.shape}"
    assert mat2.shape == (
        4 * K4,
        N,
    ), f"Expected mat2 shape ({4 * K4}, {N}), got {mat2.shape}"
    assert mat1_meta.shape == (
        M,
        K4,
    ), f"Expected mat1_meta shape ({M}, {K4}), got {mat1_meta.shape}"

    output_dtype = out_dtype if out_dtype is not None else mat1.dtype

    masked = torch.empty((M, 4 * K4), device=mat1.device, dtype=mat1.dtype)

    BLOCK_M = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K4, BLOCK_K))
    with torch_device_fn.device(mat1.device):
        _sparse_gate_kernel[grid](
            mat1,
            mat1_meta,
            masked,
            M,
            K4,
            mat1.stride(0),
            mat1.stride(1),
            mat1_meta.stride(0),
            mat1_meta.stride(1),
            masked.stride(0),
            masked.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )

    out = _gems_mm(masked, mat2)
    if out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out
