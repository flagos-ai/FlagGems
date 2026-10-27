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

from flag_gems.ops._safe_softmax import _safe_softmax as generic_safe_softmax

logger = logging.getLogger(__name__)


@triton.jit
def _safe_softmax_kernel(
    input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = input_ptr + row_id * n_cols
    out_start = output_ptr + row_id * n_cols

    m_i = -float("inf")
    l_i = 0.0

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        m_ij = tl.max(x_fp32, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_i_new))
        beta = tl.exp(x_fp32 - m_i_new)

        sum_block = tl.sum(tl.where(mask, beta, 0.0), axis=0)
        l_i = l_i * alpha + sum_block
        m_i = m_i_new

    all_neginf = m_i == -float("inf")

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        p = tl.exp(x_fp32 - m_i) / l_i
        p = tl.where(all_neginf, 0.0, p)
        tl.store(out_start + cols, p, mask=mask)


def _safe_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = None):
    # Fallback to generic implementation for non-FP32 dtypes to avoid performance regression
    if x.dtype != torch.float32:
        return generic_safe_softmax(x, dim=dim, dtype=dtype)

    logger.debug("GEMS_NVIDIA SAFE_SOFTMAX")
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.ndim >= 1, "Input tensor must have at least 1 dimension"

    dim = dim if dim >= 0 else x.ndim + dim
    assert 0 <= dim < x.ndim, "Invalid dim for softmax"

    if dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        y = x.permute(perm).contiguous()
        inv_perm = [0] * x.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
    else:
        y = x.contiguous()
        inv_perm = None

    n_cols = y.shape[-1]
    n_rows = y.numel() // n_cols

    y_fp32 = y.float()
    out_fp32 = torch.empty_like(y_fp32)

    def _next_pow2(v: int) -> int:
        if v <= 1:
            return 1
        return 1 << (v - 1).bit_length()

    BLOCK_SIZE = min(4096, _next_pow2(n_cols))

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    grid = lambda meta: (n_rows,)

    _safe_softmax_kernel[grid](
        y_fp32, out_fp32, n_rows, n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
    )

    out = out_fp32
    if dtype is not None:
        out = out.to(dtype)
    else:
        out = out.to(x.dtype)

    out = out.view(*y.shape)
    if inv_perm is not None:
        out = out.permute(inv_perm)

    return out
