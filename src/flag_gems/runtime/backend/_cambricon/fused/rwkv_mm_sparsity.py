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


@triton.jit
def rwkv_mm_sparsity_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    v_cols: tl.constexpr,
    k_size: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    col_idx = pid * block_n + tl.arange(0, block_n)
    col_mask = col_idx < v_cols
    k_idx = tl.arange(0, block_k)

    acc_lanes = tl.zeros((block_k, block_n), dtype=tl.float32)

    for k_base in range(0, k_size, block_k):
        k_offsets = k_base + k_idx
        k_mask = k_offsets < k_size
        k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        v = tl.load(
            v_ptr + k_offsets[:, None] * v_cols + col_idx[None, :],
            mask=k_mask[:, None] & col_mask[None, :] & (k[:, None] != 0.0),
            other=0.0,
        ).to(tl.float32)
        acc_lanes += k[:, None] * v

    acc = tl.sum(acc_lanes, axis=0)
    tl.store(output_ptr + col_idx, acc, mask=col_mask)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    logger.debug("GEMS_CAMBRICON RWKV MM SPARSITY")
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)

    v_cols = v.size(1)
    output = torch.empty(v_cols, device=k.device, dtype=k.dtype)
    block_k = 32
    block_n = triton.next_power_of_2(16)
    k_size = k.size(0)
    grid = (triton.cdiv(v_cols, block_n),)
    rwkv_mm_sparsity_kernel[grid](k, v, output, v_cols, k_size, block_k, block_n)
    return output
