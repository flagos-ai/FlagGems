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

from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def _next_power_of_2_or_1(x: int) -> int:
    return 1 if x <= 1 else triton.next_power_of_2(x)


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_ptr,
    combined_stride,
    lens_ptr,
    topk_ptr,
    topk_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    COMBINED_TOPK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_idx, query_end, num_workers):
        pos = start_pos + token_idx - query_start
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
        row_ptr = combined_ptr + token_idx * combined_stride

        # Scalar stores avoid the boundary-mask lowering issue on Kunlunxin.
        for col in range(COMBINED_TOPK):
            tl.store(row_ptr + col, -1)
        for col in range(TOP_K):
            if col < topk_len:
                topk_val = tl.load(topk_ptr + token_idx * topk_stride + col)
                tl.store(row_ptr + col, topk_val + M * batch_idx)
        for col in range(WINDOW_SIZE):
            if col < swa_len:
                value = M * batch_idx + N + col + pos - swa_len + 1 - gather_start
                tl.store(row_ptr + topk_len + col, value)
        tl.store(lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined = torch.empty(
        (num_tokens, combined_topk), device=topk_indices.device, dtype=torch.int32
    )
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    with torch_device_fn.device(topk_indices.device):
        _combine_topk_swa_indices_kernel[(num_reqs, 128)](
            combined,
            combined.stride(0),
            lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            COMBINED_TOPK=combined_topk,
        )
    return combined, lens


__all__ = ["combine_topk_swa_indices"]
