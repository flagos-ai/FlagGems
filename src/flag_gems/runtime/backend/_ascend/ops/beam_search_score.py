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

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver


@triton.jit
def _beam_search_score_kernel(
    log_probs_ptr,
    beam_scores_ptr,
    out_ptr,
    n_rows,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_count = tl.num_programs(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < vocab_size
    for row in range(program_id, n_rows, program_count):
        offsets = row * vocab_size + columns
        log_probs = tl.load(log_probs_ptr + offsets, mask=mask)
        beam_score = tl.load(beam_scores_ptr + row)
        tl.store(out_ptr + offsets, log_probs + beam_score, mask=mask)


def beam_search_score(log_probs, beam_scores):
    out = torch.empty_like(log_probs)
    vocab_size = log_probs.shape[-1]
    n_rows = log_probs.numel() // vocab_size
    block_size = triton.next_power_of_2(vocab_size)
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    half_vector_cores = max(1, properties["num_vectorcore"] // 2)
    grid_cap = 1 << (half_vector_cores.bit_length() - 1)
    grid = (min(n_rows, grid_cap),)
    _beam_search_score_kernel[grid](
        log_probs,
        beam_scores,
        out,
        n_rows,
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out
