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
import math

import torch
import triton
import triton.language as tl

from flag_gems.fused.unpack_seq import _select_unpack_seq_config

logger = logging.getLogger(__name__)


@triton.jit
def _lengths_to_offsets_kernel(
    lengths_ptr,
    offsets_ptr,
    B: tl.constexpr,
):
    offset = tl.full((), 0, tl.int32)
    tl.store(offsets_ptr, offset)
    for batch in tl.static_range(B):
        offset += tl.load(lengths_ptr + batch).to(tl.int32)
        tl.store(offsets_ptr + batch + 1, offset)


@triton.jit
def _unpack_seq_kunlunxin_kernel(
    packed_ptr,
    out_ptr,
    offsets_ptr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    LOG_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    out_row = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)

    low = tl.full((), 0, tl.int32)
    high = tl.full((), B, tl.int32)
    for _ in tl.static_range(LOG_B):
        active = low < high
        mid = low + (high - low) // 2
        sequence_end = tl.load(offsets_ptr + mid + 1)
        go_left = out_row < sequence_end
        high = tl.where(active & go_left, mid, high)
        low = tl.where(active & ~go_left, mid + 1, low)

    batch = low
    sequence_start = tl.load(offsets_ptr + batch)
    timestep = out_row - sequence_start
    mask = off_d < D
    packed_offsets = (batch * Lmax + timestep) * D + off_d
    values = tl.load(packed_ptr + packed_offsets, mask=mask)
    tl.store(out_ptr + out_row * D + off_d, values, mask=mask)


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    logger.debug("GEMS UNPACK_SEQ_TRITON")
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    offsets = torch.empty(B + 1, dtype=torch.int32, device=lengths.device)
    _lengths_to_offsets_kernel[(1,)](
        lengths,
        offsets,
        B=B,
        isCloseVectorization=True,
        buffer_size_limit=2048,
        num_warps=1,
        num_stages=1,
    )
    N = int(offsets[B].item())
    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)
    num_warps = 4
    num_stages = 1
    if block_t == 64 and block_d == 64:
        block_t, block_d, num_warps, num_stages = _select_unpack_seq_config(
            B, Lmax, D, packed_reshaped.element_size()
        )
        num_stages = 1

    if N > 0:
        grid = (N, triton.cdiv(D, block_d))
        _unpack_seq_kunlunxin_kernel[grid](
            packed_reshaped.contiguous(),
            out,
            offsets,
            Lmax,
            D,
            B,
            LOG_B=max(1, math.ceil(math.log2(B + 1))),
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
            isCloseVectorization=True,
            buffer_size_limit=2048,
            unroll_num=8,
        )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out
