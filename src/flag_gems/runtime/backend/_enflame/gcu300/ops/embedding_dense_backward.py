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

import flag_gems
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _embedding_dense_backward_kernel(
    grad_output_ptr,
    index_sorted_ptr,
    order_ptr,
    grad_weight_ptr,
    num_weights,
    N,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    SCALE: tl.constexpr,
):
    # GCU300 cannot legalize `tt.atomic_rmw`, so the scatter is inverted into a
    # segmented sum: `grad_weight[i] += sum_{j : indices[j] == i} grad_output[j]`.
    # `index_sorted`/`order` come from a `torch.sort` in the wrapper, so each
    # output row binary-searches the contiguous run of source rows that maps to
    # it and accumulates them. No atomics are involved anywhere.
    pid_r = tl.program_id(0)
    pid_t = tl.program_id(1)
    num_r = tl.num_programs(0)
    num_t = tl.num_programs(1)
    n_tiles = tl.cdiv(EMBED_DIM, BLOCK_D)

    for row in tl.range(pid_r, num_weights, num_r):
        # start = lower_bound(index_sorted, row)
        lo = 0
        hi = N
        while lo < hi:
            mid = lo + (hi - lo) // 2
            val = tl.load(index_sorted_ptr + mid)
            lo = tl.where(val < row, mid + 1, lo)
            hi = tl.where(val < row, hi, mid)
        start = lo

        # end = lower_bound(index_sorted, row + 1) = upper_bound(row)
        lo = start
        hi = N
        while lo < hi:
            mid = lo + (hi - lo) // 2
            val = tl.load(index_sorted_ptr + mid)
            lo = tl.where(val < row + 1, mid + 1, lo)
            hi = tl.where(val < row + 1, hi, mid)
        end = lo

        cnt = tl.maximum(end - start, 1)
        for tile in tl.range(pid_t, n_tiles, num_t):
            offs_d = tile * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs_d < EMBED_DIM

            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            j = start
            while j < end:
                src_row = tl.load(order_ptr + j)
                acc += tl.load(
                    grad_output_ptr + src_row * EMBED_DIM + offs_d,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                j += 1

            if SCALE:
                acc = acc / cnt.to(tl.float32)
            tl.store(grad_weight_ptr + row * EMBED_DIM + offs_d, acc, mask=mask)


def embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    logger.debug("GEMS_ENFLAME EMBEDDING_DENSE_BACKWARD")
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    if (
        grad_output.device.type != flag_gems.device
        or indices.device.type != flag_gems.device
        or grad_output.device != indices.device
    ):
        raise ValueError(
            f"Inputs must be {flag_gems.device} tensors on the same device."
        )

    device = grad_output.device
    assert (
        grad_output.dim() >= 2
    ), "grad_output must have embedding dimension as the last dim."

    D = grad_output.shape[-1]
    go = grad_output.contiguous().view(-1, D)  # (N, D)
    idx = indices.contiguous().view(-1)
    N = idx.numel()

    assert go.shape[0] == N, "indices number must match grad_output rows."

    if num_weights > 0 and N > 0 and D > 0:
        # Both of the rewrites below happen on the host, because neither device
        # call works on this backend's runner: `idx == pad` goes through
        # `topsatenEq` and `torch.sort` dispatches into `at::_ops::sort_stable`
        # whatever the `stable` argument, and both abort there with
        # `Check failed: 0`. `idx` is one integer per row, so the round trip is
        # O(N) against the O(N * D) gradient the kernel then streams.
        #
        # The padding rewrite has to come first: it moves those rows out of the
        # valid range in the copy the kernel sees, and the positions (hence
        # `order`) must stay aligned with `grad_output` rows.
        idx_host = idx.to("cpu")
        pad = padding_idx if padding_idx is not None else -1
        if pad >= 0:
            idx_host = torch.where(
                idx_host == pad, torch.full_like(idx_host, num_weights), idx_host
            )
        idx_host = idx_host.to(torch.int32)
        index_sorted, order = torch.sort(idx_host)
        index_sorted = index_sorted.to(device)
        order = order.to(torch.int32).to(device)

        BLOCK_D = min(1024, max(32, triton.next_power_of_2(D)))
        grid = (
            min(num_weights, 65535),
            min(triton.cdiv(D, BLOCK_D), 255),
        )

        # Every (row, tile) pair is written exactly once by the grid.
        grad_weight_fp32 = torch.empty(
            (num_weights, D), device=device, dtype=torch.float32
        )
        _embedding_dense_backward_kernel[grid](
            go,
            index_sorted,
            order,
            grad_weight_fp32,
            num_weights,
            N,
            BLOCK_D=BLOCK_D,
            EMBED_DIM=D,
            SCALE=scale_grad_by_freq,
        )
    else:
        grad_weight_fp32 = torch.zeros(
            (num_weights, D), device=device, dtype=torch.float32
        )

    if grad_output.dtype != torch.float32:
        return grad_weight_fp32.to(grad_output.dtype)
    return grad_weight_fp32
