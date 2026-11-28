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

"""Grouped matmul for Enflame GCU400.

``grouped_matmul`` computes one GEMM per group:
``out[group] = x[group] @ weight[group]`` where ``x`` is ``[M, K]``,
``weight`` is ``[G, K, N]`` and ``out`` is ``[M, N]``.  ``group_list``
holds either the inclusive cumulative row ends (``group_list_type=0``) or
the per-group row counts (``group_list_type=1``).

The kernel walks a flat ``(group, m tile, n tile)`` index space with a
fixed grid so that every GCU400 SIP stays busy regardless of how skewed
the per-group row counts are.  Group boundaries are derived on device
from ``group_list``; no host synchronization is required.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

# GCU400 has 4 DIEs x 6 SIPs; a fixed grid of 48 programs matches that
# topology and the kernel strides over the remaining tiles.
FIXED_GRID = 48

# Tuned on GCU400 with the FlagGems grouped_matmul benchmark shapes: a deep
# K block (512) keeps the DTE pipeline fed and a 128-wide N block keeps the
# per-group weight tile resident, which is where the previous 64-wide layout
# lost most of its time.
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 512
NUM_STAGES = 4
NUM_WARPS = 1


@libentry()
@triton.jit(
    do_not_specialize=["M", "K"],
)
def _grouped_matmul_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    group_list_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wk,
    stride_wn,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_LIST_P2: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GRID_DIM_X: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load group_list once per program and derive [group_start, group_end)
    # for every group.  Type 0 stores inclusive cumulative ends, type 1
    # stores counts, so the same exclusive cumsum covers both.
    lane = tl.arange(0, GROUP_LIST_P2)
    lane_mask = lane < GROUPS
    raw_value = tl.load(group_list_ptr + lane, mask=lane_mask, other=0).to(tl.int32)
    raw_prev = tl.load(
        group_list_ptr + lane - 1,
        mask=lane_mask & (lane > 0),
        other=0,
    ).to(tl.int32)
    if GROUP_LIST_TYPE == 0:
        group_start = raw_prev
        group_end = raw_value
    else:
        group_start = tl.cumsum(raw_value, axis=0) - raw_value
        group_end = group_start + raw_value
    group_rows = tl.where(lane_mask, group_end - group_start, 0)
    group_m_tiles = tl.cdiv(group_rows, BLOCK_M)
    tile_cum = tl.cumsum(group_m_tiles, axis=0)
    total_tiles = tl.sum(group_m_tiles, axis=0) * NUM_N_TILES

    for tile in tl.range(pid, total_tiles, GRID_DIM_X, num_stages=num_stages):
        m_tile_global = tile // NUM_N_TILES
        n_tile = tile % NUM_N_TILES
        # Number of groups whose cumulative tile end is not past this tile.
        group_id = tl.sum((tile_cum <= m_tile_global).to(tl.int32), axis=0)
        row_start = tl.sum(tl.where(lane == group_id, group_start, 0), axis=0)
        row_end = tl.sum(tl.where(lane == group_id, group_end, 0), axis=0)
        tiles_before = tl.sum(
            tl.where(lane == group_id, tile_cum - group_m_tiles, 0), axis=0
        )
        m_local = m_tile_global - tiles_before
        rows_in_group = row_end - row_start

        x_block = tl.make_block_ptr(
            base=x_ptr + row_start.to(tl.int64) * stride_xm,
            shape=(rows_in_group, K),
            strides=(stride_xm, stride_xk),
            offsets=(m_local * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        w_block = tl.make_block_ptr(
            base=weight_ptr + group_id.to(tl.int64) * stride_wg,
            shape=(K, N),
            strides=(stride_wk, stride_wn),
            offsets=(0, n_tile * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr + row_start.to(tl.int64) * N,
            shape=(rows_in_group, N),
            strides=(N, 1),
            offsets=(m_local * BLOCK_M, n_tile * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, K, BLOCK_K):
            a = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(w_block, boundary_check=(0, 1), padding_option="zero")
            accumulator = tl.dot(a, b, accumulator)
            x_block = tl.advance(x_block, (0, BLOCK_K))
            w_block = tl.advance(w_block, (BLOCK_K, 0))

        tl.store(
            out_block,
            accumulator.to(out_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )


def _validate_common(x, group_list, groups, group_list_type):
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 or 1")
    if x.ndim != 2:
        raise ValueError("x must have shape [M, K]")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError("grouped_matmul supports float16 and bfloat16 inputs")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if group_list.ndim != 1 or group_list.numel() != groups:
        raise ValueError("group_list must be a 1D tensor with one value per group")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_list must have dtype int32 or int64")
    if group_list.device != x.device:
        raise ValueError("x and group_list must be on the same device")
    if groups <= 0:
        raise ValueError("at least one group is required")


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Compute ``Y[group] = X[group] @ Weight[group]`` on GCU400."""
    if weight.ndim != 3:
        raise ValueError("weight must have shape [G, K, N]")
    groups, weight_k, n = weight.shape
    _validate_common(x, group_list, groups, group_list_type)
    m, k = x.shape
    if k <= 0 or n <= 0:
        raise ValueError("K and N must be positive")
    if weight_k != k:
        raise ValueError("x and weight must have the same K dimension")
    if weight.dtype != x.dtype:
        raise TypeError("x and weight must have the same dtype")
    if weight.device != x.device:
        raise ValueError("x and weight must be on the same device")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return output
    if group_list.dtype != torch.int32:
        group_list = group_list.to(torch.int32)
    if not group_list.is_contiguous():
        group_list = group_list.contiguous()

    num_n_tiles = triton.cdiv(n, BLOCK_N)
    group_list_p2 = max(triton.next_power_of_2(groups), 2)

    with torch_device_fn.device(x.device):
        _grouped_matmul_kernel[(FIXED_GRID,)](
            x,
            weight,
            output,
            group_list,
            m,
            k,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            N=n,
            GROUPS=groups,
            GROUP_LIST_P2=group_list_p2,
            GROUP_LIST_TYPE=group_list_type,
            NUM_N_TILES=num_n_tiles,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GRID_DIM_X=FIXED_GRID,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )
    return output


__all__ = ["grouped_matmul"]
