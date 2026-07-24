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

import os

import triton
from _enflame.gcu300.utils.codegen_config_utils import get_heuristics_for_num_warps

ENFLAME_GCU300_4SIPS = int(os.getenv("ENFLAME_GCU300_4SIPS", "0"))
MMU_LIMIT = 512 * 1024 * 1024


def heuristics_for_num_warps(tile_size):
    return get_heuristics_for_num_warps(tile_size)


def prev_power_of_2(n):
    """Return the largest power of 2 less than or equal to n."""
    return 1 << max(0, n.bit_length() - 1) if n >= 1 else 1


def heuristics_for_tile_size(max_tile_size, *sizes):
    ndim = len(sizes)
    tile_sizes = [0 for _ in range(ndim)]
    for i in range(ndim):
        size = sizes[ndim - 1 - i]
        tile_size = min(max_tile_size, triton.next_power_of_2(size))
        if (
            ENFLAME_GCU300_4SIPS != 1
            and triton.next_power_of_2(size) <= 512 * 1024
            and max_tile_size > 1
        ):
            tile_size = min(max_tile_size // 2, tile_size)
        if max_tile_size > 1:
            tile_size = max(2, tile_size)
        tile_sizes[ndim - 1 - i] = tile_size
        max_tile_size = max(1, max_tile_size // tile_size)
    return tuple(tile_sizes)


# This function is used to get the tile sizes with the constraint of MMU memory(512MB)
def heuristics_for_tile_size_with_mmu_constraint(
    max_tile_size, element_size, strides, *sizes
):
    mmu_size_left = MMU_LIMIT
    ndim = len(sizes)
    tile_sizes = [0] * ndim

    for i in range(ndim - 1, -1, -1):
        size, stride = sizes[i], strides[i]
        size_po2 = triton.next_power_of_2(size)

        # Calculate initial tile_size
        tile_size = min(max_tile_size, size_po2)
        if ENFLAME_GCU300_4SIPS != 1 and size_po2 <= 512 * 1024 and max_tile_size > 1:
            tile_size = min(max_tile_size // 2, tile_size)
        if max_tile_size > 1:
            tile_size = max(2, tile_size)

        # Adjust tile_size based on MMU memory constraint
        mem_cost = (tile_size - 1) * stride * element_size
        if mem_cost > mmu_size_left:
            tile_size = prev_power_of_2(
                max(1, mmu_size_left // (stride * element_size))
            )
            mem_cost = (tile_size - 1) * stride * element_size

        mmu_size_left -= mem_cost
        tile_sizes[i] = tile_size
        max_tile_size = max(1, max_tile_size // tile_size)

    return tuple(tile_sizes)


def mmu_safe_index_put_block_sizes(
    block_size0: int,
    block_size1: int,
    input_stride,
    input_shape,
    element_size: int,
) -> tuple[int, int]:
    """Clamp index_put tile sizes for GCU300 MMU scatter-store limit (512MB)."""
    index_stride = input_stride[0]
    index_dim = input_shape[0]

    # Tensor indices on dim 0 may scatter across the full indexed dimension.
    scatter_span = index_stride * max(1, index_dim - 1) * element_size
    if scatter_span >= MMU_LIMIT:
        block_size0 = 1

    # block_ptr-style tile footprint: stride * (BLOCK - 1) along each tiled axis.
    mmu_size = (
        index_stride * max(0, block_size0 - 1) + max(0, block_size1 - 1)
    ) * element_size
    if mmu_size >= MMU_LIMIT:
        block_size0 = 1
        block_size1 = max(1, MMU_LIMIT // element_size)

    return block_size0, block_size1
