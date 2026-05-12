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
