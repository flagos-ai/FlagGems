import triton
from _enflame.gcu400.utils.codegen_config_utils import get_heuristics_for_num_warps


def heuristics_for_num_warps(tile_size):
    return get_heuristics_for_num_warps(tile_size)


def heuristics_for_tile_size(max_tile_size, *sizes):
    ndim = len(sizes)
    tile_sizes = [0 for _ in range(ndim)]
    for i in range(ndim):
        size = sizes[ndim - 1 - i]
        tile_size = min(max_tile_size, triton.next_power_of_2(size))
        tile_sizes[ndim - 1 - i] = tile_size
        max_tile_size = max(1, max_tile_size // tile_size)
    return tuple(tile_sizes)
