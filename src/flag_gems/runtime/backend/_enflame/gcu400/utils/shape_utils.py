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


def heuristics_for_tile_size_notDMA(max_tile_size, *sizes):
    ndim = len(sizes)
    tile_sizes = [0 for _ in range(ndim)]
    for i in range(ndim):
        size = sizes[ndim - 1 - i]
        tile_size = min(max_tile_size, triton.next_power_of_2(size))
        tile_sizes[ndim - 1 - i] = tile_size
        if tile_sizes[ndim - 1 - i] == 1 and ndim > 1:
            tile_sizes[ndim - 1 - i] = 2
            tile_sizes[ndim - i] //= 2
        max_tile_size = max(1, max_tile_size // tile_size)
    return tuple(tile_sizes)
