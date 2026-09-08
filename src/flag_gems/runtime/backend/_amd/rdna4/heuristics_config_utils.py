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

"""RDNA4 heuristics for the non-inner reduction kernels.

Only `mean_non_inner` and `argmax_non_inner` are overridden here; every other
key keeps the `_amd` vendor value.
"""

import torch
import triton

# Tile area (TILE_N * TILE_K) a workgroup handles per iteration. argmax keeps an
# int64 index live next to every value it compares, so it needs a smaller tile
# than mean to stay within the register budget.
_MEAN_TILE_BUDGET = 16384
_ARGMAX_TILE_BUDGET = 8192

# A wider TILE_K would leave the tile budget too little room on the reduction axis.
_MAX_TILE_K = 2048

# Elements per lane before another warp earns its keep.
_ELEMS_PER_LANE = 2048

_MAX_NUM_WARPS = 16


def _prev_power_of_2(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 1


def _num_cus():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def reduction_heur_tile_k(args):
    # grid_y is cdiv(K, TILE_K), so widening TILE_K trades workgroups for
    # coalescing. Double while there is more than one wave left to give up.
    num_cus = _num_cus()
    upper_bound = min(args["K"], _MAX_TILE_K)
    tile_k = 1
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        if (num_blocks / num_cus > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def _make_tile_n(tile_budget):
    # Spend what TILE_K leaves of the tile budget on the reduction axis. No floor
    # beyond 1: rounding N up only buys lanes that are masked off anyway.
    def reduction_heur_tile_n(args):
        per_row = max(1, tile_budget // args["TILE_K"])
        return max(1, min(triton.next_power_of_2(args["N"]), per_row))

    return reduction_heur_tile_n


def reduction_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def reduction_heur_num_warps(args):
    # These kernels are memory bound, so smaller workgroups than the vendor
    # reduction rule picks keep more of them resident per CU.
    tile_size = args["TILE_N"] * args["TILE_K"]
    return max(1, min(_MAX_NUM_WARPS, _prev_power_of_2(tile_size // _ELEMS_PER_LANE)))


def _non_inner_reduction(tile_budget):
    # Order matters: triton.heuristics feeds each result into the args of the
    # next, so TILE_K has to be resolved before TILE_N, and both before
    # num_warps.
    return {
        "TILE_K": reduction_heur_tile_k,
        "TILE_N": _make_tile_n(tile_budget),
        "ONE_TILE_PER_CTA": reduction_heur_one_tile_per_cta,
        "num_warps": reduction_heur_num_warps,
    }


HEURISTICS_CONFIGS = {
    "mean_non_inner": _non_inner_reduction(_MEAN_TILE_BUDGET),
    "argmax_non_inner": _non_inner_reduction(_ARGMAX_TILE_BUDGET),
}
