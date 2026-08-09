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


def simple_elementwise_blocksize_heur(args):
    return 1024


def argmax_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmax_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


def argmin_heur_block_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def argmin_heur_block_n(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


def bmm_heur_divisible_m(args):
    return args["M"] % args["TILE_M"] == 0


def bmm_heur_divisible_n(args):
    return args["N"] % args["TILE_N"] == 0


def bmm_heur_divisible_k(args):
    return args["K"] % args["TILE_K"] == 0


def dropout_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def dropout_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def exponential_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def exponential_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def gather_heur_block_m(args):
    return min(4, triton.next_power_of_2(triton.cdiv(args["N"], 2048)))


def gather_heur_block_n(args):
    return min(2048, triton.next_power_of_2(args["N"]))


# XPU5 2026-08-31 single-variable probe (/tmp/gbq_x5/probe_perf.py):
#   PROBE A -- the nvidia default `min(2048, next_power_of_2(args["N"]))` returns 0
#   whenever args["N"] == 0, and `tl.arange(0, 0)` is a hard CompilationError
#   ("arange's end argument must be greater than the start argument").
#   Two live call sites hit N == 0 on every backend:
#     * gather_block_quantized(..., indices=not None): the wrapper passes the literal
#       0 for N (ops/gather_block_quantized.py:161 "Not used in this kernel"), so the
#       whole indices path is dead -- 6/6 probe configs raised CompilationError.
#     * gather_block_quantized(empty_tensor, ...): N = numel() = 0.
#   `max(64, ...)` repairs both without touching the generic implementation.
#
#   PROBE B -- BLOCK_SIZE_N sweep, standalone kernel, min-of-3 do_bench median.
#   Inside the official matrix (N <= 16384) the nvidia 2048 cap is already optimal,
#   so this function returns exactly `min(2048, next_power_of_2(N))` there and the
#   official benchmark cells stay bit-identical to HEAD. Beyond it the cap costs
#   1.12x-1.81x:
#     N=  32768  2048 0.00940 ms  8192 0.00838 ms  (1.12x)
#     N=  65536  2048 0.01073 ms  8192 0.00826 ms  (1.30x)
#     N= 262144  2048 0.02382 ms  8192 0.01502 ms  (1.59x)
#     N=1048576  2048 0.07591 ms  8192 0.04183 ms  (1.81x)
#     N=16777216 2048 1.09375 ms  8192 0.57240 ms  (1.91x)
#   8192 is also the elementwise BLOCK sweet spot already established on this
#   backend (dequantize 2026-08-29: 209.6 GB/s at BLOCK=8192).
def gather_block_quantized_heur_block_size_n(args):
    # gather_block_quantized_with_indices_kernel drives its trip count from
    # `index_len`, not from `N` (the wrapper hard-codes N = 0 there), so read the
    # real length when it is present.
    n = args.get("index_len", None)
    if n is None:
        n = args["N"]
    n = triton.next_power_of_2(n)
    if n <= 16384:
        return max(64, min(2048, n))
    return 8192


def index_add_heur_block_m(args):
    # BLOCK_M was previously next_power_of_2(cdiv(M, 12)) -> UNBOUNDED: it grows
    # with M, so a large M produces a giant [BLOCK_M, BLOCK_N] constexpr tile that
    # ConvertTritonXPUToLLVM materializes per element -> IR explosion (29MB/148MB
    # in ir-index_add*-devN.log) and slow launches. Cap BLOCK_M to keep the tile
    # bounded and increase program-level parallelism for wide unique-index rows.
    return min(8, triton.next_power_of_2(triton.cdiv(args["M"], 12)))


def index_add_heur_block_n(args):
    # Likewise bound BLOCK_N (was min(8192, next_pow2(N))). A smaller contiguous
    # column tile measured faster on XPU for the large (4096,4096) case and keeps
    # the 2D tile bounded together with the capped BLOCK_M.
    return min(256, triton.next_power_of_2(args["N"]))


def index_select_heur_block_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def index_select_heur_block_n(args):
    return 64


def mm_heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def rand_heur_block(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12 * 4))  # CLUSTER_NUM = 12
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def rand_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def randn_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def randn_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def softmax_heur_tile_k(args):
    MAX_TILE_K = 8192
    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    tile_k = 1
    upper_bound = min(args["K"], MAX_TILE_K)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / NUM_SMS
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def softmax_heur_tile_n_non_inner(args):
    return triton.cdiv(8192, args["TILE_K"])


def softmax_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def softmax_heur_num_warps_non_inner(args):
    tile_size = args["TILE_N"] * args["TILE_K"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def softmax_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def softmax_heur_num_warps_inner(args):
    tile_size = args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def softmax_heur_tile_n_bwd_non_inner(args):
    return max(1, 1024 // args["TILE_K"])


def softmax_heur_tile_m(args):
    return max(1, 1024 // args["TILE_N"])


def uniform_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def uniform_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def var_mean_heur_block_n(args):
    return triton.next_power_of_2(args["BLOCK_NUM"])


def upsample_nearest2d_SAME_H(args):
    return args["OH"] == args["IH"]


def upsample_nearest2d_SAME_W(args):
    return args["OW"] == args["IW"]


def batch_norm_heur_block_m(args):
    return min(2048, triton.next_power_of_2(args["batch_dim"]))


def batch_norm_heur_block_n(args):
    # A maximum of 16384 elements are loaded at once.
    BLOCK_M = batch_norm_heur_block_m(args)
    BLOCK_N = triton.next_power_of_2(args["spatial_dim"])
    return min(BLOCK_N, max(1, 2**14 // BLOCK_M))


def vdot_heur_block_size(args):
    n = args["n_elements"]
    if n < 1024:
        return 32
    elif n < 8192:
        return 256
    else:
        return 1024


HEURISTICS_CONFIGS = {
    "argmax": {
        "BLOCK_M": argmax_heur_block_m,
        "BLOCK_N": argmax_heur_block_n,
    },
    "argmin": {
        "BLOCK_M": argmin_heur_block_m,
        "BLOCK_N": argmin_heur_block_n,
    },
    "bmm": {
        "DIVISIBLE_M": bmm_heur_divisible_m,
        "DIVISIBLE_N": bmm_heur_divisible_n,
        "DIVISIBLE_K": bmm_heur_divisible_k,
    },
    "dropout": {
        "BLOCK": dropout_heur_block,
        "num_warps": dropout_heur_num_warps,
    },
    "exponential_": {
        "BLOCK": exponential_heur_block,
        "num_warps": exponential_heur_num_warps,
    },
    "gather": {
        "BLOCK_M": gather_heur_block_m,
        "BLOCK_N": gather_heur_block_n,
    },
    "gather_block_quantized": {
        "BLOCK_SIZE_N": gather_block_quantized_heur_block_size_n,
    },
    "index_select": {
        "BLOCK_M": index_select_heur_block_m,
        "BLOCK_N": index_select_heur_block_n,
    },
    "index_add": {
        "BLOCK_M": index_add_heur_block_m,
        "BLOCK_N": index_add_heur_block_n,
    },
    "mm": {
        "EVEN_K": mm_heur_even_k,
    },
    "rand": {
        "BLOCK": rand_heur_block,
        "num_warps": rand_heur_num_warps,
    },
    "randn": {
        "BLOCK": randn_heur_block,
        "num_warps": randn_heur_num_warps,
    },
    "softmax_non_inner": {
        "TILE_K": softmax_heur_tile_k,
        "TILE_N": softmax_heur_tile_n_non_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
        "num_warps": softmax_heur_num_warps_non_inner,
    },
    "softmax_inner": {
        "TILE_N": softmax_heur_tile_n_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
        "num_warps": softmax_heur_num_warps_inner,
    },
    "softmax_backward_non_inner": {
        "TILE_N": softmax_heur_tile_n_bwd_non_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
    },
    "softmax_backward_inner": {
        "TILE_M": softmax_heur_tile_m,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
    },
    "uniform": {
        "BLOCK": uniform_heur_block,
        "num_warps": uniform_heur_num_warps,
    },
    "upsample_nearest2d": {
        "SAME_H": upsample_nearest2d_SAME_H,
        "SAME_W": upsample_nearest2d_SAME_W,
    },
    "var_mean": {
        "BLOCK_N": var_mean_heur_block_n,
    },
    "batch_norm": {
        "BLOCK_M": batch_norm_heur_block_m,
        "BLOCK_N": batch_norm_heur_block_n,
    },
    "vdot": {
        "BLOCK_SIZE": vdot_heur_block_size,
    },
    "elementwise_generic": {
        "BLOCK_SIZE": simple_elementwise_blocksize_heur,
        "num_warps": lambda args: 8,
    },
}
