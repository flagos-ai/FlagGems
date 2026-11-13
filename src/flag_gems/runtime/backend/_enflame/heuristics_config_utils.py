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


def argmax_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmax_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


def argmin_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmin_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


# def bmm_heur_divisible_m(args):
#     return args["M"] % args["BLOCK_M"] == 0


# def bmm_heur_divisible_n(args):
#     return args["N"] % args["BLOCK_N"] == 0


# def bmm_heur_divisible_k(args):
#     return args["K"] % args["BLOCK_K"] == 0


def dropout_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 4096


def dropout_heur_num_warps(args):
    return 4


def exponential_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 16384


def exponential_heur_num_warps(args):
    return 4


def gather_heur_block_m(args):
    return min(4, triton.next_power_of_2(triton.cdiv(args["N"], 2048)))


def gather_heur_block_n(args):
    return min(2048, triton.next_power_of_2(args["N"]))


def index_select_heur_block_m(args):
    return min(16, triton.next_power_of_2(triton.cdiv(32768, args["N"])))


def index_select_heur_block_n(args):
    m = min(triton.next_power_of_2(triton.cdiv(args["N"], 16)), 512)
    return max(m, 16)


def mm_heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def rand_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 16384


def rand_heur_num_warps(args):
    return 4


def randn_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 16384


def randn_heur_num_warps(args):
    return 4


def softmax_heur_tile_k(args):
    MAX_TILE_K = 8192
    # NUM_SMS = torch.cuda.get_device_properties(
    #     torch.cuda.current_device()
    # ).multi_processor_count
    NUM_SMS = 64
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
    return 4


def softmax_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def softmax_heur_num_warps_inner(args):
    return 4


def softmax_heur_tile_n_bwd_non_inner(args):
    return max(1, 1024 // args["TILE_K"])


def softmax_heru_tile_m(args):
    return max(1, 1024 // args["TILE_N"])


def uniform_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 16384


def uniform_heur_num_warps(args):
    return 4


def var_mean_heur_block_n(args):
    return triton.next_power_of_2(args["BLOCK_NUM"])


def upsample_nearest2d_NUM_TILE(args):
    grid_y = triton.cdiv(args["N"] * args["C"], 4)
    if grid_y <= 128:
        num_tile = 1
    else:
        num_tile = triton.cdiv(grid_y, 128)
    return num_tile


def upsample_nearest2d_TOTAL_TILE(args):
    return triton.cdiv(args["N"] * args["C"], 4)


def upsample_nearest2d_SAME_H(args):
    return args["OH"] == args["IH"]


def upsample_nearest2d_SAME_W(args):
    return args["OW"] == args["IW"]


def upsample_nearest2d_USE_INT32_IDX(args):
    return args["N"] * args["C"] * args["OH"] * args["OW"] <= (2**31 - 1)  # INT32 MAX


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


def simple_elementwise_blocksize_heur(args):
    n = args["n_elements"]
    if n < 65535:
        return 1024
    else:
        return 16384


def argmax_heur_tile_k(args):
    # With a fixed grid + grid-stride, TILE_K prioritizes memory contiguity:
    # k is the contiguous dim, so cover the whole row K when possible
    # (long contiguous runs are friendly to DTE/DMA).
    # Measured: [64,512,512] fp16 +14x / fp32 +4.2x when TILE_K 64/128->512,
    # [1024,1024,512] fp32 +95%.
    # Cap at 512: TILE_K=1024 with a partial n-tile (e.g. (200,2560,3))
    # makes the GCU Triton compiler emit wrong indices (tile id leaks into
    # the high bits); values <= 512 are verified safe.
    K = args["K"]
    if K <= 0:
        return 1
    return min(triton.next_power_of_2(K), 512)


def argmax_heur_tile_n_non_inner(args):
    n = args["N"]
    tile_k = args["TILE_K"]

    if n <= 128:
        return triton.next_power_of_2(n)

    target_tile = min(8192, n)
    tile_n = triton.next_power_of_2(target_tile)
    tile_n = max(64, min(tile_n, 4096))

    # Measured (GCU400, grid-stride + pipelining): a 65536-element tile cap
    # beats 32768: [64,512,512] fp16 +97% / fp32 +39% / bf16 +54%,
    # [64,4096,64] +46~112%. 131072 fails to compile/run on some dtypes.
    if tile_n * tile_k > 65536:
        tile_n = max(64, 65536 // tile_k)

    return tile_n


def argmax_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def argmax_heur_num_warps_non_inner(args):
    tile_n = args["TILE_N"]
    m_size = args["M"]
    inp_dtype = args["inp"].dtype
    if inp_dtype == torch.float32:
        dtype = "fp32"
    elif inp_dtype == torch.float16:
        dtype = "fp16"
    else:
        dtype = "bf16"
    if tile_n <= 128:
        # Small tile: few warps suffice when the grid is small.
        # NOTE: measured num_warps=1 on this 2D-tile non_inner kernel
        # produces wrong results on shapes with masked partial tiles
        # (e.g. (200,2560,3) dim=0), so the floor is 2.
        if m_size <= 512:
            return 2
        # Large M + small tile (measured on [1024,1024,512]): fp16 prefers
        # 2, fp32/bf16 prefer 4
        return 2 if dtype == "fp16" else 4
    if tile_n <= 512:
        # Mid tile (measured: [64,512,512] fp32 nw=2 is optimal, +56%)
        return 2
    return 4


def argmax_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def argmax_heur_block_m_inner(args):
    # 2D tile (BLOCK_M rows x TILE_N). Measured on GCU400 (fp32/fp16/bf16):
    # - BM=8 is optimal when the tile has <=32768 elements
    #   ([1024,256] 1.54x, [4096,4096] 0.57x)
    # - BM=32 (131072-element tile) fails to compile; small M (<=128) needs
    #   a smaller BM to keep enough CTAs ([64,64] BM=4 is optimal, 0.68x)
    tile_n = args["TILE_N"]
    m = args["M"]
    cap = max(1, 32768 // tile_n)
    bm = min(8, cap)
    if m <= 128:
        bm = min(bm, max(1, triton.next_power_of_2(m // 16)))
    return bm


def argmax_heur_grid_size(args):
    return 48


def argmax_heur_num_stages_inner(args):
    return 4


def argmax_heur_num_stages_non_inner(args):
    return 4


def argmax_heur_num_warps_inner(args):
    tile_size = args["TILE_N"]
    m_size = args["M"]
    if tile_size <= 16:
        # Tiny tile: extra warps bring no gain (measured: [1024,1]/[1024,16]
        # prefer nw=1)
        return 1
    if m_size <= 128:
        # Small grid (<=128 CTAs): more warps hide memory latency
        # (measured: [64,64] prefers nw=4)
        return 4
    # Large grid: fewer warps raise occupancy (measured: [1024,256+] prefers
    # nw=1)
    return 2 if tile_size <= 128 else 1


HEURISTICS_CONFIGS = {
    "argmax_non_inner": {
        "TILE_K": argmax_heur_tile_k,
        "TILE_N": argmax_heur_tile_n_non_inner,
        "ONE_TILE_PER_CTA": argmax_heur_one_tile_per_cta,
        "GRID_SIZE": argmax_heur_grid_size,
        "NUM_STAGES": argmax_heur_num_stages_non_inner,
        "num_warps": argmax_heur_num_warps_non_inner,
    },
    "argmax_inner": {
        "TILE_N": argmax_heur_tile_n_inner,
        "BLOCK_M": argmax_heur_block_m_inner,
        "GRID_SIZE": argmax_heur_grid_size,
        "NUM_STAGES": argmax_heur_num_stages_inner,
        "num_warps": argmax_heur_num_warps_inner,
    },
    "argmax": {
        "BLOCK_M": argmax_heur_block_m,
        "BLOCK_N": argmax_heur_block_n,
    },
    "argmin": {
        "BLOCK_M": argmin_heur_block_m,
        "BLOCK_N": argmin_heur_block_n,
    },
    "bmm": {
        #     "DIVISIBLE_M": bmm_heur_divisible_m,
        #     "DIVISIBLE_N": bmm_heur_divisible_n,
        #     "DIVISIBLE_K": bmm_heur_divisible_k,
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
    "index_select": {
        "BLOCK_M": index_select_heur_block_m,
        "BLOCK_N": index_select_heur_block_n,
    },
    "mha_block_128": {
        "BLOCK_M": lambda args: 64,
        "BLOCK_N": lambda args: 32,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 1,
    },
    "mha_block_64": {
        "BLOCK_M": lambda args: 32,
        "BLOCK_N": lambda args: 64,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 1,
    },
    "mha_block_32": {
        "BLOCK_M": lambda args: 32,
        "BLOCK_N": lambda args: 64,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 1,
    },
    "mha_block_16": {
        "BLOCK_M": lambda args: 16,
        "BLOCK_N": lambda args: 64,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 1,
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
        "TILE_M": softmax_heru_tile_m,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
    },
    "uniform": {
        "BLOCK": uniform_heur_block,
        "num_warps": uniform_heur_num_warps,
    },
    "upsample_nearest2d": {
        "NUM_TILE": upsample_nearest2d_NUM_TILE,
        "TOTAL_TILE": upsample_nearest2d_TOTAL_TILE,
        "SAME_H": upsample_nearest2d_SAME_H,
        "SAME_W": upsample_nearest2d_SAME_W,
        "USE_INT32_IDX": upsample_nearest2d_USE_INT32_IDX,
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
        "num_warps": lambda args: 4,
    },
    "mha_varlen_fwd": {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 32,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
    },
}
