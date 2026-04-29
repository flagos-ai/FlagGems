"""
T-Head Zhenwu (真武) PPU Heuristics Configuration Utilities

Provides dynamic parameter selection based on input tensor shapes
and PPU hardware characteristics.

Hardware: Zhenwu 810E PPU
Key Features:
- Tensor Core with extended PTX instructions for AIU
- High memory bandwidth with optimized access patterns
- Multi-stream parallelism support
- ICN interconnect for multi-card scenarios

Heuristics are designed to:
1. Maximize Tensor Core utilization
2. Optimize memory hierarchy usage (shared memory, L2 cache)
3. Balance compute and memory bandwidth
4. Adapt to different problem sizes dynamically

Reference:
- PPU SDK v2.0.0 Documentation
- Triton support: 2.3.x - 3.4.x with AIU extensions
"""

import torch
import triton


def simple_elementwise_blocksize_heur(args):
    return 1024


def argmax_heur_block_m(args):
    """Select BLOCK_M based on M dimension size for PPU"""
    # PPU benefits from moderate parallelism
    if args["M"] < 4096:
        return 4
    elif args["M"] < 16384:
        return 8
    else:
        return 16


def argmax_heur_block_n(args):
    """Select BLOCK_N based on N dimension size for PPU"""
    # Larger blocks to utilize PPU's memory bandwidth
    return min(8192, triton.next_power_of_2(args["N"]))


def argmin_heur_block_m(args):
    """Select BLOCK_M for argmin operation on PPU"""
    return argmax_heur_block_m(args)


def argmin_heur_block_n(args):
    """Select BLOCK_N for argmin operation on PPU"""
    return argmax_heur_block_n(args)


def bmm_heur_divisible_m(args):
    """Check if M dimension is divisible by TILE_M"""
    return args["M"] % args["TILE_M"] == 0


def bmm_heur_divisible_n(args):
    """Check if N dimension is divisible by TILE_N"""
    return args["N"] % args["TILE_N"] == 0


def bmm_heur_divisible_k(args):
    """Check if K dimension is divisible by TILE_K"""
    return args["K"] % args["TILE_K"] == 0


def dropout_heur_block(args):
    """Select block size for dropout based on N dimension"""
    if args["N"] <= 512:
        return 512
    elif args["N"] <= 1024:
        return 1024
    else:
        return 2048


def dropout_heur_num_warps(args):
    """Select num_warps for dropout based on N dimension"""
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def softmax_heur_tile_k(args):
    """
    Select TILE_K for softmax on PPU.
    Considers PPU's Tensor Core capabilities and memory hierarchy.
    """
    MAX_TILE_K = 8192
    tile_k = 1
    upper_bound = min(args["K"], MAX_TILE_K)
    
    # Get PPU SM count (if available, otherwise use default)
    try:
        NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:
        NUM_SMS = 128  # Default for Zhenwu 810E
    
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / NUM_SMS
        
        # PPU benefits from higher occupancy
        if (num_waves > 2) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    
    return tile_k


def softmax_heur_tile_n_non_inner(args):
    """Select TILE_N for non-inner softmax on PPU"""
    return triton.cdiv(8192, args["TILE_K"])


def softmax_heur_one_tile_per_cta(args):
    """Determine if one tile per CTA is sufficient"""
    return args["TILE_N"] >= args["N"]


def softmax_heur_num_warps_non_inner(args):
    """Select num_warps based on tile size for PPU"""
    tile_size = args["TILE_N"] * args["TILE_K"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def softmax_heur_tile_n_inner(args):
    """Select TILE_N for inner softmax on PPU"""
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def softmax_heur_num_warps_inner(args):
    """Select num_warps for inner softmax on PPU"""
    tile_size = args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def layer_norm_heur_block_row_size(args):
    """Select block row size for layer normalization on PPU"""
    return min(32, triton.next_power_of_2(args["row_count"]))


def batch_norm_heur_block_m(args):
    """Select BLOCK_M for batch normalization on PPU"""
    return min(2048, triton.next_power_of_2(args["batch_dim"]))


def batch_norm_heur_block_n(args):
    """
    Select BLOCK_N for batch normalization on PPU.
    Optimizes for PPU's memory access patterns.
    """
    BLOCK_M = batch_norm_heur_block_m(args)
    BLOCK_N = triton.next_power_of_2(args["spatial_dim"])
    # PPU can handle larger loads efficiently
    return min(BLOCK_N, max(1, 2**15 // BLOCK_M))


def mv_heur_block_m(args):
    """Select BLOCK_M for matrix-vector multiplication on PPU"""
    return min(1024, triton.next_power_of_2(args["M"]))


def mv_heur_block_n(args):
    """Select BLOCK_N for matrix-vector multiplication on PPU"""
    return min(128, triton.next_power_of_2(args["N"]))


def attention_heur_block_m(args):
    """Select BLOCK_M for attention on PPU"""
    # Attention benefits from larger blocks on PPU
    if args["M"] < 1024:
        return 64
    else:
        return 128


def attention_heur_block_n(args):
    """Select BLOCK_N for attention on PPU"""
    if args["N"] < 1024:
        return 32
    elif args["N"] < 4096:
        return 64
    else:
        return 128


def index_select_heur_block_m(args):
    """Select BLOCK_M for index_select on PPU"""
    return min(4, triton.next_power_of_2(triton.cdiv(256, args["N"])))


def index_select_heur_block_n(args):
    """Select BLOCK_N for index_select on PPU"""
    m = min(triton.next_power_of_2(triton.cdiv(args["N"], 16)), 1024)
    return max(m, 16)


def gather_heur_block_m(args):
    """Select BLOCK_M for gather operation on PPU"""
    return min(4, triton.next_power_of_2(triton.cdiv(args["N"], 2048)))


def gather_heur_block_n(args):
    """Select BLOCK_N for gather operation on PPU"""
    return min(2048, triton.next_power_of_2(args["N"]))


def var_mean_heur_block_n(args):
    """Select BLOCK_N for var_mean on PPU"""
    return triton.next_power_of_2(args["BLOCK_NUM"])


def upsample_nearest2d_SAME_H(args):
    """Check if output height equals input height"""
    return args["OH"] == args["IH"]


def upsample_nearest2d_SAME_W(args):
    """Check if output width equals input width"""
    return args["OW"] == args["IW"]


def mm_heur_even_k(args):
    """Check if K dimension is even for mm operation"""
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def rand_heur_block(args):
    """Select block size for random number generation on PPU"""
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def rand_heur_num_warps(args):
    """Select num_warps for random number generation on PPU"""
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


# Register all heuristics configurations for PPU
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
    "layer_norm_persistent": {
        "BLOCK_ROW_SIZE": layer_norm_heur_block_row_size,
    },
    "batch_norm": {
        "BLOCK_M": batch_norm_heur_block_m,
        "BLOCK_N": batch_norm_heur_block_n,
    },
    "mv": {
        "BLOCK_M": mv_heur_block_m,
        "BLOCK_N": mv_heur_block_n,
    },
    "attention": {
        "BLOCK_M": attention_heur_block_m,
        "BLOCK_N": attention_heur_block_n,
    },
    "index_select": {
        "BLOCK_M": index_select_heur_block_m,
        "BLOCK_N": index_select_heur_block_n,
    },
    "gather": {
        "BLOCK_M": gather_heur_block_m,
        "BLOCK_N": gather_heur_block_n,
    },
    "var_mean": {
        "BLOCK_N": var_mean_heur_block_n,
    },
    "upsample_nearest2d": {
        "SAME_H": upsample_nearest2d_SAME_H,
        "SAME_W": upsample_nearest2d_SAME_W,
    },
    "mm": {
        "EVEN_K": mm_heur_even_k,
    },
    "rand": {
        "BLOCK": rand_heur_block,
        "num_warps": rand_heur_num_warps,
    },
    "randn": {
        "BLOCK": rand_heur_block,
        "num_warps": rand_heur_num_warps,
    },
    "elementwise_generic": {
        "BLOCK_SIZE": simple_elementwise_blocksize_heur,
        "num_warps": lambda args: 8,
    },
}