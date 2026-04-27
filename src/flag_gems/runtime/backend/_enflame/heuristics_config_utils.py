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
