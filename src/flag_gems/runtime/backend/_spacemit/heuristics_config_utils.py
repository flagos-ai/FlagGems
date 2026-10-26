import os

import torch
import triton

from flag_gems.runtime import config_loader

# Shared smt_rvisa eDSL plumbing for mean/max/sum/softmax/log_softmax/mv
# (they do `from ..heuristics_config_utils import HAS_SMT_RVISA, ...`).
# The eDSL only exists on newer spine-triton builds; binding it once here
# lets each op file keep its regular triton kernel when the module is
# missing, the same way ops/__init__ drops mm/flash_attention on old stacks.
try:
    import triton.language.extra.smt_rvisa as tle
    from triton.language.extra.smt_rvisa import call as _sr_call

    f16 = tle.f16
    f32 = tle.f32
    HAS_SMT_RVISA = True
except ImportError:
    tle = None
    _sr_call = None
    f16 = None
    f32 = None
    HAS_SMT_RVISA = False


def _rvisa_yaml_default() -> bool:
    # default comes from the spacemit_rvisa entry in tune_configs.yaml
    # (vendor_primitive_yaml_config is the raw yaml dict, kept on the loader
    # exactly for vendor-custom keys like this one)
    for cfg in (config_loader.vendor_primitive_yaml_config or {}).get(
        "spacemit_rvisa"
    ) or []:
        enabled = (cfg.get("META") or {}).get("enabled")
        if enabled is not None:
            return bool(enabled)
    return False


_RVISA_YAML_DEFAULT = _rvisa_yaml_default()


def smt_rvisa_enabled() -> bool:
    """Opt-in switch for the smt_rvisa eDSL kernels (default off).

    Default lives in tune_configs.yaml (spacemit_rvisa.enabled);
    SPACEMIT_SMT_RVISA=1/0 overrides it per process. The env var is read per
    call so tests can toggle it with monkeypatch.setenv.
    """
    if not HAS_SMT_RVISA:
        return False
    env = os.environ.get("SPACEMIT_SMT_RVISA")
    if env is not None:
        return env == "1"
    return _RVISA_YAML_DEFAULT


def argmax_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmax_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


def argmin_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmin_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


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



def mm_heur_even_k(args):
    return args["K"] % args["BLOCK_SIZE_K"] == 0


def rand_heur_block(args):
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
    NUM_SMS = 8
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
    # The one-tile fast path of the forward softmax kernels feeds the raw
    # half-dtype tensor straight into tl.exp, which the Triton frontend
    # rejects by design (math ops accept fp32/fp64 only, same on NVIDIA).
    # The multi-tile path keeps m/z in fp32 accumulators, so route fp16/bf16
    # inputs there; other dtypes keep the tile-size decision.
    for value in args.values():
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            if dtype in (torch.float16, torch.bfloat16):
                return False
            break
    return args["TILE_N"] >= args["N"]


def softmax_bwd_heur_one_tile_per_cta(args):
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


def softmax_heru_tile_m(args):
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


def upsample_nearest2d_USE_INT32_IDX(args):
    return args["N"] * args["C"] * args["OH"] * args["OW"] <= (2**31 - 1)


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
    },
    "exponential_": {
        "BLOCK": exponential_heur_block,
    },
    "gather": {
        "BLOCK_M": gather_heur_block_m,
        "BLOCK_N": gather_heur_block_n,
    },
    "mm": {
        "EVEN_K": mm_heur_even_k,
    },
    "rand": {
        "BLOCK": rand_heur_block,
    },
    "randn": {
        "BLOCK": randn_heur_block,
    },
    "softmax_non_inner": {
        "TILE_K": softmax_heur_tile_k,
        "TILE_N": softmax_heur_tile_n_non_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
    },
    "softmax_inner": {
        "TILE_N": softmax_heur_tile_n_inner,
        "ONE_TILE_PER_CTA": softmax_heur_one_tile_per_cta,
    },
    "softmax_backward_non_inner": {
        "TILE_N": softmax_heur_tile_n_bwd_non_inner,
        "ONE_TILE_PER_CTA": softmax_bwd_heur_one_tile_per_cta,
    },
    "softmax_backward_inner": {
        "TILE_M": softmax_heru_tile_m,
        "ONE_TILE_PER_CTA": softmax_bwd_heur_one_tile_per_cta,
    },
    "uniform": {
        "BLOCK": uniform_heur_block,
    },
    "upsample_nearest2d": {
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
}
