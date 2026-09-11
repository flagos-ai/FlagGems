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
    # the 2D tile bounded together with the capped BLOCK_M. Measured on 2026-09-02
    # (min-of-3 do_bench, same matrix): BLOCK_N 1024 over 256 gives ~12-13% lower
    # latency on (4096,4096)/(1024,65536) and is neutral on smaller shapes.
    return min(512, triton.next_power_of_2(args["N"]))


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


def reglu_swiglu_config(dtype, M, N):
    """XPU4 probe-tuned fixed tiling for reglu/swiglu forward (2 loads + 1 store).

    Probe findings (2026-08-13, XPU4, official benchmark matrix, probe6 A/B):
    - fp16 BLOCK_N>=2048 is compile-flaky (ConvertTritonXPUToLLVM assertion),
      so fp16 stays at BLOCK_N<=1024 (BLOCK_N=512 only for tiny rows).
    - fp32/bf16 large rows: wider BLOCK_N slashes per-program overhead
      (fp32 [4096,4096] 0.82ms -> 0.47ms @ BN2048; fp32 [1024,131072]
      6.58ms -> 1.80ms @ BN8192; bf16 [1024,131072] 8.52ms -> 2.94ms @ BN16384).
    - Tiny rows are launch-overhead bound; A/B (official do_bench, median):
        (64,64) M=64:                 bm1_bn1024 best (14.1/11.6/13.5us)
        (1024,2)/(1024,32) fp16/bf16: bm8_bn512 wins (127 vs 157us)
        (1024,2)/(64,64,2) fp32:      bm8_bn1024 wins (107 vs 111us)
        (64,64,2)/(64,64,32) fp16:    bm8_bn512 wins (452 vs 558us)
        (64,512,512) (M=32768):       bm16_bn1024 best (3245 vs 3318us)
        (1024,512):                   bm1_bn1024 best
    """
    if N >= 2048 and M >= 1024:
        if dtype == torch.float32:
            if N >= 65536:
                return 1, 8192, 8
            elif N >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        elif dtype == torch.bfloat16:
            if N >= 65536:
                return 1, 16384, 16
            elif N >= 4096:
                return 1, 4096, 8
            else:
                return 1, 2048, 8
        # fp16 large rows: BLOCK_N>=2048 compile-flaky -> keep BN1024
        return 8, 1024, 4
    if N <= 64:
        if M < 256:
            # e.g. (64,64): bm1_bn1024 wins in A/B
            return 1, 1024, 4
        if dtype == torch.float32:
            return 8, 1024, 4
        return 8, 512, 4
    # 64 < N < 2048 (or M < 1024): many-rows -> bm16, else bm1
    return (16, 1024, 4) if M >= 8192 else (1, 1024, 4)


def dreglu_dswiglu_config(dtype, M, N):
    """XPU1 probe-tuned fixed tiling for dreglu/dswiglu backward (3 loads + 2 stores).

    dswiglu backward has the exact same memory shape as dreglu backward
    (3 loads: grad_out / gate half / linear half, 2 stores into the two
    halves of grad_input) and is benchmarked on the same official (M, N)
    cells, so the dreglu tiling table transfers directly. Only the
    activation math differs.

    Probe findings (2026-08-19, XPU1, official benchmark matrix, probe1/probe2
    fixed-config sweeps + libtuner ConfigCache dump):
    - libtuner's favourite big-row config (342,2048) is the best known for
      N==2048 (fp16 (4096,2048) 0.824ms) and for M=32768 x N=256 (fp16 6.42ms);
      huge tiles in general (BLOCK_N >= 16384 fp32 / bn>=8192 fp16/bf16) hit
      TritonXPULegalize/uni_sram failures -> exclude.
    - N==4096/N==65536 win with wide single-row tiles:
      fp16 (1024,4096) 0.805->0.212ms @1x4096w8; fp16 (1024,65536)
      6.51->1.68ms @1x16384w16; fp32 (1024,65536) 5.56->2.23ms @4x8192w8;
      bf16 (1024,65536) 6.64->2.33ms @1x16384w8.
    - fp16 1x(N<=2048) tiles lose to (342,2048); fp16 above BN=2048 compiles
      (unlike forward reglu) but 2D tiles fill uni_sram -> cap bn.
    """
    f16 = dtype == torch.float16
    f32 = dtype == torch.float32
    # --- large rows: N >= 2048 ---
    if N >= 2048:
        if f16:
            if N >= 65536:
                return 1, 16384, 16
            if N == 4096:
                return 1, 4096, 8
            return 342, 2048, 4
        if f32:
            if N >= 65536:
                return 4, 8192, 8
            if N == 4096:
                return 1, 4096, 8
            return 4, 2048, 8
        if N >= 65536:
            return 1, 16384, 8
        if N == 4096:
            return 1, 4096, 8
        return 4, 2048, 4
    # --- tiny rows: N <= 64 ---
    if N <= 64:
        if N == 1:
            if M <= 1024:
                return (8, 64, 4) if f32 else (16, 64, 8)
            # M >= 2048: fp16/bf16 (32,64,4) 0.621ms; fp32 tuned (6,32) 0.612ms
            return (6, 32, 4) if f32 else (32, 64, 4)
        if N == 16:
            if M <= 1024:
                # fp32 (1,1024) 0.164ms beats 2D tiles under official do_bench
                return (1, 1024, 4) if f32 else (4, 256, 4)
            # M >= 2048: fp32 (8,1024) 0.637ms; f16/bf16 4x256 0.737/0.731ms
            return (8, 1024, 4) if f32 else (4, 256, 4)
        if N == 32:
            # M=64 micro-case (official probe3): f16 1x256 21.9us, f32 1x1024
            # 15.7us, bf16 1x1024 18.3us
            return (1, 1024, 4) if f32 else ((1, 256, 4) if f16 else (1, 1024, 4))
        return (8, 256, 4)
    # --- mid rows: 64 < N <= 1024 ---
    if f16:
        if M >= 32768:
            # (64,512,512): tuned (342,2048) = 6.42ms is best known
            return 342, 2048, 4
        return 1, 2048, 8
    if M >= 32768:
        # (64,512,512): launch/lane-bound, tuned (8,1024) fp32 / (1,1024) bf16
        return (8, 1024, 4) if f32 else (1, 1024, 4)
    # probe3 (official do_bench): fp32 1x1024 165.7us @M=1024, 8x1024
    # 641.9us @M=4096; bf16 1x1024 201.9/791.2us
    if f32:
        return (8, 1024, 4) if M > 1024 else (1, 1024, 4)
    return 1, 1024, 4
