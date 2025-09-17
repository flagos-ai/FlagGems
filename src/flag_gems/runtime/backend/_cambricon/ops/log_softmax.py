import copy
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from ..utils import MAX_NRAM_SIZE, TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
MAX_N = 16384


def align(max_block):
    a = triton.next_power_of_2(max_block)
    return max_block if max_block == a else a // 2


def config_prune1(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    input = named_args["input_ptr"]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        TILE_K, TILE_N, num_warps, num_stages = (
            kw["TILE_K"],
            kw["TILE_N"],
            config.num_warps,
            config.num_stages,
        )
        if N < MAX_N:
            config = copy.deepcopy(config)
            TILE_N = config.kwargs["TILE_N"] = N
            k_per_core = math.ceil(K / max(TOTAL_CORE_NUM // M, 1))
            nram_usage = (2 * TILE_N + 1) * k_per_core * 4
            if nram_usage < MAX_NRAM_SIZE:
                TILE_K = config.kwargs["TILE_K"] = k_per_core
                num_stages = config.num_stages = 1
                key = (TILE_K, TILE_N, num_warps, num_stages)
                configs_map.setdefault(key, config)
            else:
                max_tile_k_without_pipe = MAX_NRAM_SIZE // 4 // (2 * TILE_N + 1)
                TILE_K = config.kwargs["TILE_K"] = align(max_tile_k_without_pipe)
                num_stages = config.num_stages = 1
                key = (TILE_K, TILE_N, num_warps, num_stages)
                configs_map.setdefault(key, config)

                config = copy.deepcopy(config)
                max_tile_k_without_pipe = MAX_NRAM_SIZE // 4 // (3 * TILE_N + 1)
                if input.dtype == torch.float32:
                    max_tile_k_without_pipe = MAX_NRAM_SIZE // 4 // (4 * TILE_N + 1)
                TILE_K = config.kwargs["TILE_K"] = align(max_tile_k_without_pipe)
                num_stages = config.num_stages = 3
                key = (TILE_K, TILE_N, num_warps, num_stages)
                configs_map.setdefault(key, config)
        else:
            key = (TILE_K, TILE_N, num_warps, num_stages)
            configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    extra_config = copy.deepcopy(pruned_configs[0])
    extra_config.kwargs["TILE_K"] = 1
    extra_config.kwargs["TILE_N"] = N
    extra_config.num_warps = 1
    extra_config.num_stages = 3
    pruned_configs.append(extra_config)
    extra_config2 = copy.deepcopy(extra_config)
    extra_config2.num_stages = 1
    pruned_configs.append(extra_config2)
    return pruned_configs


def log_softmax_tile_mode_for_non_inner(M, N, K, TILE_N, TILE_K):
    one_tile_k = TILE_K * max(TOTAL_CORE_NUM // M, 1) >= K
    one_tile_n = TILE_N >= N
    if one_tile_n and one_tile_k:
        return 0
    elif one_tile_n and not one_tile_k:
        return 1
    else:
        return 2


@libentry()
@libtuner(
    configs=[
        triton.Config({"TILE_K": k, "TILE_N": 2**n}, num_stages=s, num_warps=1)
        for k in [1, 2, 4, 8]
        for n in range(10, 15, 1)
        for s in [1, 3]
    ],
    key=[
        "N",
        "K",
    ],
    prune_configs_by={"early_config_prune": config_prune1},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: log_softmax_tile_mode_for_non_inner(
            args["M"], args["N"], args["K"], args["TILE_N"], args["TILE_K"]
        ),
    },
)
@triton.jit
def log_softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    p_k_num = tl.num_programs(axis=1)
    split_k = tl.cdiv(K, p_k_num)
    k_start = pid_k * split_k

    log2e = 1.442695

    if TILE_MODE == 0:
        n_offset = tl.arange(0, TILE_N)
        k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)
        offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
        mask = k_offset[None, :] < K
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = inp - tl.max(inp, axis=0)[None, :]
        e = tl.exp(m)
        s = tl.sum(e, axis=0)[None, :]
        output = m - tl.log2(s) / log2e
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, output, mask=mask)
    elif TILE_MODE == 1:
        for k_idx in range(0, split_k, TILE_K):
            k_offset = k_start + k_idx + tl.arange(0, TILE_K)
            n_offset = tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
            mask = k_offset[None, :] < K
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            m = inp - tl.max(inp, axis=0)[None, :]
            e = tl.exp(m)
            s = tl.sum(e, axis=0)[None, :]
            output = m - tl.log2(s) / log2e
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, output, mask=mask)
    else:
        for k_idx in range(0, split_k, TILE_K):
            k_offset = k_start + k_idx + tl.arange(0, TILE_K)
            m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
            z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, TILE_N):
                n_offset = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
                mask = (n_offset[:, None] < N) & (k_offset[None, :] < K)
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                m_new = tl.maximum(m, inp)
                all_neg_inf = m_new == float("-inf")
                z = tl.where(
                    all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new)
                )
                m = m_new

            m_reduced = tl.max(m, 0)  # (TILE_K,)
            z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
            recip_z = 1.0 / z
            m = m_reduced

            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, TILE_N):
                n_offset = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
                mask = (n_offset[:, None] < N) & (k_offset[None, :] < K)
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                o = tl.exp(inp - m[None, :]) * recip_z[None, :]
                output = tl.log2(o) / log2e
                tl.store(output_ptr + offset, output, mask=mask)


def config_prune2(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    input = named_args["input_ptr"]
    configs_map = {}
    # When N is less than MAX_C_MLU_SOFTMAX_FORWARD, no reduction loops
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            config.num_warps,
            config.num_stages,
        )
        if N < MAX_N:
            config = copy.deepcopy(config)
            BLOCK_N = config.kwargs["BLOCK_N"] = N
            m_per_core = math.ceil(M / TOTAL_CORE_NUM)
            nram_usage = (2 * BLOCK_N + 1) * m_per_core * 4
            if nram_usage < MAX_NRAM_SIZE:
                BLOCK_M = config.kwargs["BLOCK_M"] = m_per_core
                num_stages = config.num_stages = 1
                key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
                configs_map.setdefault(key, config)
            else:
                max_block_m_without_pipe = MAX_NRAM_SIZE // 4 // (2 * BLOCK_N + 1)
                BLOCK_M = config.kwargs["BLOCK_M"] = align(max_block_m_without_pipe)
                num_stages = config.num_stages = 1
                key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
                configs_map.setdefault(key, config)

                config = copy.deepcopy(config)
                max_block_m_without_pipe = MAX_NRAM_SIZE // 4 // (4 * BLOCK_N + 1)
                if input.dtype == torch.float32:
                    max_block_m_without_pipe = MAX_NRAM_SIZE // 4 // (6 * BLOCK_N + 1)
                BLOCK_M = config.kwargs["BLOCK_M"] = align(max_block_m_without_pipe)
                num_stages = config.num_stages = 3
                key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
                configs_map.setdefault(key, config)
        key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    # Add a heuristic config.
    extra_config = copy.deepcopy(pruned_configs[0])
    extra_config.kwargs["BLOCK_M"] = 1
    extra_config.kwargs["BLOCK_N"] = N
    extra_config.num_warps = 1
    extra_config.num_stages = 3
    pruned_configs.append(extra_config)
    extra_config2 = copy.deepcopy(extra_config)
    extra_config2.num_stages = 1
    pruned_configs.append(extra_config2)
    return pruned_configs


def log_softmax_tile_mode_for_inner(M, N, BLOCK_M, BLOCK_N):
    one_tile_m = BLOCK_M * TOTAL_CORE_NUM >= M
    one_tile_n = BLOCK_N >= N
    if one_tile_n and one_tile_m:
        return 0
    elif one_tile_n and not one_tile_m:
        return 1
    else:
        return 2


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("log_softmax"),
    key=[
        "M",
        "N",
    ],
    prune_configs_by={"early_config_prune": config_prune2},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: log_softmax_tile_mode_for_inner(
            args["M"], args["N"], args["BLOCK_M"], args["BLOCK_N"]
        ),
    },
)
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    split_m = tl.cdiv(M, pnum)
    m_start = pid_m * split_m

    log2e = 1.442695

    if TILE_MODE == 0:
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offset = tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row_minus_max = inp - tl.max(inp, axis=1)[:, None]
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        recip = 1.0 / denominator
        softmax_output = numerator * recip
        output = tl.log2(softmax_output) / log2e
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, output, mask=mask)
    elif TILE_MODE == 1:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            n_offset = tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            trans_inp = tl.trans(inp)
            row_minus_max = trans_inp - tl.max(trans_inp, axis=0)[None, :]
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)[None, :]
            recip = 1.0 / denominator
            softmax_output = tl.trans(numerator * recip)
            output = tl.log2(softmax_output) / log2e
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, output, mask=mask)
    else:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            block_max = tl.full(
                [BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32
            )
            block_sum = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                cur_max = tl.maximum(block_max, inp)
                all_neg_inf = cur_max == float("-inf")
                block_sum = tl.where(
                    all_neg_inf,
                    block_sum,
                    block_sum * tl.exp(block_max - cur_max) + tl.exp(inp - cur_max),
                )
                block_max = cur_max

            trans_block_max = tl.trans(block_max)
            trans_block_sum = tl.trans(block_sum)
            max_reduced = tl.max(trans_block_max, 0)
            total_sum = tl.sum(
                trans_block_sum * tl.exp(trans_block_max - max_reduced[None, :]), 0
            )
            recip_total_sum = 1.0 / total_sum
            total_max = max_reduced

            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                o = tl.exp(inp - total_max[:, None]) * recip_total_sum[:, None]
                output = tl.log2(o) / log2e
                tl.store(output_ptr + offset, output, mask=mask)


@triton.jit
def log_softmax_kernel_inner_k_partial_stats(
    x_ptr,
    max_buf_ptr,
    sum_buf_ptr,
    M,
    N,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pnum = tl.num_programs(axis=0)
    pid = tl.program_id(0)
    total_blocks = (M // BLOCK_M) * T
    work_per_core = (total_blocks + pnum - 1) // pnum
    start = pid * work_per_core
    end = tl.minimum(start + work_per_core, total_blocks)

    for task in range(start, end):
        row_id = task // T
        tile_id = task % T

        offs_m = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        tile = tl.load(
            x_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        tile_max = tl.max(tile, axis=1)
        tile_sum = tl.sum(tl.exp(tile - tile_max[:, None]), axis=1)

        tl.store(max_buf_ptr + offs_m * T + tile_id, tile_max, mask=(offs_m < M))
        tl.store(sum_buf_ptr + offs_m * T + tile_id, tile_sum, mask=(offs_m < M))


@triton.jit
def log_softmax_kernel_inner_k_merge_stats(
    max_buf_ptr,
    sum_buf_ptr,
    gmax_ptr,
    gsum_ptr,
    M: tl.constexpr,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    tile_max = tl.load(
        max_buf_ptr + offs_m[:, None] * T + tl.arange(0, T)[None, :],
        mask=(offs_m[:, None] < M),
        other=-float("inf"),
    )
    tile_sum = tl.load(
        sum_buf_ptr + offs_m[:, None] * T + tl.arange(0, T)[None, :],
        mask=(offs_m[:, None] < M),
        other=0.0,
    ).to(tl.float32)

    gmax = tl.max(tile_max, axis=1)
    scale = tl.exp(tile_max - gmax[:, None])
    scale = tl.where(gmax[:, None] == -float("inf"), 0.0, scale)
    gsum = tl.sum(tile_sum * scale, axis=1)

    tl.store(gmax_ptr + offs_m, gmax, mask=mask_m)
    tl.store(gsum_ptr + offs_m, gsum, mask=mask_m)


@triton.jit
def log_softmax_kernel_inner_k_write_logsoftmax(
    x_ptr,
    y_ptr,
    gmax_ptr,
    gsum_ptr,
    M,
    N,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    log2e = 1.442695
    pnum = tl.num_programs(axis=0)
    pid = tl.program_id(0)
    total_blocks = (M // BLOCK_M) * T
    work_per_core = (total_blocks + pnum - 1) // pnum
    start = pid * work_per_core
    end = tl.minimum(start + work_per_core, total_blocks)

    for task in range(start, end):
        row_id = task // T
        tile_id = task % T

        offs_m = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        gmax = tl.load(gmax_ptr + offs_m, mask=(offs_m < M), other=-float("inf")).to(
            tl.float32
        )
        gsum = tl.load(gsum_ptr + offs_m, mask=(offs_m < M), other=0.0).to(tl.float32)

        tile = tl.load(
            x_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        o = tl.exp(tile - gmax[:, None]) / gsum[:, None]
        out = tl.log2(o) / log2e

        tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# ------------------------  backward -------------------------------


def config_prune3(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    output = named_args["output_ptr"]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        TILE_K, TILE_N, num_warps, num_stages = (
            kw["TILE_K"],
            kw["TILE_N"],
            config.num_warps,
            config.num_stages,
        )
        if N < MAX_N:
            config = copy.deepcopy(config)
            TILE_N = config.kwargs["TILE_N"] = N
            k_per_core = math.ceil(K / max(TOTAL_CORE_NUM // M, 1))
            nram_usage = (3 * TILE_N + 1) * k_per_core * 4
            if nram_usage < MAX_NRAM_SIZE:
                TILE_K = config.kwargs["TILE_K"] = k_per_core
                num_stages = config.num_stages = 1
                key = (TILE_K, TILE_N, num_warps, num_stages)
                configs_map.setdefault(key, config)
            else:
                max_tile_k_without_pipe = MAX_NRAM_SIZE // 4 // (3 * TILE_N + 1)
                TILE_K = config.kwargs["TILE_K"] = align(max_tile_k_without_pipe)
                num_stages = config.num_stages = 1
                key = (TILE_K, TILE_N, num_warps, num_stages)
                configs_map.setdefault(key, config)

                config = copy.deepcopy(config)
                max_tile_k_without_pipe = MAX_NRAM_SIZE // 4 // (6 * TILE_N + 1)
                if output.dtype == torch.float32:
                    max_tile_k_without_pipe = MAX_NRAM_SIZE // 4 // (7 * TILE_N + 1)
                TILE_K = config.kwargs["TILE_K"] = align(max_tile_k_without_pipe)
                num_stages = config.num_stages = 3
                key = (TILE_K, TILE_N, num_warps, num_stages)
                configs_map.setdefault(key, config)
        else:
            key = (TILE_K, TILE_N, num_warps, num_stages)
            configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    extra_config = copy.deepcopy(pruned_configs[0])
    extra_config.kwargs["TILE_K"] = 1
    extra_config.kwargs["TILE_N"] = N
    extra_config.num_warps = 1
    extra_config.num_stages = 3
    pruned_configs.append(extra_config)
    extra_config2 = copy.deepcopy(extra_config)
    extra_config2.num_stages = 1
    pruned_configs.append(extra_config2)
    return pruned_configs


@libentry()
@libtuner(
    configs=[
        triton.Config({"TILE_K": k, "TILE_N": 2**n}, num_stages=s, num_warps=1)
        for k in [1, 2, 4, 8]
        for n in range(10, 15, 1)
        for s in [1, 3]
    ],
    key=[
        "N",
        "K",
    ],
    prune_configs_by={"early_config_prune": config_prune3},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: log_softmax_tile_mode_for_non_inner(
            args["M"], args["N"], args["K"], args["TILE_N"], args["TILE_K"]
        ),
    },
)
@triton.jit
def log_softmax_backward_kernel_non_inner(
    output_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    p_k_num = tl.num_programs(axis=1)
    split_k = tl.cdiv(K, p_k_num)
    k_start = pid_k * split_k

    if TILE_MODE == 0:
        n_offset = tl.arange(0, TILE_N)
        k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)
        offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
        mask = k_offset[None, :] < K
        out_tile = tl.load(output_ptr + offset, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
        scale = tl.sum(out_grad_tile, axis=0)
        in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]
        tl.store(in_grad_ptr + offset, in_grad_tile, mask=mask)
    elif TILE_MODE == 1:
        for k_idx in range(0, split_k, TILE_K):
            k_offset = k_start + k_idx + tl.arange(0, TILE_K)
            n_offset = tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
            mask = k_offset[None, :] < K
            out_tile = tl.load(output_ptr + offset, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
            scale = tl.sum(out_grad_tile, axis=0)
            in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]
            tl.store(in_grad_ptr + offset, in_grad_tile, mask=mask)
    else:
        for k_idx in range(0, split_k, TILE_K):
            k_offset = k_start + k_idx + tl.arange(0, TILE_K)
            scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, TILE_N):
                n_offset = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
                mask = (n_offset[:, None] < N) & (k_offset[None, :] < K)
                out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
                scale += out_grad_tile
            scale = tl.sum(scale, axis=0)
            for start_n in range(0, N, TILE_N):
                n_offset = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
                mask = (n_offset[:, None] < N) & (k_offset[None, :] < K)
                out_tile = tl.load(output_ptr + offset, mask=mask).to(tl.float32)
                out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
                in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]
                tl.store(in_grad_ptr + offset, in_grad_tile, mask=mask)


def nram_usage_for_backward_inner(bm, bn, tile_mode, num_stages, dtype):
    coef = 1
    if tile_mode == 0:
        if dtype == torch.float32:
            return 5 * bn * bm * 4
        else:
            return 4 * bn * bm * 4
    elif tile_mode == 1:
        if num_stages == 1:
            coef = 3
        else:
            if dtype == torch.float32:
                coef = 8
            else:
                coef = 6
    else:
        if num_stages == 1:
            coef = 4
        else:
            if dtype == torch.float32:
                coef = 11
            else:
                coef = 8
    return (coef * bn + 1) * bm * 4


def config_prune4(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    output = named_args["output_ptr"]
    dtype = output.dtype
    m_per_core = math.ceil(M / TOTAL_CORE_NUM)
    # No need for any loop.
    if nram_usage_for_backward_inner(m_per_core, N, 0, 1, dtype) < MAX_NRAM_SIZE:
        config = copy.deepcopy(configs[0])
        config.kwargs["BLOCK_M"] = m_per_core
        config.kwargs["BLOCK_N"] = N
        config.num_stages = 1
        return [config]
    align_num = 256 // 4 if dtype == torch.float32 else 256 // 2
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            config.num_stages,
        )
        # Align the lowest dimension to 256B while loading/storing data.
        if BLOCK_N % align_num != 0:
            continue
        # nram usage shoule be smaller than MAX_NRAM_SIZE
        mode = log_softmax_tile_mode_for_inner(M, N, BLOCK_M, BLOCK_N)
        nram = nram_usage_for_backward_inner(BLOCK_M, BLOCK_N, mode, num_stages, dtype)
        if nram > MAX_NRAM_SIZE or nram < MAX_NRAM_SIZE // 2:
            continue
        pruned_configs.append(config)
    return pruned_configs


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_N": 64 * k, "BLOCK_M": 2**n}, num_stages=s, num_warps=1)
        for k in range(1, 17)
        for n in range(3, 10, 1)
        for s in [1, 3]
    ],
    key=[
        "N",
        "M",
    ],
    prune_configs_by={"early_config_prune": config_prune4},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: log_softmax_tile_mode_for_inner(
            args["M"], args["N"], args["BLOCK_M"], args["BLOCK_N"]
        ),
    },
)
@triton.jit
def log_softmax_backward_kernel_inner(
    output_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    split_m = tl.cdiv(M, pnum)
    m_start = pid_m * split_m

    if TILE_MODE == 0:
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offset = tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M
        out_tile = tl.load(output_ptr + offset, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
        scale = tl.sum(out_grad_tile, 1)
        in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
        tl.store(in_grad_ptr + offset, in_grad_tile, mask=mask)
    elif TILE_MODE == 1:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            n_offset = tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M and n_offset[None, :] < N
            out_tile = tl.load(output_ptr + offset, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
            scale = tl.sum(out_grad_tile, 1)
            in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
            tl.store(in_grad_ptr + offset, in_grad_tile, mask=mask)
    else:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            scale = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
                scale += out_grad_tile
            scale = tl.sum(scale, 1)
            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                out_tile = tl.load(
                    output_ptr + offset, mask=mask, eviction_policy="evict_first"
                ).to(tl.float32)
                out_grad_tile = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
                in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
                tl.store(in_grad_ptr + offset, in_grad_tile, mask=mask)


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_CAMBRICON LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            logger.debug("GEMS_CAMBRICON LOGSOFTMAX USE NON INNER")
            grid = lambda meta: (M, max(TOTAL_CORE_NUM // M, 1), 1)
            log_softmax_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            logger.debug("GEMS_CAMBRICON LOGSOFTMAX USE INNER")
            if M > TOTAL_CORE_NUM or N < 1024 * 8 * 8:
                log_softmax_kernel_inner[TOTAL_CORE_NUM, 1, 1](
                    out,
                    inp,
                    M,
                    N,
                )
            else:
                block_m = 1
                block_n = 8192 * 4
                if dtype is torch.float32:
                    block_n = 8192 * 2
                # workspace
                T = (N + block_n - 1) // block_n
                max_buf = torch.empty((M, T), device=self.device, dtype=torch.float32)
                sum_buf = torch.empty((M, T), device=self.device, dtype=torch.float32)
                gmax = torch.empty((M,), device=self.device, dtype=torch.float32)
                gsum = torch.empty((M,), device=self.device, dtype=torch.float32)
                # kernel 1: per-tile stats
                log_softmax_kernel_inner_k_partial_stats[(TOTAL_CORE_NUM,)](
                    self,
                    max_buf,
                    sum_buf,
                    M,
                    N,
                    T,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    bottleneck="simd",
                    num_stages=3,
                )
                # kernel 2: merge stats along N-tiles
                grid_merge = (triton.cdiv(M, block_m),)
                log_softmax_kernel_inner_k_merge_stats[grid_merge](
                    max_buf, sum_buf, gmax, gsum, M, T, BLOCK_M=block_m
                )
                block_n = block_n // 2
                T = (N + block_n - 1) // block_n
                # kernel 3: write normalized outputs
                log_softmax_kernel_inner_k_write_logsoftmax[(TOTAL_CORE_NUM,)](
                    self,
                    out,
                    gmax,
                    gsum,
                    M,
                    N,
                    T,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    bottleneck="simd",
                    num_stages=3,
                )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_CAMBRICON LOG_SOFTMAX VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            logger.debug("GEMS_CAMBRICON LOG SOFTMAX VJP USE NON INNER")
            grid = lambda meta: (M, max(TOTAL_CORE_NUM // M, 1), 1)
            log_softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                K,
            )
        else:
            logger.debug("GEMS_CAMBRICON LOG SOFTMAX VJP USE INNER")
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            log_softmax_backward_kernel_inner[TOTAL_CORE_NUM, 1, 1](
                output,
                grad_output,
                in_grad,
                M,
                N,
            )
    return in_grad
