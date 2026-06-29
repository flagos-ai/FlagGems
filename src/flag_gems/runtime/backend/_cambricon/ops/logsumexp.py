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

logger = logging.getLogger(__name__)
MAX_N = 16384


def align(max_block):
    a = triton.next_power_of_2(max_block)
    return max_block if max_block == a else a // 2


def config_prune_non_inner(configs, named_args, **kwargs):
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
    for v in configs_map.values():
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


def logsumexp_tile_mode_for_non_inner(M, N, K, TILE_N, TILE_K):
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
    prune_configs_by={"early_config_prune": config_prune_non_inner},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: logsumexp_tile_mode_for_non_inner(
            args["M"], args["N"], args["K"], args["TILE_N"], args["TILE_K"]
        ),
    },
)
@triton.jit
def logsumexp_kernel_non_inner(
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

    if TILE_MODE == 0:
        n_offsets = tl.arange(0, TILE_N)
        k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
        offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
        mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, axis=0)
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        z = tl.sum(tl.exp(inp - safe_m[None, :]), axis=0)
        out = tl.where(m == float("-inf"), m, safe_m + tl.log(z))
        tl.store(output_ptr + pid_m * K + k_offsets, out, mask=k_offsets < K)
    elif TILE_MODE == 1:
        for k_idx in range(0, split_k, TILE_K):
            n_offsets = tl.arange(0, TILE_N)
            k_offsets = k_start + k_idx + tl.arange(0, TILE_K)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m = tl.max(inp, axis=0)
            safe_m = tl.where(m == float("-inf"), 0.0, m)
            z = tl.sum(tl.exp(inp - safe_m[None, :]), axis=0)
            out = tl.where(m == float("-inf"), m, safe_m + tl.log(z))
            tl.store(output_ptr + pid_m * K + k_offsets, out, mask=k_offsets < K)
    else:
        for k_idx in range(0, split_k, TILE_K):
            k_offsets = k_start + k_idx + tl.arange(0, TILE_K)
            m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
            z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)
                offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
                mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
                inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                m_new = tl.maximum(m, inp)
                all_neg_inf = m_new == float("-inf")
                z = tl.where(
                    all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new)
                )
                m = m_new

            m_reduced = tl.max(m, axis=0)
            z = tl.sum(z * tl.exp(m - m_reduced[None, :]), axis=0)
            out = tl.where(m_reduced == float("-inf"), m_reduced, m_reduced + tl.log(z))
            tl.store(output_ptr + pid_m * K + k_offsets, out, mask=k_offsets < K)


def config_prune_inner(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    input = named_args["input_ptr"]
    configs_map = {}
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
        configs_map.setdefault(key, config)
    pruned_configs = []
    for v in configs_map.values():
        pruned_configs.append(v)
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


def logsumexp_tile_mode_for_inner(M, N, BLOCK_M, BLOCK_N):
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
    prune_configs_by={"early_config_prune": config_prune_inner},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: logsumexp_tile_mode_for_inner(
            args["M"], args["N"], args["BLOCK_M"], args["BLOCK_N"]
        ),
    },
)
@triton.jit
def logsumexp_kernel_inner(
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

    if TILE_MODE == 0:
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = tl.arange(0, BLOCK_N)
        offsets = m_offsets[:, None] * N + n_offsets[None, :]
        mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, axis=1)
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
        out = tl.where(m == float("-inf"), m, safe_m + tl.log(z))
        tl.store(output_ptr + m_offsets, out, mask=m_offsets < M)
    elif TILE_MODE == 1:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offsets = m_start + m_idx + tl.arange(0, BLOCK_M)
            n_offsets = tl.arange(0, BLOCK_N)
            offsets = m_offsets[:, None] * N + n_offsets[None, :]
            mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m = tl.max(inp, axis=1)
            safe_m = tl.where(m == float("-inf"), 0.0, m)
            z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
            out = tl.where(m == float("-inf"), m, safe_m + tl.log(z))
            tl.store(output_ptr + m_offsets, out, mask=m_offsets < M)
    else:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offsets = m_start + m_idx + tl.arange(0, BLOCK_M)
            block_max = tl.full(
                [BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32
            )
            block_sum = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)

            for start_n in range(0, N, BLOCK_N):
                n_offsets = start_n + tl.arange(0, BLOCK_N)
                offsets = m_offsets[:, None] * N + n_offsets[None, :]
                mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
                inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
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

            max_reduced = tl.max(block_max, axis=1)
            total_sum = tl.sum(
                block_sum * tl.exp(block_max - max_reduced[:, None]), axis=1
            )
            out = tl.where(
                max_reduced == float("-inf"),
                max_reduced,
                max_reduced + tl.log(total_sum),
            )
            tl.store(output_ptr + m_offsets, out, mask=m_offsets < M)


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_CAMBRICON LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            return inp.clone()
        if len(dim) == 1:
            dim = dim[0]
        else:
            sorted_dims = sorted([d % inp.ndim for d in dim], reverse=True)
            result = inp
            for d in sorted_dims:
                result = logsumexp(result, d, keepdim=True)
            if not keepdim:
                for d in sorted(sorted_dims, reverse=True):
                    result = result.squeeze(d)
            return result

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    M = 1
    N = inp.shape[dim]
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    shape = list(inp.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if K > 1:
            logger.debug("GEMS_CAMBRICON LOGSUMEXP USE NON INNER")
            grid = lambda meta: (M, max(TOTAL_CORE_NUM // M, 1), 1)
            logsumexp_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            logger.debug("GEMS_CAMBRICON LOGSUMEXP USE INNER")
            logsumexp_kernel_inner[TOTAL_CORE_NUM, 1, 1](
                out,
                inp,
                M,
                N,
            )

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
