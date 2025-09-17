import copy
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils import MAX_NRAM_SIZE, TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
MAX_N = 31744


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_last"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = tl.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = bx + tx
    col_mask = col_offset < N

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    normalized = tl.sqrt(tl.sum(v_block, axis=1) + eps)
    tl.store(norm + col_offset, normalized[:, None], mask=col_mask)
    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)

    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value / normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


def config_prune_for_first(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    configs_map = {}
    # When N is less than MAX_C_MLU_SOFTMAX_FORWARD, no reduction loops
    for config in configs:
        kw = config.kwargs
        BLOCK_ROW_SIZE, BLOCK_COL_SIZE, num_warps, num_stages = (
            kw["BLOCK_ROW_SIZE"],
            kw["BLOCK_COL_SIZE"],
            config.num_warps,
            config.num_stages,
        )
        if N < MAX_N:
            config = copy.deepcopy(config)
            BLOCK_COL_SIZE = config.kwargs["BLOCK_COL_SIZE"] = N
            m_per_core = math.ceil(M / TOTAL_CORE_NUM)
            nram_usage = (3 * BLOCK_COL_SIZE + 1) * m_per_core * 4
            if nram_usage < MAX_NRAM_SIZE:
                BLOCK_ROW_SIZE = config.kwargs["BLOCK_ROW_SIZE"] = m_per_core
                num_stages = config.num_stages = 1
                key = (BLOCK_ROW_SIZE, BLOCK_COL_SIZE, num_warps, num_stages)
                configs_map.setdefault(key, config)
            else:
                max_block_m_without_pipe = (
                    MAX_NRAM_SIZE // 4 // (3 * BLOCK_COL_SIZE + 1)
                )
                BLOCK_ROW_SIZE = config.kwargs[
                    "BLOCK_ROW_SIZE"
                ] = max_block_m_without_pipe
                num_stages = config.num_stages = 1
                key = (BLOCK_ROW_SIZE, BLOCK_COL_SIZE, num_warps, num_stages)
                configs_map.setdefault(key, config)

                config = copy.deepcopy(config)
                max_block_m_without_pipe = (
                    MAX_NRAM_SIZE // 4 // (6 * BLOCK_COL_SIZE + 4)
                )
                num_stages = config.num_stages = 3
                key = (BLOCK_ROW_SIZE, BLOCK_COL_SIZE, num_warps, num_stages)
                configs_map.setdefault(key, config)
        key = (BLOCK_ROW_SIZE, BLOCK_COL_SIZE, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    return pruned_configs


def tile_mode_for_first(args):
    one_tile_m = args["BLOCK_ROW_SIZE"] * TOTAL_CORE_NUM >= args["M"]
    one_tile_n = args["BLOCK_COL_SIZE"] >= args["N"]
    if one_tile_n and one_tile_m:
        return 0
    elif one_tile_n and not one_tile_m:
        return 1
    else:
        return 2


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_first"),
    key=["M", "N"],
    prune_configs_by={"early_config_prune": config_prune_for_first},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: tile_mode_for_first(args),
    },
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_first(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    split_m = tl.cdiv(M, pnum)
    m_start = pid_m * split_m
    if TILE_MODE == 0:
        m_offset = pid_m * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
        n_offset = tl.arange(0, BLOCK_COL_SIZE)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M
        v_value = tl.load(v + offset, mask=mask).to(tl.float32)
        normalized = tl.sqrt(tl.sum(v_value * v_value, axis=1) + eps)
        tl.store(norm + m_offset[:, None], normalized[:, None], mask=mask)
        g_value = tl.load(g + m_offset[:, None], mask=mask).to(tl.float32)
        v_vec = v_value / normalized[:, None]
        out = v_vec * g_value
        tl.store(output + offset, out, mask=mask)
    elif TILE_MODE == 1:
        for m_idx in range(0, split_m, BLOCK_ROW_SIZE):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_ROW_SIZE)
            n_offset = tl.arange(0, BLOCK_COL_SIZE)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M
            v_value = tl.load(v + offset, mask=mask).to(tl.float32)
            normalized = tl.sqrt(tl.sum(v_value * v_value, axis=1) + eps)
            tl.store(norm + m_offset[:, None], normalized[:, None], mask=mask)
            g_value = tl.load(g + m_offset[:, None], mask=mask).to(tl.float32)
            v_vec = v_value / normalized[:, None]
            out = v_vec * g_value
            tl.store(output + offset, out, mask=mask)
    else:
        for m_idx in range(0, split_m, BLOCK_ROW_SIZE):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_ROW_SIZE)
            m_mask = m_offset[:, None] < M
            v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
            for start_n in range(0, N, BLOCK_COL_SIZE):
                n_offset = start_n + tl.arange(0, BLOCK_COL_SIZE)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_mask and n_offset[None, :] < N
                v_value = tl.load(v + offset, mask=mask).to(tl.float32)
                v_block += v_value * v_value

            normalized = tl.sqrt(tl.sum(v_block, axis=1) + eps)
            tl.store(norm + m_offset[:, None], normalized[:, None], mask=m_mask)
            g_value = tl.load(g + m_offset[:, None], mask=m_mask).to(tl.float32)

            for start_n in range(0, N, BLOCK_COL_SIZE):
                n_offset = start_n + tl.arange(0, BLOCK_COL_SIZE)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_mask and n_offset[None, :] < N
                v_value = tl.load(v + offset, mask=mask).to(tl.float32)
                v_vec = v_value / normalized[:, None]
                out = v_vec * g_value
                tl.store(output + offset, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_last"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = tl.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = tx + bx
    col_mask = col_offset < N

    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)
    norm_value = tl.load(norm + col_offset, mask=col_mask).to(tl.float32)

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]

    vw_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        vw_block += v_value * w_value
    vw_sum = tl.sum(vw_block, 1)[:, None]

    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + col_offset, g_grad_value, mask=col_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_norm_kernel_first"), key=["M", "N"]
)
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_first(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = tl.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)
    norm_value = tl.load(norm + row_offset, mask=row_mask).to(tl.float32)

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]

    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * w_value
    vw_sum = tl.sum(v_block, 1)[:, None]

    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


def weight_norm_interface(v, g, dim=0):
    logger.debug("GEMS_CAMBRICON WEIGHTNORM FORWARD")
    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g)
    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        with torch_device_fn.device(v.device):
            weight_norm_kernel_first[TOTAL_CORE_NUM, 1, 1](
                output, norm, v, g, M, N, eps=torch.finfo(torch.float32).tiny
            )
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
        with torch_device_fn.device(v.device):
            weight_norm_kernel_last[grid](
                output, norm, v, g, M, N, eps=torch.finfo(torch.float32).tiny
            )
    return output, norm


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_CAMBRICON WEIGHTNORM BACKWARD")
    w_grad = w_grad.contiguous()
    saved_v = saved_v.contiguous()
    saved_g = saved_g.contiguous()
    saved_norms = saved_norms.contiguous()
    v_grad = torch.empty_like(saved_v)
    g_grad = torch.empty_like(saved_g)

    if dim == 0:
        M = saved_v.shape[0]
        N = math.prod(saved_v.shape[1:])
        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_first[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
            )
    elif dim == saved_v.ndim - 1:
        M = math.prod(saved_v.shape[:dim])
        N = saved_v.shape[dim]
        grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_last[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=torch.finfo(torch.float32).tiny,
            )
    return v_grad, g_grad
