import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

from ..utils import (
    MAX_NRAM_SIZE,
    TOTAL_CORE_NUM,
    cfggen_reduce_op,
    count_divisible_by_2,
)

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    _count = tl.maximum(count, 1)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / _count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("var_mean"), key=["M", "N"])
@triton.jit(do_not_specialize=["correction"])
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Var_ptr = Var + pid
        Mean_ptr = Mean + pid
        row_mask = pid < M

        _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            x = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)

            count = _count + mask
            cnt = tl.maximum(count, 1)
            cur_mean = (_mean * _count + x) / cnt
            _acc += (x - cur_mean) * (x - _mean) * mask
            _mean = cur_mean
            _count = count

        mean, _, acc = tl.reduce((_mean, _count, _acc), axis=1, combine_fn=welford_func)
        var = acc / (N - correction)
        mean = mean[:, None]
        var = var[:, None]
        # Write mean / var
        tl.store(Mean_ptr, mean, row_mask)
        tl.store(Var_ptr, var, row_mask)


def prune_varmean_config(configs, named_args, **kwargs):
    M = named_args["M"]
    pruned_configs = []
    for config in configs:
        BLOCK_SIZE = config.kwargs["BLOCK_SIZE"]
        num_stages = config.num_stages
        num_block = M // BLOCK_SIZE
        if num_block < 1:
            continue
        if num_block < TOTAL_CORE_NUM:
            # A core must process a BLOCK_SIZE of data.
            if num_stages > 1:
                continue
            alloc_num = 3
        else:
            alloc_num = 6
        # Set f32 as the default type.
        if BLOCK_SIZE * 4 * alloc_num < MAX_NRAM_SIZE:
            pruned_configs.append(config)
    # If M < 512, append the default config.
    if len(pruned_configs) == 0:
        pruned_configs.append(
            triton.Config({"BLOCK_SIZE": 512}, num_warps=1, num_stages=1)
        )
    return pruned_configs


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    prune_configs_by={"early_config_prune": prune_varmean_config},
    key=["M"],
    reset_to_zero=["Acc", "Average", "Count"],
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    },
)
@triton.jit
def var_mean_kernel_1(
    X, Acc, Average, Count, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    count = 0.0
    average = 0.0
    acc = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        count = tl.sum(mask.to(tl.float32))
        average = tl.sum(x) / count
        acc = tl.sum(x * x) - count * average * average
    else:
        _tmp1 = tl.zeros([BLOCK_SIZE], tl.float32)
        _tmp2 = tl.zeros([BLOCK_SIZE], tl.float32)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
            _count = tl.sum(mask.to(tl.float32))
            count = count + _count
            _tmp1 = _tmp1 + x
            _tmp2 = _tmp2 + x * x
        count = tl.maximum(count, 1)
        average = tl.sum(_tmp1) / count
        acc = tl.sum(_tmp2) - count * average * average

    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var_mean"))
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
    M,
    correction,
    BLOCK_NUM: tl.constexpr,
    ITER_NUM: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_NUM)
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc)
    average = tl.load(Average)
    count = tl.load(Count)

    for x in tl.static_range(1, ITER_NUM, 1):
        (
            average[: BLOCK_NUM // (2**x)],
            count[: BLOCK_NUM // (2**x)],
            acc[: BLOCK_NUM // (2**x)],
        ) = welford_func(
            average[: BLOCK_NUM // (2**x)],
            count[: BLOCK_NUM // (2**x)],
            acc[: BLOCK_NUM // (2**x)],
            average[BLOCK_NUM // (2**x) : (BLOCK_NUM // (2**x)) * 2],
            count[BLOCK_NUM // (2**x) : (BLOCK_NUM // (2**x)) * 2],
            acc[BLOCK_NUM // (2**x) : (BLOCK_NUM // (2**x)) * 2],
        )
    mean, _, nvar = tl.reduce(
        (
            average[: BLOCK_NUM // (2 ** (ITER_NUM - 1))],
            count[: BLOCK_NUM // (2 ** (ITER_NUM - 1))],
            acc[: BLOCK_NUM // (2 ** (ITER_NUM - 1))],
        ),
        axis=0,
        combine_fn=welford_func,
    )

    # FIXME: Reset to original reduce programming mode after optimizing the tl.reduce.
    # mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (M - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_CAMBRICON VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        M = x.numel()

        grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        acc = torch.zeros([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
        average = torch.zeros([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
        count = torch.zeros([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
        loop_num = count_divisible_by_2(TOTAL_CORE_NUM) + 1

        with torch_device_fn.device(x.device):
            var_mean_kernel_1[grid](x, acc, average, count, M)
            var_mean_kernel_2[(1,)](
                acc,
                average,
                count,
                var,
                mean,
                M,
                correction,
                BLOCK_NUM=TOTAL_CORE_NUM,
                ITER_NUM=loop_num,
            )
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        grid = lambda META: (min(triton.cdiv(M, META["BLOCK_M"]), TOTAL_CORE_NUM),)
        with torch_device_fn.device(x.device):
            var_mean_welford_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
