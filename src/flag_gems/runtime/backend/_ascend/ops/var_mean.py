import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)

        count = _count + mask
        cnt = tl.maximum(count, 1)
        cur_mean = (_mean * _count + x) / cnt
        _acc += (x - cur_mean) * (x - _mean) * mask
        _mean = cur_mean
        _count = count

    # 手动实现 tl.reduce 的功能，沿着 axis=1 进行归约
    # 使用 tl.sum 来进行归约，这等价于 welford 算法在这种情况下的行为

    # 计算每行的总计数
    total_count = tl.sum(_count, axis=1)  # shape: (BLOCK_M,)

    # 计算加权平均值
    weighted_sum = tl.sum(_mean * _count, axis=1)  # shape: (BLOCK_M,)
    mean = weighted_sum / tl.maximum(total_count, 1)  # shape: (BLOCK_M,)

    # 计算方差累积值
    # 对于每个元素，计算其对总体方差的贡献
    mean_expanded = mean[:, None]  # shape: (BLOCK_M, 1)

    # 计算每个局部统计量对总体方差的贡献
    # 这是 Welford 算法的并行化版本
    local_var_contrib = _acc + _count * (_mean - mean_expanded) * (
        _mean - mean_expanded
    )
    acc = tl.sum(local_var_contrib, axis=1)  # shape: (BLOCK_M,)

    var = acc / (N - correction)
    mean = mean[:, None]
    var = var[:, None]

    # Write mean / var
    tl.store(Mean, mean, row_mask)
    tl.store(Var, var, row_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("var_mean"), key=["M", "N"])
@triton.jit(do_not_specialize=["correction"])
def var_mean_welford_kernel_simple(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 程序ID映射
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid
    row_mask = pid < M

    # 每行单独处理
    for row in range(BLOCK_M):
        if row < BLOCK_M:
            current_row_mask = (tl.arange(0, BLOCK_M) == row)[:, None] & row_mask

            if tl.sum(current_row_mask.to(tl.int32)) > 0:
                # 初始化当前行的统计量
                running_mean = 0.0
                running_M = 0.0
                count = 0

                # 按块处理当前行
                for off in range(0, N, BLOCK_N):
                    cols = off + tl.arange(0, BLOCK_N)
                    col_mask = cols < N

                    # 加载数据
                    x_vals = tl.load(X + row * N + cols, col_mask, other=0.0).to(
                        tl.float32
                    )

                    # 对块内每个有效元素进行在线更新
                    for i in range(BLOCK_N):
                        if i < BLOCK_N and (off + i) < N:
                            count += 1
                            x = x_vals[i]

                            delta = x - running_mean
                            running_mean += delta / count
                            delta2 = x - running_mean
                            running_M += delta * delta2

                # 计算方差
                variance = running_M / (N - correction) if N > correction else 0.0

                # 存储结果
                tl.store(Mean + row, running_mean, current_row_mask[:, 0])
                tl.store(Var + row, variance, current_row_mask[:, 0])


@libentry()
@triton.jit
def var_mean_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = tle.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

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
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    # mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)
    # 手动实现 tl.reduce 的功能，沿着 axis=0 进行归约
    # 计算总计数
    total_count = tl.sum(count)

    # 计算加权平均值
    weighted_sum = tl.sum(average * count)
    mean = weighted_sum / tl.maximum(total_count, 1)

    # 计算方差累积值
    # 对于每个块，计算其对总体方差的贡献
    # 这是 Welford 算法的并行化版本
    local_var_contrib = acc + count * (average - mean) * (average - mean)
    nvar = tl.sum(local_var_contrib)

    var = nvar / (N - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ASCEND VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            var_mean_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
            var_mean_kernel_2[(1,)](
                acc, average, count, var, mean, N, correction, BLOCK_NUM
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

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            var_mean_welford_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
