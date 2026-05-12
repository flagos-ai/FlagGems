import logging

import torch
import triton
from triton import language as tl

from flag_gems.ops.mv import mv
from flag_gems.utils import libentry

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


# The outer kernel requires 3 parameters to determine the splitting method,
# but during actual tuning, you only need to determine the total size of the split blocks.
# Based on the second input length N and the total size of the split blocks,
# the 3 parameters that determine the splitting method can be calculated.
# Therefore, the conversion between these two is achieved through early_config_prune.
def early_config_prune(configs, named_args, **kwargs):
    if "N" in kwargs:
        N = kwargs["N"]
    else:
        N = named_args["N"]

    new_configs = []
    for config in configs:
        tile_size = config.kwargs["tile_size"]
        block_n = min(tile_size, N)
        block_m = triton.cdiv(tile_size, block_n)
        new_config = triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n, "NEED_LOOP_N": block_n < N},
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )
        new_configs.append(new_config)

    return new_configs


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"tile_size": 1024}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 2048}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 4096}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 8192}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 16384}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 21760}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 32768}, num_stages=3, num_warps=1),
    ],
    key=["M", "N"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def outer_kernel(
    lhs,
    rhs,
    res,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_LOOP_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)

    m_tasks_num = tl.cdiv(M, BLOCK_M)
    n_tasks_num = tl.cdiv(N, BLOCK_N)
    total_tasks_num = m_tasks_num * n_tasks_num

    if NEED_LOOP_N:
        for task_id in range(pid, total_tasks_num, num_jobs):
            start_m = task_id // n_tasks_num
            start_n = task_id % n_tasks_num

            offset_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
            lhs_val = tl.load(lhs + offset_m, mask=offset_m < M)

            offset_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
            rhs_val = tl.load(rhs + offset_n, mask=offset_n < N)

            res_val = lhs_val[:, None] * rhs_val[None, :]

            offset_r = offset_m[:, None] * N + offset_n[None, :]
            tl.store(
                res + offset_r,
                res_val,
                mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
            )
    else:
        offset_n = tl.arange(0, BLOCK_N)
        rhs_val = tl.load(rhs + offset_n)
        for task_id in range(pid, total_tasks_num, num_jobs):
            start_m = task_id // n_tasks_num

            offset_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
            lhs_val = tl.load(lhs + offset_m, mask=offset_m < M)

            res_val = lhs_val[:, None] * rhs_val[None, :]

            offset_r = offset_m[:, None] * N + offset_n[None, :]
            tl.store(
                res + offset_r,
                res_val,
                mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
            )


def outer_(lhs, rhs):
    m = lhs.shape[0]
    n = rhs.shape[0]
    res_shape = [m, n]
    res = torch.empty(res_shape, dtype=lhs.dtype, device="npu")
    grid = lambda META: (
        min(
            triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
            65535,
        ),
    )
    outer_kernel[grid](lhs, rhs, res, m, n)
    return res


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("GEMS_ASCEND OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        out = outer_(inp, weight)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_ASCEND OUTER VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_grad = mv(out_grad, weight)
        weight_grad = mv(out_grad.t(), inp)

        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
