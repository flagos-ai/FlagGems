import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim

from ..utils import TOTAL_CORE_NUM, cfggen_reduce_op, prune_reduce_config

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
pow = tl_extra_shim.pow


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _sum += a * a
        sum = tl.sum(_sum, axis=1)

        out = tl.sqrt(sum)[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
    reset_to_zero=["Out"],
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    },
)
@triton.jit
def l2_norm_kernel_1(
    X, Out, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    mid = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        mid = tl.sum(x * x)
    else:
        _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
            _tmp = _tmp + x * x
        mid = tl.sum(_tmp)

    tl.atomic_add(Out, mid.to(tl.float32))


@libentry()
@triton.jit
def l2_norm_kernel_2(
    Out,
):
    out = tl.load(Out)
    out = tl.sqrt(out)
    tl.store(Out, out)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _max = tl.maximum(tl.abs(a), _max)

        max = tl.max(_max, axis=1)
        out = max[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    },
)
@triton.jit
def max_norm_kernel_1(
    X, Out, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    mid = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        mid = tl.max(tl.abs(x))
    else:
        _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
            _x = tl.abs(x)
            _tmp = tl.where(_tmp > _x, _tmp, _x)
        mid = tl.max(_tmp)

    tl.atomic_max(Out, mid.to(tl.float32))


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=float("inf")).to(tl.float32)
            _min = tl.minimum(tl.abs(a), _min)

        min = tl.min(_min, axis=1)
        out = min[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    },
)
@triton.jit
def min_norm_kernel_1(
    X, Out, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=float("inf")).to(tl.float32)
        mid = tl.min(tl.abs(x))
    else:
        _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=float("inf")).to(tl.float32)
            _x = tl.abs(x)
            _tmp = tl.where(_tmp < _x, _tmp, _x)
        mid = tl.min(_tmp)

    tl.atomic_min(Out, mid.to(tl.float32))


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0).to(tl.float32)
            _sum += tl.where(a != 0, 1, 0)
        sum = tl.sum(_sum, axis=1)
        out = sum[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
    reset_to_zero=["Out"],
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    },
)
@triton.jit
def l0_norm_kernel_1(
    X, Out, M, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        mid = tl.sum((x != 0).to(tl.float32))
    else:
        _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
            _tmp = _tmp + (x != 0).to(tl.float32)
        mid = tl.sum(_tmp)

    tl.atomic_add(Out, mid.to(tl.float32))


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)

    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _sum += tl.extra.mlu.libdevice.pow(tl.abs(a), ord)
        sum = tl.sum(_sum, axis=1)
        out = tl.extra.mlu.libdevice.pow(sum, 1 / ord)[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by={"early_config_prune": prune_reduce_config},
    reset_to_zero=["Out"],
)
@triton.heuristics(
    values={
        "ONE_TILE_PER_CTA": lambda args: args["M"]
        <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM
    },
)
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_1(
    X, Out, M, ord, BLOCK_SIZE: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    mid = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        mid = tl.sum(pow(tl.abs(x), ord))
    else:
        _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
            _tmp = _tmp + pow(tl.abs(x), ord)
        mid = tl.sum(_tmp)

    tl.atomic_add(Out, mid.to(tl.float32))


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(
    Out,
    ord,
):
    out = tl.load(Out)
    out = pow(out, 1 / ord)
    tl.store(Out, out)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_CAMBRICON VECTOR NORM")
    if dtype is not None:
        dtype = torch.dtype(dtype)
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    with torch_device_fn.device(x.device):
        if (not dim) or len(dim) == x.ndim:
            dim = list(range(x.ndim))
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()

            grid = lambda meta: (
                min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
            )
            out = torch.zeros(shape, dtype=torch.float, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[grid](x, out, M)
                l2_norm_kernel_2[(1,)](out)
            elif ord == float("inf"):
                max_norm_kernel_1[grid](x, out, M)
            elif ord == -float("inf"):
                out = torch.full(
                    shape,
                    fill_value=torch.finfo(torch.float32).max,
                    dtype=torch.float,
                    device=x.device,
                )
                min_norm_kernel_1[grid](x, out, M)
            elif ord == 0:
                l0_norm_kernel_1[grid](x, out, M)
            else:
                l1_norm_kernel_1[grid](x, out, M, ord)
                l1_norm_kernel_2[(1,)](
                    out,
                    ord,
                )
            out = out.to(dtype)
        else:
            shape = list(x.shape)
            dim = [d % x.ndim for d in dim]
            x = dim_compress(x, dim)
            N = 1
            for i in dim:
                N *= shape[i]
                shape[i] = 1
            M = x.numel() // N
            out = torch.empty(shape, dtype=dtype, device=x.device)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            if ord == 2:
                l2_norm_kernel[grid](x, out, M, N)
            elif ord == float("inf"):
                max_norm_kernel[grid](x, out, M, N)
            elif ord == -float("inf"):
                min_norm_kernel[grid](x, out, M, N)
            elif ord == 0:
                l0_norm_kernel[grid](x, out, M, N)
            else:
                v_norm_kernel[grid](x, out, M, N, ord)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
