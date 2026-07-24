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

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim

from ..utils import TOTAL_CORE_NUM, cfggen_reduce_op, prune_reduce_config

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow


# Use a narrow-lane accumulation path for moderate explicit all-dim fp32
# reductions. Other paths keep the parallel reduction to avoid a serial-loop
# performance cliff.
CPU_ORDER_L2_MAX_EXACT_ELEMENTS = 2 * 1024 * 1024
CPU_ORDER_L2_LANES = 8


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

        inp_dtype = X.type.element_ty
        if inp_dtype == tl.float64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        sum = tl.zeros([BLOCK_M, 1], dtype=acc_dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(acc_dtype)
            sum += tl.sum(a * a, axis=1)[:, None]

        out = tl.sqrt(sum)
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

    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    mid = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
        mid = tl.sum(x * x)
    else:
        _tmp = tl.zeros([BLOCK_SIZE], acc_dtype)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
            _tmp = _tmp + x * x
        mid = tl.sum(_tmp)

    tl.atomic_add(Out, mid.to(acc_dtype))


@libentry()
@triton.jit
def l2_norm_kernel_2(Out):
    out = tl.sqrt(tl.load(Out))
    tl.store(Out, out)


@libentry()
@triton.jit
def l2_norm_all_kernel(X, Out, M, BLOCK_N: tl.constexpr):
    lanes = tl.arange(0, BLOCK_N)
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    sum = tl.zeros([BLOCK_N], dtype=acc_dtype)
    for off in range(0, M, BLOCK_N):
        offsets = off + lanes
        mask = offsets < M
        x = tl.load(X + offsets, mask=mask, other=0.0).to(acc_dtype)
        sum += x * x
    tl.store(Out, tl.sqrt(tl.sum(sum)))


@libentry()
@triton.jit
def l2_norm_kernel_seq(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < M
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    sum = tl.zeros([BLOCK_M, 1], dtype=acc_dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        x = tl.load(X + pid * N + cols, mask=mask, other=0.0).to(acc_dtype)
        sum += tl.sum(x * x, axis=1)[:, None]
    tl.store(Out + pid, tl.sqrt(sum), row_mask)


@libentry()
@triton.jit
def l2_norm_cpu_order_mid_kernel(X, Mid, M, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    sum = 0.0
    for off in range(0, M, BLOCK_N):
        idx = off + pid
        mask = idx < M
        x = tl.load(X + idx, mask=mask, other=0.0).to(acc_dtype)
        sum += x * x
    tl.store(Mid + pid, sum)


@libentry()
@triton.jit
def l2_norm_cpu_order_finalize(Mid, Out, BLOCK_N: tl.constexpr):
    offsets = tl.arange(0, BLOCK_N)
    inp_dtype = Mid.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    mid = tl.load(Mid + offsets).to(acc_dtype)
    tl.store(Out, tl.sqrt(tl.sum(mid)))


@libentry()
@triton.jit
def l2_norm_mid_kernel(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    x = tl.load(X + offsets, mask=mask, other=0.0).to(acc_dtype)
    tl.store(Mid + pid, tl.sum(x * x))


@libentry()
@triton.jit
def l2_norm_mid_finalize(Mid, Out, MID_SIZE: tl.constexpr, BLOCK_MID: tl.constexpr):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < MID_SIZE
    inp_dtype = Mid.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    mid = tl.load(Mid + offsets, mask=mask, other=0.0).to(acc_dtype)
    tl.store(Out, tl.sqrt(tl.sum(mid)))


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

        inp_dtype = X.type.element_ty
        if inp_dtype == tl.float64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(acc_dtype)
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

    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    mid = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
        mid = tl.max(tl.abs(x))
    else:
        _tmp = tl.full([BLOCK_SIZE], value=-float("inf"), dtype=acc_dtype)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
            _x = tl.abs(x)
            _tmp = tl.where(_tmp > _x, _tmp, _x)
        mid = tl.max(_tmp)

    tl.atomic_max(Out, mid.to(acc_dtype))


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

        inp_dtype = X.type.element_ty
        if inp_dtype == tl.float64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=acc_dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(X_ptr + cols, mask, other=float("inf")).to(acc_dtype)
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

    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=float("inf")).to(acc_dtype)
        mid = tl.min(tl.abs(x))
    else:
        _tmp = tl.full([BLOCK_SIZE], value=float("inf"), dtype=acc_dtype)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=float("inf")).to(acc_dtype)
            _x = tl.abs(x)
            _tmp = tl.where(_tmp < _x, _tmp, _x)
        mid = tl.min(_tmp)

    tl.atomic_min(Out, mid.to(acc_dtype))


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

        inp_dtype = X.type.element_ty
        if inp_dtype == tl.float64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(X_ptr + cols, mask, other=0).to(acc_dtype)
            _sum += (a != 0).to(acc_dtype)
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

    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
        mid = tl.sum((x != 0).to(acc_dtype))
    else:
        _tmp = tl.zeros([BLOCK_SIZE], acc_dtype)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
            _tmp = _tmp + (x != 0).to(acc_dtype)
        mid = tl.sum(_tmp)

    tl.atomic_add(Out, mid.to(acc_dtype))


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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

        inp_dtype = X.type.element_ty
        if inp_dtype == tl.float64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        sum = tl.zeros([BLOCK_M, 1], dtype=acc_dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(acc_dtype)
            sum += tl.sum(tl.extra.mlu.libdevice.pow(tl.abs(a), ord), axis=1)[:, None]
        out = tl.extra.mlu.libdevice.pow(sum, 1 / ord)
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

    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    mid = 0.0
    if ONE_TILE_PER_CTA:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
        mid = tl.sum(pow(tl.abs(x), ord))
    else:
        _tmp = tl.zeros([BLOCK_SIZE], acc_dtype)
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        for block_start_offset in range(block_start, M, step):
            offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < M
            x = tl.load(X + offsets, mask, other=0.0).to(acc_dtype)
            _tmp = _tmp + pow(tl.abs(x), ord)
        mid = tl.sum(_tmp)

    tl.atomic_add(Out, mid.to(acc_dtype))


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(
    Out,
    ord,
):
    out = tl.load(Out)
    out = pow(out, 1 / ord)
    tl.store(Out, out)


@libentry()
@triton.jit
def l1_norm_all_kernel(
    X, Out, M, SCALE: tl.constexpr, QUANT: tl.constexpr, BLOCK_N: tl.constexpr
):
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    sum = 0.0
    for off in range(0, M, BLOCK_N):
        offsets = off + tl.arange(0, BLOCK_N)
        mask = offsets < M
        x = tl.load(X + offsets, mask=mask, other=0.0).to(acc_dtype)
        sum += tl.sum(tl.abs(x))
    out = sum * SCALE
    if QUANT:
        out = ((out + 0.5).to(tl.int32)).to(tl.float32)
    tl.store(Out, out)


@libentry()
@triton.jit
def l1_norm_mid_kernel(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    x = tl.load(X + offsets, mask=mask, other=0.0).to(acc_dtype)
    tl.store(Mid + pid, tl.sum(tl.abs(x)))


@libentry()
@triton.jit
def l1_norm_mid_finalize(
    Mid, Out, SCALE: tl.constexpr, MID_SIZE: tl.constexpr, BLOCK_MID: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < MID_SIZE
    inp_dtype = Mid.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    mid = tl.load(Mid + offsets, mask=mask, other=0.0).to(acc_dtype)
    tl.store(Out, tl.sum(mid) * SCALE)


@libentry()
@triton.jit
def l1_norm_kernel_seq(
    X, Out, M, N, SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        row_mask = pid < M
        inp_dtype = X.type.element_ty
        if inp_dtype == tl.float64:
            acc_dtype = tl.float64
        else:
            acc_dtype = tl.float32
        sum = tl.zeros([BLOCK_M, 1], dtype=acc_dtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = row_mask & (cols < N)
            x = tl.load(X + pid * N + cols, mask=mask, other=0.0).to(acc_dtype)
            sum += tl.sum(tl.abs(x), axis=1)[:, None]
        tl.store(Out + pid, sum * SCALE, row_mask)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_CAMBRICON VECTOR_NORM")
    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            dtype = torch.float32
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    with torch_device_fn.device(x.device):
        explicit_full_dim = dim is not None and len(dim) == x.ndim
        if (not dim) or len(dim) == x.ndim:
            dim = list(range(x.ndim))
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()
            BLOCK_N = 1024
            grid = lambda meta: (
                min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
            )
            out_dtype = torch.float64 if dtype == torch.float64 else torch.float
            out = torch.zeros(shape, dtype=out_dtype, device=x.device)
            if ord == 2:
                if (
                    explicit_full_dim
                    and dtype == torch.float32
                    and M <= CPU_ORDER_L2_MAX_EXACT_ELEMENTS
                ):
                    mid = torch.empty(
                        [CPU_ORDER_L2_LANES], dtype=out_dtype, device=x.device
                    )
                    l2_norm_cpu_order_mid_kernel[(CPU_ORDER_L2_LANES,)](
                        x, mid, M, BLOCK_N=CPU_ORDER_L2_LANES
                    )
                    l2_norm_cpu_order_finalize[(1,)](
                        mid, out, BLOCK_N=CPU_ORDER_L2_LANES
                    )
                else:
                    l2_norm_kernel_1[grid](x, out, M)
                    l2_norm_kernel_2[(1,)](out)
            elif ord == float("inf"):
                out = torch.full(
                    shape,
                    fill_value=-float("inf"),
                    dtype=out_dtype,
                    device=x.device,
                )
                max_norm_kernel_1[grid](x, out, M)
            elif ord == -float("inf"):
                out = torch.full(
                    shape,
                    fill_value=torch.finfo(out_dtype).max,
                    dtype=out_dtype,
                    device=x.device,
                )
                min_norm_kernel_1[grid](x, out, M)
            elif ord == 0:
                l0_norm_kernel_1[grid](x, out, M)
            else:
                if ord == 1:
                    l1_norm_all_kernel[(1,)](x, out, M, 1.0, False, BLOCK_N)
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
            BLOCK_N = 1024
            if ord == 2:
                if N >= 1024:
                    l2_norm_kernel_seq[(triton.cdiv(M, 1),)](
                        x, out, M, N, BLOCK_M=1, BLOCK_N=1
                    )
                else:
                    l2_norm_kernel[grid](x, out, M, N)
            elif ord == float("inf"):
                max_norm_kernel[grid](x, out, M, N)
            elif ord == -float("inf"):
                min_norm_kernel[grid](x, out, M, N)
            elif ord == 0:
                l0_norm_kernel[grid](x, out, M, N)
            else:
                if ord == 1:
                    if N >= 1024:
                        l1_norm_kernel_seq[(triton.cdiv(M, 1),)](
                            x, out, M, N, 1.0, BLOCK_M=1, BLOCK_N=1
                        )
                    else:
                        v_norm_kernel[grid](x, out, M, N, ord)
                else:
                    v_norm_kernel[grid](x, out, M, N, ord)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
