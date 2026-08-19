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

import functools
import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.device_info import get_l2_cache_size, get_sm_count

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_metax_expand.yaml")
)


def _prune_mm_dense_configs(configs, named_args, transposed_b=False, **kwargs):
    configs = list(configs)
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    pruned_configs = []

    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        pipeline = config.kwargs["pipeline"]
        warps = config.num_warps
        stages = config.num_stages

        if block_k == 128:
            if block_m > 128 or block_n > 128 or pipeline != "cpasync":
                continue

        if (
            pipeline == "cpasync"
            and block_k >= 64
            and (block_m == 128 or block_n == 128)
        ):
            continue

        if transposed_b and K % block_k != 0 and block_m == 256 and block_n == 256:
            continue

        if (block_m == 128 or block_n == 128) and pipeline == "basic":
            stage_bytes = (block_m + block_n) * block_k * 2 * stages
            if stage_bytes > 64 * 1024:
                continue

        if M >= 1024 and N >= 128:
            if block_m not in (64, 128, 256) or block_n not in (64, 128, 256):
                continue
            if warps not in (4, 8):
                continue
        else:
            if block_m == 128 or warps == 8:
                continue
            if N <= 64 and (block_m > 64 or block_n > 64):
                continue
            if warps == 2 and not (block_m <= 32 and block_n <= 64):
                continue

        pruned_configs.append(config)

    return pruned_configs or configs


_prune_mm_dense_configs_nt = functools.partial(
    _prune_mm_dense_configs, transposed_b=True
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.heuristics(
    {
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = ext.program_id(0)
    else:
        pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        if EVEN_M:
            ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M).to(tl.int64)
        else:
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(
                tl.int64
            )
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if EVEN_M:
            ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        else:
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        if EVEN_N:
            rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N).to(tl.int64)
        else:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(
                tl.int64
            )
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if EVEN_N:
            rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        else:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc = tl.dot(
            a,
            b,
            acc,
            out_dtype=dot_out_dtype,
            allow_tf32=False,
        )
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if EVEN_M and EVEN_N:
        tl.store(C, acc)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_nn"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nn",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def mm_kernel_nn(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rbn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    result = acc.to(C.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_nt"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs_nt},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nt",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def mm_kernel_nt(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] + rbn[None, :] * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    result = acc.to(C.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


def _prune_gemv_configs(configs, named_args, **kwargs):
    configs = list(configs)
    pruned_configs = [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] == 256 and config.num_warps in (4, 8)
    ]
    return pruned_configs or configs


@libentry()
@libtuner(
    configs=[triton.Config({"BLOCK_M": 32, "BLOCK_K": 256})],
    key=["M", "K", "stride_am", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def gemv_kernel(
    A,
    B,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K
        a = tl.load(
            A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(B + offsets_k * stride_bk, mask=mask_k, other=0.0)
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    tl.store(C + offsets_m * stride_cm, acc.to(C.dtype.element_ty), mask=mask_m)


def gemv_mm(a, b, c, M, K):
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(a.device):
        gemv_kernel[grid](
            a,
            b,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            c.stride(0),
        )
    return c


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gemv_k_parallel"),
    key=["M", "K", "stride_am", "stride_bk"],
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv_k_parallel",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def gemv_kernel_k_parallel_partial(
    A,
    B,
    P,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < M

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_iter in range(k_start, k_end):
        offsets_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K
        a = tl.load(
            A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(B + offsets_k * stride_bk, mask=mask_k, other=0.0)
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    tl.store(P + pid_k * M + offsets_m, acc, mask=mask_m)


@libentry()
@triton.jit
def gemv_kernel_k_parallel_reduce(
    P,
    C,
    M,
    stride_cm,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_m = offsets_m < M
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for split_id in range(0, SPLIT_K):
        acc += tl.load(P + split_id * M + offsets_m, mask=mask_m, other=0.0)

    tl.store(
        C + offsets_m * stride_cm,
        acc.to(C.dtype.element_ty),
        mask=mask_m,
    )


_GEMV_BLOCK_M = 32
_GEMV_K_PARALLEL_BLOCK_K = 128
_GEMV_K_PARALLEL_MAX_SPLITS = 16
_GEMV_K_PARALLEL_TARGET_SM_NUMERATOR = 1
_GEMV_K_PARALLEL_TARGET_SM_DENOMINATOR = 2


def _floor_power_of_two(value):
    if value < 1:
        return 0
    return 1 << (int(value).bit_length() - 1)


def _ceil_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def _gemv_k_parallel_split_k(M, K):
    base_programs = max(1, triton.cdiv(M, _GEMV_BLOCK_M))
    target_programs = max(
        1,
        get_sm_count()
        * _GEMV_K_PARALLEL_TARGET_SM_NUMERATOR
        // _GEMV_K_PARALLEL_TARGET_SM_DENOMINATOR,
    )
    occupancy_splits = target_programs // base_programs
    k_splits = max(1, triton.cdiv(K, _GEMV_K_PARALLEL_BLOCK_K))

    split_k = min(
        _GEMV_K_PARALLEL_MAX_SPLITS,
        k_splits,
        _floor_power_of_two(occupancy_splits),
    )
    return max(1, split_k)


def gemv_mm_k_parallel(a, b, c, M, K):
    split_k = _gemv_k_parallel_split_k(M, K)
    partials = torch.empty((split_k, M), device=a.device, dtype=torch.float32)
    partial_grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        split_k,
    )
    reduce_block_size = 256
    reduce_grid = (triton.cdiv(M, reduce_block_size),)

    with torch_device_fn.device(a.device):
        gemv_kernel_k_parallel_partial[partial_grid](
            a,
            b,
            partials,
            M,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            SPLIT_K=split_k,
        )
        gemv_kernel_k_parallel_reduce[reduce_grid](
            partials,
            c,
            M,
            c.stride(0),
            SPLIT_K=split_k,
            BLOCK_SIZE=reduce_block_size,
            num_warps=4,
        )
    return c


def _reset_splitk_output(args, reset_only=False):
    c = args["C"] if isinstance(args, dict) else args[2]
    c.zero_()


def _splitk_b_is_transposed(named_args):
    return named_args.get("stride_bk") == 1


def _splitk_nt_config_aborts(config, transposed_b):
    if not transposed_b:
        return False

    pipeline = config.kwargs.get("pipeline")
    block_n = config.kwargs["BLOCK_N"]
    block_k = config.kwargs["BLOCK_K"]

    if pipeline == "cpasync" and block_k < block_n:
        return True

    return False


def _prune_mm_splitk_two_step_configs(configs, named_args, **kwargs):
    configs = list(configs)
    N = named_args["N"]
    transposed_b = _splitk_b_is_transposed(named_args)
    block_ns = (16, 32) if N == 16 else (64, 128)
    pruned_configs = [
        config
        for config in configs
        if not _splitk_nt_config_aborts(config, transposed_b)
        if config.kwargs["BLOCK_N"] in block_ns
    ]
    return pruned_configs or configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_splitk"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    pre_hook=_reset_splitk_output,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_splitk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def mm_kernel_splitk(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end):
        offsets_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    mask = (offsets_m < M)[:, None] & (offsets_n < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)


def splitk_mm(a, b, c, M, N, K):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_kernel_splitk[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
    return c


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_splitk_two_step"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    prune_configs_by={"early_config_prune": _prune_mm_splitk_two_step_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_splitk_two_step",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def mm_kernel_splitk_partial(
    A,
    B,
    P,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end):
        offsets_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    p_ptrs = P + pid_k * M * N + offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m < M)[:, None] & (offsets_n < N)[None, :]
    tl.store(p_ptrs, acc, mask=mask)


@libentry()
@triton.jit
def mm_kernel_splitk_reduce(
    P,
    C,
    M,
    N,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for split_id in range(0, SPLIT_K):
        acc += tl.load(P + split_id * M * N + offsets, mask=mask, other=0.0)

    offsets_m = offsets // N
    offsets_n = offsets % N
    c_ptrs = C + offsets_m * stride_cm + offsets_n * stride_cn
    tl.store(c_ptrs, acc, mask=mask)


_TWO_STEP_MAX_SPLITS = 16
_TWO_STEP_MIN_PROFITABLE_SPLITS = 4
_TWO_STEP_MIN_K_ITERS_PER_SPLIT = 2
_TWO_STEP_TARGET_SM_NUMERATOR = 3
_TWO_STEP_TARGET_SM_DENOMINATOR = 4
_TWO_STEP_WORKSPACE_L2_NUMERATOR = 1
_TWO_STEP_WORKSPACE_L2_DENOMINATOR = 4


@functools.lru_cache(maxsize=1)
def _two_step_tile_candidates():
    candidates = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
        )
        for config in runtime.get_tuned_config("mm_splitk_two_step")
    }

    expand_config = runtime.get_expand_config(
        "mm_splitk_two_step",
        yaml_path=EXPAND_CONFIG_FILENAME,
    )
    if expand_config != -1:
        ranges = expand_config["ranges"]
        candidates.update(
            (block_m, block_n, block_k)
            for block_m in ranges.get("BLOCK_M", ())
            for block_n in ranges.get("BLOCK_N", ())
            for block_k in ranges.get("BLOCK_K", ())
        )

    return tuple(sorted(candidates))


def _two_step_reference_tile(N):
    allowed_block_ns = (16, 32) if N == 16 else (64, 128)
    candidates = tuple(
        candidate
        for candidate in _two_step_tile_candidates()
        if candidate[1] in allowed_block_ns
    )
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda tile: (tile[0] * tile[1], tile[0], tile[1], tile[2]),
    )


def _two_step_split_k(M, N, K):
    reference_tile = _two_step_reference_tile(N)
    if reference_tile is None:
        return 1, 0

    block_m, block_n, block_k = reference_tile
    output_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    target_programs = max(
        1,
        get_sm_count()
        * _TWO_STEP_TARGET_SM_NUMERATOR
        // _TWO_STEP_TARGET_SM_DENOMINATOR,
    )

    required_splits = triton.cdiv(target_programs, output_tiles)
    occupancy_split_k = _ceil_power_of_two(required_splits)

    k_iters = triton.cdiv(K, block_k)
    max_k_split = _floor_power_of_two(k_iters // _TWO_STEP_MIN_K_ITERS_PER_SPLIT)
    split_k = min(
        occupancy_split_k,
        max_k_split,
        _TWO_STEP_MAX_SPLITS,
    )
    return max(1, split_k), output_tiles


def _launch_splitk_mm_two_step(a, b, c, M, N, K, split_k):
    partials = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    partial_grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
    reduce_block_size = 256
    reduce_grid = (triton.cdiv(M * N, reduce_block_size),)

    with torch_device_fn.device(a.device):
        mm_kernel_splitk_partial[partial_grid](
            a,
            b,
            partials,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            SPLIT_K=split_k,
        )
        mm_kernel_splitk_reduce[reduce_grid](
            partials,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            SPLIT_K=split_k,
            BLOCK_SIZE=reduce_block_size,
            num_warps=4,
        )
    return c


def splitk_mm_two_step(a, b, c, M, N, K, split_k=None):
    if split_k is None:
        split_k, _ = _two_step_split_k(M, N, K)
    return _launch_splitk_mm_two_step(a, b, c, M, N, K, split_k)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def general_mm(a, b, c, M, N, K):
    dot_out_dtype = tl.float32
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
        )
    return c


def general_mm_nn(a, b, c, M, N, K):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_nn[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            GROUP_M=8,
        )
    return c


def general_mm_nt(a, b, c, M, N, K):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_nt[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            GROUP_M=8,
        )
    return c


@functools.lru_cache(maxsize=1)
def _dense_output_tile_candidates():
    tile_candidates = set()
    for config_name in ("mm", "mm_nn", "mm_nt"):
        tile_candidates.update(
            (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"])
            for config in runtime.get_tuned_config(config_name)
            if "BLOCK_M" in config.kwargs and "BLOCK_N" in config.kwargs
        )

        expand_config = runtime.get_expand_config(
            config_name,
            yaml_path=EXPAND_CONFIG_FILENAME,
        )
        if expand_config != -1:
            ranges = expand_config["ranges"]
            block_ms = ranges.get("BLOCK_M", ())
            block_ns = ranges.get("BLOCK_N", ())
            tile_candidates.update(
                (block_m, block_n) for block_m in block_ms for block_n in block_ns
            )

    return tuple(sorted(tile_candidates))


def _max_general_mm_programs(M, N):
    tile_candidates = _dense_output_tile_candidates()
    if not tile_candidates:
        return get_sm_count()

    return max(
        triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        for block_m, block_n in tile_candidates
    )


def splitk_mm_scenario(M, N, K):
    if K < 2048:
        return False

    max_general_programs = _max_general_mm_programs(M, N)
    parallelism_budget = max(1, get_sm_count() * 3 // 4)

    atomic_working_set = M * N * 4
    atomic_l2_budget = max(1, get_l2_cache_size() // 32)

    return (
        max_general_programs <= parallelism_budget
        and atomic_working_set <= atomic_l2_budget
    )


def _splitk_mm_two_step_scenario(M, N, K):
    if K < 2048:
        return False

    if N != 256:
        return False

    split_k, output_tiles = _two_step_split_k(M, N, K)
    target_programs = max(
        1,
        get_sm_count()
        * _TWO_STEP_TARGET_SM_NUMERATOR
        // _TWO_STEP_TARGET_SM_DENOMINATOR,
    )

    if output_tiles * _TWO_STEP_MIN_PROFITABLE_SPLITS > target_programs:
        return False
    if split_k < _TWO_STEP_MIN_PROFITABLE_SPLITS:
        return False

    workspace_bytes = split_k * M * N * torch.float32.itemsize
    workspace_budget = max(
        1,
        get_l2_cache_size()
        * _TWO_STEP_WORKSPACE_L2_NUMERATOR
        // _TWO_STEP_WORKSPACE_L2_DENOMINATOR,
    )
    if workspace_bytes > workspace_budget:
        return False

    return True


def _gemv_k_parallel_scenario(M, K):
    return K >= 2048 and _gemv_k_parallel_split_k(M, K) > 1


def nn_mm_scenario(a, b, c, M, N, K):
    return (
        M > 0
        and N > 0
        and K > 0
        and a.dtype in _ordered_datatypes
        and b.dtype in _ordered_datatypes
        and c.dtype in _ordered_datatypes
        and a.stride(0) == K
        and a.stride(1) == 1
        and b.stride(0) == N
        and b.stride(1) == 1
        and c.stride(0) == N
        and c.stride(1) == 1
        and M * K < 2**31
        and K * N < 2**31
        and M * N < 2**31
    )


def nt_mm_scenario(a, b, c, M, N, K):
    return (
        M > 0
        and N > 0
        and K > 0
        and a.dtype in _ordered_datatypes
        and b.dtype in _ordered_datatypes
        and c.dtype in _ordered_datatypes
        and a.stride(0) == K
        and a.stride(1) == 1
        and b.stride(0) == 1
        and b.stride(1) == K
        and c.stride(0) == N
        and c.stride(1) == 1
        and M * K < 2**31
        and K * N < 2**31
        and M * N < 2**31
    )


def _select_two_step_split_k(M, N, K):
    if _splitk_mm_two_step_scenario(M, N, K):
        split_k, _ = _two_step_split_k(M, N, K)
        return split_k
    return None


def mm(a, b):
    logger.debug("GEMS_METAX MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if N == 1:
        if _gemv_k_parallel_scenario(M, K):
            return gemv_mm_k_parallel(a, b, c, M, K)
        return gemv_mm(a, b, c, M, K)
    two_step_split_k = _select_two_step_split_k(M, N, K)
    if two_step_split_k is not None:
        return splitk_mm_two_step(a, b, c, M, N, K, two_step_split_k)
    if splitk_mm_scenario(M, N, K):
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
    if nn_mm_scenario(a, b, c, M, N, K):
        return general_mm_nn(a, b, c, M, N, K)
    if nt_mm_scenario(a, b, c, M, N, K):
        return general_mm_nt(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS_METAX MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    if N == 1:
        if _gemv_k_parallel_scenario(M, K):
            return gemv_mm_k_parallel(a, b, c, M, K)
        return gemv_mm(a, b, c, M, K)
    two_step_split_k = _select_two_step_split_k(M, N, K)
    if two_step_split_k is not None:
        return splitk_mm_two_step(a, b, out, M, N, K, two_step_split_k)
    if splitk_mm_scenario(M, N, K):
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
    if nn_mm_scenario(a, b, c, M, N, K):
        return general_mm_nn(a, b, c, M, N, K)
    if nt_mm_scenario(a, b, c, M, N, K):
        return general_mm_nt(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)
