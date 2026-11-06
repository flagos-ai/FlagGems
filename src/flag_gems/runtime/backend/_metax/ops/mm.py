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

# MetaX C550's MACA MMA lowers to m16n8k16 fragments, so neither extent of a
# dense tile can be smaller than 16.  Triton rejects such a `tl.dot` in the
# frontend, and the MetaX dialect additionally trips
# `MACAMmaEncodingAttr::composeSharedLayoutForOperand`:
#
#   BLOCK_M < 16 -> Assertion `shape_judge && "tn and tk not meet conditon"` failed
#   BLOCK_N < 16 -> Assertion `shape_judge && "tm and tk not meet conditon"` failed
#
# Both surface as an opaque `RuntimeError: PassManager::run failed`, which
# aborts the whole operator (and stops vLLM from starting) when such a tile is
# the one that gets compiled.
_DENSE_MM_MIN_BLOCK_M = 16
_DENSE_MM_MIN_BLOCK_N = 16

# The extents that carry the floor above, in the order they are checked.
_MMA_EXTENT_FLOORS = (
    ("BLOCK_M", _DENSE_MM_MIN_BLOCK_M),
    ("BLOCK_N", _DENSE_MM_MIN_BLOCK_N),
)

# Timing reported for a tile that must never be compiled.  ``LibTuner`` treats a
# non-finite measurement as "this candidate lost", which is exactly how a tile
# that cannot compile should be ranked.
_MACA_BENCH_FAILURE = float("inf")

# Last-resort dense tiles for config spaces that contain no legal candidate at
# all (for example an externally overridden expanded YAML).  Every entry has a
# BLOCK_M at or above the MACA minimum and a shape that survives
# `_prune_mm_dense_configs` on both the small and the large branch, so the
# operator always has a tile it can actually compile.
_DENSE_MM_FALLBACK_CONFIGS = (
    triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "pipeline": "basic",
            "scenario": "",
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "pipeline": "basic",
            "scenario": "",
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "pipeline": "basic",
            "scenario": "",
        },
        num_warps=4,
        num_stages=4,
    ),
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

        # A dense tile below the MACA MMA M extent can never compile (see
        # _DENSE_MM_MIN_BLOCK_M).  Rejecting it here is what keeps one
        # unsupported candidate from aborting the whole autotune.
        if block_m < _DENSE_MM_MIN_BLOCK_M:
            continue

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

    if pruned_configs:
        return pruned_configs

    # Pruning left nothing usable.  Returning the unpruned list would hand back
    # exactly the illegal tiles that were just rejected and, for an overridden
    # config space, guarantee a "PassManager::run failed" abort.  Tiles that
    # only violate the shape heuristics stay legal for MMA, so prefer those and
    # only then fall back to the built-in safe set.
    legal_configs = [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] >= _DENSE_MM_MIN_BLOCK_M
    ]
    return legal_configs or list(_DENSE_MM_FALLBACK_CONFIGS)


_prune_mm_dense_configs_nt = functools.partial(
    _prune_mm_dense_configs, transposed_b=True
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    policy="flagtune",
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_general",
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
    policy="flagtune",
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nn",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_nn",
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
    policy="flagtune",
    prune_configs_by={"early_config_prune": _prune_mm_dense_configs_nt},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_nt",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_nt",
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
    return pruned_configs or list(configs)


@libentry()
@libtuner(
    configs=[triton.Config({"BLOCK_M": 32, "BLOCK_K": 256})],
    key=["M", "K", "stride_am", "stride_bk"],
    policy="flagtune",
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_gemv",
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
    policy="flagtune",
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv_k_parallel",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_gemv_k_parallel_partial",
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
        # Same MACA MMA floor as the dense tiles: a sub-16 BLOCK_M can never
        # compile on this backend.
        if config.kwargs["BLOCK_M"] >= _DENSE_MM_MIN_BLOCK_M
        if not _splitk_nt_config_aborts(config, transposed_b)
        if config.kwargs["BLOCK_N"] in block_ns
    ]
    legal_configs = [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] >= _DENSE_MM_MIN_BLOCK_M
    ]
    return pruned_configs or legal_configs or list(configs)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_splitk"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    policy="flagtune",
    pre_hook=_reset_splitk_output,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_splitk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_splitk",
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
    policy="flagtune",
    prune_configs_by={"early_config_prune": _prune_mm_splitk_two_step_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_splitk_two_step",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_op_id="flaggems/mm",
    flagtune_variant="metax_splitk_two_step_partial",
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
def mm_kernel_small_n_partial(
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
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nk = tl.program_id(1)
    pid_n = pid_nk % N
    pid_k = pid_nk // N

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
        b = tl.load(
            B + offsets_k * stride_bk + pid_n * stride_bn,
            mask=mask_k,
            other=0.0,
        )
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    p_offsets = pid_k * M * N + offsets_m * N + pid_n
    tl.store(P + p_offsets, acc, mask=mask_m)


@libentry()
@triton.jit
def mm_kernel_small_n_dot_partial(
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
    for k_iter in range(k_start, k_end):
        offsets_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
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

    p_offsets = pid_k * M * N + offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m < M)[:, None] & (offsets_n < N)[None, :]
    tl.store(P + p_offsets, acc, mask=mask)


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
_GENERIC_TWO_STEP_MAX_SPLITS = 32
_GENERIC_TWO_STEP_WORKSPACE_L2_DENOMINATOR = 8
_GENERIC_SPLITK_DENSE_OUTPUT_L2_DENOMINATOR = 64
_SMALL_N_DOT_BLOCK_N = 16
_SMALL_N_DOT_BLOCK_K = 64
_SMALL_N_MAX_SPLITS = 32
_SMALL_N_ELEMENTWISE_MAX_M = 8
_SMALL_N_ELEMENTWISE_MAX_BLOCK_K = 1024
_SMALL_N_ELEMENTWISE_TILE_ELEMENTS = 4096


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


def _generic_two_step_split_k(M, N, K):
    reference_tile = _two_step_reference_tile(N)
    if reference_tile is None:
        return 1

    block_m, _, block_k = reference_tile
    k_iters = triton.cdiv(K, block_k)
    max_k_split = _floor_power_of_two(k_iters // _TWO_STEP_MIN_K_ITERS_PER_SPLIT)
    split_k = min(max_k_split, _TWO_STEP_MAX_SPLITS)

    extended_split_k = min(max_k_split, _GENERIC_TWO_STEP_MAX_SPLITS)
    extended_workspace_bytes = extended_split_k * M * N * torch.float32.itemsize
    extended_workspace_budget = max(
        1,
        get_l2_cache_size() // _GENERIC_TWO_STEP_WORKSPACE_L2_DENOMINATOR,
    )
    if (
        extended_split_k > split_k
        and M <= block_m
        and extended_workspace_bytes <= extended_workspace_budget
    ):
        split_k = extended_split_k

    return max(1, split_k)


def _small_n_mm_config(M, N, K):
    if M <= _SMALL_N_ELEMENTWISE_MAX_M:
        block_m = _ceil_power_of_two(M)
        block_k = min(
            _SMALL_N_ELEMENTWISE_MAX_BLOCK_K,
            _SMALL_N_ELEMENTWISE_TILE_ELEMENTS // block_m,
        )
        base_programs = triton.cdiv(M, block_m) * N
        occupancy_split_k = _floor_power_of_two(max(1, get_sm_count() // base_programs))
        max_k_split = _floor_power_of_two(triton.cdiv(K, block_k))
        split_k = min(
            occupancy_split_k,
            max_k_split,
            _SMALL_N_MAX_SPLITS,
        )
        return "elementwise", max(1, split_k), block_m, block_k, 1

    block_m = 32 if M <= 32 else 64
    num_warps = 2 if block_m == 32 else 4
    output_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, _SMALL_N_DOT_BLOCK_N)
    required_splits = triton.cdiv(get_sm_count(), output_tiles)
    occupancy_split_k = _ceil_power_of_two(required_splits)
    k_iters = triton.cdiv(K, _SMALL_N_DOT_BLOCK_K)
    max_k_split = _floor_power_of_two(k_iters // _TWO_STEP_MIN_K_ITERS_PER_SPLIT)
    split_k = min(
        occupancy_split_k,
        max_k_split,
        _SMALL_N_MAX_SPLITS,
    )
    return "dot", max(1, split_k), block_m, _SMALL_N_DOT_BLOCK_K, num_warps


def _launch_splitk_reduce(partials, c, M, N, split_k):
    n_elements = M * N
    small_output = n_elements <= 256
    block_size = 64 if small_output else 256
    num_warps = 1 if small_output else 4
    mm_kernel_splitk_reduce[(triton.cdiv(n_elements, block_size),)](
        partials,
        c,
        M,
        N,
        c.stride(0),
        c.stride(1),
        SPLIT_K=split_k,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def _launch_splitk_mm_two_step(a, b, c, M, N, K, split_k):
    partials = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    partial_grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
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
        _launch_splitk_reduce(partials, c, M, N, split_k)
    return c


def splitk_mm_two_step(a, b, c, M, N, K, split_k=None):
    if split_k is None:
        split_k, _ = _two_step_split_k(M, N, K)
    return _launch_splitk_mm_two_step(a, b, c, M, N, K, split_k)


def small_n_mm(a, b, c, M, N, K):
    kernel, split_k, block_m, block_k, num_warps = _small_n_mm_config(M, N, K)
    partials = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)

    with torch_device_fn.device(a.device):
        if kernel == "elementwise":
            grid = (triton.cdiv(M, block_m), N * split_k)
            mm_kernel_small_n_partial[grid](
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
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                num_warps=num_warps,
            )
        else:
            grid = (
                triton.cdiv(M, block_m) * triton.cdiv(N, _SMALL_N_DOT_BLOCK_N),
                split_k,
            )
            mm_kernel_small_n_dot_partial[grid](
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
                BLOCK_M=block_m,
                BLOCK_N=_SMALL_N_DOT_BLOCK_N,
                BLOCK_K=block_k,
                num_warps=num_warps,
            )
        _launch_splitk_reduce(partials, c, M, N, split_k)
    return c


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


def _tile_is_legal(config):
    """Return whether both MMA extents of ``config`` meet the MACA minimum."""
    for name, floor in _MMA_EXTENT_FLOORS:
        value = config.kwargs.get(name)
        if isinstance(value, int) and value < floor:
            return False
    return True


def _floor_dense_tile(config, kernel_name=None):
    """Return ``config`` with its MMA tile extents raised to the MACA minimum.

    The tile has to be corrected on the ``triton.Config`` itself, not on the
    keyword arguments of a single launch.  The tuner hands the config it
    selected to the persistent ``ConfigCache``, and ``LibEntry`` reads the tile
    back out of ``LibTuner.best_config`` to build the launch metadata it caches
    (``constexprs``/``tune_constexprs``, plus the heuristics recomputed from
    them).  Correcting the config keeps all of those views describing the same
    tile as the kernel that is actually compiled.  Correcting only one launch's
    kwargs would leave both the cached config and that metadata still
    advertising a sub-16 tile, and the next cache hit would launch a tile that
    disagrees with the compiled kernel.

    An already legal config is returned unchanged, so a legal candidate keeps
    its identity and its place in the candidate list.
    """
    changes = {}
    for name, floor in _MMA_EXTENT_FLOORS:
        value = config.kwargs.get(name)
        if isinstance(value, int) and value < floor:
            changes[name] = floor
    if not changes:
        return config
    if kernel_name is not None:
        logger.warning(
            "%s: %s below the MACA MMA minimum, using %s",
            kernel_name,
            ", ".join(f"{name}={config.kwargs[name]}" for name in changes),
            ", ".join(f"{name}={floor}" for name, floor in changes.items()),
        )
    return triton.Config(
        {**config.kwargs, **changes},
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        maxnreg=config.maxnreg,
        pre_hook=config.pre_hook,
        ir_override=config.ir_override,
    )


def _install_dense_config_guard(tuner, kernel_name):
    """Keep a sub-16 MACA MMA tile out of the config the tuner selects.

    ``LibTuner.run`` compiles, caches and launches whatever the policy returns
    without checking it against the pruned candidate list.  FlagTune's cost
    model derives the tile from the problem shape
    (``BLOCK_M = next_power_of_2(M)``, ``BLOCK_N = next_power_of_2(N)``), so for
    a decode-shaped problem it can return a tile that was never a candidate and
    therefore never met the pruner's floor.

    Replacing such a proposal with the best legal candidate the policy actually
    measured normalizes the config before it is compiled and before it is
    written to the persistent ``ConfigCache``.
    """
    policy = getattr(tuner, "policy", None)
    if policy is None or getattr(policy, "_metax_tile_guard", False):
        return

    def floored_policy(bench, configs, args, kwargs):
        candidates = list(configs)
        legal = [config for config in candidates if _tile_is_legal(config)]
        if not legal:
            # Nothing legal to substitute with; the pruner's own fallback set
            # (see _prune_mm_dense_configs) is the remaining line of defence.
            return policy(bench, candidates, args, kwargs)

        measured = {}

        def measuring_bench(config):
            if not _tile_is_legal(config):
                # The cost model builds its own candidates from the problem
                # shape, so a decode-shaped problem makes it benchmark a
                # sub-16 tile that the pruner never saw.  Compiling that tile
                # can only fail, so report it as a non-candidate instead: the
                # policy then cannot select it, and no failed compile is
                # emitted for it.
                measured[config] = float("inf")
                return [_MACA_BENCH_FAILURE] * 3
            timings = bench(config)
            try:
                measured[config] = float(timings[0])
            except (TypeError, IndexError, ValueError):
                measured[config] = float("inf")
            return timings

        best, timings = policy(measuring_bench, candidates, args, kwargs)
        if best is not None and _tile_is_legal(best):
            return best, timings

        measured_legal = [config for config in legal if config in measured]
        chosen = (
            min(measured_legal, key=measured.__getitem__)
            if measured_legal
            else legal[0]
        )
        logger.warning(
            "%s: the tuning policy proposed BLOCK_M=%s below the MACA MMA "
            "minimum, falling back to a legal candidate (BLOCK_M=%s)",
            kernel_name,
            None if best is None else best.kwargs.get("BLOCK_M"),
            chosen.kwargs.get("BLOCK_M"),
        )
        return chosen, timings

    floored_policy._metax_tile_guard = True
    tuner.policy = floored_policy


def _install_dense_space_guard(tuner, kernel_name):
    """Floor the MMA extents whenever the tuner swaps its config space.

    FlagTune resolves its candidate space at run time and installs it through
    ``_set_configs_and_strategy``, which replaces ``self.configs`` and rebinds
    ``self.cache``.  Flooring the config space only at import time would be
    discarded by that swap, and the replaced cache would lose its floor too.

    Transforming the incoming list keeps ``configs_hash``/``config_table_name``
    describing the space that is actually searched, and re-arming the cache
    guard afterwards keeps the rebound cache normalized as well.  This is the
    one swap point every resolution path goes through, including the
    exhaustive-collection route that benches the space without consulting the
    policy.
    """
    original = tuner._set_configs_and_strategy
    if getattr(original, "_metax_space_guard", False):
        return

    def guarded_set_configs_and_strategy(configs, strategy, *, mode=None):
        result = original(
            [_floor_dense_tile(config) for config in configs], strategy, mode=mode
        )
        # the swap rebinds LibTuner.cache to a different ConfigCache
        _install_dense_cache_guard(tuner, kernel_name)
        return result

    guarded_set_configs_and_strategy._metax_space_guard = True
    tuner._set_configs_and_strategy = guarded_set_configs_and_strategy


def _install_dense_cache_guard(tuner, kernel_name):
    """Floor the MMA extents on every read and write of the tuner's cache.

    ``LibTuner.run`` replays the config stored in the persistent ``ConfigCache``
    straight into the compiler without consulting the pruner, so a tile that was
    written by an older config space -- before the MACA floor existed -- would
    still be replayed as-is.

    ``get`` and ``set`` are ordinary methods, so shadowing them on the instance
    normalizes the store in both directions without subclassing or replacing it:
    a stale row reads back legal, and every new selection is stored legal, so
    the cache heals itself on the next write.
    """
    cache = getattr(tuner, "cache", None)
    if cache is None or getattr(cache, "_metax_tile_guard", False):
        return
    original_get = cache.get
    original_set = cache.set

    def floored_get(key):
        config = original_get(key)
        return None if config is None else _floor_dense_tile(config, kernel_name)

    def floored_set(key, config):
        return original_set(key, _floor_dense_tile(config, kernel_name))

    cache.get = floored_get
    cache.set = floored_set
    cache._metax_tile_guard = True


# Every kernel in this module that feeds a MACA MMA.  The ones without a
# `tl.dot` (gemv, the elementwise small-N path, the split-K reduction) are left
# alone because a sub-16 extent is legal for them.  The names are passed
# explicitly because the kernels are wrapped by ``libentry``/``libtuner`` and no
# longer expose a usable ``__name__``.
for _dense_kernel_name, _dense_kernel in (
    ("mm_kernel", mm_kernel),
    ("mm_kernel_nn", mm_kernel_nn),
    ("mm_kernel_nt", mm_kernel_nt),
    ("mm_kernel_splitk", mm_kernel_splitk),
    ("mm_kernel_splitk_partial", mm_kernel_splitk_partial),
):
    _dense_tuner = _dense_kernel.fn
    # ``LibTuner.run`` uses the declared candidate list as-is when it holds a
    # single config, which is the one path that never reaches the pruner.
    _dense_tuner.configs = [
        _floor_dense_tile(config) for config in _dense_tuner.configs
    ]
    _install_dense_space_guard(_dense_tuner, _dense_kernel_name)
    _install_dense_config_guard(_dense_tuner, _dense_kernel_name)
    _install_dense_cache_guard(_dense_tuner, _dense_kernel_name)
del _dense_kernel_name, _dense_kernel, _dense_tuner


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
        mm_kernel_nn[grid](a, b, c, M, N, K, GROUP_M=8)
    return c


def general_mm_nt(a, b, c, M, N, K):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_nt[grid](a, b, c, M, N, K, GROUP_M=8)
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


def _small_n_mm_scenario(a, b, c, N, K):
    return (
        1 < N < _SMALL_N_DOT_BLOCK_N
        and K >= 2048
        and a.dtype == b.dtype == c.dtype
        and c.dtype in (torch.float16, torch.bfloat16)
    )


def _prefer_dense_nt_over_generic_splitk(M, N, K):
    fp32_output_bytes = M * N * torch.float32.itemsize
    output_threshold = max(
        1,
        get_l2_cache_size() // _GENERIC_SPLITK_DENSE_OUTPUT_L2_DENOMINATOR,
    )
    return K <= 2 * N and fp32_output_bytes >= output_threshold


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
    if M == 0 or N == 0:
        return c
    if N == 1:
        if _gemv_k_parallel_scenario(M, K):
            return gemv_mm_k_parallel(a, b, c, M, K)
        return gemv_mm(a, b, c, M, K)
    if _small_n_mm_scenario(a, b, c, N, K):
        return small_n_mm(a, b, c, M, N, K)
    two_step_split_k = _select_two_step_split_k(M, N, K)
    if two_step_split_k is not None:
        return splitk_mm_two_step(a, b, c, M, N, K, two_step_split_k)
    nt_scenario = nt_mm_scenario(a, b, c, M, N, K)
    prefer_dense_nt = (
        c.dtype in (torch.float16, torch.bfloat16)
        and nt_scenario
        and _prefer_dense_nt_over_generic_splitk(M, N, K)
    )
    if splitk_mm_scenario(M, N, K) and not prefer_dense_nt:
        if (
            c.dtype == torch.float32
            and not torch.are_deterministic_algorithms_enabled()
        ):
            c.zero_()
            return splitk_mm(a, b, c, M, N, K)
        split_k = _generic_two_step_split_k(M, N, K)
        return splitk_mm_two_step(a, b, c, M, N, K, split_k)
    if nn_mm_scenario(a, b, c, M, N, K):
        return general_mm_nn(a, b, c, M, N, K)
    if nt_scenario:
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
    if M == 0 or N == 0:
        return c
    if N == 1:
        if _gemv_k_parallel_scenario(M, K):
            return gemv_mm_k_parallel(a, b, c, M, K)
        return gemv_mm(a, b, c, M, K)
    if _small_n_mm_scenario(a, b, c, N, K):
        return small_n_mm(a, b, c, M, N, K)
    two_step_split_k = _select_two_step_split_k(M, N, K)
    if two_step_split_k is not None:
        return splitk_mm_two_step(a, b, out, M, N, K, two_step_split_k)
    nt_scenario = nt_mm_scenario(a, b, c, M, N, K)
    prefer_dense_nt = (
        c.dtype in (torch.float16, torch.bfloat16)
        and nt_scenario
        and _prefer_dense_nt_over_generic_splitk(M, N, K)
    )
    if splitk_mm_scenario(M, N, K) and not prefer_dense_nt:
        if (
            c.dtype == torch.float32
            and not torch.are_deterministic_algorithms_enabled()
        ):
            c.zero_()
            return splitk_mm(a, b, c, M, N, K)
        split_k = _generic_two_step_split_k(M, N, K)
        return splitk_mm_two_step(a, b, c, M, N, K, split_k)
    if nn_mm_scenario(a, b, c, M, N, K):
        return general_mm_nn(a, b, c, M, N, K)
    if nt_scenario:
        return general_mm_nt(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)
