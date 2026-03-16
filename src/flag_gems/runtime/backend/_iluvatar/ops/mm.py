import logging
import math

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

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
    raise AssertionError("unreachable")


def _to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 15,
    },
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
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
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    # grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # # re-order program ID for better L2 performance
    # width = GROUP_M * grid_n
    # group_id = pid // width
    # group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    # pid_m = group_id * GROUP_M + (pid % group_size)
    # pid_n = (pid % width) // (group_size)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

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
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def _launch_mm(a, b, c, M, N, K):
    """Launch Triton matmul _kernel; c must be pre-allocated."""
    ab_dtype = get_higher_dtype(a.dtype, b.dtype)
    acc_dtype_tl = tl.float32
    ab_dtype_tl = _to_tl_type(ab_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
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
            acc_dtype=acc_dtype_tl,
            input_precision=None,
            fp8_fast_accum=True,
            GROUP_M=8,
            AB_DTYPE=ab_dtype_tl,
        )
    return c


def mm(a, b):
    logger.debug("ILUVATAR GEMS MM")
    device = a.device
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    return _launch_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("ILUVATAR GEMS MM_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    return _launch_mm(a, b, out, M, N, K)
