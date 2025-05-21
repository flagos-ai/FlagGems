import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from .utils import create_tma_device_descriptor, get_triton_dtype, should_enable_sqmma

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.jit
def mm_sqmma_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    a_dtype: tl.constexpr,
    b_dtype: tl.constexpr,
    c_dtype: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    pid_z = tle.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = pid_z * BLOCK_K
    offs_am = offs_am.to(tl.int32)
    offs_bn = offs_bn.to(tl.int32)
    offs_k = offs_k.to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    atype = a_dtype
    btype = b_dtype
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        a = tl._experimental_descriptor_load(
            A, [offs_am, offs_k], [BLOCK_M, BLOCK_K], atype
        )
        b = tl._experimental_descriptor_load(
            B, [offs_k, offs_bn], [BLOCK_K, BLOCK_N], btype
        )
        if a.dtype != b.dtype:
            a = a.to(c_dtype)
            b = b.to(c_dtype)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        offs_k += BLOCK_K
    acc = acc.to(c_dtype)

    tl._experimental_descriptor_store(C, acc, [offs_am, offs_bn])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
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
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    pid_z = tle.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


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


def get_mm_config():
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "SPLIT_K": 1,
        "num_stages": 1,
        "num_warps": 4,
    }


def mm_sqmma(a, b):
    logger.debug("GEMS MM SQMMA")
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
    c_ori_dtype = get_higher_dtype(a.dtype, b.dtype)
    c_dtype = c_ori_dtype if a.dtype != torch.bfloat16 else torch.float32
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    a_dtype = get_triton_dtype(a.dtype)
    b_dtype = get_triton_dtype(b.dtype)
    c_dtype = get_triton_dtype(c_dtype)
    dot_out_dtype = tl.float32
    # prepare tma descriptor for sqmma
    mm_config = get_mm_config()
    BLOCK_M = mm_config["BLOCK_M"]
    BLOCK_N = mm_config["BLOCK_N"]
    BLOCK_K = mm_config["BLOCK_K"]
    SPLIT_K = mm_config["SPLIT_K"]
    num_stages = mm_config["num_stages"]
    num_warps = mm_config["num_warps"]
    desc_a = create_tma_device_descriptor(a, BLOCK_M, BLOCK_K, device)
    desc_b = create_tma_device_descriptor(b, BLOCK_K, BLOCK_N, device)
    desc_c = create_tma_device_descriptor(c, BLOCK_M, BLOCK_N, device)

    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_sqmma_kernel[grid](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_K=SPLIT_K,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return c.to(c_ori_dtype)


def mm_fma(a, b):
    logger.debug("GEMS MM FMA")
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
    dot_out_dtype = tl.float32
    # launch kernel
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
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
        )
    return c


def mm(a, b):
    a_dtype = a.dtype
    b_dtype = b.dtype
    M, K = a.shape
    _, N = b.shape
    use_sqmma = should_enable_sqmma(a_dtype, b_dtype, M, N, K)
    if use_sqmma:
        return mm_sqmma(a, b)
    else:
        enable_sqmma = os.environ.pop("MUSA_ENABLE_SQMMA", None)
        result = mm_fma(a, b)
        if enable_sqmma:
            os.environ["MUSA_ENABLE_SQMMA"] = enable_sqmma
        return result
