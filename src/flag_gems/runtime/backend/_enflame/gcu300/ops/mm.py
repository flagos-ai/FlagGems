import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    strategy=["log", "log", "log"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.jit
def mm_kernel(
    A_in,
    B_in,
    C_in,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    num_warps: tl.constexpr,
):
    # matrix multiplication
    pid_mn = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_num = tl.cdiv(MAX_GRID_DIM, num_warps)
    for pid in tl.range(pid_mn, grid_m * grid_n, grid_num):
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        # do matrix multiplication
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        # pointers
        A = A_in + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_in + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                mask_a = (rm < M)[:, None]
                mask_b = (rn < N)[None, :]
                a = tl.load(A, mask_a, other=0)
                b = tl.load(B, mask_b, other=0)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                mask_a = (rm < M)[:, None] & (rk < k_remaining)[None, :]
                mask_b = (rk < k_remaining)[:, None] & (rn < N)[None, :]
                a = tl.load(A, mask=mask_a, other=0)
                b = tl.load(B, mask=mask_b, other=0)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C_in.dtype.element_ty)
        C = C_in + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
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


def mm(a, b):
    logger.debug("GEMS MM")
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
    # launch kernel
    MAX_GRID_DIM = 24
    grid = lambda META: (
        min(
            triton.cdiv(MAX_GRID_DIM, META["num_warps"]),
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ),
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
            GROUP_M=8,
            MAX_GRID_DIM=MAX_GRID_DIM,
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # launch kernel
    MAX_GRID_DIM = 24
    grid = lambda META: (
        min(
            triton.cdiv(MAX_GRID_DIM, META["num_warps"]),
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            GROUP_M=8,
            MAX_GRID_DIM=MAX_GRID_DIM,
        )
    return out
