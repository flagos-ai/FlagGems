"""
Int8 Matrix Multiplicatiton - Triton kernel

API:
    Int8_matmul(a, b)->Tensor

    a: (M, K)
    b: (K, N)

    Returns: int32
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

CACHE_USAGE_THRESHOLD = 0.8

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("int8_mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=4,
    rep=10,
)
@triton.jit
def int8_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    GROUP_M: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    a_ptrs = a_ptr + ram[:, None] * stride_am
    b_ptrs = b_ptr + rbn[None, :] * stride_bn
    prev_multiple = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K

    # multiple loop
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for start_k in tl.range(0, prev_multiple, BLOCK_K, num_stages=num_stages):
        rk = start_k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptrs + rk[None, :] * stride_ak)
        b = tl.load(b_ptrs + rk[:, None] * stride_bk)
        acc += tl.dot(a, b, out_dtype=tl.int32)

    # remainder
    rk = prev_multiple + tl.arange(0, BLOCK_K)
    mask_k = rk < K
    a = tl.load(a_ptrs + rk[None, :] * stride_ak, mask=mask_k[None, :], other=0)
    b = tl.load(b_ptrs + rk[:, None] * stride_bk, mask=mask_k[:, None], other=0)
    acc += tl.dot(a, b, out_dtype=tl.int32)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)


def int8_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Int8 matrix multiplication

    Args:
        a: (M, K)    int8
        b: (K, M)    int8
    Returns:
        c: (M, N)    int32
    """
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.dim() == 2, f"A tensor rank must be 2, but got {a.dim()}"
    assert b.dim() == 2, f"B tensor rank must be 2, but got {b.dim()}"
    assert a.dtype == torch.int8, f"a tensor dtype must be int8, but got {a.dtype}"
    assert b.dtype == torch.int8, f"b tensor dtype must be int8, but got {b.dtype}"

    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    # with torch_device_fn.device(a.device):
    int8_mm_kernel[grid](
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
    )
    return c


def int8_mm_out(a, b, *, out):
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.dim() == 2, f"A tensor rank must be 2, but got {a.dim()}"
    assert b.dim() == 2, f"B tensor rank must be 2, but got {b.dim()}"
    assert a.dtype == torch.int8, f"a tensor dtype must be int8, but got {a.dtype}"
    assert b.dtype == torch.int8, f"b tensor dtype must be int8, but got {b.dtype}"

    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    M, K = a.shape
    K, N = b.shape

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    with torch_device_fn.device(a.device):
        int8_mm_kernel[grid](
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
        )
    return out
