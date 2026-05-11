import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.device_info import get_device_capability, get_sm_count

CACHE_USAGE_THRESHOLD = 0.8

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mm_kernel_general(
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
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # matrix multiplication
    pid = ext.program_id(0)
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
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if IS_FP64:
            acc += tl.dot(a, b, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
        mask=mask_k[:, None],
        other=0.0,
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    if IS_FP64:
        acc += tl.dot(a, b, allow_tf32=False)
    else:
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


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
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_general[grid](
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
            IS_FP64=a.dtype == torch.float64,
        )
    return c


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_self_transpose"),
    key=["M", "K", "stride_am", "stride_ak"],
    strategy=["align32", "align32", "align32", "align32"],
    warmup=2,
    rep=4,
)
@triton.jit
def mm_kernel_syrk(
    A,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # Packed lower-triangular launch domain:
    #   pid = row * (row + 1) / 2 + col, where 0 <= col <= row.
    #
    # Invert the triangular-number indexing by solving:
    #   row^2 + row - 2 * pid = 0
    # => row = (-1 + sqrt(1 + 8 * pid)) / 2
    #
    # We take floor(...) as the candidate row, then apply an integer +/-1 correction
    # because fp32 sqrt can be off near triangular-number boundaries.
    pid_f = pid.to(tl.float32)
    pid_m = tl.floor((tl.sqrt(8.0 * pid_f + 1.0) - 1.0) / 2.0).to(tl.int32)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_m = tl.where(tri_start > pid, pid_m - 1, pid_m)
    next_tri_start = (pid_m + 1) * (pid_m + 2) // 2
    pid_m = tl.where(next_tri_start <= pid, pid_m + 1, pid_m)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_n = pid - tri_start

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    ran = tl.max_contiguous(tl.multiple_of(rn % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            A + (rk[:, None] * stride_ak + ran[None, :] * stride_am),
            mask=mask_k[:, None],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    out = acc.to(C.dtype.element_ty)
    c_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < M)[None, :]
    tl.store(c_ptr, out, mask=mask)

    if pid_m > pid_n:
        c_t_ptr = C + (rn[:, None] * stride_cm + rm[None, :] * stride_cn)
        mask_t = (rn < M)[:, None] & (rm < M)[None, :]
        tl.store(c_t_ptr, tl.trans(out), mask=mask_t)


def is_syrk_transpose_pair(a, b):
    return (
        a.ndim == 2
        and b.ndim == 2
        and a.shape[0] == b.shape[1]
        and a.shape[1] == b.shape[0]
        and a.stride(0) == b.stride(1)
        and a.stride(1) == b.stride(0)
        and a.storage_offset() == b.storage_offset()
        and a.data_ptr() == b.data_ptr()
    )


def syrk_mm(a, c, M, K):
    grid = lambda META: (
        # Number of tile rows is tiles = ceil(M / BLOCK_M).
        # Packed lower triangle contains:
        #   1 + 2 + ... + tiles = tiles * (tiles + 1) / 2
        triton.cdiv(M, META["BLOCK_M"])
        * (triton.cdiv(M, META["BLOCK_M"]) + 1)
        // 2,
    )
    with torch_device_fn.device(a.device):
        mm_kernel_syrk[grid](
            a,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            c.stride(0),
            c.stride(1),
        )
    return c


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] == 8).
    # The optimal settings for other devices need to be determined through real testing.
    capability = get_device_capability()
    return (
        capability[0] == 8
        and a.dtype in [torch.float16, torch.bfloat16]
        and b.dtype in [torch.float16, torch.bfloat16]
        and a.is_contiguous()
        and b.is_contiguous()
        and K > M * 5
        and K > N * 5
    )


def mm(a, b):
    logger.debug("GEMS MM")

    device = a.device
    if is_syrk_transpose_pair(a, b):
        M, K = a.shape
        c = torch.empty((M, M), device=device, dtype=a.dtype)
        return syrk_mm(a, c, M, K)
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
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS MM_OUT")

    if is_syrk_transpose_pair(a, b):
        M, K = a.shape
        return syrk_mm(a, out, M, K)
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, out, M, N, K)
