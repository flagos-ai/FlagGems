import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    strategy=["log", "log", "log"],
)
@triton.jit
def mm_kernel(
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
):
    pid = tle.program_id(0)

    # --------------------------
    # match naming: num_pid_m, num_pid_n
    # --------------------------
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # reorder for L2
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --------------------------
    # match naming: offs_am, offs_bn, offs_k
    # --------------------------
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # contiguous aligned offsets (ram/rbn → offs_am/offs_bn)
    offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
    offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    # previous K multiple
    # prev_k_mult = prev_multiple_of(K, BLOCK_K)
    prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K

    # accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --------------------------
    # main K loop
    # --------------------------
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k

        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak)
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn)
        )

        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)

        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # --------------------------
    # loop peel
    # --------------------------
    rk = prev_k_mult + offs_k
    mask_k = rk < K

    a = tl.load(
        a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
    )
    b = tl.load(
        b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
        mask=mask_k[:, None],
    )

    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)

    accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # cast to output dtype
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # --------------------------
    # rematerialize offsets for store
    # (match naming: offs_cm, offs_cn)
    # --------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    tl.store(c_ptr, accumulator, mask=mask_store)


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
            GROUP_M=8,
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
    # allocates output
    c = out
    # launch kernel
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
            GROUP_M=8,
        )
    return c
