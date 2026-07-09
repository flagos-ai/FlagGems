import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

from ..utils.shape_utils import dim_compress

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

# Cap block size so local-memory tensors fit on TPU even for very large M
# (e.g. M=2^30 -> sqrt(M)=32768 would overflow). The _1 kernels hold
# ~2*BLOCK_SIZE*4B; the _2 kernels loop in fixed BLOCK_MID chunks.
MAX_BLOCK = 4096


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum = tl.sum(_sum, axis=1)

    out = tl.sqrt(sum)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def l2_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l2_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    # MID_SIZE can be very large for big M; loop over fixed BLOCK_MID chunks
    # instead of one giant tl.arange(0, BLOCK_MID) that overflows local memory.
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    for off in range(0, MID_SIZE, BLOCK_MID):
        offset = off + tl.arange(0, BLOCK_MID)
        mask = offset < MID_SIZE
        mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
        acc += mid
    out = tl.sqrt(tl.sum(acc))
    tl.store(Out, out)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _max = tl.maximum(tl.abs(a), _max)

    max = tl.max(_max, axis=1)
    out = max[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def max_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.max(tl.abs(x))
    tl.store(Mid, mid)


@libentry()
@triton.jit
def max_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    acc = tl.full([BLOCK_MID], value=float("-inf"), dtype=tl.float32)
    for off in range(0, MID_SIZE, BLOCK_MID):
        offset = off + tl.arange(0, BLOCK_MID)
        mask = offset < MID_SIZE
        mid = tl.load(Mid + offset, mask=mask, other=float("-inf")).to(tl.float32)
        acc = tl.maximum(acc, mid)
    out = tl.max(acc)
    tl.store(Out, out)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=float("inf")).to(tl.float32)
        _min = tl.minimum(tl.abs(a), _min)

    min = tl.min(_min, axis=1)
    out = min[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def min_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=float("inf")).to(tl.float32)
    mid = tl.min(tl.abs(x))
    tl.store(Mid, mid)


@libentry()
@triton.jit
def min_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    acc = tl.full([BLOCK_MID], value=float("inf"), dtype=tl.float32)
    for off in range(0, MID_SIZE, BLOCK_MID):
        offset = off + tl.arange(0, BLOCK_MID)
        mask = offset < MID_SIZE
        mid = tl.load(Mid + offset, mask=mask, other=float("inf")).to(tl.float32)
        acc = tl.minimum(acc, mid)
    out = tl.min(acc)
    tl.store(Out, out)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0).to(tl.float32)
        _sum += tl.where(a != 0, 1, 0)
    sum = tl.sum(_sum, axis=1)
    out = sum[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def l0_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    cnt = (x != 0).to(tl.float32)
    mid = tl.sum(cnt)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l0_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    for off in range(0, MID_SIZE, BLOCK_MID):
        offset = off + tl.arange(0, BLOCK_MID)
        mask = offset < MID_SIZE
        mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
        acc += mid
    out = tl.sum(acc)
    tl.store(Out, out)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += pow(tl.abs(a), ord)
    sum = tl.sum(_sum, axis=1)
    out = pow(sum, 1 / ord)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_1(X, Mid, ord, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(pow(tl.abs(x), ord))
    tl.store(Mid, mid)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr):
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    for off in range(0, MID_SIZE, BLOCK_MID):
        offset = off + tl.arange(0, BLOCK_MID)
        mask = offset < MID_SIZE
        mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
        acc += mid
    out = pow(tl.sum(acc), 1 / ord)
    tl.store(Out, out)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS VECTOR NORM")
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
            BLOCK_SIZE = min(triton.next_power_of_2(math.ceil(math.sqrt(M))), MAX_BLOCK)
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = min(MAX_BLOCK, triton.next_power_of_2(MID_SIZE))

            mid = torch.empty([MID_SIZE], dtype=dtype, device=x.device)
            out = torch.empty(shape, dtype=dtype, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(MID_SIZE,)](x, mid, M, BLOCK_SIZE)
                l2_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            elif ord == float("inf"):
                max_norm_kernel_1[(MID_SIZE,)](x, mid, M, BLOCK_SIZE)
                max_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            elif ord == -float("inf"):
                min_norm_kernel_1[(MID_SIZE,)](x, mid, M, BLOCK_SIZE)
                min_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            elif ord == 0:
                l0_norm_kernel_1[(MID_SIZE,)](x, mid, M, BLOCK_SIZE)
                l0_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            else:
                l1_norm_kernel_1[(MID_SIZE,)](x, mid, ord, M, BLOCK_SIZE)
                l1_norm_kernel_2[(1,)](mid, out, ord, MID_SIZE, BLOCK_MID)
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
