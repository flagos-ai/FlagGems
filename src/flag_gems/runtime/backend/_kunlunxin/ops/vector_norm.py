import builtins
import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
pow = tl_extra_shim.pow


def heur_block_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def heur_block_n(args):
    return builtins.min(args["N"], 8192)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
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
def l2_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
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
def l2_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
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
def max_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
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
def max_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.max(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
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
def min_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
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
def min_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=float("inf")).to(tl.float32)
    out = tl.min(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
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
def l0_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
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
def l0_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sum(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ord = ord.to(tl.float32)
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
def l1_norm_kernel_1(
    X, Mid, ord, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    ord = ord.to(tl.float32)
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
def l1_norm_kernel_2(
    Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    ord = ord.to(tl.float32)
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = pow(tl.sum(mid), 1 / ord)
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
            cluster_num = 12
            BLOCK_SIZE = min(
                triton.next_power_of_2(triton.cdiv(M, cluster_num)),
                int(1024 * 64 / x.element_size()),
            )
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            mid = torch.empty([MID_SIZE], dtype=dtype, device=x.device)
            out = torch.empty(shape, dtype=dtype, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                l2_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == float("inf"):
                max_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                max_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == -float("inf"):
                min_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                min_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == 0:
                l0_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                l0_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            else:
                l1_norm_kernel_1[(MID_SIZE,)](
                    x, mid, ord, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                l1_norm_kernel_2[(1,)](
                    mid, out, ord, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
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
                v_norm_kernel[grid](x, out, M, N, ord, isCloseUnrollControl=True)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
