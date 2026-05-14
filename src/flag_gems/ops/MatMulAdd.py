import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("addmm", pre_hook=None)
    if os.environ.get("USE_FLAGTUNE") == "1"
    else runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=runtime.get_expand_config("addmm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def matmuladd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_m,
    stride_bias_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_ptrs = (
        bias_ptr + stride_bias_m * offs_cm[:, None] + stride_bias_n * offs_cn[None, :]
    )
    bias_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    accumulator = accumulator + bias
    c = accumulator.to(bias.dtype)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=bias_mask)


def matmuladd(input, other, bias):
    """
    Matrix multiplication with addition: output = matmul(input, other) + bias
    """
    assert input.shape[1] == other.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], other.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = other.shape

    logger.debug(
        "GEMS MatMulAdd, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        input.stride(0) == 1,
        other.stride(0) == 1,
        bias.stride(0) == 1,
    )
    input = input.contiguous()
    # other = other.contiguous()
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        matmuladd_kernel[grid](
            input,
            other,
            out,
            bias,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            bias.stride(0),
            bias.stride(1),
        )
    return out


def matmuladd_out(input, other, bias, *, out=None):
    """
    Matrix multiplication with addition (out variant): output = matmul(input, other) + bias
    """
    assert input.shape[1] == other.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], other.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = other.shape
    if out is None:
        out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    logger.debug(
        "GEMS MatMulAdd_OUT, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        input.stride(0) == 1,
        other.stride(0) == 1,
        bias.stride(0) == 1,
    )
    input = input.contiguous()
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        matmuladd_kernel[grid](
            input,
            other,
            out,
            bias,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            bias.stride(0),
            bias.stride(1),
        )
    return out
