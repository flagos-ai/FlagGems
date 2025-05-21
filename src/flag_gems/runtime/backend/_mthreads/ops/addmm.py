import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

from .utils import create_tma_device_descriptor, get_triton_dtype, should_enable_sqmma

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_sqmma_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    a_dtype: tl.constexpr,
    b_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    offs_am = offs_am.to(tl.int32)
    offs_bn = offs_bn.to(tl.int32)
    offs_k = offs_k.to(tl.int32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    atype = a_dtype
    btype = b_dtype
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], atype
        )
        b = tl._experimental_descriptor_load(
            b_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], btype
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += BLOCK_SIZE_K

    c_ptrs = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(offs_am, offs_bn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )
    i_ptrs = tl.make_block_ptr(
        base=i_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(offs_am, offs_bn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )
    alpha_ptrs = tl.make_block_ptr(
        base=alpha,
        shape=(M, N),
        strides=(N, 1),
        offsets=(offs_am, offs_bn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )
    bias = tl.load(i_ptrs)
    alpha = tl.load(alpha_ptrs)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(atype)
    tl.store(c_ptrs, c)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
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
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def get_mm_config():
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_stages": 1,
        "num_warps": 4,
    }


def addmm_sqmma(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMM SQMMA")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    # allocates output
    device = mat1.device
    c_dtype = mat1.dtype
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    a_dtype = get_triton_dtype(mat1.dtype)
    b_dtype = get_triton_dtype(mat2.dtype)
    c_dtype = get_triton_dtype(c_dtype)
    # prepare tma descriptor for sqmma
    mm_config = get_mm_config()
    BLOCK_M = mm_config["BLOCK_M"]
    BLOCK_N = mm_config["BLOCK_N"]
    BLOCK_K = mm_config["BLOCK_K"]
    num_stages = mm_config["num_stages"]
    num_warps = mm_config["num_warps"]
    desc_a = create_tma_device_descriptor(mat1, BLOCK_M, BLOCK_K, device)
    desc_b = create_tma_device_descriptor(mat2, BLOCK_K, BLOCK_N, device)

    bias = bias.broadcast_to(c.shape).contiguous()
    alpha = torch.full(c.shape, alpha, device=device, dtype=c.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_sqmma_kernel[grid](
            desc_a,
            desc_b,
            bias,
            c,
            alpha,
            beta,
            M,
            N,
            K,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return c


def addmm_fma(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMM FMA")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    a_dtype = mat1.dtype
    b_dtype = mat2.dtype
    M, K = mat1.shape
    _, N = mat2.shape
    use_sqmma = should_enable_sqmma(a_dtype, b_dtype, M, N, K)
    if use_sqmma:
        return addmm_sqmma(bias, mat1, mat2, alpha=alpha, beta=beta)
    else:
        enable_sqmma = os.environ.pop("MUSA_ENABLE_SQMMA", None)
        result = addmm_fma(bias, mat1, mat2, alpha=alpha, beta=beta)
        if enable_sqmma:
            os.environ["MUSA_ENABLE_SQMMA"] = enable_sqmma
        return result
