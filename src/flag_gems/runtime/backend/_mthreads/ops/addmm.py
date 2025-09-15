import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from .utils import create_tma_device_descriptor, should_enable_sqmma

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@libentry()
@libtuner(
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


def addmm_fma(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM(FMA)")
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


@triton.jit
def addmm_sqmma_kernel(
    a_desc_ptr,
    b_desc_ptr,
    bias_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    ab_type: tl.constexpr,
    c_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    input_type = ab_type
    output_type = c_type
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], input_type
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], input_type
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    bias = tl._experimental_descriptor_load(
        bias_desc_ptr, [offs_am, offs_bn], [BLOCK_SIZE_M, BLOCK_SIZE_N], input_type
    )
    result = (alpha * accumulator.to(output_type) + beta * bias.to(output_type)).to(
        output_type
    )
    tl._experimental_descriptor_store(c_desc_ptr, result, [offs_am, offs_bn])


def get_triton_type(elem_type):
    type_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
    }
    return type_map.get(elem_type, None)


def addmm_sqmma(
    A,
    B,
    Bias,
    elem_type,
    alpha,
    beta,
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    num_stages,
):
    device = "musa"
    ab_type = elem_type
    c_type = elem_type if (elem_type != torch.bfloat16) else torch.float16
    C = torch.empty((M, N), dtype=torch.float16, device=device).to(c_type)
    desc_a = create_tma_device_descriptor(A, BLOCK_M, BLOCK_K, device)
    desc_b = create_tma_device_descriptor(B, BLOCK_K, BLOCK_N, device)
    desc_bias = create_tma_device_descriptor(Bias, BLOCK_M, BLOCK_N, device)
    desc_c = create_tma_device_descriptor(C, BLOCK_M, BLOCK_N, device)
    addmm_sqmma_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        desc_a,
        desc_b,
        desc_bias,
        desc_c,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        alpha,
        beta,
        get_triton_type(ab_type),
        get_triton_type(c_type),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


def addmm(bias, mat1, mat2, *, beta=0.5, alpha=0.5):
    a_dtype = mat1.dtype
    b_dtype = mat2.dtype
    M, K = mat1.shape
    _, N = mat2.shape
    use_sqmma = should_enable_sqmma(a_dtype, b_dtype, M, N, K)
    if use_sqmma:
        BLOCK_M = 256 if M % 256 == 0 else 128
        BLOCK_N = BLOCK_M
        BLOCK_K = 64
        num_warps = 16 if BLOCK_M == 256 else 4
        num_stages = 1
        return addmm_sqmma(
            mat1,
            mat2,
            bias,
            a_dtype,
            alpha,
            beta,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps,
            num_stages,
        )
    else:
        enable_sqmma = os.environ.pop("MUSA_ENABLE_SQMMA", None)
        result = addmm_fma(bias, mat1, mat2, alpha=alpha, beta=beta)
        if enable_sqmma:
            os.environ["MUSA_ENABLE_SQMMA"] = enable_sqmma
        return result
