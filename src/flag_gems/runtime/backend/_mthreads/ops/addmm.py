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


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
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
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID with swizzling for better L2 cache utilization
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Re-order program ID for better L2 performance
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # Compute block offsets
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Use memory hints for better access
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), BLOCK_SIZE_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    # Compute previous multiple of BLOCK_SIZE_K for loop peeling
    prev_multiple = prev_multiple_of(K, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop: iterate over K dimension
    for start_k in range(0, prev_multiple, BLOCK_SIZE_K):
        rk = (start_k + tl.arange(0, BLOCK_SIZE_K)).to(tl.int64)
        a = tl.load(a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Loop peeling: handle remaining K elements
    rk = (prev_multiple + tl.arange(0, BLOCK_SIZE_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
        mask=mask_k[:, None],
        other=0.0,
    )
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Load bias
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    i_ptrs = i_ptr + (rm[:, None] * stride_im + rn[None, :] * stride_in)
    bias = tl.load(i_ptrs, mask=mask, other=0.0)

    # Apply alpha and beta
    acc = acc * alpha + bias * beta

    # Store result
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c = acc.to(bias.dtype)
    tl.store(c_ptrs, c, mask=mask)


def addmm_fma(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM(FMA)")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    # Handle non-contiguous inputs
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()

    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)

    # Use 1D grid with GROUP_SIZE_M for better L2 cache utilization
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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
            GROUP_SIZE_M=8,
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
    GROUP_SIZE_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    ab_type: tl.constexpr,
    c_type: tl.constexpr,
    is_transpose_a: tl.constexpr = False,
    is_transpose_b: tl.constexpr = False,
):
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Re-order program ID for better L2 performance
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    offs_am = offs_am.to(tl.int32)
    offs_bn = offs_bn.to(tl.int32)
    offs_k = offs_k.to(tl.int32)

    input_type = ab_type
    output_type = c_type
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [offs_am, offs_k],
            [BLOCK_SIZE_M, BLOCK_SIZE_K],
            input_type,
            is_transpose_a,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [offs_k, offs_bn],
            [BLOCK_SIZE_K, BLOCK_SIZE_N],
            input_type,
            is_transpose_b,
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
    logger.debug("GEMS_MTHREADS ADDMM(SQMMA)")
    device = "musa"
    assert broadcastable_to(
        Bias.shape, (A.shape[0], B.shape[1])
    ), "Incompatible input shape"
    # handle non-contiguous inputs if necessary
    is_transpose_a = False
    is_transpose_b = False
    if not A.is_contiguous():
        if A.stride(0) == 1 and A.stride(1) == A.shape[0]:
            is_transpose_a = True
        else:
            A = A.contiguous()
    if not B.is_contiguous():
        if B.stride(0) == 1 and B.stride(1) == B.shape[0]:
            is_transpose_b = True
        else:
            B = B.contiguous()
    ab_type = elem_type
    a_type = A.dtype
    b_type = B.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_type = a_type
    C = torch.empty((M, N), dtype=c_type, device=device)
    Bias = Bias.broadcast_to(C.shape).contiguous()
    desc_a = create_tma_device_descriptor(A, BLOCK_M, BLOCK_K, device)
    desc_b = create_tma_device_descriptor(B, BLOCK_K, BLOCK_N, device)
    desc_bias = create_tma_device_descriptor(Bias, BLOCK_M, BLOCK_N, device)
    desc_c = create_tma_device_descriptor(C, BLOCK_M, BLOCK_N, device)

    GROUP_M = 8
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
        GROUP_M,
        alpha,
        beta,
        get_triton_type(ab_type),
        get_triton_type(c_type),
        is_transpose_a,
        is_transpose_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
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
