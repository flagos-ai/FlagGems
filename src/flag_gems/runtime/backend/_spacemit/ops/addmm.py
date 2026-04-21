import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

logger = logging.getLogger(__name__)


@triton.jit
def addmm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
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
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    SUB_BLK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
    accumulator = smt.view(
        accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N)
    )
    sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
    for k in tl.range(0, sub_num):
        a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
        a = smt.view(
            a_descriptor_load,
            (0, k * SUB_BLK_K),
            (BLOCK_SIZE_M, SUB_BLK_K),
            (MICRO_M, MICRO_K),
        )
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(
            b_descriptor_load,
            (k * SUB_BLK_K, 0),
            (SUB_BLK_K, BLOCK_SIZE_N),
            (MICRO_K, MICRO_N),
        )
        accumulator += smt.dot(a, b)
    accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))

    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=[M, N],
        strides=[stride_im, stride_in],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    bias = tl.load(bias_block_ptr, boundary_check=(0, 1))
    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )

    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_SPACEMIT ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape).contiguous()

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    BLOCK_SIZE_K = triton.next_power_of_2(K)
    SUB_BLK_K = min(1024, BLOCK_SIZE_K)

    if mat1.dtype == torch.float32:
        MICRO_M = 8
        MICRO_N = 32
        MICRO_K = 32
    else:
        MICRO_M = 16
        MICRO_N = 32
        MICRO_K = 8

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
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=MICRO_M,
        MICRO_N=MICRO_N,
        MICRO_K=MICRO_K,
        SUB_BLK_K=SUB_BLK_K,
    )
    return out
