"""
Sophgo TPU 专用 addmm 算子实现

修复说明：
原始 addmm kernel 在 Sophgo TPU 上编译失败，原因是：
1. 循环中使用指针更新 (a_ptrs += ...) 导致生成 scf.for iter_args 模式，
   PPL ShapeInference 无法正确处理
2. 生成了 ppl.get_value 操作（标量指针解引用），TPU 上会失败

解决方案：
1. 移除 autotune，使用固定 block size
2. 每次迭代重新计算地址，避免指针更新
3. 使用 2D tensor 模式进行加载和存储
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


def select_addmm_block_sizes(M, N, K):
    if M <= 1:
        block_m = 1
    elif M <= 8:
        block_m = 4
    elif M <= 32:
        block_m = 8
    else:
        block_m = 16

    if N >= 2048:
        block_n = 128
    elif N >= 1024:
        block_n = 64
    else:
        block_n = 32

    if K >= 512:
        block_k = 32
    elif K >= 128:
        block_k = 16
    else:
        block_k = 8

    return block_m, block_n, block_k


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
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
):
    """
    Sophgo TPU 专用 addmm kernel
    计算: beta * bias + alpha * (mat1 @ mat2)

    关键修改点：
    - 每次循环迭代重新计算地址，避免指针更新产生 iter_args
    - 使用 2D tensor 模式进行加载和存储
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    # 计算当前 block 的行列偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K 维度循环 - 每次迭代重新计算地址，避免指针更新
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k_idx in range(0, num_k_blocks):
        # 计算当前 K block 的偏移
        offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # 重新计算 A 和 B 的地址（避免指针更新）
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # 2D tensor 加载 with mask
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘法累加
        accumulator += tl.dot(a, b, allow_tf32=False)

    # 加载 bias（使用 2D 广播后的 bias）
    bias_ptrs = bias_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias = tl.load(bias_ptrs, mask=out_mask, other=0.0)

    # 计算最终结果: beta * bias + alpha * (mat1 @ mat2)
    c = accumulator * alpha + bias * beta
    c = c.to(bias.dtype)

    # 存储结果（使用 2D tensor 存储）
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=out_mask)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    """
    Sophgo TPU 专用 addmm 实现
    计算: beta * bias + alpha * (mat1 @ mat2)

    Args:
        bias: 偏置张量，形状为 (N,) 或 (M, N)
        mat1: 第一个矩阵，形状为 (M, K)
        mat2: 第二个矩阵，形状为 (K, N)
        beta: bias 的缩放系数，默认为 1
        alpha: 矩阵乘法结果的缩放系数，默认为 1

    Returns:
        输出张量，形状为 (M, N)
    """
    logging.debug("GEMS ADDMM (Sophgo TPU)")

    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"

    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)

    # 推理里常见的是 1D bias；这里保留广播视图，避免额外 materialize/copy kernel。
    bias = bias.broadcast_to(out.shape)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = select_addmm_block_sizes(M, N, K)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

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
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    return out
