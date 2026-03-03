try:
    import paddle
    paddle.compat.enable_torch_proxy()
except:
    pass

import torch
import triton
import triton.language as tl
import flag_gems

import time
import torch
from triton import runtime 

import triton
import triton.language as tl
import torch


@triton.jit
def add_kernel(
    a_ptr,   # (M, N)
    b_ptr,   # (M, N)
    c_ptr,   # (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    矩阵相加 Kernel: C = A + B
    """

    # 1️⃣ 当前 block 在输出矩阵中的位置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2️⃣ 当前 block 覆盖的行列范围
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 3️⃣ 计算全局内存地址（按行主序）
    ptrs = offs_m[:, None] * N + offs_n[None, :]

    a_ptrs = a_ptr + ptrs
    b_ptrs = b_ptr + ptrs
    c_ptrs = c_ptr + ptrs

    # 4️⃣ 边界 mask
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 5️⃣ 加载 + 相加
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    c = a + b

    # 6️⃣ 写回
    tl.store(c_ptrs, c, mask=mask)

def run_matmul(M=512, N=512, a=None, b=None):
    if a is None:
        a = torch.randn((M, N), dtype=torch.float32, device='cuda')
    if b is None:
        b = torch.randn((M, N), dtype=torch.float32, device='cuda')

    c = torch.empty((M, N), dtype=torch.float32, device='cuda')

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    # print(grid)
    add_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return c