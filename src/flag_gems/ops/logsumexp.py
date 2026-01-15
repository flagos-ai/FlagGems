import logging

import torch
import triton
import triton.language as tl

from typing import Optional
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def logsumexp_kernel(
    input_ptr,      # *fp16 / *fp32
    output_ptr,     # *fp32
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # 当前 program 负责的 M 行
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 初始化 max 和 sum
    m_val = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    z_val = tl.zeros((BLOCK_M,), tl.float32)

    # ===== 第一遍：算 max =====
    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)

        offsets = (
            m_offsets[:, None] * N * K
            + n_offsets[None, :] * K
            + pid_k
        )

        mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
        x = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)

        m_val = tl.maximum(m_val, tl.max(x, axis=1))

    # ===== 第二遍：算 sum(exp(x - m)) =====
    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)

        offsets = (
            m_offsets[:, None] * N * K
            + n_offsets[None, :] * K
            + pid_k
        )

        mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
        x = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)

        z_val += tl.sum(tl.exp(x - m_val[:, None]), axis=1)

    # ===== 写回 logsumexp =====
    out = m_val + tl.log(z_val)

    out_offsets = m_offsets * K + pid_k
    out_mask = m_offsets < M
    tl.store(output_ptr + out_offsets, out, mask=out_mask)

@libentry()
@triton.jit
def logsumexp_backward_kernel(
    input_ptr,        # *fp16 / *fp32, [M, N, K]
    out_ptr,          # *fp32, [M, K]   (logsumexp)
    grad_out_ptr,     # *fp32, [M, K]
    grad_in_ptr,      # *fp32, [M, N, K]
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 读取 logsumexp 和上游梯度
    out_offsets = m_offsets * K + pid_k
    mask_m = m_offsets < M

    lse = tl.load(out_ptr + out_offsets, mask=mask_m, other=0.0)
    grad_out = tl.load(grad_out_ptr + out_offsets, mask=mask_m, other=0.0)

    # 遍历 N
    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)

        offsets = (
            m_offsets[:, None] * N * K
            + n_offsets[None, :] * K
            + pid_k
        )

        mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)

        x = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)

        # dx = grad_out * exp(x - logsumexp)
        dx = grad_out[:, None] * tl.exp(x - lse[:, None])

        tl.store(grad_in_ptr + offsets, dx, mask=mask)


def logsumexp(self, dim: Optional[int|tuple[int]], keepdim: bool = False) -> torch.Tensor:
    logger.debug("OPS LOGSUMEXP")

    if isinstance(dim, tuple) or isinstance(dim, list) and len(dim) == 1:
        dim = dim[0]

    dim = dim % self.ndim
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    K = self.numel() // (M * N)

    x = self.contiguous()
    out = torch.empty((M, K), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(x.device):
        logsumexp_kernel[grid](
            x,
            out,
            M,
            N,
            K,
            BLOCK_M=8,
            BLOCK_N=64,
            num_warps=8,
        )

    if keepdim:
        out_shape = list(self.shape)
        out_shape[dim] = 1
        out = out.view(*out_shape)
    else:
        out_shape = list(self.shape)
        out_shape.pop(dim)
        out = out.view(*out_shape)
    
    out = out.to(self.dtype)

    return out


def logsumexp_backward(self, grad_output, output, dim, input_dtype, keep_dim: bool = False):
    logger.info("OPS LOGSUMEXP BACKWARD")

    if isinstance(dim, tuple) or isinstance(dim, list) and len(dim) == 1:
        dim = dim[0]

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim

    if keep_dim:
        grad_output = grad_output.squeeze(dim)
        output = output.squeeze(dim)

    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    K = self.numel() // (M * N)

    x = self.contiguous()
    grad_output = grad_output.contiguous()
    output = output.contiguous()

    grad_in = torch.empty_like(x, dtype=input_dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )

    with torch_device_fn.device(x.device):
        logsumexp_backward_kernel[grid](
            x,
            output,
            grad_output,
            grad_in,
            M,
            N,
            K,
            BLOCK_M=8,
            BLOCK_N=64,
            num_warps=8,
        )

    return grad_in