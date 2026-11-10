"""t9 4/5 候选：addmm_dtype / addmm_dtype_out 的 gcu400 分档实现。

生产 flag_gems.addmm_dtype 走公共 flag_gems/ops/addmm.py 的 @libtuner
autotune kernel；gcu400/ops/addmm.py 已有静态分档 kernel，但只注册了
addmm / addmm_out。本实现保留公共 addmm_dtype_out 的全部校验与 fp32
累加语义，仅对「规整大 GEMM + 连续输入」改走 gcu400 kernel（bn=64，
省掉 bias.broadcast_to(...).contiguous() 多余拷贝），小维度 / 非连续 /
标量或向量 bias / beta=0 一律回退公共实现，保证语义与数值不变。
"""

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.addmm import _select_addmm_out_config
from flag_gems.runtime.backend._enflame.gcu400.ops.addmm import (
    addmm_kernel as _g4_addmm_kernel,
)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    """分配 out_dtype 输出后转交 addmm_dtype_out。"""
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    """校验与公共实现一致；按 shape / 内存布局分派到静态分档 kernel。"""
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    # beta=0 时 PyTorch 语义忽略 bias（含 NaN/Inf），保持公共实现的处理方式。
    bias_c = bias if beta == 0 else bias.to(out_dtype)
    M, K = mat1.shape
    N = mat2.shape[1]

    # 只有规整大 GEMM 且所有张量连续、bias 已是完整 (M,N) 矩阵时才走
    # gcu400 kernel；其余情况回退公共实现，避免 stride / broadcast 语义差异。
    use_g4 = (
        beta != 0
        and M >= 64
        and N >= 64
        and K >= 64
        and mat1.is_contiguous()
        and mat2.is_contiguous()
        and out.is_contiguous()
        and bias_c.is_contiguous()
        and tuple(bias_c.shape) == (M, N)
    )
    if not use_g4:
        from flag_gems.ops.addmm import addmm_out as _public_addmm_out

        return _public_addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)

    block_m, block_n, block_k, num_warps, num_stages = _select_addmm_out_config(
        M, N, K, mat1.dtype
    )
    # C 族 11 轮：bn=64 三次复现 1.386/1.391/1.398x，优于默认 256 的 1.298x。
    block_n = 64
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    with torch_device_fn.device(mat1.device):
        _g4_addmm_kernel[grid](
            mat1,
            mat2,
            bias_c,
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
            bias_c.stride(0),
            bias_c.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_M=8,
            EVEN=even,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
