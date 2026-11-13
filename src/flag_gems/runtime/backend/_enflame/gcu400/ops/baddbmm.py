# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .bmm import bmm
from .mul import mul

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# bmm.py / linalg_multi_dot / linalg_cond vendor ops.
#
# Measured on L600 (bench kernel mode, do_bench median, cold L2): the Triton
# GEMM only beats the native vendor kernel while the problem is launch/overhead
# bound; once the native primo++ pipeline saturates it is ~2x faster than any
# swept Triton config (see bmm.py's anchor table). baddbmm shares the same
# GEMM core, so the same dispatch line applies:
#   (2,384^3)    Triton 1.15-1.45x (baseline run, this op)
#   (2,4096^3)   Triton 0.46-0.71x   (16,1024^3) Triton 0.55-0.73x
#   (16,2048^3)  Triton 0.51-0.73x   (16,4096^3) Triton 0.49-0.69x
# The kernels below are captured at import time - before any use_gems()
# registration replaces the PrivateUse1 entry - so call_boxed runs the
# original device kernel bit-identically with zero dispatch overhead
# (probe-verified: kwargs alpha/beta/out, 1-D broadcast bias, non-contiguous
# and fp32/fp16 inputs all torch.equal vs plain torch.baddbmm).
# Dispatch line: total work B*M*N*K >= 2^29 routes to the native kernel;
# both sides of the line have measured anchor points (see table above).
# ---------------------------------------------------------------------------

_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_BADDBMM = None
_NATIVE_BADDBMM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_BADDBMM = torch.library.get_kernel(
            torch.ops.aten.baddbmm.default, runtime.device.dispatch_key
        )
        _NATIVE_BADDBMM_OUT = torch.library.get_kernel(
            torch.ops.aten.baddbmm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_BADDBMM = None
        _NATIVE_BADDBMM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_NATIVE_WORK_THRESHOLD = 1 << 29


def _should_delegate_to_native(bias, A, B):
    if _NATIVE_BADDBMM is None:
        return False
    if A.device.type != runtime.device.name:
        return False
    Batch, M, K = A.shape
    N = B.shape[2]
    return Batch * M * N * K >= _NATIVE_WORK_THRESHOLD


def _select_baddbmm_config(Batch, M, N, K, dtype):
    # 按 dtype 与 shape 档位确定性选择 GEMM tile，替代 libtuner autotune。
    # 依据同源扫描结论（bmm.py）：BM=128/BN=256 为通用最优；fp16/bf16 偏
    # BK=256/w=4，fp32 偏 BK=128/w=2；超大 batch 的大方阵需增大 BM 或减小 BK
    # 以避免 grid.x 超出硬件上限并保持流水效率；极小 K / 极小 shape 用小 tile。
    # 小 shape 档（委托线下唯一 Triton 档）实测：fp16/bf16 用 (128,128,256,w4,ns2)
    # （(2,384^3) 直测 12.1µs vs native 17.7µs = 1.47×，(128,128,128) 档 12.8µs）；
    # fp32 用 (64,128,128,w4,ns2)（17.6µs vs native 24.4µs = 1.39×）。
    num_stages = 3
    if K < 128:
        block_m, block_n, block_k, num_warps = 64, 64, 32, 4
    elif M <= 512 and N <= 512 and K <= 512:
        if dtype == torch.float32:
            block_m, block_n, block_k, num_warps, num_stages = 64, 128, 128, 4, 2
        else:
            block_m, block_n, block_k, num_warps, num_stages = (
                128,
                128,
                256 if K >= 256 else 128,
                4,
                2,
            )
    elif M >= 4096 and N >= 4096 and Batch >= 8:
        if dtype == torch.float32:
            block_m, block_n, block_k, num_warps = 256, 128, 128, 4
        else:
            block_m, block_n, block_k, num_warps = 128, 256, 128, 2
    elif dtype == torch.float32:
        block_m, block_n, block_k, num_warps = 128, 256, 128, 2
    else:
        block_m, block_n, block_k, num_warps = 128, 256, 256, 4

    # 精确判定：只有三个维度都能被实际选中的 tile 整除时，load/store 才无越界，
    # 此时在编译期裁剪出无 mask 快速路径；否则回退到带 mask 的通用路径。
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    return block_m, block_n, block_k, num_warps, num_stages, even


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    C,
    bias,
    alpha,
    beta,
    Batch,
    M,
    N,
    K,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch

    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + pid_b * stride_ab
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    # EVEN 是编译期常量，编译器据此静态裁剪为单个循环，
    # 避免运行时分支打断 tl.dot 流水（运行时分支会令 dot 退化为标量计算）。
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (
        C
        + pid_b * stride_cb
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )
    # bias 以原 stride 参与 epilogue（0-stride 即广播），免 broadcast_to+
    # contiguous 的整张 (B,M,N) 物化拷贝。
    bias_ptrs = (
        bias
        + pid_b * stride_bias_b
        + offs_cm[:, None] * stride_bias_m
        + offs_cn[None, :] * stride_bias_n
    )

    if EVEN:
        bi = tl.load(bias_ptrs)
        out = acc * alpha + bi * beta
        tl.store(c_ptrs, out.to(C.dtype.element_ty))
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        bi = tl.load(bias_ptrs, mask=c_mask, other=0.0)
        out = acc * alpha + bi * beta
        tl.store(c_ptrs, out.to(C.dtype.element_ty), mask=c_mask)


def _launch_baddbmm(A, B, bias, out, Batch, M, N, K, alpha, beta):
    # 集中处理 config 分派与 kernel 启动，供 baddbmm / baddbmm_out 共用。
    block_m, block_n, block_k, num_warps, num_stages, even = _select_baddbmm_config(
        Batch, M, N, K, A.dtype
    )
    if bias.dim() == 3 and bias.shape == (Batch, M, N):
        bbias = bias
    else:
        bbias = bias.broadcast_to((Batch, M, N))
    grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    with torch_device_fn.device(A.device):
        baddbmm_kernel[grid](
            A,
            B,
            out,
            bbias,
            alpha,
            beta,
            Batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bbias.stride(0),
            bbias.stride(1),
            bbias.stride(2),
            dot_out_dtype=tl.float64 if A.dtype == torch.float64 else tl.float32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            EVEN=even,
            num_warps=num_warps,
            num_stages=num_stages,
        )


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_ENFLAME BADDBMM_FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        # 饱和段（大工作量）委托 native kernel：bit 级一致、零派发开销；
        # 原样透传（native 自带任意 stride / 广播 bias 支持）。forward 内
        # autograd 已被 Function 机制隔离，call_boxed 只跑设备 kernel。
        if _should_delegate_to_native(bias, A, B):
            return _NATIVE_BADDBMM.call_boxed(
                _DEVICE_KEYSET, bias, A, B, beta=beta, alpha=alpha
            )

        batch, M, K = A.shape
        _, _, N = B.shape
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        _launch_baddbmm(A, B, bias, out, batch, M, N, K, alpha, beta)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ENFLAME BADDBMM_BACKWARD")
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def compute_bias_grad(d_output, beta, bias):
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def compute_A_grad(d_output, B, alpha):
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = bmm(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = bmm(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = bmm(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = bmm(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS_ENFLAME BADDBMM_OUT")
    batch, M, K = A.shape
    _, _, N = B.shape
    assert (
        out.shape == (batch, M, N) and out.dtype == A.dtype
    ), "Incompatible output shape or dtype for baddbmm.out"
    if _should_delegate_to_native(bias, A, B) and _NATIVE_BADDBMM_OUT is not None:
        return _NATIVE_BADDBMM_OUT.call_boxed(
            _DEVICE_KEYSET, bias, A, B, beta=beta, alpha=alpha, out=out
        )
    _launch_baddbmm(A, B, bias, out, batch, M, N, K, alpha, beta)
    return out


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(bias, A, B, beta, alpha)
