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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# _linalg_eigvals / linalg_multi_dot / linalg_cond vendor ops.
#
# Measured on L600 (bench kernel mode, do_bench median w1000/r100, cold L2):
# the Triton kernel beats the native vendor GEMM only while the problem is
# launch/overhead bound; once the native primo++ pipeline saturates it is
# ~2x faster than any swept Triton config ((128,256,256) is the tile sweet
# spot, matching the matrix_power job's 50+ combo sweep):
#   (2,384^3)   Triton 1.08-1.19x   (2,512^3)  Triton 1.12-1.17x
#   (2,640^3)   Triton 0.96x        (2,768^3)  Triton 0.93x
#   (4,512^3)   Triton 0.64x        (16,384^3) Triton 0.57x
#   (16,1024^3) Triton 0.51x        (16,4096^3) Triton 0.55x
# The kernels below are captured at import time - before any use_gems()
# registration replaces the PrivateUse1 entry - so call_boxed runs the
# original device kernel bit-identically with zero dispatch overhead
# (verified 1.000-1.006x vs plain torch.bmm on the shapes above).
# Dispatch line: total work B*M*N*K >= 2^29 routes to the native kernel;
# both sides of the line have measured anchor points (see table above).
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_BMM = None
_NATIVE_BMM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_BMM = torch.library.get_kernel(
            torch.ops.aten.bmm.default, runtime.device.dispatch_key
        )
        _NATIVE_BMM_OUT = torch.library.get_kernel(
            torch.ops.aten.bmm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_BMM = None
        _NATIVE_BMM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_NATIVE_WORK_THRESHOLD = 1 << 29


def _should_delegate_to_native(A, B):
    if _NATIVE_BMM is None:
        return False
    if A.device.type != runtime.device.name:
        return False
    Batch, M, K = A.shape
    N = B.shape[2]
    return Batch * M * N * K >= _NATIVE_WORK_THRESHOLD


def _select_bmm_config(Batch, M, N, K, dtype):
    # 按 dtype 与 shape 档位确定性选择 GEMM tile，替代 libtuner autotune。
    # 依据同源扫描结论：BM=128/BN=256 为通用最优；fp16/bf16 偏 BK=256/w=4，
    # fp32 偏 BK=128/w=2；超大 batch 的大方阵需增大 BM 或减小 BK 以避免
    # grid.x 超出硬件上限并保持流水效率；极小 K / 极小 shape 用小 tile 减少浪费。
    # 小 shape 档（委托线下唯一 Triton 档）实测：fp16/bf16 用 (128,128,128,w4)；
    # fp32 在 (2,384^3) 上 (64,128,128,w4,ns2) 快 ~7%（16.0 -> 14.9µs，median）。
    num_stages = 3
    if K < 128:
        block_m, block_n, block_k, num_warps = 64, 64, 32, 4
    elif M <= 512 and N <= 512 and K <= 512:
        if dtype == torch.float32:
            block_m, block_n, block_k, num_warps, num_stages = 64, 128, 128, 4, 2
        else:
            block_m, block_n, block_k, num_warps = 128, 128, 128, 4
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
@triton.jit
def bmm_kernel(
    A,
    B,
    C,
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
    c = acc.to(C.dtype.element_ty)
    if EVEN:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _launch_bmm(A, B, out, Batch, M, N, K):
    # 集中处理 config 分派与 kernel 启动，供 bmm / bmm_out 共用。
    block_m, block_n, block_k, num_warps, num_stages, even = _select_bmm_config(
        Batch, M, N, K, A.dtype
    )
    grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    bmm_kernel[grid](
        A,
        B,
        out,
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
        dot_out_dtype=tl.float32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EVEN=even,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def bmm(A, B):
    logger.debug("GEMS_ENFLAME BMM")
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    # 饱和段（大工作量）委托 native kernel：bit 级一致、零派发开销；
    # 原样透传（native 自带任意 stride 支持，非连续输入免 contiguous 拷贝）。
    if _should_delegate_to_native(A, B):
        return _NATIVE_BMM.call_boxed(_DEVICE_KEYSET, A, B)
    Batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((Batch, M, N), dtype=A.dtype, device=A.device)
    _launch_bmm(A, B, out, Batch, M, N, K)
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS_ENFLAME BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    if _should_delegate_to_native(A, B) and _NATIVE_BMM_OUT is not None:
        return _NATIVE_BMM_OUT.call_boxed(_DEVICE_KEYSET, A, B, out=out)
    Batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    _launch_bmm(A, B, out, Batch, M, N, K)
    return out
