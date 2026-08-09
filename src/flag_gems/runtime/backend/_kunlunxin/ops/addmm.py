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
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


autotune_decorator = triton.autotune(
    configs=[],
    generate_configs="addmm",
    key=["M", "N", "K"],
)


KLX_USE_AUTOTUNE = os.environ.get("KLX_USE_AUTOTUNE", "1") == "1"

if not KLX_USE_AUTOTUNE:

    # XPU tile sweep probe (2026-08-13, XPU 7, 4 unique core shapes x 3 dtypes,
    # direct do_bench): BM=BN=256 / warps=8 / stages=3 wins on all dtypes; the
    # reduction tile BK is dtype-dependent on this backend - fp16 prefers BK=256
    # (4096^3: 0.83x vs 0.56x at BK=128), while bf16/fp32 prefer BK=128
    # (4096^3: 0.81x/0.95x vs 0.54x/0.29x at BK=256). fp32 BK=256 collapses
    # (4.7ms vs 1.46ms on 4096^3). Baseline (128x128x128, no swizzle, warps=4)
    # equal-weight mean speedup ~0.53x vs candidate ~0.67x direct A/B.
    # Small shapes (M,N <= 512) keep the 128-tile warps=4 config: the 256-tile
    # warps=8 launch overhead regresses 384^3 by ~6%.

    def heur_block_m(args):
        M = args["M"]
        if M <= 512:
            return 128
        return 256

    def heur_block_n(args):
        N = args["N"]
        if N <= 512:
            return 128
        return 256

    def heur_block_k(args):
        # Wrapper passes BLOCK_K_CHOICE (fp16 -> 256, else 128).
        if args.get("BLOCK_K_CHOICE", 128) == 256:
            return 256
        return 128

    def heur_warps(args):
        if args["M"] <= 512 and args["N"] <= 512:
            return 4
        return 8

    autotune_decorator = triton.heuristics(
        {
            "BLOCK_SIZE_M": heur_block_m,
            "BLOCK_SIZE_N": heur_block_n,
            "BLOCK_SIZE_K": heur_block_k,
            "num_warps": heur_warps,
        }
    )


@libentry()
@autotune_decorator
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
    GROUP_M: tl.constexpr,
    BLOCK_K_CHOICE,
):
    pid = ext.program_id(0)
    if GROUP_M > 1:
        grid_m = tl.cdiv(M, BLOCK_SIZE_M)
        grid_n = tl.cdiv(N, BLOCK_SIZE_N)
        # re-order program ID for better L2 reuse along the N dimension
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
    else:
        pid_m = ext.program_id(1)
        pid_n = ext.program_id(2)

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
    # Let tl.store convert to the output pointer dtype. The dtype-out variant
    # may use fp32 output with fp16/bf16 inputs and an input-dtype bias.
    tl.store(c_ptrs, accumulator, mask=c_mask)


def addmm(bias, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    # mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)

    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
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
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            num_stages=3,
        )
    return out


def addmm_out(bias, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_KUNLUNXIN ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"

    mat1 = mat1.contiguous()
    bias = bias.broadcast_to(out.shape)

    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
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
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            num_stages=3,
        )
    return out


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]), device=mat1.device, dtype=out_dtype
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_KUNLUNXIN ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the provided out tensor dtype"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")
    if mat1.shape[1] != mat2.shape[0]:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")
    if not broadcastable_to(bias.shape, (mat1.shape[0], mat2.shape[1])):
        raise RuntimeError("self is not broadcastable to the result shape")
    if out.shape != (mat1.shape[0], mat2.shape[1]):
        raise RuntimeError("out has an incompatible shape")

    M, K = mat1.shape
    _, N = mat2.shape
    mat1 = mat1.contiguous()
    bias = bias.broadcast_to(out.shape)
    block_k_choice = 256 if mat1.dtype == torch.float16 else 128
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
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
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            num_stages=3,
        )
    return out
