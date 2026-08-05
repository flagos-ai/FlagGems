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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_SIZE_M"] * args["BLOCK_SIZE_N"])
        ).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
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
    GROUP_M: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
):
    if UPGRADE:
        pid = ext.program_id(0)
    else:
        pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Visit neighboring M tiles before advancing N to improve B-tile reuse.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    if UPGRADE_A_OFFS:
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    else:
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    if UPGRADE_B_OFFS:
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    else:
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if UPGRADE_C_OFFS:
        store_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        store_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    else:
        store_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        store_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * store_m[:, None] + stride_cn * store_n[None, :]
    mask = (store_m[:, None] < M) & (store_n[None, :] < N)
    if BIAS_IS_VECTOR:
        bias_tile = tl.load(
            i_ptr + stride_in * store_n,
            mask=store_n < N,
            other=0.0,
        )[None, :]
    elif BIAS_IS_SCALAR:
        bias_tile = tl.load(i_ptr)
    else:
        i_ptrs = i_ptr + stride_im * store_m[:, None] + stride_in * store_n[None, :]
        bias_tile = tl.load(i_ptrs, mask=mask, other=0.0)
    accumulator = accumulator * alpha + bias_tile.to(accumulator.dtype) * beta
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


def _addmm_impl(bias, mat1, mat2, out, beta, alpha):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    logger.debug(
        "GEMS_METAX ADDMM, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        mat2.stride(0) == 1,
        bias.ndim > 0 and bias.stride(0) == 1,
    )

    # MetaX lowers the GEMM load efficiently when B is contiguous in N.
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(1) != 1:
        mat2 = mat2.contiguous()
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    # Keep vector/scalar bias compact; broadcast strides cover other valid shapes.
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == N
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        bias_stride_m = 0
        bias_stride_n = bias.stride(0)
    elif bias_is_scalar:
        bias_stride_m = 0
        bias_stride_n = 0
    else:
        bias = bias.broadcast_to(out.shape)
        bias_stride_m = bias.stride(0)
        bias_stride_n = bias.stride(1)
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
            bias_stride_m,
            bias_stride_n,
            out.stride(0),
            out.stride(1),
            GROUP_M=8,
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_METAX ADDMM")
    return _addmm_impl(bias, mat1, mat2, None, beta, alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_METAX ADDMM_OUT")
    return _addmm_impl(bias, mat1, mat2, out, beta, alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_METAX ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_METAX ADDMM_DTYPE_OUT")
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

    return _addmm_impl(bias.to(out_dtype), mat1, mat2, out, beta, alpha)
