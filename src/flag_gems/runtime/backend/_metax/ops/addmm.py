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
from flag_gems.runtime.backend._metax import heuristics_config_utils as _hcu
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
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["addmm"])
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
    # PyTorch ignores bias, including NaN and Inf values, when beta is zero.
    if beta == 0:
        result = accumulator * alpha
    else:
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
        result = accumulator * alpha + bias_tile.to(accumulator.dtype) * beta
    tl.store(c_ptrs, result.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_fallback_kernel(
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
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid // N
    col = pid % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = 0.0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        a = tl.load(
            a_ptr + row * stride_am + k_offsets * stride_ak,
            mask=k_offsets < K,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + k_offsets * stride_bk + col * stride_bn,
            mask=k_offsets < K,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(a * b, axis=0)

    if beta == 0:
        result = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(i_ptr + col * stride_in)
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            bias = tl.load(i_ptr + row * stride_im + col * stride_in)
        result = accumulator * alpha + bias.to(tl.float32) * beta
    tl.store(
        c_ptr + row * stride_cm + col * stride_cn,
        result.to(c_ptr.dtype.element_ty),
    )


def _prepare_bias(bias, out):
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == out.shape[1]
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        return bias, 0, bias.stride(0), True, False
    if bias_is_scalar:
        return bias, 0, 0, False, True
    bias = bias.broadcast_to(out.shape)
    return bias, bias.stride(0), bias.stride(1), False, False


def _fallback_addmm(bias, mat1, mat2, out, beta, alpha):
    M, K = mat1.shape
    N = mat2.shape[1]
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    if M == 0 or N == 0:
        return out

    bias, stride_im, stride_in, bias_is_vector, bias_is_scalar = _prepare_bias(
        bias, out
    )
    block_size_k = min(256, triton.next_power_of_2(K)) if K > 0 else 1
    with torch_device_fn.device(mat1.device):
        addmm_fallback_kernel[(M * N,)](
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
            stride_im,
            stride_in,
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_K=block_size_k,
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
        )
    return out


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

    # Avoid MetaX dot lowering for output dimensions smaller than one tile.
    MIN_TILE = 32
    if M < MIN_TILE or N < MIN_TILE:
        logger.debug(
            "GEMS_METAX ADDMM using scalar fallback (small M=%s or N=%s)", M, N
        )
        return _fallback_addmm(bias, mat1, mat2, out, beta, alpha)

    fallback_bias = bias
    fallback_mat1 = mat1
    fallback_mat2 = mat2
    fallback_out = out
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
    bias, bias_stride_m, bias_stride_n, bias_is_vector, bias_is_scalar = _prepare_bias(
        bias, out
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    try:
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
    except RuntimeError as e:
        # Retry without dot tiling when MetaX async-pipeline lowering rejects
        # a shape/config combination.
        logger.warning(
            "GEMS_METAX ADDMM kernel compilation failed for shape (%s,%s,%s), "
            "using scalar fallback: %s",
            M,
            N,
            K,
            e,
        )
        return _fallback_addmm(
            fallback_bias,
            fallback_mat1,
            fallback_mat2,
            fallback_out,
            beta,
            alpha,
        )


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

    # beta=0 must not read bias; otherwise cast it directly to the output dtype.
    bias_c = bias if beta == 0 else bias.to(out_dtype)
    return _addmm_impl(bias_c, mat1, mat2, out, beta, alpha)
