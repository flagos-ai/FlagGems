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
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import broadcastable_to, libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
)
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["mm"])
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    A,
    B,
    bias,
    C,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Visit neighboring M tiles before advancing N to improve B-tile reuse.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A += ram[:, None] * stride_am + rk[None, :] * stride_ak
    B += rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
            b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(
                A,
                mask=(ram < M)[:, None] & (rk < k_remaining)[None, :],
                other=0.0,
            )
            b = tl.load(
                B,
                mask=(rk < k_remaining)[:, None] & (rbn < N)[None, :],
                other=0.0,
            )
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    C += ram[:, None] * stride_cm + rbn[None, :] * stride_cn
    mask = (ram < M)[:, None] & (rbn < N)[None, :]
    if BIAS_IS_VECTOR:
        # Load a 1-D bias once per output-column tile.
        bias_tile = tl.load(
            bias + stride_in * rbn,
            mask=rbn < N,
            other=0.0,
        )[None, :]
    elif BIAS_IS_SCALAR:
        bias_tile = tl.load(bias)
    else:
        bias += stride_im * ram[:, None] + stride_in * rbn[None, :]
        bias_tile = tl.load(bias, mask=mask, other=0.0)
    acc = acc * alpha + bias_tile.to(acc.dtype) * beta
    tl.store(C, acc.to(C.dtype.element_ty), mask=mask)


def _launch_addmm(bias, mat1, mat2, out, alpha, beta):
    M, K = mat1.shape
    _, N = mat2.shape
    # Keep row- or column-contiguous views and materialize only general strides.
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()

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
    dot_out_dtype = tl.float32
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META.get("SPLIT_K", 1),
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
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_ASCEND ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M = mat1.shape[0]
    N = mat2.shape[1]
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    return _launch_addmm(bias, mat1, mat2, out, alpha, beta)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_ASCEND ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M = mat1.shape[0]
    N = mat2.shape[1]
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    return _launch_addmm(bias, mat1, mat2, out, alpha, beta)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_ASCEND ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_ASCEND ADDMM_DTYPE_OUT")
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

    bias_c = bias.to(out_dtype)
    return addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)
