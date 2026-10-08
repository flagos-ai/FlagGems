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
from numbers import Number

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to

from .kernels import (
    _transpose_addmm_b,
    addmm_column_kernel,
    addmm_fp32_kernel,
    addmm_kernel,
    addmm_medium_kernel,
    addmm_rowmajor_kernel,
    addmm_small_kernel,
)
from .ring import addmm_ring_kernel

logger = logging.getLogger(__name__)


def _scalar_eq(value, target):
    return isinstance(value, Number) and value == target


def _validate_addmm_shapes(bias, mat1, mat2):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    output_shape = (mat1.shape[0], mat2.shape[1])
    assert broadcastable_to(bias.shape, output_shape), "Incompatible input shape"


def _is_col_major(tensor):
    return tensor.dim() >= 1 and tensor.stride(0) == 1


def _finish_bias_only(out, bias, beta):
    bias = bias.broadcast_to(out.shape)
    if _scalar_eq(beta, 0):
        return out.zero_()
    if _scalar_eq(beta, 1):
        return out.copy_(bias)
    return torch.mul(bias, beta, out=out)


def _allow_tf32(mat1, mat2):
    if mat1.dtype != torch.float32 or mat2.dtype != torch.float32:
        return False
    if mat1.device.type != "cuda":
        return False
    cuda_backend = getattr(torch.backends, "cuda", None)
    return bool(cuda_backend and cuda_backend.matmul.allow_tf32)


def _launch_addmm_generic(bias, mat1, mat2, out, *, beta=1, alpha=1):
    M, K = mat1.shape
    _, N = mat2.shape

    if M == 0 or N == 0:
        return out

    if K == 0 or _scalar_eq(alpha, 0):
        return _finish_bias_only(out, bias, beta)

    logger.debug(
        "GEMS ADDMM, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        _is_col_major(mat2),
        _is_col_major(bias),
    )

    kernel = _select_addmm_kernel(bias, mat1, mat2)
    mat1 = mat1.contiguous()
    beta_zero = _scalar_eq(beta, 0)
    bias_scalar = False
    bias_row = False
    bias_col = False
    if not beta_zero:
        bias = bias.broadcast_to(out.shape)
        stride_im = bias.stride(0)
        stride_in = bias.stride(1)
        bias_scalar = stride_im == 0 and stride_in == 0
        bias_row = stride_im == 0 and stride_in != 0
        bias_col = stride_im != 0 and stride_in == 0
    else:
        stride_im = 0
        stride_in = 0

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        if kernel is addmm_rowmajor_kernel or kernel is addmm_column_kernel:
            # Materialize on every call: inputs may be mutated between launches.
            transposed_b = torch.empty((N, K), device=mat2.device, dtype=mat2.dtype)
            _transpose_addmm_b[(triton.cdiv(K, 64), triton.cdiv(N, 64))](
                mat2, transposed_b, K, N, num_warps=4
            )
            mat2 = transposed_b.t()
        kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat2.stride(0),
            mat2.stride(1),
            stride_im,
            stride_in,
            out.stride(0),
            out.stride(1),
            ALPHA_ONE=_scalar_eq(alpha, 1),
            BETA_ZERO=beta_zero,
            BETA_ONE=_scalar_eq(beta, 1),
            BIAS_SCALAR=bias_scalar,
            BIAS_ROW=bias_row,
            BIAS_COL=bias_col,
            B_CONTIGUOUS=mat2.stride(0) == N and mat2.stride(1) == 1,
            ALLOW_TF32=_allow_tf32(mat1, mat2),
            IS_FP64=mat1.dtype == torch.float64,
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    _validate_addmm_shapes(bias, mat1, mat2)
    M = mat1.shape[0]
    N = mat2.shape[1]
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    return _launch_addmm(bias, mat1, mat2, out, beta=beta, alpha=alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    _validate_addmm_shapes(bias, mat1, mat2)
    M = mat1.shape[0]
    N = mat2.shape[1]
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    return _launch_addmm(bias, mat1, mat2, out, beta=beta, alpha=alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS ADDMM_DTYPE_OUT")
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

    bias_c = bias if _scalar_eq(beta, 0) else bias.to(out_dtype)
    return addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)


def _select_addmm_kernel(bias, mat1, mat2):
    m, k = mat1.shape
    n = mat2.shape[1]
    dense_bias = (
        bias.ndim == 2
        and bias.shape == (m, n)
        and (bias.stride(0) != 0)
        and (bias.stride(1) != 0)
    )
    if not dense_bias or mat2.stride() != (n, 1) or mat1.dtype != mat2.dtype:
        return addmm_kernel
    if mat1.dtype in (torch.float16, torch.bfloat16):
        if min(m, n, k) >= 128 and max(m, n, k) <= 512:
            return addmm_small_kernel
        if min(m, n, k) >= 512 and max(m, n, k) <= 1536:
            return addmm_medium_kernel
        if min(m, n, k) >= 1536 and max(m, n, k) <= 3072:
            return addmm_column_kernel
        if min(m, n, k) >= 4096 and m % 256 == 0 and n % 256 == 0 and k % 128 == 0:
            return addmm_rowmajor_kernel
    elif (
        mat1.dtype == torch.float32
        and min(m, n, k) >= 3072
        and (m % 128 == 0)
        and (n % 128 == 0)
        and (k % 64 == 0)
    ):
        return addmm_fp32_kernel
    return addmm_kernel


def _use_ring_addmm(bias, mat1, mat2, out):
    """Use the gfx936 ring for supported dense row-major GEMMs."""
    m, k = mat1.shape
    n = mat2.shape[1]
    if (
        mat1.dtype not in (torch.float16, torch.bfloat16)
        or mat2.dtype != mat1.dtype
        or out.dtype not in (mat1.dtype, torch.float32)
        or bias.ndim != 2
        or bias.shape != (m, n)
        or not all(bias.stride())
        or not mat1.is_contiguous()
        or mat2.stride() != (n, 1)
        or mat1.data_ptr() % 16
        or mat2.data_ptr() % 16
        or min(m, n) < 256
        or k < 256
        or k % 64
        or n % 8
    ):
        return False
    gm, gn = triton.cdiv(m, 256), triton.cdiv(n, 256)
    if gm * gn < 24 or 2 * m * n < gm * gn * 256**2:
        return False
    if max(m * k, k * n, gm * 256 * n) * 2 >= 2**31:
        return False
    arch = getattr(torch.cuda.get_device_properties(mat1.device), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx936"


def _launch_addmm(bias, mat1, mat2, out, *, beta=1, alpha=1):
    if _scalar_eq(alpha, 0) or not _use_ring_addmm(bias, mat1, mat2, out):
        return _launch_addmm_generic(bias, mat1, mat2, out, beta=beta, alpha=alpha)
    m, k = mat1.shape
    n = mat2.shape[1]
    with torch_device_fn.device(mat1.device):
        addmm_ring_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
            mat1,
            mat2,
            out,
            bias,
            alpha,
            beta,
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            _scalar_eq(alpha, 1),
            _scalar_eq(beta, 0),
            _scalar_eq(beta, 1),
            m,
            n,
            k,
            BF16=mat1.dtype == torch.bfloat16,
            LOGK=min(k.bit_length() - 1, 12),
            allow_flush_denorm=True,
            enable_fp_fusion=True,
        )
    return out
