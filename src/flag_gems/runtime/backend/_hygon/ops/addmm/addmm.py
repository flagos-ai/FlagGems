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
from contextlib import nullcontext
from numbers import Number

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.flagtune import _flagtune_state_key
from flag_gems.utils import broadcastable_to
from flag_gems.utils.libentry import LibTunerRunMode

from ..bmm.bmm import check_out_autograd, copy_tensor
from ..bmm.splitk import launch_small_splitk, small_splitk_count
from .kernels import (
    _transpose_addmm_b,
    addmm_column_kernel,
    addmm_fp32_kernel,
    addmm_kernel,
    addmm_medium_kernel,
    addmm_rowmajor_kernel,
    addmm_small_kernel,
    dense_addmm_medium,
    dense_addmm_small,
    dense_addmm_small_fp32,
)

logger = logging.getLogger(__name__)
_VECTOR_BIAS_LAUNCHES = {}


def _launch_vector_bias(args, options, grid):
    kernel = addmm_kernel
    tuner = kernel.fn
    specialize = kernel._tensor_spec_fn
    tensors = args[:4]
    alpha, beta = args[4:6]
    if (
        specialize is None
        or kernel._flagtune_tuners
        or tuner._run_mode != LibTunerRunMode.NORMAL
        or getattr(tuner, "shared_config_pre_hook", None) is not None
        or type(alpha) not in (int, float, bool)
        or type(beta) not in (int, float, bool)
        or any(type(t) is not torch.Tensor for t in tensors)
    ):
        kernel.run(*args, grid=grid, warmup=False, **options)
        return
    signature = tuple((t.device, t.dtype, specialize(t, align=True)) for t in tensors)
    key = (
        _flagtune_state_key(),
        signature,
        args[4:],
        tuple(options.items()),
        type(alpha),
        type(beta),
    )
    plan = _VECTOR_BIAS_LAUNCHES.get(key)
    if plan is not None:
        launch, constants = plan
        launch(*args, *constants)
        return

    binary, meta = kernel.run(*args, grid=grid, warmup=False, **options)
    if all(config.pre_hook is None for config in tuner.configs):
        names = kernel.arg_names
        launch_grid = grid({**dict(zip(names, args)), **options, **meta})
        bound = {**meta, **options}
        constants = tuple(bound[name] for name in names[len(args) :])
        if len(_VECTOR_BIAS_LAUNCHES) >= 128:
            _VECTOR_BIAS_LAUNCHES.clear()
        _VECTOR_BIAS_LAUNCHES[key] = (binary[(launch_grid + (1, 1))[:3]], constants)


def _scalar_eq(value, target):
    if isinstance(value, (int, float)):
        return value == target
    return isinstance(value, Number) and value == target


def _validate_addmm_shapes(bias, mat1, mat2):
    m, k = mat1.shape
    k2, n = mat2.shape
    assert k == k2, "Incompatible dimensions"
    output_shape = (m, n)
    if bias.shape != output_shape:
        assert broadcastable_to(bias.shape, output_shape), "Incompatible input shape"
    return m, n, k


def _is_col_major(tensor):
    return tensor.dim() >= 1 and tensor.stride(0) == 1


def _finish_bias_only(out, bias, beta):
    bias = bias.broadcast_to(out.shape)
    return copy_tensor(bias, out, beta, zero=_scalar_eq(beta, 0))


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

    if logger.isEnabledFor(logging.DEBUG):
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
    if not mat1.is_contiguous():
        contiguous = torch.empty(mat1.shape, dtype=mat1.dtype, device=mat1.device)
        copy_tensor(mat1, contiguous)
        mat1 = contiguous
    beta_zero = _scalar_eq(beta, 0)
    bias_scalar = False
    bias_row = False
    bias_col = False
    if not beta_zero:
        bias_shape = bias.shape
        if bias_shape == out.shape:
            stride_im, stride_in = bias.stride()
        elif not bias_shape:
            stride_im = stride_in = 0
        elif len(bias_shape) == 1:
            stride_im = 0
            stride_in = bias.stride(0) if bias_shape[0] == N else 0
        else:
            # The API already validated broadcasting. The kernel can address
            # the original storage using zero strides in expanded dimensions.
            stride_im = bias.stride(0) if bias_shape[0] == M else 0
            stride_in = bias.stride(1) if bias_shape[1] == N else 0
        bias_scalar = stride_im == 0 and stride_in == 0
        bias_row = stride_im == 0 and stride_in != 0
        bias_col = stride_im != 0 and stride_in == 0
    else:
        stride_im = 0
        stride_in = 0

    grid = lambda META: (
        ((M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"])
        * ((N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"]),
    )
    device = mat1.device
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device.index
        else torch_device_fn.device(device)
    )
    with guard:
        if kernel is addmm_rowmajor_kernel or kernel is addmm_column_kernel:
            # Materialize on every call: inputs may be mutated between launches.
            transposed_b = torch.empty((N, K), device=mat2.device, dtype=mat2.dtype)
            _transpose_addmm_b[((K + 63) // 64, (N + 63) // 64)](
                mat2, transposed_b, K, N, num_warps=4
            )
            mat2 = transposed_b.t()
        args = (
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
        )
        options = dict(
            ALPHA_ONE=_scalar_eq(alpha, 1),
            BETA_ZERO=beta_zero,
            BETA_ONE=_scalar_eq(beta, 1),
            BIAS_SCALAR=bias_scalar,
            BIAS_ROW=bias_row,
            BIAS_COL=bias_col,
            B_CONTIGUOUS=mat2.stride(0) == N and mat2.stride(1) == 1,
            ALLOW_TF32=_allow_tf32(mat1, mat2),
            IS_FP64=mat1.dtype == torch.float64,
            **(
                {"ALPHA_FP64": alpha, "BETA_FP64": beta}
                if mat1.dtype == torch.float64
                else {}
            ),
        )
        if (
            kernel is addmm_kernel
            and bias_row
            and bias.ndim == 1
            and mat1.dtype in (torch.float16, torch.bfloat16)
        ):
            _launch_vector_bias(args, options, grid)
        else:
            kernel[grid](*args, **options)
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    m, n, k = _validate_addmm_shapes(bias, mat1, mat2)
    device, dtype = mat1.device, mat1.dtype
    out = torch.empty((m, n), device=device, dtype=dtype)
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)
    small = 128 <= min_dim and max_dim <= 512
    half = dtype in (torch.float16, torch.bfloat16)
    # Keep the dense epilogue in the existing small and medium tile domains.
    medium = (
        half
        and 512 <= min_dim
        and max_dim <= 1536
        and triton.cdiv(m, 256) * triton.cdiv(n, 256) < 24
    )
    if (
        (
            (
                small
                and (half or (dtype == torch.float32 and not _allow_tf32(mat1, mat2)))
            )
            or medium
        )
        and _scalar_eq(alpha, 1)
        and _scalar_eq(beta, 1)
        and mat2.dtype == dtype == bias.dtype
        and mat2.device == device == bias.device
        and bias.shape == (m, n)
        and mat1.is_contiguous()
        and mat2.is_contiguous()
        and bias.is_contiguous()
    ):
        guard = (
            nullcontext()
            if torch_device_fn.current_device() == device.index
            else torch_device_fn.device(device)
        )
        with guard:
            if dtype == torch.float32:
                dense_addmm_small_fp32[(triton.cdiv(m, 32) * triton.cdiv(n, 32),)](
                    mat1, mat2, bias, out, m, n, k, num_warps=4, num_stages=2
                )
            else:
                kernel = dense_addmm_small if small else dense_addmm_medium
                kernel[
                    lambda cfg: (
                        triton.cdiv(m, cfg["BLOCK_SIZE_M"])
                        * triton.cdiv(n, cfg["BLOCK_SIZE_N"]),
                    )
                ](mat1, mat2, bias, out, m, n, k, not small)
        return out
    return _launch_addmm(bias, mat1, mat2, out, beta=beta, alpha=alpha)


def _check_addmm_out(bias, mat1, mat2, out, out_dtype):
    check_out_autograd("addmm", bias, mat1, mat2, out)
    if not (bias.device == mat1.device == mat2.device == out.device):
        raise RuntimeError("addmm inputs and output must be on the same device")
    if mat1.dtype != mat2.dtype or out.dtype != out_dtype:
        raise RuntimeError("addmm input or output dtype does not match")
    if bias.dtype not in (mat1.dtype, out_dtype):
        raise RuntimeError("addmm bias dtype does not match the inputs or output")
    m, n, _ = _validate_addmm_shapes(bias, mat1, mat2)
    if out.shape != (m, n):
        raise RuntimeError("Incompatible output shape")


def _launch_addmm_out(bias, mat1, mat2, out, beta, alpha):
    if any(torch._C._overlaps(out, value) for value in (bias, mat1, mat2)):
        temporary = torch.empty(out.shape, device=out.device, dtype=out.dtype)
        _launch_addmm(bias, mat1, mat2, temporary, beta=beta, alpha=alpha)
        return copy_tensor(temporary, out)
    return _launch_addmm(bias, mat1, mat2, out, beta=beta, alpha=alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    M, N, K = _validate_addmm_shapes(bias, mat1, mat2)
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    _check_addmm_out(bias, mat1, mat2, out, mat1.dtype)
    return _launch_addmm_out(bias, mat1, mat2, out, beta, alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_HYGON ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_HYGON ADDMM_DTYPE_OUT")
    _check_addmm_out(bias, mat1, mat2, out, out_dtype)
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

    bias_c = bias
    if not _scalar_eq(beta, 0) and bias.dtype != out_dtype:
        bias_c = torch.empty(bias.shape, dtype=out_dtype, device=bias.device)
        copy_tensor(bias, bias_c)
    return _launch_addmm_out(bias_c, mat1, mat2, out, beta, alpha)


def _select_addmm_kernel(bias, mat1, mat2):
    m, k = mat1.shape
    n = mat2.shape[1]
    matrix_bias = (
        bias.ndim == 2
        and bias.shape == (m, n)
        and (bias.stride(0) != 0)
        and (bias.stride(1) != 0)
    )
    if not matrix_bias or mat2.stride() != (n, 1) or mat1.dtype != mat2.dtype:
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


def _launch_addmm(bias, mat1, mat2, out, *, beta=1, alpha=1):
    m, k = mat1.shape
    n = mat2.shape[1]
    splits = small_splitk_count(1, m, n, k, mat1.dtype)
    if splits and not _scalar_eq(alpha, 0):
        device = mat1.device
        guard = (
            nullcontext()
            if torch_device_fn.current_device() == device.index
            else torch_device_fn.device(device)
        )
        with guard:
            launch_small_splitk(
                mat1,
                mat2,
                out,
                bias if bias.shape == out.shape else bias.broadcast_to(out.shape),
                alpha,
                beta,
                splits=splits,
            )
        return out
    return _launch_addmm_generic(bias, mat1, mat2, out, beta=beta, alpha=alpha)
