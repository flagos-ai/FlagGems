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

"""Weight-gradient GEMM with in-place accumulation (Apex-aligned).

Matches Apex ``fused_weight_gradient_mlp_cuda`` semantics used by Megatron
``LinearWithGradAccumulationAndAsyncCommunication`` when
``gradient_accumulation_fusion`` is enabled.

Each update performs ``main_grad += grad_output.T @ input`` (after collapsing
leading dimensions).

``wgrad_gemm_accum_fp32`` (including half/bf16 activations into fp32
``main_grad``) calls ``cublasGemmEx`` with the same layout / dtype / algo as
Apex, via a compiled CUDA extension (FlagGems ``c_operators`` when present,
otherwise ``torch.utils.cpp_extension`` JIT).  ``wgrad_gemm_accum_fp16`` uses
``torch.addmm`` (cuBLAS) for same-dtype accumulation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

import flag_gems

logger = logging.getLogger(__name__)


def _collapse_to_2d(input: torch.Tensor, grad_output: torch.Tensor):
    if input.dim() > 2:
        input_2d = input.reshape(-1, input.size(-1))
    else:
        input_2d = input

    if grad_output.dim() > 2:
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
    else:
        grad_output_2d = grad_output

    if input_2d.size(0) != grad_output_2d.size(0):
        raise RuntimeError(
            "input and grad_output must have the same number of rows after collapse"
        )

    return input_2d, grad_output_2d


def _validate_device(*tensors: torch.Tensor) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RuntimeError("All tensors must be on the same device")
    device = devices.pop()
    if device.type != flag_gems.device:
        raise RuntimeError(
            f"Expected tensors on {flag_gems.device}, but got {device.type}"
        )


@lru_cache(None)
def _load_wgrad_gemm_ext():
    """Return a module exposing ``wgrad_gemm_accum_fp32(input, grad, main)``.

    Prefer the official FlagGems C extension when installed; otherwise JIT-compile
    ``flag_gems/csrc/wgrad_gemm_accum.cpp`` (cublas headers + Torch BLAS handle,
    no ctypes / ``libcublas.so`` path hunting).
    """
    try:
        from flag_gems import c_operators

        if hasattr(c_operators, "wgrad_gemm_accum_fp32"):
            logger.debug("wgrad_gemm_accum_fp32: using flag_gems.c_operators")
            return c_operators
    except ImportError:
        pass

    try:
        from torch.ops import flag_gems as flag_gems_ops

        if hasattr(flag_gems_ops, "wgrad_gemm_accum_fp32"):
            logger.debug("wgrad_gemm_accum_fp32: using torch.ops.flag_gems")
            return flag_gems_ops
    except (ImportError, AttributeError):
        pass

    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent.parent / "csrc" / "wgrad_gemm_accum.cpp"
    if not src.is_file():
        raise RuntimeError(
            f"Missing wgrad GemmEx source: {src}. "
            "Rebuild FlagGems cpp package or restore "
            "flag_gems/csrc/wgrad_gemm_accum.cpp"
        )

    # Do NOT link -lcublas here. The extension must call the same libcublas that
    # owns PyTorch's BLAS handle (via dlsym); a second copy -> INVALID_VALUE.
    logger.info(
        "wgrad_gemm_accum_fp32: JIT-compiling CUDA extension from %s (first call only)",
        src,
    )
    return load(
        name="flag_gems_wgrad_gemm_accum",
        sources=[str(src)],
        extra_ldflags=["-ldl"],
        with_cuda=True,
        verbose=False,
    )


def _wgrad_fp32_strict_accum(
    grad_output_2d: torch.Tensor,
    input_2d: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """fp32 activations with TF32 off: match CPU fp64 reference semantics.

    GemmEx + DEFAULT_TENSOR_OP may still use TF32 tensor cores on Ampere+ even
    when ``allow_tf32=False``.  Mirror ``_ref_wgrad_gemm_accum_fp32_cpu``:
    fp64 GEMM -> cast to fp32 -> add into main_grad.
    """
    input_c = input_2d.contiguous()
    grad_c = grad_output_2d.contiguous()
    wgrad_fp32 = (grad_c.t().contiguous().double() @ input_c.double()).float()
    if main_grad.is_contiguous():
        main_grad.add_(wgrad_fp32)
        return
    weight = main_grad.contiguous()
    weight.add_(wgrad_fp32)
    main_grad.copy_(weight)


def _cublas_wgrad_gemm_accum_fp32(
    input_2d: torch.Tensor,
    grad_output_2d: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    strict_cpu_ref: bool = False,
) -> None:
    """Apex ``wgrad_gemm_accum_fp32_cuda`` layout via compiled ``cublasGemmEx``."""
    if main_grad.dtype != torch.float32:
        raise RuntimeError("main_grad must be float32 for GemmEx fp32-accum path")
    if input_2d.dtype != grad_output_2d.dtype:
        raise RuntimeError(
            "input and grad_output dtype must match, "
            f"got {input_2d.dtype} vs {grad_output_2d.dtype}"
        )

    # Test-only: fp64 GEMM to match CPU reference when TF32 is off.
    # Default (bench / training): GemmEx like Apex regardless of allow_tf32 default.
    if strict_cpu_ref and input_2d.dtype == torch.float32:
        _wgrad_fp32_strict_accum(grad_output_2d, input_2d, main_grad)
        return

    ext = _load_wgrad_gemm_ext()
    ext.wgrad_gemm_accum_fp32(input_2d, grad_output_2d, main_grad)


def _matmul_operands(
    grad_output_2d: torch.Tensor, input_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_output.T, input)`` for same-dtype ``torch.addmm``.

    Densify first, then take a transpose view so contiguous / non-contiguous
    callers share one cuBLAS ``OP_T`` path.
    """
    if not grad_output_2d.is_contiguous():
        grad_output_2d = grad_output_2d.contiguous()
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()
    return grad_output_2d.t(), input_2d


def _fused_addmm_cublas(
    main_grad: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    """Same-dtype fused ``main_grad += mat1 @ mat2`` via PyTorch cuBLAS addmm."""
    if main_grad.is_contiguous():
        torch.addmm(main_grad, mat1, mat2, beta=1, alpha=1, out=main_grad)
        return
    # Non-contiguous out: compute into a dense buffer then copy back.
    weight = main_grad.contiguous()
    torch.addmm(weight, mat1, mat2, beta=1, alpha=1, out=weight)
    main_grad.copy_(weight)


def _accum_wgrad(
    grad_output_2d: torch.Tensor,
    input_2d: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    fp32_accum: bool,
    strict_cpu_ref: bool = False,
) -> None:
    # Empty GEMM: K==0 or zero feature dims contribute nothing; leave main_grad.
    # (Also handled inside the CUDA extension; keep here to avoid JIT on no-ops.)
    if input_2d.size(0) == 0 or input_2d.size(1) == 0 or grad_output_2d.size(1) == 0:
        return

    if fp32_accum:
        # Match Apex fused_weight_gradient path (half/bf16/fp32 -> fp32 C).
        _cublas_wgrad_gemm_accum_fp32(
            input_2d, grad_output_2d, main_grad, strict_cpu_ref=strict_cpu_ref
        )
        return

    grad_output_T, input_c = _matmul_operands(grad_output_2d, input_2d)
    _fused_addmm_cublas(main_grad, grad_output_T, input_c)


def wgrad_gemm_accum_fp32(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    strict_cpu_ref: bool = False,
) -> None:
    """Accumulate weight gradient into ``main_grad`` using fp32 storage.

    ``strict_cpu_ref`` is for tests only: fp64 GEMM matching CPU reference when
    TF32 is disabled.  Production / bench paths leave it False (GemmEx, ~1× Apex).
    """
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP32")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype != torch.float32:
        raise RuntimeError(
            "main_grad must be float32 for wgrad_gemm_accum_fp32, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Unsupported input dtype for wgrad_gemm_accum_fp32: " f"{input.dtype}"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "grad_output dtype must match input dtype, "
            f"but got {grad_output.dtype} vs {input.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=True,
        strict_cpu_ref=strict_cpu_ref,
    )


def wgrad_gemm_accum_fp16(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Accumulate weight gradient into ``main_grad`` using fp16/bf16 storage."""
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP16")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "main_grad must be float16 or bfloat16 for wgrad_gemm_accum_fp16, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Unsupported input dtype for wgrad_gemm_accum_fp16: " f"{input.dtype}"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "grad_output dtype must match input dtype, "
            f"but got {grad_output.dtype} vs {input.dtype}"
        )
    if main_grad.dtype != input.dtype:
        raise RuntimeError(
            "main_grad dtype must match input dtype, "
            f"but got {main_grad.dtype} vs {input.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=False,
    )
