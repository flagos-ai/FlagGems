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

from flag_gems import runtime
from flag_gems.deprecated.cutlass_scaled_mm import (
    _scaled_mm_fp8_kernel as scaled_mm_kernel,
)
from flag_gems.deprecated.cutlass_scaled_mm import cutlass_scaled_mm_fp8 as _csmm
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)

# GCU400 tile: 15 sweeps + a second retest confirm (64,128,32) is 3.4x faster than
# (32,64,32) at 512^3, ties at 16^3/128^3, with zero accuracy failures. BLOCK_K must stay
# 32: 64/128 makes GCU400's fp8->fp16 dot produce completely wrong results at 512^3.
_TILE_M = 64
_TILE_N = 128
_TILE_K = 32


def _check_inputs(self, mat2):
    if self.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError("mat1 and mat2 must be matrices")
    if self.shape[1] != mat2.shape[0]:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")
    if self.device != mat2.device:
        raise RuntimeError("mat1 and mat2 must be on the same device")
    if self.dtype not in FP8_DTYPES or mat2.dtype not in FP8_DTYPES:
        raise RuntimeError("mat1 and mat2 must be Float8 matrices")


def _resolve_out_dtype(self, out_dtype, out=None):
    dtype = (
        out_dtype
        if out_dtype is not None
        else (out.dtype if out is not None else self.dtype)
    )
    if dtype not in (*FP8_DTYPES, torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("out_dtype must be Float8, Float16, BFloat16 or Float32")
    if out is not None and out.dtype != dtype:
        raise RuntimeError("out_dtype must match output matrix type")
    return dtype


def _normalize_scale(scale, expected_size, *, is_left_scale):
    name = "scale_a" if is_left_scale else "scale_b"
    if scale.dtype != torch.float32:
        raise RuntimeError(f"{name} must be Float32")
    if scale.numel() == 1:
        return scale if scale.ndim == 1 else scale.reshape(1), False
    shape = (expected_size, 1) if is_left_scale else (1, expected_size)
    if scale.shape != shape or not scale.is_contiguous():
        raise RuntimeError(
            f"{name} must be a scalar or a contiguous tensor of shape {shape}"
        )
    return scale.reshape(expected_size), True


def _normalize_bias(bias, cols):
    if bias is not None:
        if bias.numel() != cols:
            raise RuntimeError(f"Bias must be size {cols}")
        if bias.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("Bias must be BFloat16 or Half")
        if bias.ndim != 1:
            bias = bias.reshape(cols)
        bias = bias.contiguous()
    return bias


def _can_use_cutlass_scaled_mm(self, mat2, scale_a, scale_b, bias, out):
    if runtime.device.vendor_name != "nvidia" or self.device.type != "cuda":
        return False
    if self.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or mat2.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return False
    major, _ = torch.cuda.get_device_capability(self.device)
    return (
        major == 9
        and self.shape[1] > 0
        and self.stride(1) == 1
        and mat2.stride(0) == 1
        and out.stride(1) == 1
        and out.stride(0) % 16 == 0
        and mat2.stride(1) % 16 == 0
        and scale_a.is_contiguous()
        and scale_b.is_contiguous()
    )


def _prepare_scaled_mm(
    self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out
):
    _check_inputs(self, mat2)
    M, K = self.shape
    N = mat2.shape[1]
    output_dtype = _resolve_out_dtype(self, out_dtype, out)
    for tensor in (scale_a, scale_b, bias, scale_result, out):
        if tensor is not None and tensor.device != self.device:
            raise RuntimeError("All tensors must be on the same device")
    scale_a, row_a = _normalize_scale(scale_a, M, is_left_scale=True)
    scale_b, row_b = _normalize_scale(scale_b, N, is_left_scale=False)
    bias = _normalize_bias(bias, N)
    if bias is not None:
        if output_dtype == torch.float32:
            raise RuntimeError("Bias is not supported when out_dtype is set to Float32")
        if (
            output_dtype in (torch.float16, torch.bfloat16)
            and bias.dtype != output_dtype
        ):
            raise RuntimeError("Bias dtype must match output dtype")
    if scale_result is not None and (
        scale_result.numel() != 1 or scale_result.dtype != torch.float32
    ):
        raise RuntimeError("scale_result must be a float scalar")
    if out is None:
        out = torch.empty((M, N), dtype=output_dtype, device=self.device)
    elif out.shape != (M, N):
        out.resize_(M, N)
    return scale_a, scale_b, bias, out, row_a, row_b


def _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out):
    scale_a, scale_b, bias, out, row_a, row_b = _prepare_scaled_mm(
        self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out
    )
    M, K = self.shape
    N = mat2.shape[1]
    if M == 0 or N == 0:
        return out
    if K == 0:
        # With K=0 matmul is all zeros, so no FP8 kernel launch is needed; on GCU400 that kernel
        # triggers an llvm.fptrunc -> i8 compile error at the f32->fp8 conversion.
        with torch_device_fn.device(self.device):
            out.zero_()
        return out
    with torch_device_fn.device(self.device):
        if _can_use_cutlass_scaled_mm(self, mat2, scale_a, scale_b, bias, out):
            _csmm(out, self, mat2, scale_a, scale_b, bias)
        else:
            # GCU400's Triton backend lowers epilogue stores to a row-major
            # matrix-write intrinsic; when `out` is non-contiguous (e.g. the
            # transposed view passed to scaled_mm_out) it writes wrong results, so compute into a contiguous
            # temporary buffer first and copy back to `out`.
            store_target = out
            if not out.is_contiguous():
                store_target = torch.empty((M, N), dtype=out.dtype, device=self.device)
            # FP8 output stores use GCU's m32 matrix-write intrinsic, so BLOCK_M/BLOCK_N
            # must stay 32/64; only fp16/bf16/fp32 outputs can use the larger 64x128 tile.
            if out.dtype in FP8_DTYPES:
                bm, bn, bk = 32, 64, 32
            else:
                bm, bn, bk = _TILE_M, _TILE_N, _TILE_K
            scaled_mm_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
                self,
                mat2,
                scale_a,
                scale_b,
                bias,
                store_target,
                M,
                N,
                K,
                *self.stride(),
                *mat2.stride(),
                *store_target.stride(),
                row_a,
                row_b,
                bias is not None,
                bm,
                bn,
                bk,
                False,
            )
            if store_target is not out:
                out.copy_(store_target)
    return out


def scaled_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_ENFLAME_GCU400 SCALED_MM")
    # Match ATen GPU semantics: scale_result and the fast accumulation hint
    # do not change the mathematical result in this implementation.
    return _scaled_mm_impl(
        self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, None
    )


def scaled_mm_out(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    logger.debug("GEMS_ENFLAME_GCU400 SCALED_MM_OUT")
    return _scaled_mm_impl(
        self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out
    )
