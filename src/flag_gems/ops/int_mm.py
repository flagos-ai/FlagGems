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

logger = logging.getLogger(__name__)

_GROUP_M = 8
_IO_BLOCK_SIZE = 256
_MAX_DOT_K_WITHOUT_INT32_OVERFLOW = 131071


@libentry()
@triton.jit
def _int_mm_dot_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    EXPLICIT_OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        if EVEN_M and EVEN_K:
            a = tl.load(a_ptrs)
        else:
            a_mask = (offs_k[None, :] < k_remaining) & (offs_m[:, None] < M)
            a = tl.load(a_ptrs, mask=a_mask, other=0)

        if EVEN_N and EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0)

        if EXPLICIT_OUT_DTYPE:
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc)
    else:
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)


@libentry()
@triton.jit
def _int_mm_outer_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    row = (pid // grid_n).to(tl.int64)
    cols = ((pid % grid_n) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k in range(0, K):
        k64 = k.to(tl.int64)
        a = tl.load(a_ptr + row * stride_am + k64 * stride_ak).to(tl.int32)
        b = tl.load(
            b_ptr + k64 * stride_bk + cols * stride_bn,
            mask=col_mask,
            other=0,
        ).to(tl.int32)
        acc += a * b

    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        acc,
        mask=col_mask,
    )


@libentry()
@triton.jit
def _int_mm_zero_kernel(
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    offsets64 = offsets.to(tl.int64)
    tl.store(
        out_ptr + offsets64,
        tl.zeros((BLOCK_SIZE,), dtype=tl.int32),
        mask=mask,
    )


@libentry()
@triton.jit
def _int_mm_copy_kernel(
    src_ptr,
    out_ptr,
    N,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask = cols < N
    values = tl.load(src_ptr + row * N + cols, mask=mask)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        values,
        mask=mask,
    )


def _check_inputs(self, mat2):
    if self.ndim != 2:
        raise RuntimeError("_int_mm: self must be a matrix")
    if mat2.ndim != 2:
        raise RuntimeError("_int_mm: mat2 must be a matrix")
    if self.dtype != torch.int8 or mat2.dtype != torch.int8:
        raise RuntimeError(
            "_int_mm: expected both inputs to have dtype torch.int8, "
            f"but got {self.dtype} and {mat2.dtype}"
        )
    if self.device != mat2.device:
        raise RuntimeError("_int_mm: self and mat2 must be on the same device")
    if self.shape[1] != mat2.shape[0]:
        raise RuntimeError(
            "_int_mm: mat1 and mat2 shapes cannot be multiplied "
            f"({self.shape[0]}x{self.shape[1]} and "
            f"{mat2.shape[0]}x{mat2.shape[1]})"
        )


def _check_out(self, mat2, out):
    expected_shape = (self.shape[0], mat2.shape[1])
    if out.dtype != torch.int32:
        raise RuntimeError(
            "_int_mm.out: expected out to have dtype torch.int32, "
            f"but got {out.dtype}"
        )
    if out.shape != expected_shape:
        raise RuntimeError(
            f"_int_mm.out: expected out shape {expected_shape}, "
            f"but got {tuple(out.shape)}"
        )
    if out.device != self.device:
        raise RuntimeError("_int_mm.out: out must be on the same device as the inputs")


def _prepare_inputs(self, mat2):
    if 0 in self.stride() or (self.stride(0) > 1 and self.stride(1) > 1):
        self = self.contiguous()
    if 0 in mat2.stride() or (mat2.stride(0) > 1 and mat2.stride(1) > 1):
        mat2 = mat2.contiguous()
    return self, mat2


def _launch_dot(
    self,
    mat2,
    out,
    M,
    N,
    K,
    *,
    block_m=64,
    block_n=64,
    block_k=64,
    group_m=_GROUP_M,
    num_warps=None,
    num_stages=None,
    explicit_out_dtype=False,
    **compiler_options,
):
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    if num_warps is not None:
        compiler_options["num_warps"] = num_warps
    if num_stages is not None:
        compiler_options["num_stages"] = num_stages
    _int_mm_dot_kernel[grid](
        self,
        mat2,
        out,
        M,
        N,
        K,
        self.stride(0),
        self.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        EVEN_M=M % block_m == 0,
        EVEN_N=N % block_n == 0,
        EVEN_K=K % block_k == 0,
        EXPLICIT_OUT_DTYPE=explicit_out_dtype,
        **compiler_options,
    )


def _launch_outer(self, mat2, out, M, N, K, *, block_n=64):
    _int_mm_outer_kernel[(M * triton.cdiv(N, block_n),)](
        self,
        mat2,
        out,
        M,
        N,
        K,
        self.stride(0),
        self.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
    )


def _zero_output(out, numel):
    _int_mm_zero_kernel[(triton.cdiv(numel, _IO_BLOCK_SIZE),)](
        out,
        numel,
        BLOCK_SIZE=_IO_BLOCK_SIZE,
    )


def _copy_output(src, out, M, N):
    _int_mm_copy_kernel[(M, triton.cdiv(N, _IO_BLOCK_SIZE))](
        src,
        out,
        N,
        out.stride(0),
        out.stride(1),
        BLOCK_N=_IO_BLOCK_SIZE,
    )


def _launch_conservative(self, mat2, out, M, N, K):
    if K > _MAX_DOT_K_WITHOUT_INT32_OVERFLOW:
        _launch_outer(self, mat2, out, M, N, K)
    else:
        _launch_dot(self, mat2, out, M, N, K, explicit_out_dtype=True)


def _launch_nvidia(self, mat2, out, M, N, K):
    if K > _MAX_DOT_K_WITHOUT_INT32_OVERFLOW:
        _launch_outer(self, mat2, out, M, N, K)
    elif M <= 32:
        _launch_dot(
            self,
            mat2,
            out,
            M,
            N,
            K,
            block_m=32,
            block_n=64 if N >= 64 else 32,
            block_k=32,
            num_warps=4,
            num_stages=3,
            explicit_out_dtype=True,
        )
    elif M <= 128:
        _launch_dot(
            self,
            mat2,
            out,
            M,
            N,
            K,
            block_m=64,
            block_n=64 if N >= 64 else 32,
            block_k=32,
            num_warps=4,
            num_stages=4,
            explicit_out_dtype=True,
        )
    else:
        _launch_dot(
            self,
            mat2,
            out,
            M,
            N,
            K,
            block_m=128,
            block_n=128 if N >= 128 else 64,
            block_k=32,
            num_warps=8,
            num_stages=4,
            explicit_out_dtype=True,
        )


def _launch_iluvatar(self, mat2, out, M, N, K):
    if M <= 32:
        _launch_dot(self, mat2, out, M, N, K, block_k=32)
    elif M <= 64:
        _launch_dot(self, mat2, out, M, N, K)
    else:
        _launch_dot(
            self,
            mat2,
            out,
            M,
            N,
            K,
            block_m=128,
            block_n=128,
            block_k=64,
            num_warps=8,
        )


# Backend identity is process-stable, so resolve the launcher once at import.
_BACKEND_LAUNCHER = {
    "ascend": _launch_conservative,
    "iluvatar": _launch_iluvatar,
    "metax": _launch_conservative,
    "nvidia": _launch_nvidia,
}.get(runtime.device.vendor_name, _launch_conservative)


def _int_mm_impl(self, mat2, launcher, out=None):
    _check_inputs(self, mat2)
    M, K = self.shape
    N = mat2.shape[1]

    if out is None:
        out = torch.empty((M, N), dtype=torch.int32, device=self.device)
    else:
        _check_out(self, mat2, out)

    if M == 0 or N == 0:
        return out

    kernel_out = out
    if not out.is_contiguous():
        kernel_out = torch.empty((M, N), dtype=torch.int32, device=self.device)

    with torch_device_fn.device(self.device):
        if K == 0:
            _zero_output(kernel_out, M * N)
        else:
            kernel_self, kernel_mat2 = _prepare_inputs(self, mat2)
            launcher(kernel_self, kernel_mat2, kernel_out, M, N, K)
        if kernel_out is not out:
            _copy_output(kernel_out, out, M, N)
    return out


def int_mm(self, mat2):
    logger.debug("GEMS INT_MM")
    return _int_mm_impl(self, mat2, _BACKEND_LAUNCHER)


def int_mm_out(self, mat2, *, out):
    logger.debug("GEMS INT_MM_OUT")
    return _int_mm_impl(self, mat2, _BACKEND_LAUNCHER, out=out)


__all__ = ["int_mm", "int_mm_out"]
