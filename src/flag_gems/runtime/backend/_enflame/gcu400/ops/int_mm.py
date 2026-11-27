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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_GROUP_M = 8
_IO_BLOCK_SIZE = 256
_MAX_DOT_K_WITHOUT_INT32_OVERFLOW = 131071
_INT32_ELEMENT_LIMIT = 2**31
_GRID_LIMIT = 65535

_INT_MM_CONFIGS = [
    # decode / tiny-M arm (M = token batch)
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": _GROUP_M},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": _GROUP_M},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": _GROUP_M},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": _GROUP_M},
        num_warps=2,
        num_stages=2,
    ),
    # dominant winner on every M >= 64 shape swept
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": _GROUP_M},
        num_warps=2,
        num_stages=2,
    ),
    # large-shape variant (won 4096^3 by ~3%)
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": _GROUP_M},
        num_warps=2,
        num_stages=3,
    ),
    # universal safety net: smallest grid, generic tile
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": _GROUP_M},
        num_warps=4,
        num_stages=2,
    ),
]


def _int_mm_heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


def _int_mm_heur_full_even(args):
    return (
        args["K"] % args["BLOCK_K"] == 0
        and args["M"] % args["BLOCK_M"] == 0
        and args["N"] % args["BLOCK_N"] == 0
    )


def _int_mm_config_prune(configs, named_args, **kwargs):
    # Keep only configs whose 1D grid fits the 65535 program hard limit; the
    # host pre-check guarantees the (64,64,64) safety net survives, so this
    # list is never empty on the fast path.
    M = named_args["M"]
    N = named_args["N"]
    kept = [
        config
        for config in configs
        if triton.cdiv(M, config.kwargs["BLOCK_M"])
        * triton.cdiv(N, config.kwargs["BLOCK_N"])
        <= _GRID_LIMIT
    ]
    return kept


@libentry()
@libtuner(
    configs=_INT_MM_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    prune_configs_by={"early_config_prune": _int_mm_config_prune},
    warmup=5,
    rep=10,
    disable_aabs=True,
)
@triton.heuristics(
    {
        "EVEN_K": _int_mm_heur_even_k,
        "FULL_EVEN": _int_mm_heur_full_even,
    }
)
@triton.jit
def _int_mm_dot_i32_kernel(
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
    EVEN_K: tl.constexpr,
    FULL_EVEN: tl.constexpr,
):
    # Fast path: i32 addressing only (the host guarantees every in-bounds
    # linear offset stays below 2**31 and every stride is non-negative).
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    # Dynamic K loop: the int32 tl.dot accumulator is not affected by the
    # fp32 loop-carried-dot miscompile (A/B-measured equal to static unroll).
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if FULL_EVEN:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        elif EVEN_K:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0)
            b = tl.load(b_ptrs, mask=(offs_n[None, :] < N), other=0)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0,
            )
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if FULL_EVEN:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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


def _extent_i32_safe(tensor, outer, inner):
    # Exact linear-extent guard for the i32 fast kernel: non-negative
    # strides and max in-bounds element offset below 2**31.
    stride_outer = tensor.stride(0)
    stride_inner = tensor.stride(1)
    if stride_outer < 0 or stride_inner < 0:
        return False
    extent = (outer - 1) * stride_outer + (inner - 1) * stride_inner + 1
    return extent < _INT32_ELEMENT_LIMIT


def _launch_dot(self, mat2, out, M, N, K):
    if (
        _extent_i32_safe(self, M, K)
        and _extent_i32_safe(mat2, K, N)
        and _extent_i32_safe(out, M, N)
        # the (64,64,64) safety-net config must fit the grid hard limit so
        # the tuner's pruned config list is never empty on this path
        and triton.cdiv(M, 64) * triton.cdiv(N, 64) <= _GRID_LIMIT
    ):
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        _int_mm_dot_i32_kernel[grid](
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
        )
    else:
        _launch_dot_i64(self, mat2, out, M, N, K)


def _launch_dot_i64(self, mat2, out, M, N, K):
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        GROUP_M=_GROUP_M,
        EVEN_M=M % 64 == 0,
        EVEN_N=N % 64 == 0,
        EVEN_K=K % 64 == 0,
        EXPLICIT_OUT_DTYPE=True,
    )


def _launch_outer(self, mat2, out, M, N, K):
    _int_mm_outer_kernel[(M * triton.cdiv(N, 64),)](
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
        BLOCK_N=64,
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


def _int_mm_impl(self, mat2, out=None):
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
            if K > _MAX_DOT_K_WITHOUT_INT32_OVERFLOW:
                _launch_outer(kernel_self, kernel_mat2, kernel_out, M, N, K)
            else:
                _launch_dot(kernel_self, kernel_mat2, kernel_out, M, N, K)
        if kernel_out is not out:
            _copy_output(kernel_out, out, M, N)
    return out


def int_mm(self, mat2):
    logger.debug("GEMS_ENFLAME INT_MM")
    return _int_mm_impl(self, mat2)


def int_mm_out(self, mat2, *, out):
    # Each entry point announces itself: tests/test_int_mm.py::test_int_mm_out_gems_route
    # asserts that the out variant logs "...INT_MM_OUT" (the tag used to be
    # emitted by int_mm, so the out entry point looked unrouted).
    logger.debug("GEMS_ENFLAME INT_MM_OUT")
    return _int_mm_impl(self, mat2, out=out)


__all__ = ["int_mm", "int_mm_out"]
