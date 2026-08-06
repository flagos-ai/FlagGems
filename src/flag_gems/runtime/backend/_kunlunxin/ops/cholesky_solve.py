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

import importlib
import itertools

import torch
import triton
import triton.language as tl


_generic = importlib.import_module("flag_gems.ops.cholesky_solve")


@triton.jit
def _cholesky_solve_row_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs,
    batch_stride_L,
    batch_stride_B,
    batch_stride_X,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    stride_X_row,
    stride_X_col,
    ROW: tl.constexpr,
    FORWARD: tl.constexpr,
    upper: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    rhs_pid = tl.program_id(1)
    cols = rhs_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    X_base = batch_pid * batch_stride_X

    if FORWARD:
        value = tl.load(
            B_ptr + B_base + ROW * stride_B_row + cols * stride_B_col,
            mask=cols_mask,
            other=0.0,
        )
        for col in range(ROW):
            if upper:
                factor = tl.load(L_ptr + L_base + col * stride_L_row + ROW * stride_L_col)
            else:
                factor = tl.load(L_ptr + L_base + ROW * stride_L_row + col * stride_L_col)
            previous = tl.load(
                X_ptr + X_base + col * stride_X_row + cols * stride_X_col,
                mask=cols_mask,
                other=0.0,
            )
            value -= factor * previous
    else:
        value = tl.load(
            X_ptr + X_base + ROW * stride_X_row + cols * stride_X_col,
            mask=cols_mask,
            other=0.0,
        )
        for col in range(ROW + 1, N):
            if upper:
                factor = tl.load(L_ptr + L_base + ROW * stride_L_row + col * stride_L_col)
            else:
                factor = tl.load(L_ptr + L_base + col * stride_L_row + ROW * stride_L_col)
            previous = tl.load(
                X_ptr + X_base + col * stride_X_row + cols * stride_X_col,
                mask=cols_mask,
                other=0.0,
            )
            value -= factor * previous

    diagonal = tl.load(L_ptr + L_base + ROW * stride_L_row + ROW * stride_L_col)
    tl.store(
        X_ptr + X_base + ROW * stride_X_row + cols * stride_X_col,
        value / diagonal,
        mask=cols_mask,
    )


@triton.jit
def _cholesky_solve_complex_row_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs,
    batch_stride_L,
    batch_stride_B,
    batch_stride_X,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    stride_X_row,
    stride_X_col,
    ROW: tl.constexpr,
    FORWARD: tl.constexpr,
    upper: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    rhs_pid = tl.program_id(1)
    cols = rhs_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    X_base = batch_pid * batch_stride_X

    if FORWARD:
        b_offset = B_base + ROW * stride_B_row + cols * stride_B_col
        value_real = tl.load(B_ptr + b_offset, mask=cols_mask, other=0.0)
        value_imag = tl.load(B_ptr + b_offset + 1, mask=cols_mask, other=0.0)
        start = 0
        end = ROW
    else:
        x_offset = X_base + ROW * stride_X_row + cols * stride_X_col
        value_real = tl.load(X_ptr + x_offset, mask=cols_mask, other=0.0)
        value_imag = tl.load(X_ptr + x_offset + 1, mask=cols_mask, other=0.0)
        start = ROW + 1
        end = N

    for col in range(start, end):
        if FORWARD:
            if upper:
                factor_offset = L_base + col * stride_L_row + ROW * stride_L_col
            else:
                factor_offset = L_base + ROW * stride_L_row + col * stride_L_col
        else:
            if upper:
                factor_offset = L_base + ROW * stride_L_row + col * stride_L_col
            else:
                factor_offset = L_base + col * stride_L_row + ROW * stride_L_col
        factor_real = tl.load(L_ptr + factor_offset)
        factor_imag = tl.load(L_ptr + factor_offset + 1)
        if (FORWARD and upper) or ((not FORWARD) and (not upper)):
            factor_imag = -factor_imag
        previous_offset = X_base + col * stride_X_row + cols * stride_X_col
        previous_real = tl.load(X_ptr + previous_offset, mask=cols_mask, other=0.0)
        previous_imag = tl.load(X_ptr + previous_offset + 1, mask=cols_mask, other=0.0)
        value_real -= factor_real * previous_real - factor_imag * previous_imag
        value_imag -= factor_real * previous_imag + factor_imag * previous_real

    diagonal_offset = L_base + ROW * stride_L_row + ROW * stride_L_col
    diagonal_real = tl.load(L_ptr + diagonal_offset)
    diagonal_imag = tl.load(L_ptr + diagonal_offset + 1)
    denominator = diagonal_real * diagonal_real + diagonal_imag * diagonal_imag
    out_real = (value_real * diagonal_real + value_imag * diagonal_imag) / denominator
    out_imag = (value_imag * diagonal_real - value_real * diagonal_imag) / denominator
    out_offset = X_base + ROW * stride_X_row + cols * stride_X_col
    tl.store(X_ptr + out_offset, out_real, mask=cols_mask)
    tl.store(X_ptr + out_offset + 1, out_imag, mask=cols_mask)


def _can_use_row_kernel(B: torch.Tensor, L: torch.Tensor) -> bool:
    if B.dtype != torch.float32 or L.dtype != torch.float32 or B.ndim < 2 or L.ndim < 2:
        return False
    if B.shape[:-2] != L.shape[:-2] or L.shape[-2] != L.shape[-1]:
        return False
    n, nrhs = B.shape[-2:]
    return n == L.shape[-1] and ((n <= 32 and nrhs <= 16) or (n < 64 and nrhs == 1))


def _can_use_complex_row_kernel(B: torch.Tensor, L: torch.Tensor) -> bool:
    if B.dtype != torch.complex64 or L.dtype != torch.complex64:
        return False
    if B.ndim < 2 or L.ndim < 2 or B.is_conj() or L.is_conj():
        return False
    if B.shape[:-2] != L.shape[:-2] or L.shape[-2] != L.shape[-1]:
        return False
    n, nrhs = B.shape[-2:]
    return n == L.shape[-1] and ((n <= 32 and nrhs <= 16) or (n < 64 and nrhs == 1))


def _cholesky_solve_complex_rows(B, L, upper, out):
    B_real = torch.view_as_real(B).reshape(-1, B.shape[-2], B.shape[-1], 2)
    L_real = torch.view_as_real(L).reshape(-1, L.shape[-2], L.shape[-1], 2)
    X_real = torch.view_as_real(out).reshape(-1, out.shape[-2], out.shape[-1], 2)
    batch_size = B_real.shape[0]
    n, nrhs = B.shape[-2:]
    block_rhs = triton.next_power_of_2(nrhs)
    grid = (batch_size, triton.cdiv(nrhs, block_rhs))

    for row in range(n):
        _cholesky_solve_complex_row_kernel[
            grid
        ](
            L_real,
            B_real,
            X_real,
            n,
            nrhs,
            L_real.stride(0) if L_real.ndim > 3 else 0,
            B_real.stride(0) if B_real.ndim > 3 else 0,
            X_real.stride(0) if X_real.ndim > 3 else 0,
            L_real.stride(-3),
            L_real.stride(-2),
            B_real.stride(-3),
            B_real.stride(-2),
            X_real.stride(-3),
            X_real.stride(-2),
            ROW=row,
            FORWARD=True,
            upper=upper,
            BLOCK_RHS=block_rhs,
            num_warps=1,
            num_stages=1,
        )
    for row in range(n - 1, -1, -1):
        _cholesky_solve_complex_row_kernel[
            grid
        ](
            L_real,
            B_real,
            X_real,
            n,
            nrhs,
            L_real.stride(0) if L_real.ndim > 3 else 0,
            B_real.stride(0) if B_real.ndim > 3 else 0,
            X_real.stride(0) if X_real.ndim > 3 else 0,
            L_real.stride(-3),
            L_real.stride(-2),
            B_real.stride(-3),
            B_real.stride(-2),
            X_real.stride(-3),
            X_real.stride(-2),
            ROW=row,
            FORWARD=False,
            upper=upper,
            BLOCK_RHS=block_rhs,
            num_warps=1,
            num_stages=1,
        )
    return out


def cholesky_solve(B, L, upper=False, *, _out=None):
    try:
        batch_shape = torch.broadcast_shapes(B.shape[:-2], L.shape[:-2])
    except RuntimeError:
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)

    result_shape = batch_shape + B.shape[-2:]
    if B.shape[:-2] != batch_shape or L.shape[:-2] != batch_shape:
        B_expanded = B.expand(result_shape)
        L_expanded = L.expand(batch_shape + L.shape[-2:])
        X = torch.empty(result_shape, dtype=B.dtype, device=B.device)
        for index in itertools.product(*(range(dim) for dim in batch_shape)):
            cholesky_solve(
                B_expanded[index], L_expanded[index], upper=upper, _out=X[index]
            )
        if _out is None:
            return X
        return _generic._copy_cholesky_solve_out(X, _out)

    X = torch.empty_like(B) if _out is None else _out
    if _out is not None and (
        X.shape != B.shape
        or X.dtype != B.dtype
        or X.device != B.device
        or torch._C._is_alias_of(X, B)
        or torch._C._is_alias_of(X, L)
    ):
        result = cholesky_solve(B, L, upper=upper)
        return _generic._copy_cholesky_solve_out(result, X)
    if _can_use_complex_row_kernel(B, L):
        if X.shape == B.shape and X.dtype == B.dtype and X.device == B.device:
            return _cholesky_solve_complex_rows(B, L, upper, X)
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)
    if not _can_use_row_kernel(B, L):
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)

    if X.shape != B.shape or X.dtype != B.dtype or X.device != B.device:
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)

    n, nrhs = B.shape[-2:]
    output = X
    L = L.reshape(-1, n, n)
    B = B.reshape(-1, n, nrhs)
    X = X.reshape(-1, n, nrhs)
    batch_size = B.shape[0]
    block_rhs = triton.next_power_of_2(nrhs)
    grid = (batch_size, triton.cdiv(nrhs, block_rhs))

    for row in range(n):
        _cholesky_solve_row_kernel[grid](
            L,
            B,
            X,
            n,
            nrhs,
            L.stride(0) if L.ndim > 2 else 0,
            B.stride(0) if B.ndim > 2 else 0,
            X.stride(0) if X.ndim > 2 else 0,
            L.stride(-2),
            L.stride(-1),
            B.stride(-2),
            B.stride(-1),
            X.stride(-2),
            X.stride(-1),
            ROW=row,
            FORWARD=True,
            upper=upper,
            BLOCK_RHS=block_rhs,
            num_warps=1,
            num_stages=1,
        )
    for row in range(n - 1, -1, -1):
        _cholesky_solve_row_kernel[grid](
            L,
            B,
            X,
            n,
            nrhs,
            L.stride(0) if L.ndim > 2 else 0,
            B.stride(0) if B.ndim > 2 else 0,
            X.stride(0) if X.ndim > 2 else 0,
            L.stride(-2),
            L.stride(-1),
            B.stride(-2),
            B.stride(-1),
            X.stride(-2),
            X.stride(-1),
            ROW=row,
            FORWARD=False,
            upper=upper,
            BLOCK_RHS=block_rhs,
            num_warps=1,
            num_stages=1,
        )
    return output


def cholesky_solve_out(B, L, upper=False, *, out):
    _generic._check_cholesky_solve_out(B, out)
    return cholesky_solve(B, L, upper=upper, _out=out)
