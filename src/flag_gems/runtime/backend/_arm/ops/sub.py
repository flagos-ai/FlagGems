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
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)
_PREWARM_SUB_DONE = False

_SUPPORTED_FAST_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_SUPPORTED_INT_FAST_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@triton.jit(do_not_specialize=["alpha", "n_elements"])
def _sub_contiguous_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x - y * alpha, mask=mask)


@triton.jit(do_not_specialize=["alpha", "n_elements"])
def _sub_contiguous_single_program_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = tl.load(y_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, x - y * alpha, mask=mask)


@triton.jit(do_not_specialize=["alpha", "rows", "cols"])
def _sub_broadcast_lastdim1_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    y = tl.load(y_ptr + row)
    offs = tl.arange(0, BLOCK_SIZE)
    row_start = row * cols
    for base in range(0, cols, BLOCK_SIZE):
        col = base + offs
        mask = col < cols
        x = tl.load(x_ptr + row_start + col, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + col, x - y * alpha, mask=mask)


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
def _sub_tensor_scalar_kernel(
    x_ptr,
    scalar,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x - scalar * alpha, mask=mask)


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
def _sub_tensor_scalar_single_program_kernel(
    x_ptr,
    scalar,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, x - scalar * alpha, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _sub_tensor_scalar_int_kernel(
    x_ptr,
    scalar,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x - scalar, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _sub_tensor_scalar_int_single_program_kernel(
    x_ptr,
    scalar,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, x - scalar, mask=mask)


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
def _sub_scalar_tensor_kernel(
    scalar,
    y_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, scalar - y * alpha, mask=mask)


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
def _sub_scalar_tensor_single_program_kernel(
    scalar,
    y_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        y = tl.load(y_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, scalar - y * alpha, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _sub_scalar_tensor_int_kernel(
    scalar,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, scalar - y, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _sub_scalar_tensor_int_single_program_kernel(
    scalar,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        y = tl.load(y_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, scalar - y, mask=mask)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


def _select_block_size(n_elements, dtype):
    if n_elements <= 32:
        return 32
    if n_elements <= 1024:
        return 32
    if n_elements <= 8192:
        return 64
    return 256 if dtype in (torch.float16, torch.bfloat16) else 128


def _single_program_block(n_elements):
    if n_elements <= 256:
        return 32
    if n_elements <= 2048:
        return 128
    return 256


def _launch_sub_tensor_tensor(x, y, out, alpha, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _sub_contiguous_single_program_kernel[(1,)](
            x,
            y,
            out,
            alpha,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (triton.cdiv(n_elements, block_size),)
    _sub_contiguous_kernel[grid](
        x,
        y,
        out,
        alpha,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_sub_tensor_scalar(x, scalar, out, alpha, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _sub_tensor_scalar_single_program_kernel[(1,)](
            x,
            scalar,
            out,
            alpha,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (triton.cdiv(n_elements, block_size),)
    _sub_tensor_scalar_kernel[grid](
        x,
        scalar,
        out,
        alpha,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_sub_tensor_scalar_int(x, scalar, out, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _sub_tensor_scalar_int_single_program_kernel[(1,)](
            x,
            scalar,
            out,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (triton.cdiv(n_elements, block_size),)
    _sub_tensor_scalar_int_kernel[grid](
        x,
        scalar,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_sub_broadcast_lastdim1(x, y, out, alpha):
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    if rows == 0 or cols == 0:
        return
    if cols <= 1024:
        block_size = 64
    elif cols <= 4096:
        block_size = 128
    else:
        block_size = 256
    grid = (rows,)
    _sub_broadcast_lastdim1_kernel[grid](
        x,
        y,
        out,
        alpha,
        rows,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_sub_scalar_tensor(scalar, y, out, alpha, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _sub_scalar_tensor_single_program_kernel[(1,)](
            scalar,
            y,
            out,
            alpha,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (triton.cdiv(n_elements, block_size),)
    _sub_scalar_tensor_kernel[grid](
        scalar,
        y,
        out,
        alpha,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_sub_scalar_tensor_int(scalar, y, out, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _sub_scalar_tensor_int_single_program_kernel[(1,)](
            scalar,
            y,
            out,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (triton.cdiv(n_elements, block_size),)
    _sub_scalar_tensor_int_kernel[grid](
        scalar,
        y,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _can_use_contiguous_fastpath(a, b):
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.device.type == "cpu"
        and b.device == a.device
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape == b.shape
        and a.dtype == b.dtype
        and a.dtype in _SUPPORTED_FAST_DTYPES
    )


def _can_use_broadcast_lastdim1_fastpath(a, b):
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.device.type == "cpu"
        and b.device == a.device
        and a.is_contiguous()
        and b.is_contiguous()
        and a.ndim >= 1
        and b.ndim == a.ndim
        and a.shape[:-1] == b.shape[:-1]
        and b.shape[-1] == 1
        and a.dtype == b.dtype
        and a.dtype in _SUPPORTED_FAST_DTYPES
    )


def _can_use_tensor_scalar_int_fastpath(a, scalar, alpha):
    return (
        isinstance(a, torch.Tensor)
        and a.device.type == "cpu"
        and a.is_contiguous()
        and a.dtype in _SUPPORTED_INT_FAST_DTYPES
        and isinstance(scalar, int)
        and int(alpha) == 1
        and float(alpha) == 1.0
    )


def _can_use_scalar_tensor_fastpath(b, scalar):
    return (
        isinstance(b, torch.Tensor)
        and b.device.type == "cpu"
        and b.is_contiguous()
        and b.dtype in _SUPPORTED_FAST_DTYPES
        and isinstance(scalar, (int, float))
    )


def _can_use_scalar_tensor_int_fastpath(b, scalar, alpha):
    return (
        isinstance(b, torch.Tensor)
        and b.device.type == "cpu"
        and b.is_contiguous()
        and b.dtype in _SUPPORTED_INT_FAST_DTYPES
        and isinstance(scalar, int)
        and int(alpha) == 1
        and float(alpha) == 1.0
    )


def _maybe_scalar(v):
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    if isinstance(v, (int, float)):
        return v
    return None


def _maybe_prewarm_sub_kernels():
    global _PREWARM_SUB_DONE
    if _PREWARM_SUB_DONE:
        return
    if os.environ.get("GEMS_ARM_SUB_PREWARM", "1") != "1":
        _PREWARM_SUB_DONE = True
        return
    try:
        x = torch.zeros(8, dtype=torch.float32, device="cpu")
        y = torch.ones(8, dtype=torch.float32, device="cpu")
        out = torch.empty_like(x)
        _launch_sub_tensor_tensor(x, y, out, 1.0, x.numel(), 32)
        _launch_sub_tensor_scalar(x, 1.0, out, 1.0, x.numel(), 32)
        _launch_sub_scalar_tensor(1.0, x, out, 1.0, x.numel(), 32)

        xi = torch.arange(8, dtype=torch.int64, device="cpu")
        oi = torch.empty_like(xi)
        _launch_sub_tensor_scalar_int(xi, 1, oi, xi.numel(), 32)
        _launch_sub_scalar_tensor_int(1, xi, oi, xi.numel(), 32)

        xb = torch.zeros((1, 5, 32), dtype=torch.float32, device="cpu")
        yb = torch.zeros((1, 5, 1), dtype=torch.float32, device="cpu")
        ob = torch.empty_like(xb)
        _launch_sub_broadcast_lastdim1(xb, yb.view(-1), ob, 1.0)
    except Exception:
        logger.debug("GEMS_ARM sub prewarm failed", exc_info=True)
    _PREWARM_SUB_DONE = True


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_ARM SUB")
    _maybe_prewarm_sub_kernels()

    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if _can_use_contiguous_fastpath(A, B):
            out = torch.empty_like(A)
            block_size = _select_block_size(A.numel(), A.dtype)
            _launch_sub_tensor_tensor(A, B, out, float(alpha), A.numel(), block_size)
            return out
        if _can_use_broadcast_lastdim1_fastpath(A, B):
            out = torch.empty_like(A)
            _launch_sub_broadcast_lastdim1(A, B.view(-1), out, float(alpha))
            return out
        return sub_func(A, B, alpha)

    if isinstance(A, torch.Tensor):
        scalar = _maybe_scalar(B)
        if (
            scalar is not None
            and A.device.type == "cpu"
            and A.is_contiguous()
            and A.dtype in _SUPPORTED_FAST_DTYPES
        ):
            out = torch.empty_like(A)
            block_size = _select_block_size(A.numel(), A.dtype)
            _launch_sub_tensor_scalar(
                A, float(scalar), out, float(alpha), A.numel(), block_size
            )
            return out
        if _can_use_tensor_scalar_int_fastpath(A, scalar, alpha):
            out = torch.empty_like(A)
            block_size = _select_block_size(A.numel(), A.dtype)
            _launch_sub_tensor_scalar_int(A, int(scalar), out, A.numel(), block_size)
            return out
        return sub_func_tensor_scalar(A, B, alpha)

    if isinstance(B, torch.Tensor):
        scalar = _maybe_scalar(A)
        if _can_use_scalar_tensor_fastpath(B, scalar):
            out = torch.empty_like(B)
            block_size = _select_block_size(B.numel(), B.dtype)
            _launch_sub_scalar_tensor(
                float(scalar), B, out, float(alpha), B.numel(), block_size
            )
            return out
        if _can_use_scalar_tensor_int_fastpath(B, scalar, alpha):
            out = torch.empty_like(B)
            block_size = _select_block_size(B.numel(), B.dtype)
            _launch_sub_scalar_tensor_int(int(scalar), B, out, B.numel(), block_size)
            return out
        return sub_func_scalar_tensor(A, B, alpha)

    return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_ARM SUB_")
    _maybe_prewarm_sub_kernels()

    if isinstance(B, torch.Tensor):
        if _can_use_contiguous_fastpath(A, B):
            if A.untyped_storage().data_ptr() == B.untyped_storage().data_ptr():
                return sub_func(A, B, alpha, out0=A)
            block_size = _select_block_size(A.numel(), A.dtype)
            _launch_sub_tensor_tensor(A, B, A, float(alpha), A.numel(), block_size)
            return A
        if _can_use_broadcast_lastdim1_fastpath(A, B):
            _launch_sub_broadcast_lastdim1(A, B.view(-1), A, float(alpha))
            return A
        return sub_func(A, B, alpha, out0=A)

    scalar = _maybe_scalar(B)
    if (
        scalar is not None
        and isinstance(A, torch.Tensor)
        and A.device.type == "cpu"
        and A.is_contiguous()
        and A.dtype in _SUPPORTED_FAST_DTYPES
    ):
        block_size = _select_block_size(A.numel(), A.dtype)
        _launch_sub_tensor_scalar(
            A, float(scalar), A, float(alpha), A.numel(), block_size
        )
        return A
    if _can_use_tensor_scalar_int_fastpath(A, scalar, alpha):
        block_size = _select_block_size(A.numel(), A.dtype)
        _launch_sub_tensor_scalar_int(A, int(scalar), A, A.numel(), block_size)
        return A

    return sub_func_tensor_scalar(A, B, alpha, out0=A)


_maybe_prewarm_sub_kernels()
