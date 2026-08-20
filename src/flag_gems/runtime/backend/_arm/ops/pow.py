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

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.ops.pow import pow_scalar as base_pow_scalar
from flag_gems.ops.pow import pow_tensor_scalar as base_pow_tensor_scalar
from flag_gems.ops.pow import pow_tensor_scalar_ as base_pow_tensor_scalar_
from flag_gems.ops.pow import pow_tensor_tensor as base_pow_tensor_tensor
from flag_gems.ops.pow import pow_tensor_tensor_ as base_pow_tensor_tensor_

logger = logging.getLogger(__name__)

# For small tensors, bypass Triton entirely via numpy (zero-copy views).
_POW_NATIVE_THRESHOLD = 4096

_PREWARM_POW_DONE = False
_POW_SQUARE_HOT_ENABLED = os.environ.get("GEMS_ARM_POW_SQUARE_HOT", "1") == "1"
_POW_TRITON_ENABLED = os.environ.get("GEMS_ARM_POW_TRITON", "1") == "1"
_POW_PREWARM_ENABLED = os.environ.get("GEMS_ARM_POW_PREWARM", "1") == "1"


@triton.jit
def _pow_square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_prog * BLOCK_SIZE
    for off in range(start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * x, mask=mask)


@triton.jit
def _pow_square_single_program_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, x * x, mask=mask)


@triton.jit
def _pow_square_1024_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 1024, 256):
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, x * x)


@triton.jit
def _pow_square_2048_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 2048, 256):
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, x * x)


@triton.jit(do_not_specialize=["rows"])
def _pow_square_rows128_hot_kernel(
    x_ptr,
    out_ptr,
    rows,
    MAX_ROWS: tl.constexpr,
):
    offs = tl.arange(0, 128)
    for row in range(0, MAX_ROWS):
        if row < rows:
            base = row * 128
            x = tl.load(x_ptr + base + offs)
            tl.store(out_ptr + base + offs, x * x)


@triton.jit(do_not_specialize=["rows"])
def _pow_square_rows1024_hot_kernel(
    x_ptr,
    out_ptr,
    rows,
    MAX_ROWS: tl.constexpr,
):
    offs = tl.arange(0, 256)
    for row in range(0, MAX_ROWS):
        if row < rows:
            base = row * 1024
            for k in range(0, 1024, 256):
                x = tl.load(x_ptr + base + k + offs)
                tl.store(out_ptr + base + k + offs, x * x)


@triton.jit
def _pow_square_3584_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 3584, 256):
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, x * x)


@triton.jit(do_not_specialize=["rows"])
def _pow_square_rows3584_hot_kernel(
    x_ptr,
    out_ptr,
    rows,
    MAX_ROWS: tl.constexpr,
):
    offs = tl.arange(0, 256)
    for row in range(0, MAX_ROWS):
        if row < rows:
            base = row * 3584
            for k in range(0, 3584, 256):
                x = tl.load(x_ptr + base + k + offs)
                tl.store(out_ptr + base + k + offs, x * x)


@triton.jit
def _pow_sqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_prog * BLOCK_SIZE
    for off in range(start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.sqrt(x.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _pow_sqrt_single_program_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = tl.sqrt(x.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def _pow_rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_prog * BLOCK_SIZE
    for off in range(start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = (1.0 / tl.sqrt(x.to(tl.float32))).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _pow_rsqrt_single_program_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = (1.0 / tl.sqrt(x.to(tl.float32))).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + idx, y, mask=mask)


def _select_block_size(n_elements, dtype):
    # Tuned for Qwen decode hotspot shapes on triton-cpu.
    if n_elements <= 32:
        return 32
    if n_elements <= 1024:
        return 128
    if n_elements <= 2048:
        return 128
    if n_elements <= 4096:
        return 128
    if n_elements <= (1 << 16):
        return 128
    return 256 if dtype in (torch.float16, torch.bfloat16) else 128


def _single_program_block(n_elements):
    if n_elements <= 256:
        return 32
    if n_elements <= 2048:
        return 128
    return 256


def _maybe_scalar(v):
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.item())
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _is_supported_tensor(t):
    return (
        isinstance(t, torch.Tensor)
        and t.device.type == "cpu"
        and t.dtype
        in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    )


def _launch_pow_kernel(
    multi_kernel, single_kernel, x, out_tensor, n_elements, block_size
):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        single_kernel[(1,)](
            x,
            out_tensor,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return
    grid = (triton.cdiv(n_elements, block_size),)
    multi_kernel[grid](
        x,
        out_tensor,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _maybe_launch_pow_square_hotshape(x, out_tensor, n_elements):
    if not _POW_SQUARE_HOT_ENABLED:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    if x.ndim == 0:
        return False
    last_dim = x.shape[-1]
    if last_dim == 128:
        rows = n_elements // 128
        if rows > 0 and rows <= 96 and rows * 128 == n_elements:
            _pow_square_rows128_hot_kernel[(1,)](
                x,
                out_tensor,
                rows,
                MAX_ROWS=96,
                num_warps=1,
                num_stages=1,
            )
            return True
    if last_dim == 1024:
        rows = n_elements // 1024
        if rows > 0 and rows <= 16 and rows * 1024 == n_elements:
            _pow_square_rows1024_hot_kernel[(1,)](
                x,
                out_tensor,
                rows,
                MAX_ROWS=16,
                num_warps=1,
                num_stages=1,
            )
            return True
    if last_dim == 3584:
        rows = n_elements // 3584
        if rows > 0 and rows <= 128 and rows * 3584 == n_elements:
            _pow_square_rows3584_hot_kernel[(1,)](
                x,
                out_tensor,
                rows,
                MAX_ROWS=128,
                num_warps=1,
                num_stages=1,
            )
            return True
    return False


def _pow_tensor_scalar_special(x, exponent, out=None):
    if not _is_supported_tensor(x):
        return None
    if not x.is_contiguous():
        return None
    if out is not None and not out.is_contiguous():
        return None
    if not _POW_TRITON_ENABLED:
        return None

    if exponent == 2.0:
        kernel = _pow_square_kernel
        single_kernel = _pow_square_single_program_kernel
    elif exponent == 0.5:
        kernel = _pow_sqrt_kernel
        single_kernel = _pow_sqrt_single_program_kernel
    elif exponent == -0.5:
        kernel = _pow_rsqrt_kernel
        single_kernel = _pow_rsqrt_single_program_kernel
    else:
        return None

    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out

    block_size = _select_block_size(n_elements, x.dtype)
    out_tensor = torch.empty_like(x) if out is None else out
    if exponent == 2.0:
        if n_elements == 1024 and x.is_contiguous():
            _pow_square_1024_hot_kernel[(1,)](
                x,
                out_tensor,
                num_warps=1,
                num_stages=1,
            )
            return out_tensor
        if n_elements == 3584 and x.is_contiguous():
            _pow_square_3584_hot_kernel[(1,)](
                x,
                out_tensor,
                num_warps=1,
                num_stages=1,
            )
            return out_tensor
        if n_elements == 2048 and x.is_contiguous():
            _pow_square_2048_hot_kernel[(1,)](
                x,
                out_tensor,
                num_warps=1,
                num_stages=1,
            )
            return out_tensor
        if _maybe_launch_pow_square_hotshape(x, out_tensor, n_elements):
            return out_tensor
    _launch_pow_kernel(kernel, single_kernel, x, out_tensor, n_elements, block_size)
    return out_tensor


def _maybe_prewarm_pow_kernels():
    global _PREWARM_POW_DONE
    if _PREWARM_POW_DONE:
        return
    if not _POW_PREWARM_ENABLED:
        _PREWARM_POW_DONE = True
        return
    try:
        for dt in (torch.float32, torch.bfloat16):
            x1024 = torch.ones((1, 1, 1024), dtype=dt, device="cpu")
            out1024 = torch.empty_like(x1024)
            _pow_square_1024_hot_kernel[(1,)](
                x1024,
                out1024,
                num_warps=1,
                num_stages=1,
            )

            x2048 = torch.ones((1, 16, 1, 128), dtype=dt, device="cpu")
            out2048 = torch.empty_like(x2048)
            _pow_square_2048_hot_kernel[(1,)](
                x2048,
                out2048,
                num_warps=1,
                num_stages=1,
            )

            rows = x2048.numel() // 128
            _pow_square_rows128_hot_kernel[(1,)](
                x2048,
                out2048,
                rows,
                MAX_ROWS=96,
                num_warps=1,
                num_stages=1,
            )

            x3584 = torch.ones((1, 1, 3584), dtype=dt, device="cpu")
            out3584 = torch.empty_like(x3584)
            _pow_square_3584_hot_kernel[(1,)](
                x3584,
                out3584,
                num_warps=1,
                num_stages=1,
            )

            x_rows3584 = torch.ones((1, 128, 3584), dtype=dt, device="cpu")
            out_rows3584 = torch.empty_like(x_rows3584)
            _pow_square_rows3584_hot_kernel[(1,)](
                x_rows3584,
                out_rows3584,
                128,
                MAX_ROWS=128,
                num_warps=1,
                num_stages=1,
            )

            block1024 = _select_block_size(x1024.numel(), x1024.dtype)
            _launch_pow_kernel(
                _pow_square_kernel,
                _pow_square_single_program_kernel,
                x1024,
                out1024,
                x1024.numel(),
                block1024,
            )
    except Exception:
        logger.debug("GEMS_ARM pow prewarm failed", exc_info=True)
    _PREWARM_POW_DONE = True


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_ARM POW_TENSOR_TENSOR")
    if (
        isinstance(A, torch.Tensor)
        and A.numel() < _POW_NATIVE_THRESHOLD
        and A.is_contiguous()
    ):
        return torch.from_numpy(
            np.power(
                A.detach().numpy(),
                (
                    float(exponent)
                    if not isinstance(exponent, torch.Tensor)
                    else exponent.detach().numpy()
                ),
            )
        )
    _maybe_prewarm_pow_kernels()
    scalar_exp = _maybe_scalar(exponent)
    if scalar_exp is not None:
        special = _pow_tensor_scalar_special(A, scalar_exp)
        if special is not None:
            return special
        return base_pow_tensor_scalar(A, scalar_exp)
    return base_pow_tensor_tensor(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_ARM POW_TENSOR_TENSOR_")
    _maybe_prewarm_pow_kernels()
    scalar_exp = _maybe_scalar(exponent)
    if scalar_exp is not None:
        special = _pow_tensor_scalar_special(A, scalar_exp, out=A)
        if special is not None:
            return special
        return base_pow_tensor_scalar_(A, scalar_exp)
    return base_pow_tensor_tensor_(A, exponent)


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_ARM POW_TENSOR_SCALAR")
    if (
        isinstance(A, torch.Tensor)
        and A.numel() < _POW_NATIVE_THRESHOLD
        and A.is_contiguous()
    ):
        exp = (
            float(exponent)
            if not isinstance(exponent, torch.Tensor)
            else exponent.item()
        )
        if exp == 2.0:
            an = A.detach().numpy()
            return torch.from_numpy(np.multiply(an, an))
        return torch.from_numpy(np.power(A.detach().numpy(), exp))
    _maybe_prewarm_pow_kernels()
    scalar_exp = _maybe_scalar(exponent)
    if scalar_exp is not None:
        special = _pow_tensor_scalar_special(A, scalar_exp)
        if special is not None:
            return special
        return base_pow_tensor_scalar(A, scalar_exp)
    return base_pow_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_ARM POW_TENSOR_SCALAR_")
    _maybe_prewarm_pow_kernels()
    scalar_exp = _maybe_scalar(exponent)
    if scalar_exp is not None:
        special = _pow_tensor_scalar_special(A, scalar_exp, out=A)
        if special is not None:
            return special
        return base_pow_tensor_scalar_(A, scalar_exp)
    return base_pow_tensor_scalar_(A, exponent)


def pow_scalar(A, exponent):
    logger.debug("GEMS_ARM POW_SCALAR")
    _maybe_prewarm_pow_kernels()
    return base_pow_scalar(A, exponent)


_maybe_prewarm_pow_kernels()
