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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def powsum_kernel(
    inp,
    out,
    M,
    N,
    ord,
    IS_COMPLEX: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    inp_dtype = inp.type.element_ty
    acc_dtype = tl.float64 if inp_dtype == tl.float64 else tl.float32
    ord = ord.to(acc_dtype)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        offsets = rows[:, None] * N + cols[None, :]
        if IS_COMPLEX:
            real = tl.load(inp + 2 * offsets, mask, other=0.0).to(acc_dtype)
            imag = tl.load(inp + 2 * offsets + 1, mask, other=0.0).to(acc_dtype)
            value = tl.sqrt(real * real + imag * imag)
        else:
            value = tl.load(inp + offsets, mask, other=0.0)
            value = tl.abs(value.to(acc_dtype))
        if MODE == 0:
            term = pow(value, ord)
        elif MODE == 1:
            term = tl.full(value.shape, 1.0, acc_dtype)
        elif MODE == 2:
            term = value
        elif MODE == 3:
            term = value * value
        else:
            term = value * value * value
        acc += tl.where(mask, term, 0.0)

    result = tl.sum(acc, axis=1)
    tl.store(out + rows, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def powsum_kernel_1(
    inp,
    mid,
    size,
    ord,
    IS_COMPLEX: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    inp_dtype = inp.type.element_ty
    acc_dtype = tl.float64 if inp_dtype == tl.float64 else tl.float32
    ord = ord.to(acc_dtype)
    if IS_COMPLEX:
        real = tl.load(inp + 2 * offsets, mask, other=0.0).to(acc_dtype)
        imag = tl.load(inp + 2 * offsets + 1, mask, other=0.0).to(acc_dtype)
        value = tl.sqrt(real * real + imag * imag)
    else:
        value = tl.load(inp + offsets, mask, other=0.0)
        value = tl.abs(value.to(acc_dtype))
    if MODE == 0:
        term = pow(value, ord)
    elif MODE == 1:
        term = tl.full(value.shape, 1.0, acc_dtype)
    elif MODE == 2:
        term = value
    elif MODE == 3:
        term = value * value
    else:
        term = value * value * value
    term = tl.where(mask, term, 0.0)
    tl.store(mid + pid, tl.sum(term))


@libentry()
@triton.jit
def powsum_kernel_2(mid, out, mid_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(mid + offsets, offsets < mid_size, other=0.0)
    tl.store(out, tl.sum(values))


def _normalize_dims(dim, ndim):
    if dim is None or dim == [] or dim == ():
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    if ndim == 0:
        if any(d not in (-1, 0) for d in dim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [-1, 0])"
            )
        if len(dim) > 1:
            raise RuntimeError("dim 0 appears multiple times in the list of dims")
        return []
    if any(d < -ndim or d >= ndim for d in dim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}])"
        )
    dims = [d % ndim for d in dim]
    if len(set(dims)) != len(dims):
        raise RuntimeError("dim appears multiple times in the list of dims")
    return dims


def _output_dtype(inp, dtype):
    dtype = inp.dtype if dtype is None else dtype
    if dtype == torch.complex64:
        return torch.complex64, torch.float32
    if dtype == torch.complex128:
        return torch.complex128, torch.float64
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(f"linalg__powsum not implemented for {dtype}")
    return dtype, dtype


def linalg__powsum(inp, ord=2, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS LINALG__POWSUM")
    compute_dtype, output_dtype = _output_dtype(inp, dtype)
    if inp.dtype != compute_dtype and not (
        inp.is_complex() and not compute_dtype.is_complex
    ):
        inp = inp.to(compute_dtype)
    is_complex = inp.is_complex()

    dims = _normalize_dims(dim, inp.ndim)
    shape = list(inp.shape)
    reduced_shape = list(shape)
    for d in dims:
        reduced_shape[d] = 1
    out_shape = (
        reduced_shape
        if keepdim
        else [size for i, size in enumerate(shape) if i not in set(dims)]
    )

    if inp.numel() == 0:
        return torch.zeros(out_shape, dtype=output_dtype, device=inp.device)

    mode = 1 if ord == 0 else 2 if ord == 1 else 3 if ord == 2 else 4 if ord == 3 else 0
    with torch_device_fn.device(inp.device):
        if len(dims) == inp.ndim:
            inp = inp.contiguous()
            size = inp.numel()
            if is_complex:
                inp = torch.view_as_real(inp)
            block_size = triton.next_power_of_2(math.ceil(math.sqrt(size)))
            mid_size = triton.cdiv(size, block_size)
            block_mid = triton.next_power_of_2(mid_size)
            accumulation_dtype = (
                torch.float64 if compute_dtype == torch.float64 else torch.float32
            )
            if size <= 65536:
                out = torch.empty([], dtype=output_dtype, device=inp.device)
                powsum_kernel_1[(1,)](
                    inp,
                    out,
                    size,
                    ord,
                    is_complex,
                    mode,
                    BLOCK_SIZE=triton.next_power_of_2(size),
                )
                if keepdim:
                    out = out.reshape([1] * inp.ndim)
                return out
            mid = torch.empty(mid_size, dtype=accumulation_dtype, device=inp.device)
            out = torch.empty([], dtype=output_dtype, device=inp.device)
            powsum_kernel_1[(mid_size,)](
                inp, mid, size, ord, is_complex, mode, BLOCK_SIZE=block_size
            )
            powsum_kernel_2[(1,)](mid, out, mid_size, BLOCK_SIZE=block_mid)
            if keepdim:
                out = out.reshape([1] * inp.ndim)
            return out

        inp = dim_compress(inp, dims)
        reduction_size = math.prod(shape[d] for d in dims)
        rows = inp.numel() // reduction_size
        if is_complex:
            inp = torch.view_as_real(inp)
        out = torch.empty(rows, dtype=output_dtype, device=inp.device)
        grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
        powsum_kernel[grid](inp, out, rows, reduction_size, ord, is_complex, mode)
    return out.reshape(out_shape)
