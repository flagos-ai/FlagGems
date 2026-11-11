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

logger = logging.getLogger(__name__)

_REAL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


# ---------------------------------------------------------------------------
# Real dtype path: every element of a real tensor is real (unconditionally
# True), including NaN/infinity.  The input is never read, so this is a pure
# constant fill.  Writing 4 booleans at a time through an int32 view
# (`0x01010101` -> 4 x 0x01 == True) widens the store and cuts the number of
# store instructions 4x, which on XPU is the difference between a byte-at-a-time
# fill and a near-block-DMA-rate fill (measured: 16.7M bool 21.8us vs 26us for
# the byte-wide store).
# ---------------------------------------------------------------------------
@triton.jit
def isreal_true_pack_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, 0x01010101)


@triton.jit
def isreal_true_pack_masked_kernel(out_ptr, n32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, 0x01010101, mask=offsets < n32)


@triton.jit
def isreal_true_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, 1, mask=offsets < n_elements)


# ---------------------------------------------------------------------------
# complex64 path: the interleaved (real, imag) fp32 storage is read as a single
# contiguous int64 stream, and the imaginary component is recovered from the
# high 32 bits in registers.  This is fully coalesced (no stride-2 load) and is
# the only fast way to reach the imaginary plane on this backend (the plain
# stride-2 plane view cannot vectorize).  The bit-level test
# `(imag_bits & 0x7FFFFFFF) == 0` keeps negative zero classified as real.
# ---------------------------------------------------------------------------
@triton.jit
def isreal_complex64_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    packed = tl.load(in_ptr + offsets)
    imag_bits = (packed >> 32).to(tl.int32) & 0x7FFFFFFF
    tl.store(out_ptr + offsets, imag_bits == 0)


@triton.jit
def isreal_complex64_masked_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    packed = tl.load(in_ptr + offsets, mask=mask, other=0)
    imag_bits = (packed >> 32).to(tl.int32) & 0x7FFFFFFF
    tl.store(out_ptr + offsets, imag_bits == 0, mask=mask)


# ---------------------------------------------------------------------------
# complex128 path: no 16-byte integer supertype exists in Triton, so the
# imaginary component is loaded with a stride-2 fp64 load and tested bitwise.
# ---------------------------------------------------------------------------
@triton.jit
def isreal_complex128_kernel(in_real_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    imag = tl.load(in_real_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    bits = imag.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
    tl.store(out_ptr + offsets, bits == 0, mask=mask)


def _pick_int32_fill_block(n32: int) -> int:
    if n32 >= 1 << 20:
        return 32768
    if n32 >= 1 << 16:
        return 8192
    if n32 >= 1 << 12:
        return 2048
    return 1024


def _pick_complex64_block(n: int) -> int:
    if n >= 1 << 22:
        return 32768
    if n >= 1 << 18:
        return 16384
    if n >= 1 << 14:
        return 8192
    return 1024


def _fill_true(out: torch.Tensor, n: int) -> None:
    if n % 4 == 0:
        out32 = out.reshape(-1).view(torch.int32)
        n32 = n // 4
        block = _pick_int32_fill_block(n32)
        if n32 % block == 0:
            grid = (n32 // block,)
            isreal_true_pack_kernel[grid](out32, BLOCK_SIZE=block, num_warps=4)
        else:
            grid = (triton.cdiv(n32, block),)
            isreal_true_pack_masked_kernel[grid](out32, n32, BLOCK_SIZE=block, num_warps=4)
    else:
        block = 1024
        grid = (triton.cdiv(n, block),)
        isreal_true_kernel[grid](out, n, BLOCK_SIZE=block, num_warps=4)


def _isreal_complex64(packed: torch.Tensor, out: torch.Tensor, n: int) -> None:
    block = _pick_complex64_block(n)
    if n % block == 0:
        grid = (n // block,)
        isreal_complex64_kernel[grid](packed, out, BLOCK_SIZE=block, num_warps=8)
    else:
        grid = (triton.cdiv(n, block),)
        isreal_complex64_masked_kernel[grid](packed, out, n, BLOCK_SIZE=block, num_warps=8)


def isreal(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISREAL")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"isreal expects a Tensor, got {type(input).__name__}")
    if input.dtype not in _REAL_DTYPES + _COMPLEX_DTYPES:
        raise TypeError(
            "isreal only supports floating point, complex, integral and bool "
            f"inputs, got {input.dtype}"
        )

    out = torch.empty(input.size(), dtype=torch.bool, device=input.device)
    n = input.numel()
    if n == 0:
        return out

    with torch_device_fn.device(input.device):
        if input.dtype in _COMPLEX_DTYPES:
            # A lazy conjugate or negation only flips the sign of the imaginary
            # component, which cannot turn a zero into a non-zero, so both bits
            # may be ignored.  They must still be resolved before taking the
            # storage view, since torch.view_as_real rejects lazy
            # conjugate/negation tensors.
            if input.is_conj():
                input = input.resolve_conj()
            if input.is_neg():
                input = input.resolve_neg()
            if not input.is_contiguous():
                input = input.contiguous()

            if input.dtype == torch.complex64:
                rv = torch.view_as_real(input)
                if not rv.is_contiguous():
                    rv = rv.contiguous()
                packed = rv.reshape(-1).view(torch.int64)
                _isreal_complex64(packed, out, n)
            else:
                isreal_complex128_kernel[(triton.cdiv(n, 1024),)](
                    torch.view_as_real(input),
                    out,
                    n,
                    BLOCK_SIZE=1024,
                    num_warps=4,
                )
        else:
            _fill_true(out, n)
    return out
