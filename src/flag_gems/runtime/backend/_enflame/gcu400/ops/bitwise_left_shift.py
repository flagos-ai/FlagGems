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

"""GCU400-specific bitwise left shift: same-shape contiguous tensors use a 1D flat kernel.

The codegen path (tile 65536 + block pointer) on a 48-SIP Fixed Grid has uneven tail waves at medium sizes
(e.g. 16M elements, 256 tiles / 48 CTAs = 5.33 waves), measuring 5~14% slower than
torch native; a 1D flat kernel (smaller BLOCK, grid-stride balancing) can
match or beat it. For very large int32 (>=128M) the bptr DMA streaming path is faster, so fall back to codegen.

int64: the codegen i64 same-shape tensor-tensor path reads the wrong operand on this machine (wrong values / occasional
SIP exception), so use an i32 view + u32 half-word emulation (x86 shl semantics), see
_lshift_i64_flat_kernel.

int64 without TORCH_GCU_ENABLE_INT64_AND_UINT64 (P17): the torch_gcu transport narrows
int64 to "the first N int32 words (low 32 bits only)" storage; `.view(torch.int32)`
gives 2N elements and codegen reads 8B-wide interleaved garbage. In that case use
_narrowed_i32_alias (reinterpret as a same-shape int32 view by the real physical layout) and reuse the int32
path: the input was already truncated at H2D, so int32 computation matches the reference semantics after D2H (shift
counts >=32 or int32 result overflow are inherent losses of the narrowing platform, not recoverable here).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

FLAT_SIPS = 24
# Beyond this int32 scale the codegen bptr/DMA path is faster (measured 256M/1G: 1.047x/1.055x
# vs flat32768w2 1.038x/1.045x), so fall back; int16 flat wins at all scales, so no threshold there.
FLAT_INT32_MAX_NUMEL = 128 * 1024 * 1024
FLAT_MIN_BLOCK = 256
FLAT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
FLAT_I64_BLOCK = 8192

_I64_NARROWED = None


def _int64_narrowed(sample):
    """Detect whether the transport narrows int64 into an int32 word layout (P17, cached once per process)."""
    global _I64_NARROWED
    if _I64_NARROWED is None:
        probe = torch.tensor([(1 << 32) + 1], dtype=torch.int64, device="cpu")
        _I64_NARROWED = probe.to(sample.device).cpu().item() != (1 << 32) + 1
    return _I64_NARROWED


def _narrowed_i32_alias(t):
    """Narrowed-layout int64 tensor -> same-shape int32 view (element i is physical word i)."""
    alias = torch.empty(0, dtype=torch.int32, device=t.device)
    alias.set_(t.untyped_storage(), t.storage_offset(), t.shape, t.stride())
    return alias


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lshift_flat_kernel(A, B, Out, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    num_programs = tl.num_programs(0)
    for block in tl.range(tl.program_id(0), tl.cdiv(N, BLOCK_SIZE), num_programs):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN_N:
            a = tl.load(A + offsets)
            b = tl.load(B + offsets)
            tl.store(Out + offsets, a << b)
        else:
            mask = offsets < N
            a = tl.load(A + offsets, mask=mask, other=0)
            b = tl.load(B + offsets, mask=mask, other=0)
            tl.store(Out + offsets, a << b, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lshift_i64_flat_kernel(
    A, B, Out, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr
):
    # i32 safe-domain emulation of int64 left shift: GCU codegen's i64 same-shape tensor-tensor path
    # reads the wrong operand (measured wrong values / occasional SIP exception), so treat i64 as little-endian
    # (lo, hi) i32 pairs (host-side .view(torch.int32)) and rebuild x86 shl semantics exactly with pure i32 shifts
    # (count takes the low 6 bits). The only logical right shift uses a uint32 bitcast.
    num_programs = tl.num_programs(0)
    for block in tl.range(tl.program_id(0), tl.cdiv(N, BLOCK_SIZE), num_programs):
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        o32 = offs * 2
        if EVEN_N:
            a_lo = tl.load(A + o32)
            a_hi = tl.load(A + o32 + 1)
            b_lo = tl.load(B + o32)
        else:
            m32 = o32 < 2 * N
            a_lo = tl.load(A + o32, mask=m32, other=0)
            a_hi = tl.load(A + o32 + 1, mask=m32, other=0)
            b_lo = tl.load(B + o32, mask=m32, other=0)
        k = b_lo & 63
        kk = k & 31
        small = k < 32
        amt = tl.minimum(32 - kk, 31)
        carry_u = a_lo.to(tl.uint32, bitcast=True) >> amt
        carry = tl.where(kk == 0, 0, carry_u.to(tl.int32, bitcast=True))
        r_lo = tl.where(small, a_lo << kk, 0)
        r_hi = tl.where(small, (a_hi << kk) | carry, a_lo << kk)
        if EVEN_N:
            tl.store(Out + o32, r_lo)
            tl.store(Out + o32 + 1, r_hi)
        else:
            m32 = o32 < 2 * N
            tl.store(Out + o32, r_lo, mask=m32)
            tl.store(Out + o32 + 1, r_hi, mask=m32)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_func(x, y):
    return x << y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_func_scalar(x, y):
    return x << y


def _lshift_use_flat(A, B):
    return (
        isinstance(B, torch.Tensor)
        and B.shape == A.shape
        and B.dtype == A.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.dim() >= 1
        and A.numel() > 0
        and A.dtype in FLAT_DTYPES
        and not (
            A.element_size() == 4 and A.numel() >= FLAT_INT32_MAX_NUMEL
        )  # Large int32 uses codegen bptr/DMA
    )


def _lshift_flat_params(numel, itemsize):
    if itemsize <= 2:
        block = 65536
    elif itemsize == 4:
        block = 32768
    else:
        block = FLAT_I64_BLOCK
    if numel < block:
        block = max(triton.next_power_of_2(numel), FLAT_MIN_BLOCK)
    num_warps = 2 if block >= 32768 else 4
    grid = (1 if numel <= block else min(triton.cdiv(numel, block), FLAT_SIPS),)
    return block, num_warps, grid


def _lshift_flat_launch(A, B, out):
    numel = A.numel()
    block, num_warps, grid = _lshift_flat_params(numel, A.element_size())
    with torch_device_fn.device(A.device):
        if A.element_size() == 8:
            _lshift_i64_flat_kernel[grid](
                A.view(torch.int32),
                B.view(torch.int32),
                out.view(torch.int32),
                numel,
                BLOCK_SIZE=block,
                EVEN_N=(numel % block == 0),
                num_warps=num_warps,
            )
        else:
            _lshift_flat_kernel[grid](
                A,
                B,
                out,
                numel,
                BLOCK_SIZE=block,
                EVEN_N=(numel % block == 0),
                num_warps=num_warps,
            )
    return out


def bitwise_left_shift(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_LEFT_SHIFT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64 and B.dtype == torch.int64 and _int64_narrowed(A):
            a32 = _narrowed_i32_alias(A)
            b32 = _narrowed_i32_alias(B)
            out = torch.empty_like(A)
            out32 = _narrowed_i32_alias(out)
            if _lshift_use_flat(a32, b32):
                _lshift_flat_launch(a32, b32, out32)
            else:
                bitwise_left_shift_func(a32, b32, out0=out32)
            return out
        if _lshift_use_flat(A, B):
            return _lshift_flat_launch(A, B, torch.empty_like(A))
        return bitwise_left_shift_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64 and _int64_narrowed(A):
            out = torch.empty_like(A)
            bitwise_left_shift_func_scalar(
                _narrowed_i32_alias(A), B, out0=_narrowed_i32_alias(out)
            )
            return out
        return bitwise_left_shift_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64 and _int64_narrowed(B):
            out = torch.empty_like(B)
            bitwise_left_shift_func_scalar(
                _narrowed_i32_alias(B), A, out0=_narrowed_i32_alias(out)
            )
            return out
        return bitwise_left_shift_func_scalar(B, A)
    else:
        return torch.tensor(A * B)


def bitwise_left_shift_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_LEFT_SHIFT_")
    if isinstance(B, torch.Tensor):
        if A.dtype == torch.int64 and B.dtype == torch.int64 and _int64_narrowed(A):
            a32 = _narrowed_i32_alias(A)
            b32 = _narrowed_i32_alias(B)
            if _lshift_use_flat(a32, b32):
                _lshift_flat_launch(a32, b32, a32)
            else:
                bitwise_left_shift_func(a32, b32, out0=a32)
            return A
        if _lshift_use_flat(A, B):
            # Common same-shape same-dtype in-place cases use the 1D flat kernel, avoiding codegen
            # wrapper index decomposition and dtype promotion (small dtypes are slower on the codegen path).
            return _lshift_flat_launch(A, B, A)
        return bitwise_left_shift_func(A, B, out0=A)
    else:
        if A.dtype == torch.int64 and _int64_narrowed(A):
            a32 = _narrowed_i32_alias(A)
            bitwise_left_shift_func_scalar(a32, B, out0=a32)
            return A
        return bitwise_left_shift_func_scalar(A, B, out0=A)
