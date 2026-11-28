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

"""Element-wise less-than, GCU400-optimized.

The ``pointwise_dynamic`` path (gcu400-local codegen: 48-CTA grid cap,
65536-lane block-pointer tiles, fp32 up-cast) keeps the small, strided,
broadcast and non-float cases.

Large contiguous same-dtype float inputs take a flat capped grid-stride kernel:

* comparisons run in the native input dtype (the fp32 up-cast only wastes ALU);
* the scalar overload casts the Python scalar to ``A.dtype`` before comparing,
  matching torch TensorIterator semantics (the fp32-compare pointwise kernel
  deviates from torch on fp16 round-down scalars, e.g. 0.1);
* mask-free (``EVEN``) arm whenever N is an exact multiple of BLOCK;
* grid capped to a single 24-SM wave above 2**18 elements — the previous
  per-block-per-CTA flat kernel and the COV/CHUNK loop kernel both quantised
  2**22 / 2**24 fp32 into exactly 16 CTAs (< 24 SMs) and lost 0.79-0.91x
  against native.

2026-11-17 A/B (same board, interleaved 3-round medians, do_bench median),
replacing the old flat/loop ladder with this kernel at BLOCK 32768 (fp32) /
65536 (fp16/bf16), num_warps 2, grid cap 24:

  tensor (speedup vs torch native, before -> after):
    fp32 2**22  0.856x -> 1.102x   fp32 2**24  0.793x -> 1.033x
    fp32 2**26  0.913x -> 1.011x
    fp16 2**22  0.987x -> 1.210x   fp16 2**24  0.838x -> 1.069x
    fp16 2**26  0.808x -> 1.022x
    bf16 2**22  0.958x -> 1.189x   bf16 2**24  0.829x -> 1.059x
    bf16 2**26  0.805x -> 1.019x
"""

import logging
import struct

import numpy as np
import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
# Single-wave grid threshold: from >= 2**18 elements, shrink the grid to 24 CTAs (one SM wave);
# smaller shapes still allow two waves, avoiding single-digit CTA counts for small blocks.
_MID_N = 2**18
_NUM_SIPS = 24
_TWO_WAVE_CAP = 2 * _NUM_SIPS
_INT32_SAFE_LIMIT = 0x7FFFFFFF


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


@triton.jit(do_not_specialize=["N"])
def _lt_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(A + off)
            y = tl.load(B + off)
            tl.store(OUT + off, (x < y).to(OUT.dtype.element_ty))
        else:
            mask = off < N
            x = tl.load(A + off, mask=mask)
            y = tl.load(B + off, mask=mask)
            tl.store(OUT + off, (x < y).to(OUT.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["N"])
def _lt_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(A + off)
            tl.store(OUT + off, (x < Y.to(x.dtype)).to(OUT.dtype.element_ty))
        else:
            mask = off < N
            x = tl.load(A + off, mask=mask)
            tl.store(OUT + off, (x < Y.to(x.dtype)).to(OUT.dtype.element_ty), mask=mask)


def round_scalar_to_dtype(value, dtype):
    """Rounds a Python scalar to ``dtype`` following torch semantics.

    The pointwise fallback compares in the fp32 domain; without rounding the scalar to
    ``A.dtype`` first, ``x < 0.1`` under fp16/bf16 disagrees with torch (which rounds the
    scalar to fp16/bf16). Reproduced at fp16 with n = 65536 / 131072 / 65537.
    """
    v = float(value)
    if dtype == torch.float32 or v != v:
        return v
    if dtype == torch.float16:
        return float(np.float16(v))
    # bfloat16: the fp32 mantissa is truncated from 23 bits to 8 bits, round-to-nearest-even.
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _stride_config(n, itemsize):
    """(tile, grid, EVEN) triple: 32768 for 4-byte lanes, 65536 for 2-byte lanes.

    For small shapes tile is squeezed to ``np2(n // 48)`` so the grid still has two waves of
    CTAs (a pure launch-overhead regime where the tile size does not affect bandwidth).
    """
    block = 32768 if itemsize >= 4 else 65536
    if n < _MID_N:
        block = min(
            block, max(1024, triton.next_power_of_2(max(1, n // _TWO_WAVE_CAP)))
        )
    cap = _NUM_SIPS if n >= _MID_N else _TWO_WAVE_CAP
    grid = (min(triton.cdiv(n, block), cap),)
    return block, grid, n % block == 0


def _lt_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    block, grid, even = _stride_config(n, A.element_size())
    if triton.cdiv(n, block) * block > _INT32_SAFE_LIMIT:
        return False
    _lt_cmp_kernel[grid](A, B, out, n, BLOCK=block, EVEN=even, num_warps=2)
    return True


def lt(A, B):
    logger.debug("GEMS_ENFLAME LT")
    if (
        A.numel() >= _SMALL_N
        and A.device == B.device
        and A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _lt_cmp_run(A, B, out):
            return out
    return lt_func(A, B)


def lt_scalar(A, B):
    logger.debug("GEMS_ENFLAME LT_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n > 0:
            block, grid, even = _stride_config(n, A.element_size())
            if triton.cdiv(n, block) * block <= _INT32_SAFE_LIMIT:
                out = torch.empty_like(A, dtype=torch.bool)
                _lt_cmp_scalar_kernel[grid](
                    A, out, n, float(B), BLOCK=block, EVEN=even, num_warps=2
                )
                return out
    # The fallback path compares in the fp32 domain; scalars must first be rounded to A.dtype following torch semantics.
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES:
        B = round_scalar_to_dtype(B, A.dtype)
    return lt_func_scalar(A, B)
