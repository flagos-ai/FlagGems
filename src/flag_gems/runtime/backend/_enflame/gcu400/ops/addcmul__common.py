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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# Hyperparameters of the flat kernel (selected from R1-R6 measurements; changes require a resweep)
_GRID_CAP = 48  # Fixed grid cap corresponding to the GCU400 SIP topology
_TINY_ELEMS = 4096  # Smaller shapes only fill one program, so pick the block from the actual element count
_BLOCK_MID = 4096
_BLOCK_WIDE_FP32 = 32768
_BLOCK_WIDE_16BIT = 65536
# When the iteration count is large enough, making the trip count constexpr is needed to get full bandwidth (measured
# with 1G elements:
# ~18% difference); small shapes use a runtime trip count to avoid repeated recompilation per shape.
_EXACT_TRIP_THRESHOLD = 512


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def _addcmul_kernel(x, t1, t2, value):
    return x + value * t1 * t2


@triton.jit
def _addcmul_flat_dyn(
    X,
    T1,
    T2,
    Y,
    VALUE,
    NUM_TILE,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    HP: tl.constexpr,
):
    """Mask-free fast path with a runtime trip count.

    Only used when num_tile < _EXACT_TRIP_THRESHOLD, where it matches the constexpr version but
    does not recompile per shape. Do not add num_stages pipelining here: prefetching across the
    strided loop on GCU reads past the last tile and faults on sizes like (1,256,56,56).
    """
    pid = tl.program_id(0)
    for tile in tl.range(pid, NUM_TILE, GRID):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + off)
        t1 = tl.load(T1 + off)
        t2 = tl.load(T2 + off)
        if HP:
            y = (x.to(tl.float32) + VALUE * t1.to(tl.float32) * t2.to(tl.float32)).to(
                Y.dtype.element_ty
            )
        else:
            y = x + VALUE * t1 * t2
        tl.store(Y + off, y)


@triton.jit
def _addcmul_flat_exact(
    X,
    T1,
    T2,
    Y,
    VALUE,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    HP: tl.constexpr,
    NUM_TILE: tl.constexpr,
):
    """Mask-free fast path with a constexpr trip count, fully unrolled for large shapes."""
    pid = tl.program_id(0)
    for tile in tl.range(pid, NUM_TILE, GRID):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + off)
        t1 = tl.load(T1 + off)
        t2 = tl.load(T2 + off)
        if HP:
            y = (x.to(tl.float32) + VALUE * t1.to(tl.float32) * t2.to(tl.float32)).to(
                Y.dtype.element_ty
            )
        else:
            y = x + VALUE * t1 * t2
        tl.store(Y + off, y)


@triton.jit
def _addcmul_flat_masked(
    X,
    T1,
    T2,
    Y,
    N,
    VALUE,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    HP: tl.constexpr,
):
    """Masked version for when n is not a multiple of the block size."""
    pid = tl.program_id(0)
    num_tile = (N + BLOCK - 1) // BLOCK
    for tile in tl.range(pid, num_tile, GRID):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0)
        t1 = tl.load(T1 + off, mask=mask, other=0)
        t2 = tl.load(T2 + off, mask=mask, other=0)
        if HP:
            y = (x.to(tl.float32) + VALUE * t1.to(tl.float32) * t2.to(tl.float32)).to(
                Y.dtype.element_ty
            )
        else:
            y = x + VALUE * t1 * t2
        tl.store(Y + off, y, mask=mask)


@triton.jit
def _addcmul_flat_single(
    X,
    T1,
    T2,
    Y,
    VALUE,
    BLOCK: tl.constexpr,
    HP: tl.constexpr,
):
    """Single-tile one-shot version: no loop, no mask (requires n == BLOCK).

    Small shapes are almost entirely fixed overhead, so dropping the grid-stride loop beats the generic version.
    """
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    t1 = tl.load(T1 + off)
    t2 = tl.load(T2 + off)
    if HP:
        y = (x.to(tl.float32) + VALUE * t1.to(tl.float32) * t2.to(tl.float32)).to(
            Y.dtype.element_ty
        )
    else:
        y = x + VALUE * t1 * t2
    tl.store(Y + off, y)


def _flat_block_and_warps(n, element_size):
    """Picks block / num_warps from element count and element width (measured tiers).

    Small shapes fit in one tile and num_warps=8 squeezes the fixed overhead; for large shapes the
    block cap is set by element width (fp32 32768 / 16-bit 65536, the measured bandwidth optimum),
    and the middle tier requires at least 2*GRID tiles so the grid is never underfilled.
    """
    if n <= _TINY_ELEMS:
        return max(1024, triton.next_power_of_2(n)), 8
    cap = _BLOCK_WIDE_FP32 if element_size == 4 else _BLOCK_WIDE_16BIT
    block = max(_BLOCK_MID, triton.next_power_of_2(triton.cdiv(n, 2 * _GRID_CAP)))
    return min(cap, block), 4


def try_addcmul_flat(inp, tensor1, tensor2, value, out=None):
    """Takes the flat kernel when shape, dtype match and full contiguity all hold.

    When ``out`` is None it allocates one; returning the output tensor means handled, returning
    None means the caller must fall back to the pointwise codegen path. Checks are ordered
    cheapest-first to avoid extra Python overhead on the fallback path and small shapes.
    """
    if not isinstance(value, (int, float)):
        return None
    shape = inp.shape
    if tensor1.shape != shape or tensor2.shape != shape:
        return None
    if inp.dtype != tensor1.dtype or inp.dtype != tensor2.dtype:
        return None
    if not (
        inp.is_contiguous() and tensor1.is_contiguous() and tensor2.is_contiguous()
    ):
        return None
    if out is None:
        out = torch.empty_like(inp)
    elif out.shape != shape:
        return None
    elif out.dtype != inp.dtype or not out.is_contiguous():
        return None

    n = inp.numel()
    if n == 0:
        return out

    block, num_warps = _flat_block_and_warps(n, inp.element_size())
    num_tile = triton.cdiv(n, block)
    grid = min(_GRID_CAP, num_tile)
    high_precision = inp.dtype in (torch.float16, torch.bfloat16)
    if num_tile == 1 and n == block:
        _addcmul_flat_single[(1,)](
            inp,
            tensor1,
            tensor2,
            out,
            value,
            block,
            high_precision,
            num_warps=num_warps,
        )
    elif num_tile == 1 or n % block:
        _addcmul_flat_masked[(grid,)](
            inp,
            tensor1,
            tensor2,
            out,
            n,
            value,
            block,
            _GRID_CAP,
            high_precision,
            num_warps=num_warps,
        )
    elif n // block >= _EXACT_TRIP_THRESHOLD:
        _addcmul_flat_exact[(grid,)](
            inp,
            tensor1,
            tensor2,
            out,
            value,
            block,
            _GRID_CAP,
            high_precision,
            n // block,
            num_warps=num_warps,
        )
    else:
        _addcmul_flat_dyn[(grid,)](
            inp,
            tensor1,
            tensor2,
            out,
            value,
            n // block,
            block,
            _GRID_CAP,
            high_precision,
            num_warps=num_warps,
        )
    return out


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    logger.debug("GEMS_ENFLAME ADDCMUL_OUT")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    if list(out.shape) != list(broadcast_shape):
        out.resize_(broadcast_shape)
    flat_out = try_addcmul_flat(inp, tensor1, tensor2, value, out)
    if flat_out is not None:
        return flat_out
    _addcmul_kernel(inp, tensor1, tensor2, value, out0=out)
    return out


def addcmul(inp, tensor1, tensor2, *, value=1.0):
    """Functional entry; keep alongside ``addcmul.out`` for dispatch coverage."""
    logger.debug("GEMS_ENFLAME ADDCMUL")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    dtype = torch.promote_types(
        inp.dtype, torch.promote_types(tensor1.dtype, tensor2.dtype)
    )
    out = torch.empty(broadcast_shape, device=inp.device, dtype=dtype)
    return addcmul_out(inp, tensor1, tensor2, value=value, out=out)


def addcmul_(inp, tensor1, tensor2, *, value=1.0):
    logger.debug("GEMS_ENFLAME ADDCMUL_")
    if try_addcmul_flat(inp, tensor1, tensor2, value, inp) is not None:
        return inp
    _addcmul_kernel(inp, tensor1, tensor2, value, out0=inp)
    return inp
