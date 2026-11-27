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
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Streaming regime (>= 16MB per tensor, re-tuned 2026-11-16 on the
# DEFAULT_SHAPES benchmark mix, interleaved A/B): deep DMA bursts with a
# doubled tile beat 48-way shallow streams once the working set leaves the
# small-workload bandwidth ramp and the DSM footprint (3 streams x BLOCK x
# itemsize, 786KB/CTA) exceeds the 448KB per-SPC budget, which pins one
# program per SPC (24-way resident on 48 SPCs).
#   int16 1G   2.215 -> 2.108 ms (3.05 TB/s, at the DRAM wall)
#   int16 16M  42.2  -> 40.1 us;  int32 16M 76.6 -> 73.4 us
_STREAM_BLOCK = {1: 131072, 2: 131072, 4: 65536, 8: 4096}
FLAT_SIPS = 24
FLAT_MIN_BLOCK = 256
# Below the threshold the ramp is latency-bound and parallelism wins (crossover
# sweep 2026-11-16, interleaved 3 rounds): the 16384/8192-lane ladder under 48
# CTAs is 19-40% ahead at 0.5-4M elements (e.g. int16 524288: 5.6 vs 7.3 us)
# while the streaming tile only takes over from ~16MB per tensor on.
_STREAM_MIN_BYTES = 16 * 1024 * 1024
# Ladder tiles for the mid regime (the rshift family ladder): mask-free when
# the size divides the tile exactly; 8-bit data keeps the predicate (a masked
# 16384-lane tile measured 1.5-2% ahead of the mask-free one there).
_LADDER_EVEN_TARGET = {1: 16384, 2: 16384, 4: 8192, 8: 4096}
_LADDER_MASKED_TARGET = {1: 16384, 2: 8192, 4: 8192, 8: 4096}
_LADDER_MAX_GRID = 48
# The scalar variant moves one tensor fewer: its streaming crossover matches
# the two-tensor one (interleaved 3 rounds, 2026-11-16: int16 1G 1.341 vs
# 1.445 ms, int32 1G 2.656 vs 2.892 ms for stream vs ladder), so it shares
# the byte threshold; below it the rshift scalar ladder applies.
_SCALAR_STREAM_BLOCK = {1: 131072, 2: 131072, 4: 65536, 8: 8192}
_SCALAR_EVEN_TARGET = {1: 16384, 2: 16384, 4: 8192, 8: 8192}
_SCALAR_MASKED_TARGET = {1: 16384, 2: 8192, 4: 8192, 8: 8192}
_NUM_WARPS = 4
_SCALAR_MAX_GRID = 48

_FLAT_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "shift"])
def _bitwise_right_shift_scalar_kernel_jit(
    x_ptr,
    out_ptr,
    n_elements,
    shift,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Capped grid with a grid-stride loop: one CTA per tile overflows grid.x
    # (65535) past ~268M elements.
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            value = tl.load(x_ptr + offsets)
            # `shift` arrives as an int64 kernel argument; bring it to the
            # element width so the backend never needs a mixed-width shift.
            tl.store(out_ptr + offsets, value >> shift.to(value.dtype))
        else:
            mask = offsets < n_elements
            value = tl.load(x_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, value >> shift.to(value.dtype), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _bitwise_right_shift_tensor_kernel_jit(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            value = tl.load(x_ptr + offsets)
            shift = tl.load(y_ptr + offsets)
            tl.store(out_ptr + offsets, value >> shift)
        else:
            mask = offsets < n_elements
            value = tl.load(x_ptr + offsets, mask=mask)
            shift = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, value >> shift, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func(x, y):
    return x >> y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func_scalar(x, y):
    return x >> y


# ATen's Scalar_Tensor overload treats the scalar as the shift amount, so the tensor-scalar kernel cannot be reused.
@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func_scalar_tensor(x, y):
    return x >> y


def _use_flat(value, other, out):
    """Whether the 1-D fast path applies to this call."""
    if value.dtype not in _FLAT_DTYPES:
        return False
    tensors = (value,)
    if torch.is_tensor(other):
        if other.dtype != value.dtype or other.device != value.device:
            # Mixed integer widths need the promotion machinery.
            return False
        tensors += (other,)
    if out is not None:
        tensors += (out,)
    # The flat kernel walks every operand with the same linear index, so
    # broadcasting and the out-shape validation must stay on the pointwise
    # path; a linear walk over a broadcast operand would read out of bounds.
    if any(t.shape != value.shape or t.device != value.device for t in tensors):
        return False
    # The flat kernel addresses storage linearly, so every operand must be
    # dense; anything else keeps the stride-aware pointwise path.
    return all(t.is_contiguous() for t in tensors)


def _shift_fits(value_dtype, shift):
    """ATen wraps a Scalar shift to the tensor dtype; keep exotic values on
    the promotion path instead of truncating them in the flat kernel."""
    bits = value_dtype.itemsize * 8
    limit = 1 << (bits - 1)
    return -limit <= shift <= limit - 1


def _tensor_flat_params(numel, element_size):
    """Two-tensor launch: streaming tiles once the working set is
    bandwidth-bound, the parallelism ladder below the crossover."""
    if numel * element_size >= _STREAM_MIN_BYTES:
        block = _STREAM_BLOCK.get(element_size, 4096)
        if numel < block:
            block = max(triton.next_power_of_2(numel), FLAT_MIN_BLOCK)
        grid = 1 if numel <= block else min(triton.cdiv(numel, block), FLAT_SIPS)
        return block, grid, numel % block == 0
    limit = triton.next_power_of_2(numel)
    block = min(_LADDER_EVEN_TARGET.get(element_size, 4096), limit)
    if numel % block == 0 and element_size > 1:
        return block, min(triton.cdiv(numel, block), _LADDER_MAX_GRID), True
    block = min(_LADDER_MASKED_TARGET.get(element_size, 4096), limit)
    even = numel % block == 0 and element_size > 1
    return block, min(triton.cdiv(numel, block), _LADDER_MAX_GRID), even


def _scalar_flat_params(numel, element_size):
    if numel * element_size >= _STREAM_MIN_BYTES:
        block = _SCALAR_STREAM_BLOCK.get(element_size, 8192)
        if numel < block:
            block = max(triton.next_power_of_2(numel), FLAT_MIN_BLOCK)
        grid = 1 if numel <= block else min(triton.cdiv(numel, block), FLAT_SIPS)
        return block, grid, numel % block == 0
    limit = triton.next_power_of_2(numel)
    block = min(_SCALAR_EVEN_TARGET.get(element_size, 8192), limit)
    if numel % block != 0 or element_size == 1:
        block = min(_SCALAR_MASKED_TARGET.get(element_size, 8192), limit)
    even = numel % block == 0 and element_size > 1
    return block, min(triton.cdiv(numel, block), _SCALAR_MAX_GRID), even


def _launch(value, other, out):
    n_elements = value.numel()
    if out is None:
        out = torch.empty_like(value, memory_format=torch.contiguous_format)
    if n_elements == 0:
        return out
    element_size = max(value.element_size(), 1)
    if torch.is_tensor(other):
        block, grid_size, even = _tensor_flat_params(n_elements, element_size)
        grid = (grid_size,)
        with torch_device_fn.device(value.device):
            _bitwise_right_shift_tensor_kernel_jit[grid](
                value,
                other,
                out,
                n_elements,
                BLOCK=block,
                GRID_SIZE=grid_size,
                EVEN=even,
                num_warps=_NUM_WARPS,
            )
    else:
        block, grid_size, even = _scalar_flat_params(n_elements, element_size)
        grid = (grid_size,)
        with torch_device_fn.device(value.device):
            _bitwise_right_shift_scalar_kernel_jit[grid](
                value,
                out,
                n_elements,
                int(other),
                BLOCK=block,
                GRID_SIZE=grid_size,
                EVEN=even,
                num_warps=_NUM_WARPS,
            )
    return out


def _flat_tensor_or_scalar(value, other, out):
    if _use_flat(value, other, out):
        if torch.is_tensor(other) or _shift_fits(value.dtype, int(other)):
            return _launch(value, other, out)
    return None


def bitwise_right_shift(A, B, *, out=None):
    if isinstance(A, torch.Tensor):
        if isinstance(B, torch.Tensor):
            if _use_flat(A, B, out):
                return _launch(A, B, out)
            if out is None:
                return bitwise_right_shift_func(A, B)
            bitwise_right_shift_func(A, B, out0=out)
            return out
        flat = _flat_tensor_or_scalar(A, B, out)
        if flat is not None:
            return flat
        if out is None:
            return bitwise_right_shift_func_scalar(A, B)
        bitwise_right_shift_func_scalar(A, B, out0=out)
        return out
    elif isinstance(B, torch.Tensor):
        # Scalar_Tensor: the scalar is the amount to shift by; broadcasting is handled by the pointwise path.
        if out is None:
            return bitwise_right_shift_func_scalar_tensor(A, B)
        bitwise_right_shift_func_scalar_tensor(A, B, out0=out)
        return out
    else:
        # With two scalars, construct a 0-D tensor directly following Python right-shift semantics.
        return torch.tensor(A >> B)


def bitwise_right_shift_(A, B, *, out=None):
    if isinstance(B, torch.Tensor):
        if _use_flat(A, B, A):
            return _launch(A, B, A)
        return bitwise_right_shift_func(A, B, out0=A)
    flat = _flat_tensor_or_scalar(A, B, A)
    if flat is not None:
        return flat
    return bitwise_right_shift_func_scalar(A, B, out0=A)
