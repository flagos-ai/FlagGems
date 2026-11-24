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

"""GCU400-specific implementation of ``aten::_stack``.

Output layout is ``out[pre, t, post] = in_t[pre, post]``, where
``outer = prod(shape[:dim])``, ``seg = prod(shape[dim:])``, ``P = len(tensors)``.
The original implementation splits the flat input index with runtime int64 division into ``idx // seg`` and ``idx %
seg``;
on GCU400 software division plus interleaved writes degrades large shapes to 0.015x. This implementation uses
three layout-based paths whose target addresses need only multiplication and addition, with no division:

* ``outer == 1``: each input is one contiguous block of the output, copied directly by a 2D grid in one launch;
* ``seg == 1``: contiguous load + stride-P store;
* otherwise: a 2D tile directly gives the pre row and post slot, and the row offset is pure multiplication.

When ``outer == 1`` and dtype itemsize <= 4, use the wide channel: reinterpret input and output as
int64 views so one load/store moves 8 bytes, reducing per-element address arithmetic and memory requests
(probe: 16.7M elements fp32 210 -> 139us, fp16 85 -> 74us). Within that,
``lanes <= 8192`` (<=64KB/input) uses exact lane-based single-block single-launch tiling;
``lanes > 65536`` (>512KB/input) uses fixed 32768-lane blocks (grid cap 256 +
block stride loop); the middle tier keeps the element path.

bool / int64 / uint64 / fp8 2D masked stores are unreliable on GCU400, so these
dtypes fall back to the verified legacy kernel (outside this operator's benchmark scope).
"""

import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# The constants below are A/B parameters per iteration round; change one tier per round and keep the best.
_FLAT_BLOCK = 16384
_FLAT_GRID_CAP = 256
_FLAT_WARPS = 8
_STRIDE_BLOCK = 1024
_STRIDE_WARPS = 4
_TILE_BR = 64
_TILE_BT = 256
_TILE_WARPS = 4
_LEGACY_BLOCK = 1024

# Wide-channel path (int64 view, 8-byte lanes) tiers (probe calibrated 2026-11-13):
#   lanes <= _WIDE_SMALL_LANES: exact lane-based tiling, single grid=(1, P) launch;
#   lanes  > _WIDE_LARGE_LANES: 32768-lane large blocks + grid cap + block stride loop;
#   in between: keep the element path (too few large-block programs at medium sizes, element path is not worse).
_WIDE_SMALL_LANES = 8192
_WIDE_LARGE_LANES = 65536
_WIDE_BLOCK = 32768
_WIDE_GRID_CAP = 256
_WIDE_SMALL_WARPS = 4
_WIDE_LARGE_WARPS = 8

# The external /opt/triton_gcu converter occasionally SIGSEGVs on cold compilation (compiler_problems/P02),
# probabilistically, leaving no cache entry; a retry succeeds. Deterministic failures are re-raised unchanged after
# retries are exhausted.
# As in row_stack: wrap only the wide-channel launch.
_TOOLCHAIN_COMPILE_RETRIES = 3

# GCU400 resource limits: grid.y must not exceed 255, and 2D tile bytes must stay within the DSM budget.
_MAX_GRID_Y = 255
_DSM_LIMIT_BYTES = 900000
_MAX_POINTERS = 4

# 2D masked load/store fails or reads undefined values with i1 pointers and 64-bit integers,
# and fp8 masked loads cannot produce a legal default, so all these dtypes take the legacy path.
_TILE_EXCLUDED_DTYPES = {
    torch.bool,
    torch.int64,
    torch.uint64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


@triton.jit
def stack_copy_func_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    """Original interleaved kernel: correctness fallback only when dtype / resource limits require it."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    block_start = pid_x.to(tl.int64) * BLOCK_X
    offsets = tl.arange(0, BLOCK_X).to(tl.int64)
    idx = block_start + offsets
    scalar_zero = offsets * 0

    dim_size_out = dim_size_out + scalar_zero
    dim_prod_post = dim_prod_post + scalar_zero
    dim_offset = dim_offset + scalar_zero
    total_elements = total_elements + scalar_zero

    mask = idx < total_elements

    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


@triton.jit
def _stack_flat_copy_kernel(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    num_elements,
    batch_base,
    chunks_per_program,
    BLOCK_X: tl.constexpr,
):
    """``outer == 1`` path: input t is written wholly into the t-th contiguous output block.

    grid.y directly gives the in-batch input index, so no index division is needed; the store side is also
    a contiguous address range, so both reads and writes are coalesced. Indices are uniformly int32 (the host
    already guards overflow by output bytes), avoiding expensive 64-bit address arithmetic on GCU400.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    num_programs = tl.num_programs(0)

    if pid_y == 0:
        source_ptr = in_ptr_a
    elif pid_y == 1:
        source_ptr = in_ptr_b
    elif pid_y == 2:
        source_ptr = in_ptr_c
    else:
        source_ptr = in_ptr_d

    output_base = (batch_base + pid_y) * num_elements
    for chunk_index in range(chunks_per_program):
        block_index = pid_x + chunk_index * num_programs
        offsets = block_index * BLOCK_X + tl.arange(0, BLOCK_X)
        active = offsets < num_elements
        tl.store(
            out_ptr + output_base + offsets,
            tl.load(source_ptr + offsets, mask=active),
            mask=active,
        )


@triton.jit
def _stack_wide_copy_kernel(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    num_lanes,
    batch_base,
    chunks_per_program,
    BLOCK_X: tl.constexpr,
    MASKED: tl.constexpr,
):
    """``outer == 1`` wide-channel path: int64 view moves whole 8-byte lanes.

    Input and output are both reinterpreted with ``view(torch.int64)``, each lane moving 8 bytes,
    reducing per-element address arithmetic and memory requests to 1/2 (fp32) or 1/4 (fp16/bf16/int16).
    grid.y directly gives the in-batch input index; the host already guards int32 overflow by output bytes,
    so all indices remain int32. Exact division lets ``MASKED == False`` specialize away the tail mask.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    num_programs = tl.num_programs(0)

    if pid_y == 0:
        source_ptr = in_ptr_a
    elif pid_y == 1:
        source_ptr = in_ptr_b
    elif pid_y == 2:
        source_ptr = in_ptr_c
    else:
        source_ptr = in_ptr_d

    output_base = (batch_base + pid_y) * num_lanes
    for chunk_index in range(chunks_per_program):
        block_index = pid_x + chunk_index * num_programs
        offsets = block_index * BLOCK_X + tl.arange(0, BLOCK_X)
        if MASKED:
            active = offsets < num_lanes
            tl.store(
                out_ptr + output_base + offsets,
                tl.load(source_ptr + offsets, mask=active),
                mask=active,
            )
        else:
            tl.store(
                out_ptr + output_base + offsets,
                tl.load(source_ptr + offsets),
            )


@triton.jit
def _stack_stride_copy_kernel(
    out_ptr,
    in_ptr,
    num_elements,
    output_stride,
    output_offset,
    BLOCK_X: tl.constexpr,
):
    """``seg == 1`` path: contiguous load + stride-P store, one launch per input."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_X + tl.arange(0, BLOCK_X).to(
        tl.int64
    )
    active = offsets < num_elements
    value = tl.load(in_ptr + offsets, mask=active)
    tl.store(out_ptr + offsets * output_stride + output_offset, value, mask=active)


@triton.jit
def _stack_tile_copy_kernel(
    out_ptr,
    in_ptr,
    outer,
    seg,
    out_row_stride,
    output_offset,
    BR: tl.constexpr,
    BT: tl.constexpr,
):
    """Generic path: a 2D tile directly gives the pre row and post slot.

    The row offset contains only multiplication; the original 1D kernel's ``idx // seg`` runtime int64 software division
    is the main bottleneck for large shapes on GCU400, and is eliminated here at the algorithmic level.
    """
    row_index = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    slot_index = (tl.program_id(1) * BT + tl.arange(0, BT)).to(tl.int64)
    active = (row_index[:, None] < outer) & (slot_index[None, :] < seg)
    source = in_ptr + row_index[:, None] * seg + slot_index[None, :]
    destination = (
        out_ptr
        + row_index[:, None] * out_row_stride
        + output_offset
        + slot_index[None, :]
    )
    tl.store(destination, tl.load(source, mask=active), mask=active)


def _prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


def _launch_legacy(tensors, out, dim_size_out, dim_prod_post) -> None:
    """Fall back to the original interleaved kernel, preserving semantics for bool / int64 / fp8 / resource overflow."""
    block_size = _LEGACY_BLOCK
    tensor_index = 0
    while tensor_index < len(tensors):
        batch = tensors[tensor_index : tensor_index + _MAX_POINTERS]
        batch_size = len(batch)
        pointer_args = []
        offset_args = []
        length_args = []
        for slot in range(_MAX_POINTERS):
            if slot < batch_size:
                tensor = batch[slot].contiguous()
                pointer_args.append(tensor)
                offset_args.append(tensor_index + slot)
                length_args.append(tensor.numel())
            else:
                pointer_args.append(batch[0])
                offset_args.append(0)
                length_args.append(0)
        max_elements = max(length_args)
        grid = (triton.cdiv(max_elements, block_size), batch_size)
        stack_copy_func_kernel_4[grid](
            out,
            pointer_args[0],
            pointer_args[1],
            pointer_args[2],
            pointer_args[3],
            dim_size_out,
            dim_prod_post,
            offset_args[0],
            offset_args[1],
            offset_args[2],
            offset_args[3],
            length_args[0],
            length_args[1],
            length_args[2],
            length_args[3],
            BLOCK_X=block_size,
        )
        tensor_index += batch_size


def _launch_with_toolchain_retry(kernel, grid, args, launch_kwargs):
    """P02: retry the launch on an occasional cold-compile SIGSEGV; deterministic failures are re-raised unchanged
    after retries are exhausted."""
    for attempt in range(_TOOLCHAIN_COMPILE_RETRIES):
        try:
            return kernel[grid](*args, **launch_kwargs)
        except Exception as error:
            if attempt == _TOOLCHAIN_COMPILE_RETRIES - 1:
                raise
            logger.debug(
                "_stack: retrying Triton launch after toolchain failure "
                "(attempt %d): %r",
                attempt,
                error,
            )


def _launch_wide(tensors, out, num_lanes) -> None:
    """Wide-channel path: int64 view moves 8-byte lanes, one launch per 4 inputs."""
    if num_lanes <= _WIDE_SMALL_LANES:
        block_lanes = _next_power_of_two(num_lanes)
        grid_cap = 1
        num_warps = _WIDE_SMALL_WARPS
    else:
        block_lanes = _WIDE_BLOCK
        grid_cap = _WIDE_GRID_CAP
        num_warps = _WIDE_LARGE_WARPS
    grid_x = min(triton.cdiv(num_lanes, block_lanes), grid_cap)
    chunks_per_program = triton.cdiv(num_lanes, block_lanes * grid_x)
    masked = (num_lanes % block_lanes != 0) or chunks_per_program > 1
    launch_kwargs = {
        "BLOCK_X": block_lanes,
        "MASKED": masked,
        "num_warps": num_warps,
    }
    out_lanes = out.view(-1).view(torch.int64)
    for batch_base in range(0, len(tensors), _MAX_POINTERS):
        batch = tensors[batch_base : batch_base + _MAX_POINTERS]
        batch_size = len(batch)
        padded_pointers = list(batch) + [batch[0]] * (_MAX_POINTERS - batch_size)
        source_lanes = [t.view(-1).view(torch.int64) for t in padded_pointers]
        _launch_with_toolchain_retry(
            _stack_wide_copy_kernel,
            (grid_x, batch_size),
            (
                out_lanes,
                source_lanes[0],
                source_lanes[1],
                source_lanes[2],
                source_lanes[3],
                num_lanes,
                batch_base,
                chunks_per_program,
            ),
            launch_kwargs,
        )


def _launch_flat(tensors, out, num_elements) -> None:
    """Contiguous-block path: byte-aligned dtypes <= 4 bytes use the int64 wide channel, otherwise element tiling."""
    itemsize = tensors[0].element_size()
    total_bytes = num_elements * itemsize
    if itemsize <= 4 and total_bytes % 8 == 0:
        num_lanes = total_bytes // 8
        if num_lanes <= _WIDE_SMALL_LANES or num_lanes > _WIDE_LARGE_LANES:
            _launch_wide(tensors, out, num_lanes)
            return
    grid_x = min(triton.cdiv(num_elements, _FLAT_BLOCK), _FLAT_GRID_CAP)
    chunks_per_program = triton.cdiv(num_elements, _FLAT_BLOCK * grid_x)
    for batch_base in range(0, len(tensors), _MAX_POINTERS):
        batch = tensors[batch_base : batch_base + _MAX_POINTERS]
        batch_size = len(batch)
        padded_pointers = list(batch) + [batch[0]] * (_MAX_POINTERS - batch_size)
        grid = (grid_x, batch_size)
        _stack_flat_copy_kernel[grid](
            out,
            padded_pointers[0],
            padded_pointers[1],
            padded_pointers[2],
            padded_pointers[3],
            num_elements,
            batch_base,
            chunks_per_program,
            BLOCK_X=_FLAT_BLOCK,
            num_warps=_FLAT_WARPS,
        )


def _launch_stride(tensors, out, num_elements) -> None:
    """stride-P path: one contiguous read + equal-stride write per input."""
    grid = (triton.cdiv(num_elements, _STRIDE_BLOCK),)
    tensor_count = len(tensors)
    for tensor_index, tensor in enumerate(tensors):
        _stack_stride_copy_kernel[grid](
            out,
            tensor,
            num_elements,
            tensor_count,
            tensor_index,
            BLOCK_X=_STRIDE_BLOCK,
            num_warps=_STRIDE_WARPS,
        )


def _launch_tile(tensors, out, outer, seg, element_size) -> bool:
    """2D tile path; returns False on resource overflow so the caller falls back to legacy."""
    rows_per_block = min(_TILE_BR, _next_power_of_two(outer))
    slots_per_block = min(_TILE_BT, _next_power_of_two(seg))
    grid_y = triton.cdiv(seg, slots_per_block)
    tile_bytes = rows_per_block * slots_per_block * element_size
    if grid_y > _MAX_GRID_Y or tile_bytes > _DSM_LIMIT_BYTES:
        return False
    grid = (triton.cdiv(outer, rows_per_block), grid_y)
    out_row_stride = len(tensors) * seg
    for tensor_index, tensor in enumerate(tensors):
        _stack_tile_copy_kernel[grid](
            out,
            tensor,
            outer,
            seg,
            out_row_stride,
            tensor_index * seg,
            BR=rows_per_block,
            BT=slots_per_block,
            num_warps=_TILE_WARPS,
        )
    return True


def _stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME_GCU400 _STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    # With a single input the per-item loop below never runs, so the dim range must be validated
    # unconditionally to match ATen's out-of-range behavior.
    ndim0 = tensors[0].dim()
    if (dim < -ndim0 - 1) or (dim > ndim0):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -ndim0 - 1, ndim0, dim
            )
        )
    for i, s in enumerate(inp_shapes[1:]):
        ndim = tensors[i + 1].dim()
        if (dim < -ndim - 1) or (dim > ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -ndim - 1, ndim, dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    # Type promotion: all inputs are cast to a common dtype.
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    # Triton has no complex scalar type; move data through a zero-copy real view and restore complex at the end.
    is_complex = dtype.is_complex
    if is_complex:
        complex_dtype = dtype
        dtype = torch.view_as_real(torch.empty((), dtype=dtype, device="cpu")).dtype
        tensors = [torch.view_as_real(t.to(complex_dtype)) for t in tensors]
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    work_shape = list(tensors[0].shape)
    out_shape = work_shape[:dim] + [len(tensors)] + work_shape[dim:]
    # empty_strided bypasses GEMS's empty.memory_format override (which would launch an extra
    # zero-init kernel); the stack kernel writes every output element, so no initialization is needed.
    contig_strides = []
    stride = 1
    for dim_sz in reversed(out_shape):
        contig_strides.append(stride)
        stride *= dim_sz
    contig_strides.reverse()
    out = torch.empty_strided(out_shape, contig_strides, dtype=dtype, device=device)

    # Zero-volume outputs return directly, avoiding an illegal launch with a zero-dimensional grid.
    if out.numel() == 0:
        return torch.view_as_complex(out) if is_complex else out

    outer = _prod(work_shape[:dim])
    seg = _prod(work_shape[dim:])
    num_elements = outer * seg
    contiguous_tensors = [tensor.contiguous() for tensor in tensors]

    # The int32-indexed flat path requires output byte offsets below 2^31; very large tensors fall back to
    # the legacy kernel with int64 indices, keeping both semantics and address computation safe.
    needs_int64_index = out.numel() * out.element_size() > (1 << 31) - 1
    if dtype in _TILE_EXCLUDED_DTYPES or (outer == 1 and needs_int64_index):
        _launch_legacy(contiguous_tensors, out, len(tensors), seg)
    elif outer == 1:
        _launch_flat(contiguous_tensors, out, num_elements)
    elif seg == 1:
        _launch_stride(contiguous_tensors, out, num_elements)
    elif not _launch_tile(
        contiguous_tensors, out, outer, seg, contiguous_tensors[0].element_size()
    ):
        _launch_legacy(contiguous_tensors, out, len(tensors), seg)

    if is_complex:
        out = torch.view_as_complex(out)
    return out
