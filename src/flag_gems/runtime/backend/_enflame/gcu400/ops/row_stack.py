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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.cat__opt import (
    _can_fast_path,
    _linear_cat_kernel,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

# Linear fast-path gate/geometry (libra L600, 2026-11-10 probes): the generic
# row_stack_kernel is element-ALU bound on large inputs (int64 index math +
# per-program pointer-select chains at BLOCK_SIZE<=1024, ~13ns/element, fp16
# and fp32 identical).  Copying through 8-byte (int64-view) lanes instead
# reaches ~2.7TB/s, within ~5% of the native cat kernel.
_FAST_LARGE = 1 << 20
_WIDE_BLOCK = 32768  # int64 lanes -> 256KB per program (within DSM budget)
_PLAIN_BLOCK = 65536  # element lanes, dtypes <= 4 bytes -> 256KB per program

# The external /opt/triton_gcu converter SIGSEGVs on a fraction of cold
# compiles (compiler_problems/P02, flaggems_row_stack_toolchain_flaky_segv).
# The crash is per-invocation probabilistic, leaves no Triton cache entry,
# and a fresh attempt compiles fine, so retry the launch. Shared machines
# routinely wipe ~/.triton/cache between runs, so gates must survive cold
# compiles; a deterministic failure is re-raised after the retries.
_TOOLCHAIN_COMPILE_RETRIES = 3


def _launch_with_toolchain_retry(launch):
    for attempt in range(_TOOLCHAIN_COMPILE_RETRIES):
        try:
            return launch()
        except Exception as error:
            if attempt == _TOOLCHAIN_COMPILE_RETRIES - 1:
                raise
            logger.debug(
                "row_stack: retrying Triton launch after toolchain failure "
                "(attempt %d): %r",
                attempt,
                error,
            )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("vstack"),
    key=[
        "max_tile_elems",
    ],
)
@triton.jit
def row_stack_kernel(
    itensor_ptr0,
    itensor_ptr1,
    itensor_ptr2,
    itensor_ptr3,
    output_ptr,
    local_row0,
    local_row1,
    local_row2,
    local_row3,
    exc_row_offset0,
    exc_row_offset1,
    exc_row_offset2,
    exc_row_offset3,
    total_row_offset,
    row_stride,
    max_tile_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = ext.program_id(axis=0)
    tensor_idx = ext.program_id(axis=1)
    col_idx = tl.arange(0, BLOCK_SIZE)

    intensor_ptr = tl.where(tensor_idx == 0, itensor_ptr0, itensor_ptr1)
    intensor_ptr = tl.where(tensor_idx == 2, itensor_ptr2, intensor_ptr)
    intensor_ptr = tl.where(tensor_idx == 3, itensor_ptr3, intensor_ptr)
    base_exc_row_idx = tl.where(tensor_idx == 0, exc_row_offset0, exc_row_offset1)
    base_exc_row_idx = tl.where(tensor_idx == 2, exc_row_offset2, base_exc_row_idx)
    base_exc_row_idx = tl.where(tensor_idx == 3, exc_row_offset3, base_exc_row_idx)
    local_row = tl.where(tensor_idx == 0, local_row0, local_row1)
    local_row = tl.where(tensor_idx == 2, local_row2, local_row)
    local_row = tl.where(tensor_idx == 3, local_row3, local_row)

    # Widen through the tensor operands rather than through ``row_stride``:
    # Triton specializes a scalar ``1`` (and ``0``) to a constexpr, which has no
    # ``.to`` method, so casting the scalar itself would fail for width-1 rows.
    end_idx = local_row.to(tl.int64) * row_stride
    idx = (pid_x * BLOCK_SIZE + col_idx).to(tl.int64)
    offset_mask = idx < end_idx
    in_offset = intensor_ptr + idx
    row_stride_offset = (total_row_offset + base_exc_row_idx).to(tl.int64) * row_stride
    out_offset = output_ptr + row_stride_offset + idx
    out = tl.load(in_offset, mask=offset_mask)
    tl.store(out_offset, out, mask=offset_mask)


@triton.jit
def _row_stack_wide_cat_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    len0,
    len1,
    len2,
    len3,
    base0,
    base1,
    base2,
    base3,
    BLOCK: tl.constexpr,
):
    """Linear dim-0 concat over 8-byte lanes (int64 view of the buffers).

    One program copies one BLOCK-wide chunk of one input; pid_y selects the
    input through a uniform branch (no per-lane pointer select chains, no
    int64 index arithmetic, int32 lane offsets only).
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    off = tl.arange(0, BLOCK)
    idx = pid_x * BLOCK + off
    if pid_y == 0:
        src, length, base = In0, len0, base0
    elif pid_y == 1:
        src, length, base = In1, len1, base1
    elif pid_y == 2:
        src, length, base = In2, len2, base2
    else:
        src, length, base = In3, len3, base3
    mask = idx < length
    data = tl.load(src + idx, mask=mask)
    tl.store(Out + base + idx, data, mask=mask)


def _row_width(shape):
    """Number of elements in one row = product of the trailing dimensions."""
    width = 1
    for size in shape[1:]:
        width *= size
    return width


def _row_strides(shape):
    """Row-major contiguous strides, matching ATen for zero-sized dimensions.

    ATen skips the stride multiply for a zero-sized dimension, so ``(5, 0)``
    gets stride ``(1, 1)`` rather than ``(0, 1)``.  Reproducing that keeps the
    output metadata identical to the reference for empty inputs.
    """
    strides = [0] * len(shape)
    step = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = step
        if shape[dim] != 0:
            step *= shape[dim]
    return strides


def _channels_last_strides(shape):
    """channels_last (rank 4) / channels_last_3d (rank 5) strides."""
    if len(shape) == 4:
        _, c, h, w = shape
        return [c * h * w, 1, w * c, c]
    _, c, d, h, w = shape
    return [c * d * h * w, 1, h * w * c, w * c, c]


def _strides_for(shape, memory_format):
    """Strides ATen's allocator uses for ``shape`` in ``memory_format``.

    For channels-last the allocated strides are what the kernel needs: dim 0 is
    stepped by ``row_width`` (the product of the trailing sizes) and each row is
    a dense run of ``row_width`` elements, tiled back to back.
    """
    if memory_format == torch.channels_last and len(shape) == 4:
        return _channels_last_strides(shape)
    if memory_format == torch.channels_last_3d and len(shape) == 5:
        return _channels_last_strides(shape)
    return _row_strides(shape)


def _promote_dtype(tensors):
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        if tensor.dtype != dtype:
            dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _suggest_memory_format(tensor):
    """Mirror ATen's ``TensorImpl::suggest_memory_format``.

    Only an exact channels-last / channels-last-3d layout (rank 4 / rank 5) is
    reported; anything else maps to the default contiguous format.
    """
    from torch._prims_common import suggest_memory_format

    if tensor.layout != torch.strided:
        return torch.contiguous_format
    return suggest_memory_format(tensor)


def _derive_memory_format(tensors):
    """ATen's ``cat_compute_output_memory_format``: all inputs must agree."""
    fmt = None
    for tensor in tensors:
        current = _suggest_memory_format(tensor)
        if current == torch.contiguous_format:
            return torch.contiguous_format
        if fmt is not None and fmt != current:
            return torch.contiguous_format
        fmt = current
    return fmt if fmt is not None else torch.contiguous_format


def _allocate(shape, strides, dtype, device):
    return torch.empty_strided(tuple(shape), tuple(strides), dtype=dtype, device=device)


def _normalize_tensors(tensors):
    """``atleast_2d`` plus ATen-compatible validation."""
    tensors = list(torch.atleast_2d(tensors))
    if len(tensors) == 0:
        raise RuntimeError("vstack expects a non-empty TensorList")

    device = tensors[0].device
    ndim = tensors[0].dim()
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.device != device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got "
                f"{tensor.device} and {device}"
            )
        if tensor.dim() != ndim:
            raise RuntimeError(
                "Tensors must have same number of dimensions: got "
                f"{ndim} and {tensor.dim()}"
            )
        for dim, (expected, got) in enumerate(zip(tensors[0].shape, tensor.shape)):
            if dim != 0 and expected != got:
                raise RuntimeError(
                    "Sizes of tensors must match except in dimension 0. "
                    f"Expected size {expected} but got size {got} for tensor "
                    f"number {i} in the list."
                )
    return tensors


def _cast_tensors(tensors, dtype, memory_format):
    """Bring every input to ``dtype`` with a row block dense in layout order.

    The kernel copies row blocks positionally, so each input must be dense in
    the *same* trailing layout as the output; a channels-last input that was
    flattened to contiguous order would scatter elements to wrong positions.
    """
    cast = []
    for tensor in tensors:
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        if not tensor.is_contiguous(memory_format=memory_format):
            tensor = tensor.contiguous(memory_format=memory_format)
        cast.append(tensor)
    return cast


_I64_COPY_BLOCK = 4096  # int64 lanes (8 bytes) -> 32KB per program


def _flat_i64_view(tensor):
    """Flat int64-lane view of a dense row-tiled 8-byte tensor (zero-copy).

    Same-itemsize dtype reinterpretation keeps size/strides/storage offset,
    and every tensor reaching this path is dense in row-tile layout order
    (row-major or channels-last from ``_cast_tensors`` / ``_strides_for``),
    so a flat stride-1 walk over ``numel`` lanes is address-correct.
    """
    viewed = tensor.view(torch.int64)
    return torch.as_strided(viewed, (viewed.numel(),), (1,))


def _run_kernel_i64_bitcopy(tensors, output):
    """Bit-exact int64-lane copy (complex64: both halves are live data)."""
    num_tensors = len(tensors)
    out_flat = _flat_i64_view(output)
    for start in range(0, num_tensors, 4):
        batch = tensors[start : start + 4]
        num = len(batch)
        lens = [tensor.numel() for tensor in batch]
        lens = lens + [0] * (4 - num)
        srcs = [_flat_i64_view(tensor) for tensor in batch]
        srcs = srcs + [srcs[0]] * (4 - num)
        bases = [0] * 4
        batch_lanes = 0
        for j in range(4):
            bases[j] = batch_lanes
            batch_lanes += lens[j]
        grid = (triton.cdiv(max(max(lens), 1), _I64_COPY_BLOCK), num)
        _launch_with_toolchain_retry(
            lambda: _row_stack_wide_cat_kernel[grid](
                out_flat,
                srcs[0],
                srcs[1],
                srcs[2],
                srcs[3],
                lens[0],
                lens[1],
                lens[2],
                lens[3],
                bases[0],
                bases[1],
                bases[2],
                bases[3],
                BLOCK=_I64_COPY_BLOCK,
                num_warps=4,
            )
        )
    return output


def _run_kernel_i64_lanes(tensors, output, row_width):
    """8-byte dtype dispatch on GCU.

    fp64/int64 are "Float/Long replace" containers on this backend: the
    value lives in the low 4 bytes while the high 4 bytes hold allocator
    garbage, so bit-level 8-byte-lane copies corrupt values once the result
    is read back (and fp64 element-typed kernels abort the compiler,
    compiler_problems/P02). Down-cast to the real 4-byte dtype (native
    value-level cast reads the low half), copy through the regular 4-byte
    kernels, then cast back into ``output``. complex64 stores two live
    float32 halves, so it keeps the bit-exact int64-lane copy.
    """
    if tensors[0].dtype == torch.complex64:
        return _run_kernel_i64_bitcopy(tensors, output)

    real_dtype = torch.float32 if tensors[0].dtype == torch.float64 else torch.int32
    four = [tensor.to(real_dtype) for tensor in tensors]
    staging = torch.empty(output.shape, dtype=real_dtype, device=output.device)
    if max(tensor.numel() for tensor in four) >= _FAST_LARGE:
        _launch_fast_stack(four, staging)
    else:
        _run_kernel(four, staging, row_width, staging.stride(0))
    output.copy_(staging)
    return output


def _run_kernel(tensors, output, row_width, out_row_stride):
    """Copy every input row block into ``output`` through the Triton kernel."""
    if tensors[0].element_size() == 8:
        return _run_kernel_i64_lanes(tensors, output, row_width)
    num_tensors = len(tensors)
    outer_iters = triton.cdiv(num_tensors, 4)
    total_row_offset = 0
    for i in range(outer_iters):
        max_rows = 1
        itensors = []
        exclusive_row = []
        local_row = []
        array_row_offset = 0
        scheduled_num_tensors = 0
        for j in range(4):
            tensor_idx = i * 4 + j
            if tensor_idx < num_tensors:
                scheduled_num_tensors += 1
                tensor = tensors[tensor_idx]
                itensors.append(tensor)
                local_row.append(tensor.shape[0])
                exclusive_row.append(array_row_offset)
                array_row_offset += tensor.shape[0]
                max_rows = max(max_rows, tensor.shape[0])
            else:
                empty_tensor = torch.empty(
                    0, dtype=tensors[0].dtype, device=tensors[0].device
                )
                itensors.append(empty_tensor)
                local_row.append(local_row[-1])
                exclusive_row.append(exclusive_row[-1])
        max_tile_elems = max_rows * row_width
        grid = lambda META: (
            triton.cdiv(max_tile_elems, META["BLOCK_SIZE"]),
            scheduled_num_tensors,
        )
        # Launch the kernel
        with torch_device_fn.device(tensors[0].device):
            _launch_with_toolchain_retry(
                lambda: row_stack_kernel[grid](
                    itensors[0],
                    itensors[1],
                    itensors[2],
                    itensors[3],
                    output,
                    local_row[0],
                    local_row[1],
                    local_row[2],
                    local_row[3],
                    exclusive_row[0],
                    exclusive_row[1],
                    exclusive_row[2],
                    exclusive_row[3],
                    total_row_offset,
                    out_row_stride,
                    max_tile_elems,
                )
            )
            total_row_offset += array_row_offset
    return output


def _output_shape(tensors):
    total_rows = sum(tensor.shape[0] for tensor in tensors)
    shape = list(tensors[0].shape)
    shape[0] = total_rows
    return shape


def _fast_stack_eligible(tensors):
    """Return the aligned tensor list when the linear fast path applies.

    Only strict, cheap-to-verify cases are taken over: a list/tuple of 1-4
    tensors that are already >=2D, at least one is large, and all share
    dtype/device, contiguity and trailing shape (cat dim 0 fast-path rules).
    Everything else (1-D inputs, small inputs, mixed dtypes, non-contiguous,
    channels-last, float8, >4 tensors) keeps the generic production logic.
    """
    if not isinstance(tensors, (list, tuple)):
        return None
    if not 1 <= len(tensors) <= 4:
        return None
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return None
    aligned = list(tensors)
    first = aligned[0]
    if first.ndim < 2:
        return None
    if first.element_size() > 4:
        # 8-byte dtypes (fp64/int64/complex64) go through the production path,
        # whose ``_run_kernel`` copies them through int64 lanes (see
        # ``_run_kernel_i64_lanes``): fp64 element-typed kernels hit an
        # in-process llvm cast<VectorType> assert (compiler_problems/P02) and
        # emulated int64 arithmetic miscomputes on this backend.
        return None
    if max(tensor.numel() for tensor in aligned) < _FAST_LARGE:
        return None
    if not _can_fast_path(aligned, 0):
        return None
    return aligned


def _launch_fast_stack(aligned, out):
    """Copy the aligned tensors into ``out`` with the linear concat kernels."""
    num = len(aligned)
    itemsize = aligned[0].element_size()
    numels = [tensor.numel() for tensor in aligned]
    ptrs = [aligned[j] if j < num else aligned[0] for j in range(4)]

    factor = 8 // itemsize
    if all(length % factor == 0 for length in numels):
        # 8-byte lane path: bit-exact copy through int64 views of the flat
        # buffers (reshape(-1) is zero-copy for the contiguous fast-path inputs).
        lanes = [length // factor for length in numels]
        lens = [lanes[j] if j < num else 0 for j in range(4)]
        bases = [0] * 4
        acc = 0
        for j in range(4):
            bases[j] = acc
            acc += lens[j]
        out_lanes = out.view(-1).view(torch.int64)
        src_lanes = [tensor.view(-1).view(torch.int64) for tensor in ptrs]
        grid = (triton.cdiv(max(lens), _WIDE_BLOCK), num)
        _launch_with_toolchain_retry(
            lambda: _row_stack_wide_cat_kernel[grid](
                out_lanes,
                src_lanes[0],
                src_lanes[1],
                src_lanes[2],
                src_lanes[3],
                lens[0],
                lens[1],
                lens[2],
                lens[3],
                bases[0],
                bases[1],
                bases[2],
                bases[3],
                BLOCK=_WIDE_BLOCK,
                num_warps=8 if itemsize == 4 else 4,
            )
        )
    else:
        # Plain-lane path: reuse cat's linear kernel (masked, element lanes).
        ns = [numels[j] if j < num else 0 for j in range(4)]
        bases = [0] * 4
        acc = 0
        for j in range(4):
            bases[j] = acc
            acc += ns[j]
        grid = (triton.cdiv(max(ns), _PLAIN_BLOCK), num)
        _launch_with_toolchain_retry(
            lambda: _linear_cat_kernel[grid](
                out,
                ptrs[0],
                ptrs[1],
                ptrs[2],
                ptrs[3],
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                bases[0],
                bases[1],
                bases[2],
                bases[3],
                NT=num,
                BLOCK=_PLAIN_BLOCK,
                num_warps=4,
            )
        )
    return out


def _prepare(tensors):
    """Shared validation/shape work for both entry points."""
    tensors = _normalize_tensors(tensors)
    memory_format = _derive_memory_format(tensors)
    dtype = _promote_dtype(tensors)
    # A zero-sized trailing dimension means zero work: ``stride(0)`` can still
    # be non-zero for such inputs (e.g. ``(3, 0)`` has stride ``(1, 1)``), so a
    # launch based on ``rows * stride(0)`` would run out of bounds.
    row_width = _row_width(tensors[0].shape)
    output_shape = _output_shape(tensors)
    return tensors, memory_format, dtype, row_width, output_shape


def row_stack(tensors: list):
    logger.debug("GEMS ROW_STACK")

    aligned = _fast_stack_eligible(tensors)
    if aligned is not None:
        out_shape = (sum(t.shape[0] for t in aligned),) + tuple(aligned[0].shape[1:])
        out = torch.empty(out_shape, dtype=aligned[0].dtype, device=aligned[0].device)
        return _launch_fast_stack(aligned, out)

    tensors, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    if dtype == torch.float64:
        # gcu400 has no fp64 arithmetic: the copy kernel cannot be compiled
        # ("Pipeline run failed").  A float64 device tensor only holds
        # float-precision data on this backend, so stacking float32 inputs is
        # exact and the result is widened back afterwards.
        stacked = row_stack([t.to(torch.float32) for t in tensors])
        return stacked.to(torch.float64)
    device = tensors[0].device
    if row_width == 0 or output_shape[0] == 0:
        # Nothing to copy; still hand back a correctly shaped, format-correct
        # tensor rather than launching a kernel that would read out of bounds.
        return _allocate(
            output_shape, _strides_for(output_shape, memory_format), dtype, device
        )

    strides = _strides_for(output_shape, memory_format)
    output = _allocate(output_shape, strides, dtype, device)
    c_tensors = _cast_tensors(tensors, dtype, memory_format)
    return _run_kernel(c_tensors, output, row_width, strides[0])


def _check_out(out, output_shape, dtype, memory_format, tensors):
    """Validate/prepare ``out`` the way ATen's ``cat_out`` meta function does.

    The ordering mirrors ATen: the castability check runs before the output is
    resized, and the overlap checks run afterwards.
    """
    if not torch.can_cast(dtype, out.dtype):
        raise TypeError(
            "torch.cat(): input types can't be cast to the desired output type "
            f"{out.dtype}"
        )

    if list(out.shape) != list(output_shape):
        out.resize_(output_shape)
        # ``resize_`` always installs contiguous strides, whereas ATen's
        # ``set_output_raw_strided`` re-imposes the derived memory format; for
        # channels-last that changes the strides, so restore them here.
        strides = _strides_for(list(out.shape), memory_format)
        if tuple(out.stride()) != tuple(strides):
            out.as_strided_(tuple(out.shape), tuple(strides))

    if has_internal_overlapping(out) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    for tensor in tensors:
        if torch._C._overlaps(out, tensor):
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
    return out


def row_stack_out(tensors: list, *, out: torch.Tensor):
    logger.debug("GEMS ROW_STACK_OUT")

    aligned = _fast_stack_eligible(tensors)
    if aligned is not None:
        first = aligned[0]
        out_shape = (sum(t.shape[0] for t in aligned),) + tuple(first.shape[1:])
        if (
            out.dtype == first.dtype
            and out.device == first.device
            and out.is_contiguous()
            and tuple(out.shape) == tuple(out_shape)
            and not any(torch._C._overlaps(out, tensor) for tensor in aligned)
        ):
            return _launch_fast_stack(aligned, out)

    tensors, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    _check_out(out, output_shape, dtype, memory_format, tensors)

    if dtype == torch.float64 or out.dtype == torch.float64:
        # gcu400 has no fp64 arithmetic (see the note in row_stack); stage the
        # result in float32 and let copy_ widen it into the caller's buffer.
        staged = row_stack([t.to(torch.float32) for t in tensors])
        out.copy_(staged.to(out.dtype))
        return out

    if row_width == 0 or output_shape[0] == 0:
        return out

    c_tensors = _cast_tensors(tensors, dtype, memory_format)

    # Fast path: ``out`` stores rows as dense runs of ``row_width`` elements and
    # its trailing strides match the layout the kernel assumes.  This holds for
    # contiguous and for channels-last outputs, so only an arbitrary strided
    # view falls through to the staging copy below.
    fast = out.dtype == dtype and tuple(out.stride()) == tuple(
        _strides_for(list(out.shape), memory_format)
    )
    if fast:
        return _run_kernel(c_tensors, out, row_width, out.stride(0))

    # Otherwise scatter through a dense staging buffer; ``copy_`` is
    # stride-aware and also honours ``out``'s dtype.
    strides = _strides_for(output_shape, memory_format)
    staging = _allocate(output_shape, strides, dtype, out.device)
    _run_kernel(c_tensors, staging, row_width, strides[0])
    out.copy_(staging)
    return out
