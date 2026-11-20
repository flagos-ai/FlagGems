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
"""Kunlunxin override of :func:`flag_gems.masked_select_backward`.

The public KernelGen implementation is numerically correct on CUDA but its
kernels read the *tail* of the final (partial) tile with a masked load and
rely on ``other=0`` to neutralise the out-of-bounds lanes.  On this XPU stack
a ``tl.load(..., mask=valid, other=0)`` does **not** apply ``other`` -- the
out-of-bounds lanes read whatever lives past the allocation -- so any
reduction that consumes the loaded tile (``tl.sum`` for the total mask count)
is poisoned by garbage.  In the test suite that surfaces as a spurious
``RuntimeError: Number of elements of source < number of ones in mask`` for
shapes whose last tile is partial (32769, 65537, 262145, ...) even though the
gradient really is long enough.

The fix applied here is to keep every load inside the allocation:
``tl.minimum(offsets, n - 1)`` clamps the address (``n >= 1`` is guaranteed by
the host early return) and the lanes past ``n`` are zeroed with ``tl.where``
before any reduction.  The grad gather is clamped the same way, so no kernel
ever touches OOB memory.  ``tl.cumsum`` itself is verified correct on this
stack for every tile width used here.

The host wrapper also returns a *contiguous* output: ATen's
``masked_select_backward`` materialises a dense result (verified for
transposed and channels-last inputs), while ``torch.empty_like(...,
preserve_format)`` on this backend produces an F-contiguous / mis-strided
buffer for such inputs, failing the layout assertion of the test suite.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.ops.masked_select_backward import _broadcast_views
from flag_gems.ops.neg import neg
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _masked_select_backward_single_pass_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    total_count_ptr,
    grad_numel,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    # Clamp instead of masking the load: OOB lanes must never be read (see
    # module docstring), the clamp keeps the address in the allocation and the
    # `tl.where` below zeroes the lanes past n_elements.
    safe_mask = tl.minimum(offsets, n_elements - 1)
    selected = tl.where(valid, tl.load(mask_ptr + safe_mask) != 0, False).to(tl.int32)
    tl.store(total_count_ptr, tl.sum(selected, axis=0))
    grad_offsets = tl.cumsum(selected, axis=0) - 1
    source_valid = (selected != 0) & (grad_offsets < grad_numel)
    safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
    values = tl.load(grad_ptr + safe_grad)
    tl.store(out_ptr + offsets, tl.where(source_valid, values, 0), mask=valid)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_redundant_prefix_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    last_block_id = tl.minimum(num_blocks - 1, start_block + num_blocks_per_row - 1)
    lane = tl.arange(0, BLOCK_SIZE)

    # For medium tensors, redundant mask reads are cheaper than allocating and
    # synchronizing a global prefix-sum workspace.  Every block except the last
    # is fully in-bounds, but keep the clamp everywhere for uniformity.
    #
    # This kernel only scatters the gradient.  The global count used for the
    # ``source < mask ones`` validation comes from
    # ``_masked_select_backward_count_kernel``: on the flagtree XPU compiler
    # the masked tail store below corrupts a ``tl.sum`` over the same block
    # (even though the reduction precedes the store in source order), so the
    # count must not be computed here.
    grad_advance = 0
    prefix_offsets = lane
    for _ in range(0, start_block):
        grad_advance += tl.sum(
            tl.load(mask_ptr + tl.minimum(prefix_offsets, n_elements - 1)).to(tl.int32),
            axis=0,
        )
        prefix_offsets += BLOCK_SIZE

    offsets = start_block * BLOCK_SIZE + lane
    for _ in range(start_block, last_block_id):
        selected = tl.where(
            offsets < n_elements,
            tl.load(mask_ptr + tl.minimum(offsets, n_elements - 1)) != 0,
            False,
        ).to(tl.int32)
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        # Every block handled by this loop is fully inside ``n_elements`` (the
        # only partial block is the tail below), so this store needs no mask.
        tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0))
        grad_advance += tl.sum(selected, axis=0)
        offsets += BLOCK_SIZE

    valid = offsets < n_elements
    selected = tl.where(
        valid, tl.load(mask_ptr + tl.minimum(offsets, n_elements - 1)) != 0, False
    ).to(tl.int32)
    grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
    safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
    values = tl.load(grad_ptr + safe_grad)
    tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0), mask=valid)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_count_kernel(
    mask_ptr,
    part_sums_ptr,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block_id = tl.minimum(num_blocks - 1, start_block + num_blocks_per_row - 1)

    count = 0
    for _ in range(start_block, last_block_id):
        count += tl.sum(
            tl.load(mask_ptr + tl.minimum(offsets, n_elements - 1)).to(tl.int32),
            axis=0,
        )
        offsets += BLOCK_SIZE
    valid = offsets < n_elements
    count += tl.sum(
        tl.where(
            valid,
            tl.load(mask_ptr + tl.minimum(offsets, n_elements - 1)),
            False,
        ).to(tl.int32),
        axis=0,
    )
    tl.store(part_sums_ptr + row_id, count)


@libentry()
@triton.jit(do_not_specialize=["num_programs"])
def _masked_select_backward_prefix_kernel(
    part_sums_ptr,
    total_count_ptr,
    num_programs,
    NP_BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, NP_BLOCK)
    valid = offsets < num_programs
    safe = tl.minimum(offsets, num_programs - 1)
    counts = tl.where(valid, tl.load(part_sums_ptr + safe), 0)
    tl.store(total_count_ptr, tl.sum(counts, axis=0))
    prefixes = tl.cumsum(counts, axis=0) - counts
    tl.store(part_sums_ptr + offsets, prefixes, mask=valid)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "num_blocks", "num_blocks_per_row"])
def _masked_select_backward_scatter_kernel(
    grad_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start_block = row_id * num_blocks_per_row
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block_id = tl.minimum(num_blocks - 1, start_block + num_blocks_per_row - 1)

    grad_advance = tl.load(part_sums_ptr + row_id)

    for _ in range(start_block, last_block_id):
        selected = tl.where(
            offsets < n_elements,
            tl.load(mask_ptr + tl.minimum(offsets, n_elements - 1)) != 0,
            False,
        ).to(tl.int32)
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        # Full block: unmasked store (see the medium-path kernel for why a
        # masked store here breaks the following ``tl.sum`` on XPU).
        tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0))
        grad_advance += tl.sum(selected, axis=0)
        offsets += BLOCK_SIZE

    valid = offsets < n_elements
    selected = tl.where(
        valid, tl.load(mask_ptr + tl.minimum(offsets, n_elements - 1)) != 0, False
    ).to(tl.int32)
    grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
    safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
    values = tl.load(grad_ptr + safe_grad)
    tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0), mask=valid)


def _masked_select_backward_real(grad, mask, out, validate=True):
    n_elements = out.numel()
    if n_elements == 0:
        return out

    grad = contiguous(grad)
    # Zero-initialise: a missed kernel store must never leave garbage that the
    # host-side ``grad.numel() < total_count`` validation would misread.
    total_count = torch.zeros((), dtype=torch.int64, device=out.device)
    if n_elements <= 32768:
        block_size = triton.next_power_of_2(n_elements)
        num_warps = 4 if block_size < 2048 else min(16, block_size // 256)
        with torch_device_fn.device(out.device):
            _masked_select_backward_single_pass_kernel[(1,)](
                grad,
                mask,
                out,
                total_count,
                grad.numel(),
                n_elements,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
            raise RuntimeError("Number of elements of source < number of ones in mask")
        return out

    block_size = bracket_next_power_of_2(n_elements, 128, 4096)
    num_warps = min(16, block_size // 32)
    sm_count = torch_device_fn.get_device_properties(mask.device).multi_processor_count
    num_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(num_blocks, sm_count)
    num_blocks_per_row = triton.cdiv(num_blocks, num_programs)
    num_programs = triton.cdiv(num_blocks, num_blocks_per_row)

    scan_block = triton.next_power_of_2(num_programs)
    index_dtype = torch.int32 if n_elements < 2**31 else torch.int64

    if n_elements <= 262144:
        with torch_device_fn.device(out.device):
            part_sums = torch.empty(num_programs, dtype=index_dtype, device=out.device)
            # Count with a store-free kernel first (see the scatter kernel's
            # note on the flagtree masked-store defect), then reduce the
            # per-program counts into ``total_count``.
            _masked_select_backward_count_kernel[(num_programs,)](
                mask,
                part_sums,
                n_elements,
                num_blocks,
                num_blocks_per_row,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            _masked_select_backward_prefix_kernel[(1,)](
                part_sums,
                total_count,
                num_programs,
                NP_BLOCK=scan_block,
                num_warps=4,
            )
            _masked_select_backward_redundant_prefix_kernel[(num_programs,)](
                grad,
                mask,
                out,
                grad.numel(),
                n_elements,
                num_blocks,
                num_blocks_per_row,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
            raise RuntimeError("Number of elements of source < number of ones in mask")
        return out

    with torch_device_fn.device(out.device):
        part_sums = torch.empty(num_programs, dtype=index_dtype, device=out.device)
        _masked_select_backward_count_kernel[(num_programs,)](
            mask,
            part_sums,
            n_elements,
            num_blocks,
            num_blocks_per_row,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        _masked_select_backward_prefix_kernel[(1,)](
            part_sums,
            total_count,
            num_programs,
            NP_BLOCK=scan_block,
            num_warps=4,
        )
        _masked_select_backward_scatter_kernel[(num_programs,)](
            grad,
            mask,
            part_sums,
            out,
            grad.numel(),
            n_elements,
            num_blocks,
            num_blocks_per_row,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    if validate and grad.numel() < n_elements and grad.numel() < total_count.item():
        raise RuntimeError("Number of elements of source < number of ones in mask")
    return out


def masked_select_backward(grad, input, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_scatter_ only supports boolean masks, but got {mask.dtype}"
        )
    if grad.dtype != input.dtype:
        raise RuntimeError(
            "masked_scatter: expected self and source to have same dtypes but got "
            f"{input.dtype} and {grad.dtype}"
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("grad, input, and mask must be on the same device")

    input_expanded, mask_expanded = _broadcast_views(input, mask)
    mask_contiguous = contiguous(mask_expanded)
    # ATen materialises a dense (contiguous) result; empty_like(preserve_format)
    # would produce an F-contiguous / mis-strided buffer on this backend.
    with torch_device_fn.device(input.device):
        result = torch.zeros(
            input_expanded.shape, dtype=input.dtype, device=input.device
        )
    if result.numel() == 0:
        return result
    if grad.numel() == 0:
        # No selected elements: the output is all zeros.  Skipping the kernels
        # also avoids a grad_numel == 0 clamp (min(x, -1) would go OOB).
        return result

    if input.is_complex():
        physical_grad = grad.conj() if grad.is_conj() else grad
        # ``.real`` / ``.imag`` are metadata views of the complex storage
        # (equivalent to indexing ``view_as_real``); no torch compute call is
        # involved and the actual gather runs in the Triton kernels above.
        real_dtype = physical_grad.real.dtype
        real = torch.empty(input_expanded.shape, dtype=real_dtype, device=grad.device)
        imag = torch.empty_like(real)
        _masked_select_backward_real(physical_grad.real, mask_contiguous, real)
        _masked_select_backward_real(
            neg(physical_grad.imag) if grad.is_conj() else physical_grad.imag,
            mask_contiguous,
            imag,
            validate=False,
        )
        copy_(result.real, real)
        copy_(result.imag, imag)
        return result
    else:
        _masked_select_backward_real(grad, mask_contiguous, result)
        return result
