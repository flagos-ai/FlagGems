"""
Sophgo TPU-specific vstack operator implementation.

Fix notes:
Adapted for Sophgo TPU by:
1. Using fixed block sizes instead of autotune (Sophgo TPU does not support autotune).
2. Recomputing addresses on each loop iteration, avoiding pointer updates
   that produce scf.for iter_args pattern which PPL ShapeInference cannot handle.
3. Using 2D tensor mode for loading and storing.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def vstack_kernel(
    itensor_ptr0,
    itensor_ptr1,
    itensor_ptr2,
    itensor_ptr3,
    output_ptr,
    numel0,
    numel1,
    numel2,
    numel3,
    out_offset0,
    out_offset1,
    out_offset2,
    out_offset3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0).to(tl.int32)
    col_idx = tl.arange(0, BLOCK_SIZE)

    tiles0 = tl.cdiv(numel0, BLOCK_SIZE)
    tiles1 = tl.cdiv(numel1, BLOCK_SIZE)
    tiles2 = tl.cdiv(numel2, BLOCK_SIZE)
    tile_end0 = tiles0
    tile_end1 = tile_end0 + tiles1
    tile_end2 = tile_end1 + tiles2

    is_tensor0 = pid < tile_end0
    is_tensor1 = (pid >= tile_end0) & (pid < tile_end1)
    is_tensor2 = (pid >= tile_end1) & (pid < tile_end2)

    local_tile_tail = tl.where(is_tensor2, pid - tile_end1, pid - tile_end2).to(
        tl.int32
    )
    local_tile_rest = tl.where(is_tensor1, pid - tile_end0, local_tile_tail).to(
        tl.int32
    )
    local_tile = tl.where(is_tensor0, pid, local_tile_rest).to(tl.int32)
    intensor_ptr = tl.where(
        is_tensor0,
        itensor_ptr0,
        tl.where(
            is_tensor1,
            itensor_ptr1,
            tl.where(is_tensor2, itensor_ptr2, itensor_ptr3),
        ),
    )
    numel = tl.where(
        is_tensor0,
        numel0,
        tl.where(is_tensor1, numel1, tl.where(is_tensor2, numel2, numel3)),
    ).to(tl.int64)
    out_base = tl.where(
        is_tensor0,
        out_offset0,
        tl.where(
            is_tensor1,
            out_offset1,
            tl.where(is_tensor2, out_offset2, out_offset3),
        ),
    ).to(tl.int64)

    # Recompute addresses on each access, avoid pointer updates (Sophgo TPU compatible)
    idx = (local_tile * BLOCK_SIZE + col_idx).to(tl.int64)
    offset_mask = idx < numel
    out = tl.load(intensor_ptr + idx, mask=offset_mask)
    tl.store(output_ptr + out_base + idx, out, mask=offset_mask)


def _select_block_size(num_elems: int, elem_size: int) -> int:
    """Select a fixed block size for the Sophgo TPU (no autotune)."""
    num_bytes = num_elems * elem_size
    if num_bytes >= 4 * 1024 * 1024:
        return 4096
    if num_bytes >= 256 * 1024:
        return 2048
    if num_bytes >= 32 * 1024:
        return 1024
    return 512


def vstack(tensors: list):
    """
    Sophgo TPU-specific vstack implementation.
    Stacks tensors vertically (row-wise).

    Args:
        tensors: List of tensors to stack vertically.

    Returns:
        Output tensor with tensors stacked along dim 0.
    """
    logger.debug("GEMS VSTACK (Sophgo TPU)")

    tensors = torch.atleast_2d(tensors)
    num_tensors = len(tensors)
    assert num_tensors > 0

    # Ensure all tensors are on the same device and have the same dtype
    device = tensors[0].device
    dtype = tensors[0].dtype
    for tensor in tensors:
        assert (
            tensor.device == device
            and tensor.dtype == dtype
            and tensors[0].shape[1:] == tensor.shape[1:]
        )

    c_tensors = [t if t.is_contiguous() else t.contiguous() for t in tensors]
    # Calculate the output shape
    total_rows = sum(tensor.shape[0] for tensor in c_tensors)
    output_shape = list(c_tensors[0].shape)
    output_shape[0] = total_rows
    output = torch.empty(output_shape, device=device, dtype=dtype)

    outer_iters = triton.cdiv(num_tensors, 4)
    total_elem_offset = 0
    for i in range(outer_iters):
        itensors = []
        numel = []
        out_offsets = []
        group_elems = 0
        for j in range(4):
            tensor_idx = i * 4 + j
            if tensor_idx < num_tensors:
                tensor = c_tensors[tensor_idx]
                tensor_numel = tensor.numel()
                itensors.append(tensor)
                numel.append(tensor_numel)
                out_offsets.append(total_elem_offset + group_elems)
                group_elems += tensor_numel
            else:
                itensors.append(c_tensors[0])
                numel.append(0)
                out_offsets.append(0)
        if group_elems == 0:
            continue
        block_size = _select_block_size(group_elems, output.element_size())
        grid = (sum(triton.cdiv(n, block_size) for n in numel),)
        # Launch the kernel
        with torch_device_fn.device(c_tensors[0].device):
            vstack_kernel[grid](
                itensors[0],
                itensors[1],
                itensors[2],
                itensors[3],
                output,
                numel[0],
                numel[1],
                numel[2],
                numel[3],
                out_offsets[0],
                out_offsets[1],
                out_offsets[2],
                out_offsets[3],
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
            total_elem_offset += group_elems
    return output
