from typing import List, Tuple

import torch
import triton
import triton.language as tl


# 1
@triton.jit
def _meshgrid_kernel_2d(
    out0_ptr,
    out1_ptr,
    in0_ptr,
    in1_ptr,
    size0,
    size1,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    row_idx = offsets // size1
    col_idx = offsets - row_idx * size1

    vals0 = tl.load(in0_ptr + row_idx, mask=mask)
    vals1 = tl.load(in1_ptr + col_idx, mask=mask)

    tl.store(out0_ptr + offsets, vals0, mask=mask)
    tl.store(out1_ptr + offsets, vals1, mask=mask)


@triton.jit
def _meshgrid_kernel_2d_tiled(
    out0_ptr,
    out1_ptr,
    in0_ptr,
    in1_ptr,
    size0,
    size1,
    BLOCK_SIZE0: tl.constexpr,
    BLOCK_SIZE1: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    row_offsets = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)
    col_offsets = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)

    row_mask = row_offsets < size0
    col_mask = col_offsets < size1

    in0_vals = tl.load(in0_ptr + row_offsets, mask=row_mask)
    in1_vals = tl.load(in1_ptr + col_offsets, mask=col_mask)

    out0_vals = tl.broadcast_to(in0_vals[:, None], (BLOCK_SIZE0, BLOCK_SIZE1))
    out1_vals = tl.broadcast_to(in1_vals[None, :], (BLOCK_SIZE0, BLOCK_SIZE1))

    row_idx = row_offsets[:, None]
    col_idx = col_offsets[None, :]
    out_offset = row_idx * size1 + col_idx

    combined_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out0_ptr + out_offset, out0_vals, mask=combined_mask)
    tl.store(out1_ptr + out_offset, out1_vals, mask=combined_mask)


@triton.jit
def _meshgrid_kernel_2d_small(
    out0_ptr,
    out1_ptr,
    in0_ptr,
    in1_ptr,
    size0,
    size1,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    row_offsets = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_mask = row_offsets < size0
    col_mask = col_offsets < size1

    row_vals = tl.load(in0_ptr + row_offsets, mask=row_mask)
    col_vals = tl.load(in1_ptr + col_offsets, mask=col_mask)

    out0_vals = tl.broadcast_to(row_vals[:, None], (BLOCK_SIZE, BLOCK_SIZE))
    out1_vals = tl.broadcast_to(col_vals[None, :], (BLOCK_SIZE, BLOCK_SIZE))

    row_idx = row_offsets[:, None]
    col_idx = col_offsets[None, :]
    out_offset = row_idx * size1 + col_idx

    combined_mask = row_mask[:, None] & col_mask[None, :]

    tl.store(out0_ptr + out_offset, out0_vals, mask=combined_mask)
    tl.store(out1_ptr + out_offset, out1_vals, mask=combined_mask)


@triton.jit
def _meshgrid_kernel_3d(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    size0,
    size1,
    size2,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    size12 = size1 * size2
    idx0 = offsets // size12
    remaining = offsets - idx0 * size12
    idx1 = remaining // size2
    idx2 = remaining - idx1 * size2

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    val1 = tl.load(in1_ptr + idx1, mask=mask)
    val2 = tl.load(in2_ptr + idx2, mask=mask)

    tl.store(out0_ptr + offsets, val0, mask=mask)
    tl.store(out1_ptr + offsets, val1, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)


@triton.jit
def _meshgrid_kernel_3d_strided(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    size0,
    size1,
    size2,
    stride0_out,
    stride1_out,
    stride2_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    size12 = size1 * size2
    total_elements = size0 * size1 * size2
    mask = offsets < total_elements

    idx0 = offsets // size12
    remaining = offsets - idx0 * size12
    idx1 = remaining // size2
    idx2 = remaining - idx1 * size2

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    val1 = tl.load(in1_ptr + idx1, mask=mask)
    val2 = tl.load(in2_ptr + idx2, mask=mask)

    out_offset = idx0 * stride0_out + idx1 * stride1_out + idx2 * stride2_out

    tl.store(out0_ptr + out_offset, val0, mask=mask)
    tl.store(out1_ptr + out_offset, val1, mask=mask)
    tl.store(out2_ptr + out_offset, val2, mask=mask)


@triton.jit
def _meshgrid_kernel_4d(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    size0,
    size1,
    size2,
    size3,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    size23 = size2 * size3
    size123 = size1 * size2 * size3

    idx0 = offsets // size123
    remaining1 = offsets - idx0 * size123
    idx1 = remaining1 // size23
    remaining2 = remaining1 - idx1 * size23
    idx2 = remaining2 // size3
    idx3 = remaining2 - idx2 * size3

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    val1 = tl.load(in1_ptr + idx1, mask=mask)
    val2 = tl.load(in2_ptr + idx2, mask=mask)
    val3 = tl.load(in3_ptr + idx3, mask=mask)

    tl.store(out0_ptr + offsets, val0, mask=mask)
    tl.store(out1_ptr + offsets, val1, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)
    tl.store(out3_ptr + offsets, val3, mask=mask)


@triton.jit
def _meshgrid_kernel_4d_strided(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    size0,
    size1,
    size2,
    size3,
    stride0_out,
    stride1_out,
    stride2_out,
    stride3_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    size23 = size2 * size3
    size123 = size1 * size2 * size3
    total_elements = size0 * size1 * size2 * size3
    mask = offsets < total_elements

    idx0 = offsets // size123
    remaining1 = offsets - idx0 * size123
    idx1 = remaining1 // size23
    remaining2 = remaining1 - idx1 * size23
    idx2 = remaining2 // size3
    idx3 = remaining2 - idx2 * size3

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    val1 = tl.load(in1_ptr + idx1, mask=mask)
    val2 = tl.load(in2_ptr + idx2, mask=mask)
    val3 = tl.load(in3_ptr + idx3, mask=mask)

    out_offset = (
        idx0 * stride0_out
        + idx1 * stride1_out
        + idx2 * stride2_out
        + idx3 * stride3_out
    )

    tl.store(out0_ptr + out_offset, val0, mask=mask)
    tl.store(out1_ptr + out_offset, val1, mask=mask)
    tl.store(out2_ptr + out_offset, val2, mask=mask)
    tl.store(out3_ptr + out_offset, val3, mask=mask)


def meshgrid(
    tensors: List[torch.Tensor], indexing: str = "ij"
) -> Tuple[torch.Tensor, ...]:
    if not tensors:
        raise ValueError("tensors must be a non-empty list or tuple")

    rank = len(tensors)
    if rank > 4:
        raise NotImplementedError("Currently only supports up to 4 dimensions")

    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"tensors[{i}] must be a torch.Tensor")
        if t.dim() != 1:
            raise ValueError(f"tensors[{i}] must be 1D, got shape {t.shape}")

    if indexing not in ["ij", "xy"]:
        raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing}")

    device = tensors[0].device

    if device.type != "cuda":
        raise RuntimeError(f"All tensors must be on CUDA devices, got {device}")

    for t in tensors:
        if t.device.type != "cuda":
            raise RuntimeError(f"All tensors must be on CUDA devices, got {t.device}")

    if rank == 1:
        return tensors

    if rank == 2:
        return _meshgrid_2d(tensors, indexing)
    elif rank == 3:
        return _meshgrid_3d(tensors, indexing)
    elif rank == 4:
        return _meshgrid_4d(tensors, indexing)
    else:
        raise ValueError(f"Unsupported rank: {rank}")


def _meshgrid_2d(tensors, indexing):
    x, y = tensors
    size0, size1 = x.size(0), y.size(0)

    if indexing == "xy":
        out_shape = (size1, size0)
        in0, in1 = y, x
        actual_size0, actual_size1 = size1, size0
    else:
        out_shape = (size0, size1)
        in0, in1 = x, y
        actual_size0, actual_size1 = size0, size1

    out0 = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out1 = torch.empty(out_shape, device=y.device, dtype=y.dtype)

    total_elements = actual_size0 * actual_size1

    if total_elements <= 16384:
        if total_elements <= 4096:
            BLOCK_SIZE = 16
        else:
            BLOCK_SIZE = 32

        grid0 = triton.cdiv(actual_size0, BLOCK_SIZE)
        grid1 = triton.cdiv(actual_size1, BLOCK_SIZE)

        _meshgrid_kernel_2d_small[grid0, grid1](
            out0, out1, in0, in1, actual_size0, actual_size1, BLOCK_SIZE
        )
    elif in0.is_contiguous() and in1.is_contiguous():
        if total_elements > 4 * 1024 * 1024:
            BLOCK_SIZE = 1024
        elif total_elements > 2 * 1024 * 1024:
            BLOCK_SIZE = 768
        elif total_elements > 1024 * 1024:
            BLOCK_SIZE = 512
        elif total_elements > 256 * 256:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 128

        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

        _meshgrid_kernel_2d[grid](
            out0, out1, in0, in1, actual_size0, actual_size1, total_elements, BLOCK_SIZE
        )
    else:
        if actual_size0 > 64 and actual_size1 > 64:
            BLOCK_SIZE0, BLOCK_SIZE1 = 32, 32
        else:
            BLOCK_SIZE0, BLOCK_SIZE1 = 16, 16

        grid0 = triton.cdiv(actual_size0, BLOCK_SIZE0)
        grid1 = triton.cdiv(actual_size1, BLOCK_SIZE1)

        _meshgrid_kernel_2d_tiled[grid0, grid1](
            out0, out1, in0, in1, actual_size0, actual_size1, BLOCK_SIZE0, BLOCK_SIZE1
        )

    if indexing == "xy":
        return out1, out0
    return out0, out1


def _meshgrid_3d(tensors, indexing):
    x, y, z = tensors

    if indexing == "xy":
        size0, size1, size2 = y.size(0), x.size(0), z.size(0)
        in0, in1, in2 = y, x, z
    else:
        size0, size1, size2 = x.size(0), y.size(0), z.size(0)
        in0, in1, in2 = x, y, z

    out0 = torch.empty(size0, size1, size2, device=x.device, dtype=x.dtype)
    out1 = torch.empty(size0, size1, size2, device=y.device, dtype=y.dtype)
    out2 = torch.empty(size0, size1, size2, device=z.device, dtype=z.dtype)

    num_elements = size0 * size1 * size2

    if num_elements <= 4096:
        BLOCK_SIZE = 64
    elif num_elements > 4 * 1024 * 1024:
        BLOCK_SIZE = 1024
    elif num_elements > 1024 * 1024:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    all_contiguous = all(t.is_contiguous() for t in [in0, in1, in2])

    if all_contiguous:
        _meshgrid_kernel_3d[grid](
            out0,
            out1,
            out2,
            in0,
            in1,
            in2,
            size0,
            size1,
            size2,
            num_elements,
            BLOCK_SIZE,
        )
    else:
        stride0_out, stride1_out, stride2_out = out0.stride()

        _meshgrid_kernel_3d_strided[grid](
            out0,
            out1,
            out2,
            in0,
            in1,
            in2,
            size0,
            size1,
            size2,
            stride0_out,
            stride1_out,
            stride2_out,
            BLOCK_SIZE,
        )

    if indexing == "xy":
        return out1, out0, out2
    return out0, out1, out2


def _meshgrid_4d(tensors, indexing):
    x, y, z, w = tensors

    if indexing == "xy":
        size0, size1, size2, size3 = y.size(0), x.size(0), z.size(0), w.size(0)
        in0, in1, in2, in3 = y, x, z, w
    else:
        size0, size1, size2, size3 = x.size(0), y.size(0), z.size(0), w.size(0)
        in0, in1, in2, in3 = x, y, z, w

    out0 = torch.empty(size0, size1, size2, size3, device=x.device, dtype=x.dtype)
    out1 = torch.empty(size0, size1, size2, size3, device=y.device, dtype=y.dtype)
    out2 = torch.empty(size0, size1, size2, size3, device=z.device, dtype=z.dtype)
    out3 = torch.empty(size0, size1, size2, size3, device=w.device, dtype=w.dtype)

    num_elements = size0 * size1 * size2 * size3

    if num_elements <= 4096:
        BLOCK_SIZE = 64
    elif num_elements > 4 * 1024 * 1024:
        BLOCK_SIZE = 1024
    elif num_elements > 1024 * 1024:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    all_contiguous = all(t.is_contiguous() for t in [in0, in1, in2, in3])

    if all_contiguous:
        _meshgrid_kernel_4d[grid](
            out0,
            out1,
            out2,
            out3,
            in0,
            in1,
            in2,
            in3,
            size0,
            size1,
            size2,
            size3,
            num_elements,
            BLOCK_SIZE,
        )
    else:
        stride0_out, stride1_out, stride2_out, stride3_out = out0.stride()

        _meshgrid_kernel_4d_strided[grid](
            out0,
            out1,
            out2,
            out3,
            in0,
            in1,
            in2,
            in3,
            size0,
            size1,
            size2,
            size3,
            stride0_out,
            stride1_out,
            stride2_out,
            stride3_out,
            BLOCK_SIZE,
        )

    if indexing == "xy":
        return out1, out0, out2, out3
    return out0, out1, out2, out3


__all__ = ["meshgrid"]
 
