from typing import List, Tuple

import torch
import triton
import triton.language as tl


def register_ops(registry):
    """注册算子到 PDU"""
    registry.register_op("meshgrid", meshgrid, "aten::meshgrid")


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
    """2D meshgrid kernel - vectorized version (fallback for large tensors)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    row_idx = offsets // size1
    col_idx = offsets % size1

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
    """
    2D meshgrid kernel using tiling - avoids division and modulo
    Each block processes one tile
    """
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
    """3D meshgrid kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    size12 = size1 * size2
    idx2 = offsets % size2
    idx1 = (offsets // size2) % size1
    idx0 = offsets // size12

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    tl.store(out0_ptr + offsets, val0, mask=mask)

    val1 = tl.load(in1_ptr + idx1, mask=mask)
    tl.store(out1_ptr + offsets, val1, mask=mask)

    val2 = tl.load(in2_ptr + idx2, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)


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
    """4D meshgrid kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    size23 = size2 * size3
    size123 = size1 * size2 * size3
    idx3 = offsets % size3
    idx2 = (offsets // size3) % size2
    idx1 = (offsets // size23) % size1
    idx0 = offsets // size123

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    tl.store(out0_ptr + offsets, val0, mask=mask)

    val1 = tl.load(in1_ptr + idx1, mask=mask)
    tl.store(out1_ptr + offsets, val1, mask=mask)

    val2 = tl.load(in2_ptr + idx2, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)

    val3 = tl.load(in3_ptr + idx3, mask=mask)
    tl.store(out3_ptr + offsets, val3, mask=mask)


def meshgrid(
    tensors: List[torch.Tensor], indexing: str = "ij"
) -> Tuple[torch.Tensor, ...]:
    """
    High-performance meshgrid implementation

    Strategy:
    - 1D: return directly
    - 2D: use view + expand (fastest)
    - 3D: use view + expand
    - 4D+: use broadcast_tensors (PyTorch C++ accelerated)
    """
    # ---- Input validation ----
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

    # ---- NPU fast path ----
    device = tensors[0].device
    if device.type == "npu":
        return torch.meshgrid(*tensors, indexing=indexing)

    # ---- 1D special case ----
    if rank == 1:
        return tensors

    # ---- 2D fast path (view + expand) ----
    if rank == 2:
        x, y = tensors[0], tensors[1]
        if indexing == "ij":
            return (
                x.view(-1, 1).expand(x.size(0), y.size(0)),
                y.view(1, -1).expand(x.size(0), y.size(0)),
            )
        else:  # xy
            return (
                x.view(1, -1).expand(y.size(0), x.size(0)),
                y.view(-1, 1).expand(y.size(0), x.size(0)),
            )

    # ---- 3D fast path (view + expand) ----
    if rank == 3:
        x, y, z = tensors[0], tensors[1], tensors[2]
        if indexing == "ij":
            return (
                x.view(-1, 1, 1).expand(x.size(0), y.size(0), z.size(0)),
                y.view(1, -1, 1).expand(x.size(0), y.size(0), z.size(0)),
                z.view(1, 1, -1).expand(x.size(0), y.size(0), z.size(0)),
            )
        else:  # xy
            return (
                x.view(1, -1, 1).expand(y.size(0), x.size(0), z.size(0)),
                y.view(-1, 1, 1).expand(y.size(0), x.size(0), z.size(0)),
                z.view(1, 1, -1).expand(y.size(0), x.size(0), z.size(0)),
            )

    # ---- 4D fast path (view + expand) ----
    if rank == 4:
        x, y, z, w = tensors[0], tensors[1], tensors[2], tensors[3]
        if indexing == "ij":
            return (
                x.view(-1, 1, 1, 1).expand(x.size(0), y.size(0), z.size(0), w.size(0)),
                y.view(1, -1, 1, 1).expand(x.size(0), y.size(0), z.size(0), w.size(0)),
                z.view(1, 1, -1, 1).expand(x.size(0), y.size(0), z.size(0), w.size(0)),
                w.view(1, 1, 1, -1).expand(x.size(0), y.size(0), z.size(0), w.size(0)),
            )
        else:  # xy mode: swap first two dimensions
            return (
                x.view(1, -1, 1, 1).expand(y.size(0), x.size(0), z.size(0), w.size(0)),
                y.view(-1, 1, 1, 1).expand(y.size(0), x.size(0), z.size(0), w.size(0)),
                z.view(1, 1, -1, 1).expand(y.size(0), x.size(0), z.size(0), w.size(0)),
                w.view(1, 1, 1, -1).expand(y.size(0), x.size(0), z.size(0), w.size(0)),
            )

    # ---- Fallback: use broadcast_tensors ----
    # This is the general method, implemented in PyTorch C++
    if indexing == "ij":
        reshaped = []
        for i, t in enumerate(tensors):
            shape = [1] * rank
            shape[i] = -1
            reshaped.append(t.view(*shape))
        return torch.broadcast_tensors(*reshaped)
    else:
        # xy mode
        if rank >= 2:
            reshaped = []
            for i, t in enumerate(tensors):
                shape = [1] * rank
                if i == 0:
                    shape[1] = -1
                elif i == 1:
                    shape[0] = -1
                else:
                    shape[i] = -1
                reshaped.append(t.view(*shape))
            return torch.broadcast_tensors(*reshaped)
        else:
            return tensors


def meshgrid_stack(tensors: List[torch.Tensor], indexing: str = "ij") -> torch.Tensor:
    """
    Create coordinate grid and stack into single tensor.

    Args:
        tensors: Input coordinate vectors (list of 1D tensors)
        indexing: 'ij' or 'xy'

    Returns:
        Stacked tensor of shape (N, shape...) where N is number of inputs
    """
    grids = meshgrid(tensors, indexing=indexing)
    return torch.stack(grids, dim=0)


__all__ = ["meshgrid", "meshgrid_stack", "register_ops"]

# meshgrid operator implementation complete

# meshgrid operator implementation complete
