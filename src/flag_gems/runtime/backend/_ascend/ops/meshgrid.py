from typing import List, Tuple

import torch


def _meshgrid_2d_npu(tensors, indexing):
    """NPU-specific implementation for 2D meshgrid using as_strided."""
    x, y = tensors
    nx, ny = x.numel(), y.numel()

    if indexing == "ij":
        out0 = x.as_strided((nx, ny), (1, 0))
        out1 = y.as_strided((nx, ny), (0, 1))
    else:
        out0 = x.as_strided((ny, nx), (0, 1))
        out1 = y.as_strided((ny, nx), (1, 0))

    return out0, out1


def _meshgrid_3d_npu(tensors, indexing):
    """NPU-specific implementation for 3D meshgrid using as_strided."""
    x, y, z = tensors
    nx, ny, nz = x.numel(), y.numel(), z.numel()

    if indexing == "ij":
        out0 = x.as_strided((nx, ny, nz), (1, 0, 0))
        out1 = y.as_strided((nx, ny, nz), (0, 1, 0))
        out2 = z.as_strided((nx, ny, nz), (0, 0, 1))
    else:
        out0 = x.as_strided((ny, nx, nz), (0, 1, 0))
        out1 = y.as_strided((ny, nx, nz), (1, 0, 0))
        out2 = z.as_strided((ny, nx, nz), (0, 0, 1))

    return out0, out1, out2


def _meshgrid_4d_npu(tensors, indexing):
    """NPU-specific implementation for 4D meshgrid using as_strided."""
    x, y, z, w = tensors
    nx, ny, nz, nw = x.numel(), y.numel(), z.numel(), w.numel()

    if indexing == "ij":
        out0 = x.as_strided((nx, ny, nz, nw), (1, 0, 0, 0))
        out1 = y.as_strided((nx, ny, nz, nw), (0, 1, 0, 0))
        out2 = z.as_strided((nx, ny, nz, nw), (0, 0, 1, 0))
        out3 = w.as_strided((nx, ny, nz, nw), (0, 0, 0, 1))
    else:
        out0 = x.as_strided((ny, nx, nz, nw), (0, 1, 0, 0))
        out1 = y.as_strided((ny, nx, nz, nw), (1, 0, 0, 0))
        out2 = z.as_strided((ny, nx, nz, nw), (0, 0, 1, 0))
        out3 = w.as_strided((ny, nx, nz, nw), (0, 0, 0, 1))

    return out0, out1, out2, out3


def _meshgrid_nd_npu(tensors, indexing):
    """NPU-specific implementation for N-D meshgrid using as_strided."""
    ndim = len(tensors)

    if indexing == "xy":
        tensors_ordered = list(tensors)
        tensors_ordered[0], tensors_ordered[1] = tensors_ordered[1], tensors_ordered[0]
        sizes = [t.numel() for t in tensors_ordered]
        out_shape = list(sizes)
        out_shape[0], out_shape[1] = out_shape[1], out_shape[0]
        in_tensors = tensors_ordered
    else:
        sizes = [t.numel() for t in tensors]
        out_shape = sizes
        in_tensors = tensors

    strides = []
    for i in range(ndim):
        stride = [0] * ndim
        stride[i] = 1
        strides.append(tuple(stride))

    out_tensors = []
    for i, t in enumerate(in_tensors):
        out_tensors.append(t.as_strided(tuple(out_shape), strides[i]))

    if indexing == "xy":
        out_tensors[0], out_tensors[1] = out_tensors[1], out_tensors[0]

    return tuple(out_tensors)


def _dispatch_npu_meshgrid(tensors, indexing, rank):
    """Dispatch NPU meshgrid based on rank."""
    if rank == 1:
        return tuple(tensors)
    elif rank == 2:
        return _meshgrid_2d_npu(tensors, indexing)
    elif rank == 3:
        return _meshgrid_3d_npu(tensors, indexing)
    elif rank == 4:
        return _meshgrid_4d_npu(tensors, indexing)
    else:
        return _meshgrid_nd_npu(tensors, indexing)


def meshgrid(
    tensors: List[torch.Tensor], indexing: str = "ij"
) -> Tuple[torch.Tensor, ...]:
    """
    Create coordinate grids from 1D tensors.

    Uses as_strided for zero-copy operations.
    """
    if not tensors:
        raise ValueError("tensors must be a non-empty list or tuple")

    rank = len(tensors)
    if rank > 8:
        raise NotImplementedError("Currently only supports up to 8 dimensions")

    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"tensors[{i}] must be a torch.Tensor")
        if t.dim() != 1:
            raise ValueError(f"tensors[{i}] must be 1D, got shape {t.shape}")

    if indexing not in ["ij", "xy"]:
        raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing}")

    device = tensors[0].device
    for t in tensors[1:]:
        if t.device != device:
            raise RuntimeError(
                f"All tensors must be on the same device, got {device} and {t.device}"
            )

    return _dispatch_npu_meshgrid(tensors, indexing, rank)
