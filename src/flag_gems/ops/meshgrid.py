import torch
import triton
import triton.language as tl
from typing import List, Optional, Union


# ============ Platform Detection ============
def get_backend():
    """Detect the current running platform"""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu"
    except:
        pass
    return "cpu"

BACKEND = get_backend()


# ============ Triton Kernels (CUDA only) ============

if BACKEND == "cuda":
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["total_elements"],
    )
    @triton.jit
    def _meshgrid_kernel_2d(
        out0_ptr, out1_ptr,
        x_ptr, y_ptr,
        size_x, size_y,
        xy_mode,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        if xy_mode:
            col = offsets % size_x
            row = offsets // size_x
            x_vals = tl.load(x_ptr + col, mask=mask)
            y_vals = tl.load(y_ptr + row, mask=mask)
        else:
            col = offsets % size_y
            row = offsets // size_y
            x_vals = tl.load(x_ptr + row, mask=mask)
            y_vals = tl.load(y_ptr + col, mask=mask)
        
        tl.store(out0_ptr + offsets, x_vals, mask=mask)
        tl.store(out1_ptr + offsets, y_vals, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["total_elements"],
    )
    @triton.jit
    def _meshgrid_kernel_3d(
        out0_ptr, out1_ptr, out2_ptr,
        x_ptr, y_ptr, z_ptr,
        size_x, size_y, size_z,
        xy_mode,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        if xy_mode:
            size_x_size_z = size_x * size_z
            i = offsets // size_x_size_z
            rem = offsets - i * size_x_size_z
            j = rem // size_z
            k = rem - j * size_z
            x_vals = tl.load(x_ptr + j, mask=mask)
            y_vals = tl.load(y_ptr + i, mask=mask)
            z_vals = tl.load(z_ptr + k, mask=mask)
        else:
            size_y_size_z = size_y * size_z
            i = offsets // size_y_size_z
            rem = offsets - i * size_y_size_z
            j = rem // size_z
            k = rem - j * size_z
            x_vals = tl.load(x_ptr + i, mask=mask)
            y_vals = tl.load(y_ptr + j, mask=mask)
            z_vals = tl.load(z_ptr + k, mask=mask)
        
        tl.store(out0_ptr + offsets, x_vals, mask=mask)
        tl.store(out1_ptr + offsets, y_vals, mask=mask)
        tl.store(out2_ptr + offsets, z_vals, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["total_elements"],
    )
    @triton.jit
    def _meshgrid_kernel_4d(
        out0_ptr, out1_ptr, out2_ptr, out3_ptr,
        x_ptr, y_ptr, z_ptr, w_ptr,
        size_x, size_y, size_z, size_w,
        xy_mode,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        if xy_mode:
            size_x_size_z_size_w = size_x * size_z * size_w
            i = offsets // size_x_size_z_size_w
            rem1 = offsets - i * size_x_size_z_size_w
            size_z_size_w = size_z * size_w
            j = rem1 // size_z_size_w
            rem2 = rem1 - j * size_z_size_w
            k = rem2 // size_w
            l = rem2 - k * size_w
            x_vals = tl.load(x_ptr + j, mask=mask)
            y_vals = tl.load(y_ptr + i, mask=mask)
            z_vals = tl.load(z_ptr + k, mask=mask)
            w_vals = tl.load(w_ptr + l, mask=mask)
        else:
            size_y_size_z_size_w = size_y * size_z * size_w
            i = offsets // size_y_size_z_size_w
            rem1 = offsets - i * size_y_size_z_size_w
            size_z_size_w = size_z * size_w
            j = rem1 // size_z_size_w
            rem2 = rem1 - j * size_z_size_w
            k = rem2 // size_w
            l = rem2 - k * size_w
            x_vals = tl.load(x_ptr + i, mask=mask)
            y_vals = tl.load(y_ptr + j, mask=mask)
            z_vals = tl.load(z_ptr + k, mask=mask)
            w_vals = tl.load(w_ptr + l, mask=mask)
        
        tl.store(out0_ptr + offsets, x_vals, mask=mask)
        tl.store(out1_ptr + offsets, y_vals, mask=mask)
        tl.store(out2_ptr + offsets, z_vals, mask=mask)
        tl.store(out3_ptr + offsets, w_vals, mask=mask)


# ============ Main Function ============

def meshgrid(
    tensors: List[torch.Tensor],
    indexing: str = 'ij'
) -> List[torch.Tensor]:
    """
    Create coordinate grids from 1D tensors.
    Functionality is identical to torch.meshgrid.
    
    Args:
        tensors: List of 1D tensors
        indexing: 'ij' (matrix indexing) or 'xy' (cartesian indexing)
    
    Returns:
        List of output tensors
    """
    # ========== Input Validation ==========
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"tensors must be list or tuple, got {type(tensors)}")
    
    if len(tensors) < 1:
        raise ValueError("tensors must be a non-empty list or tuple")
    
    if indexing not in ['ij', 'xy']:
        raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing}")
    
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Each tensor must be a torch.Tensor, got {type(t)} at position {i}")
        if t.dim() > 1:
            raise ValueError(f"All tensors must be 1D, got tensor with {t.dim()} dimensions at position {i}")
    
    dtype = tensors[0].dtype
    for i, t in enumerate(tensors):
        if t.dtype != dtype:
            raise ValueError(
                f"All tensors must have the same dtype, got {dtype} and {t.dtype} at position {i}"
            )
    
    device = tensors[0].device
    for i, t in enumerate(tensors):
        if t.device != device:
            raise ValueError(
                f"All tensors must be on the same device, got {device} and {t.device} at position {i}"
            )
    
    ndim = len(tensors)
    
    # ========== Dimension Limit Check ==========
    # Only restrict dimensions for CUDA platform, since only 2-4D have Triton kernels
    # Other platforms are unrestricted (using PyTorch native implementation)
    if BACKEND == "cuda" and str(device) == "cuda":
        if ndim > 4:
            # For >4 dimensions, fallback to PyTorch native implementation
            return list(torch.meshgrid(*tensors, indexing=indexing))
    else:
        # NPU/CPU directly use PyTorch native implementation
        return list(torch.meshgrid(*tensors, indexing=indexing))
    
    # ========== Compute Shape ==========
    sizes = [t.numel() for t in tensors]
    
    if indexing == 'ij':
        shape = tuple(sizes)
    else:  # 'xy'
        if ndim >= 2:
            shape = tuple([sizes[1], sizes[0]] + sizes[2:])
        else:
            shape = tuple(sizes)
    
    # ========== Create Output Tensors ==========
    outputs = [torch.empty(shape, dtype=dtype, device=device) for _ in range(ndim)]
    
    # ========== Special Case: All Dimensions are 1 ==========
    if all(s == 1 for s in sizes):
        for i, t in enumerate(tensors):
            outputs[i].fill_(t.item() if t.numel() == 1 else t)
        return outputs
    
    # ========== Launch Triton Kernel ==========
    total_elements = 1
    for s in shape:
        total_elements *= s
    
    min_block_size = 64
    num_blocks = (total_elements + min_block_size - 1) // min_block_size
    
    if ndim == 2:
        _meshgrid_kernel_2d[(num_blocks,)](
            outputs[0], outputs[1],
            tensors[0], tensors[1],
            sizes[0], sizes[1],
            indexing == 'xy',
            total_elements,
        )
    elif ndim == 3:
        _meshgrid_kernel_3d[(num_blocks,)](
            outputs[0], outputs[1], outputs[2],
            tensors[0], tensors[1], tensors[2],
            sizes[0], sizes[1], sizes[2],
            indexing == 'xy',
            total_elements,
        )
    elif ndim == 4:
        _meshgrid_kernel_4d[(num_blocks,)](
            outputs[0], outputs[1], outputs[2], outputs[3],
            tensors[0], tensors[1], tensors[2], tensors[3],
            sizes[0], sizes[1], sizes[2], sizes[3],
            indexing == 'xy',
            total_elements,
        )
    
    return outputs


# ============ Registration Function ============

def register_ops(registry):
    """Register meshgrid operators to PDU."""
    registry.register_op("meshgrid", meshgrid, "aten::meshgrid")
