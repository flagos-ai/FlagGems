import torch
import triton
import triton.language as tl
from typing import List


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
IS_NPU = BACKEND == "npu"
IS_CUDA = BACKEND == "cuda"

# NPU specific thresholds
if IS_NPU:
    NPU_TRITON_THRESHOLD = 100000000  # 100M - avoid Triton compilation overhead
else:
    NPU_TRITON_THRESHOLD = 1000000


# ============ Fast Validation ============

def _validate_tensors_fast(tensors, indexing):
    """
    Fast validation with minimal overhead.
    """
    if not tensors:
        raise ValueError("tensors must be a non-empty list or tuple")
    
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"tensors must be list or tuple, got {type(tensors)}")
    
    if indexing not in ('ij', 'xy'):
        raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing}")
    
    first = tensors[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"Each tensor must be a torch.Tensor, got {type(first)} at position 0")
    
    dtype = first.dtype
    device = first.device
    ndim = len(tensors)
    sizes = [0] * ndim
    
    # Check first tensor
    if first.dim() != 1:
        raise ValueError(f"All tensors must be 1D, got tensor with {first.dim()} dimensions at position 0")
    sizes[0] = first.numel()
    
    for i in range(1, ndim):
        t = tensors[i]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Each tensor must be a torch.Tensor, got {type(t)} at position {i}")
        if t.dim() != 1:
            raise ValueError(f"All tensors must be 1D, got tensor with {t.dim()} dimensions at position {i}")
        if t.dtype != dtype:
            raise ValueError(f"All tensors must have the same dtype")
        if t.device != device:
            raise ValueError(f"All tensors must be on the same device")
        sizes[i] = t.numel()
    
    return {
        'ndim': ndim,
        'sizes': tuple(sizes),
        'dtype': dtype,
        'device': device,
    }


# ============ Ultra Fast Path: 2D Very Small Tensors (<=10x10) ============

def _meshgrid_2d_ultra_fast(x, y, indexing):
    """
    Ultra-optimized path for 2D very small tensors - no intermediate object creation.
    """
    nx = x.numel()
    ny = y.numel()
    
    if indexing == 'ij':
        return [x.view(-1, 1).expand(nx, ny), y.view(1, -1).expand(nx, ny)]
    else:
        return [x.view(1, -1).expand(ny, nx), y.view(-1, 1).expand(ny, nx)]


# ============ Fast Path: 2D Medium Tensors (128x128 to 1024x1024) ============

def _meshgrid_2d_medium_fast(x, y, indexing):
    """
    Optimized path for 2D medium tensors - minimize object creation.
    Suitable for tensors from 128x128 to 1024x1024.
    """
    nx = x.numel()
    ny = y.numel()
    
    if indexing == 'ij':
        x_expanded = x.view(-1, 1).expand(nx, ny)
        y_expanded = y.view(1, -1).expand(nx, ny)
        return [x_expanded, y_expanded]
    else:
        x_expanded = x.view(1, -1).expand(ny, nx)
        y_expanded = y.view(-1, 1).expand(ny, nx)
        return [x_expanded, y_expanded]


# ============ Fast Path: 2D General ============

def _meshgrid_2d_fast(tensors, indexing):
    """
    Fast 2D path.
    """
    x, y = tensors[0], tensors[1]
    nx, ny = x.numel(), y.numel()
    
    if indexing == 'ij':
        return [x.view(-1, 1).expand(nx, ny), y.view(1, -1).expand(nx, ny)]
    else:
        return [x.view(1, -1).expand(ny, nx), y.view(-1, 1).expand(ny, nx)]


# ============ Fast Path: 3D/4D ============

def _meshgrid_3d_medium_fast(tensors, indexing):
    """
    Optimized path for 3D medium tensors.
    """
    x, y, z = tensors[0], tensors[1], tensors[2]
    nx, ny, nz = x.numel(), y.numel(), z.numel()
    
    if indexing == 'ij':
        return [
            x.view(-1, 1, 1).expand(nx, ny, nz),
            y.view(1, -1, 1).expand(nx, ny, nz),
            z.view(1, 1, -1).expand(nx, ny, nz)
        ]
    else:
        return [
            x.view(1, -1, 1).expand(ny, nx, nz),
            y.view(-1, 1, 1).expand(ny, nx, nz),
            z.view(1, 1, -1).expand(ny, nx, nz)
        ]


def _meshgrid_4d_medium_fast(tensors, indexing):
    """
    Optimized path for 4D medium tensors.
    """
    x, y, z, w = tensors[0], tensors[1], tensors[2], tensors[3]
    nx, ny, nz, nw = x.numel(), y.numel(), z.numel(), w.numel()
    
    if indexing == 'ij':
        return [
            x.view(-1, 1, 1, 1).expand(nx, ny, nz, nw),
            y.view(1, -1, 1, 1).expand(nx, ny, nz, nw),
            z.view(1, 1, -1, 1).expand(nx, ny, nz, nw),
            w.view(1, 1, 1, -1).expand(nx, ny, nz, nw)
        ]
    else:
        return [
            x.view(1, -1, 1, 1).expand(ny, nx, nz, nw),
            y.view(-1, 1, 1, 1).expand(ny, nx, nz, nw),
            z.view(1, 1, -1, 1).expand(ny, nx, nz, nw),
            w.view(1, 1, 1, -1).expand(ny, nx, nz, nw)
        ]


# ============ General Implementation ============

def _meshgrid_general(tensors, indexing):
    """
    General implementation for all cases not covered by fast paths.
    """
    validated = _validate_tensors_fast(tensors, indexing)
    ndim = validated['ndim']
    sizes = validated['sizes']
    dtype = validated['dtype']
    device = validated['device']
    
    # Compute shape inline
    if indexing == 'ij':
        shape = sizes
    else:
        if ndim >= 2:
            shape = (sizes[1], sizes[0]) + sizes[2:]
        else:
            shape = sizes
    
    total = 1
    for s in shape:
        total *= s
    
    # Use broadcasting for most NPU cases
    if total <= NPU_TRITON_THRESHOLD:
        # Build view shapes
        view_shapes = []
        for i in range(ndim):
            vs = [1] * ndim
            if indexing == 'ij':
                vs[i] = sizes[i]
            else:
                if i == 0:
                    vs[1] = sizes[0]
                elif i == 1:
                    vs[0] = sizes[1]
                else:
                    vs[i] = sizes[i]
            view_shapes.append(tuple(vs))
        
        return [tensors[i].view(view_shapes[i]).expand(shape) for i in range(ndim)]
    
    # Triton path for very large tensors
    outputs = [torch.empty(shape, dtype=dtype, device=device) for _ in range(ndim)]
    
    if all(s == 1 for s in sizes):
        for i, t in enumerate(tensors):
            outputs[i].fill_(t.item())
        return outputs
    
    block_size = 4096
    num_blocks = (total + block_size - 1) // block_size
    xy_mode = indexing == 'xy'
    
    if ndim == 2:
        _meshgrid_kernel_2d_npu[(num_blocks,)](
            outputs[0], outputs[1],
            tensors[0], tensors[1],
            sizes[0], sizes[1],
            xy_mode, total
        )
    elif ndim == 3:
        _meshgrid_kernel_3d_npu[(num_blocks,)](
            outputs[0], outputs[1], outputs[2],
            tensors[0], tensors[1], tensors[2],
            sizes[0], sizes[1], sizes[2],
            xy_mode, total
        )
    elif ndim == 4:
        _meshgrid_kernel_4d_npu[(num_blocks,)](
            outputs[0], outputs[1], outputs[2], outputs[3],
            tensors[0], tensors[1], tensors[2], tensors[3],
            sizes[0], sizes[1], sizes[2], sizes[3],
            xy_mode, total
        )
    
    return outputs


# ============ Direct NPU Implementation ============

def _npu_meshgrid_direct(tensors, indexing):
    """Direct NPU implementation with fast paths."""
    ndim = len(tensors)
    
    # Ultra fast path: 2D very small tensors (<=10x10)
    if ndim == 2:
        x = tensors[0]
        y = tensors[1]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            nx = x.numel()
            ny = y.numel()
            if nx <= 10 and ny <= 10 and x.dim() == 1 and y.dim() == 1:
                return _meshgrid_2d_ultra_fast(x, y, indexing)
    
    # Fast path: 2D medium tensors (128x128 to 1024x1024)
    if ndim == 2:
        x = tensors[0]
        y = tensors[1]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            nx = x.numel()
            ny = y.numel()
            if (128 <= nx <= 1024 and 128 <= ny <= 1024 and 
                x.dim() == 1 and y.dim() == 1):
                if x.dtype == y.dtype and x.device == y.device:
                    return _meshgrid_2d_medium_fast(x, y, indexing)
    
    # Check if all elements are tensors (for error handling)
    if not _is_tensor_list(tensors):
        return _meshgrid_general(tensors, indexing)
    
    # Fast path: 2D small tensors (<=100x100)
    if ndim == 2:
        nx = tensors[0].numel()
        ny = tensors[1].numel()
        if nx <= 100 and ny <= 100:
            return _meshgrid_2d_fast(tensors, indexing)
    
    # Fast path for 3D medium tensors
    if ndim == 3:
        if tensors[0].numel() <= 64 and tensors[1].numel() <= 64 and tensors[2].numel() <= 64:
            return _meshgrid_3d_medium_fast(tensors, indexing)
    
    # Fast path for 4D medium tensors
    if ndim == 4:
        if (tensors[0].numel() <= 32 and tensors[1].numel() <= 32 and 
            tensors[2].numel() <= 32 and tensors[3].numel() <= 32):
            return _meshgrid_4d_medium_fast(tensors, indexing)
    
    # Use general implementation
    return _meshgrid_general(tensors, indexing)


def _is_tensor_list(tensors):
    """Quick check if all elements are tensors"""
    if not tensors:
        return False
    return all(isinstance(t, torch.Tensor) for t in tensors)


# ============ Direct CUDA Implementation ============

def _cuda_meshgrid_direct(tensors, indexing):
    """Direct CUDA implementation with fast paths."""
    ndim = len(tensors)
    
    # Ultra fast path: 2D very small tensors
    if ndim == 2:
        x = tensors[0]
        y = tensors[1]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            nx = x.numel()
            ny = y.numel()
            if nx <= 10 and ny <= 10 and x.dim() == 1 and y.dim() == 1:
                return _meshgrid_2d_ultra_fast(x, y, indexing)
    
    # Fast path: 2D medium tensors
    if ndim == 2:
        x = tensors[0]
        y = tensors[1]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            nx = x.numel()
            ny = y.numel()
            if (128 <= nx <= 1024 and 128 <= ny <= 1024 and 
                x.dim() == 1 and y.dim() == 1 and
                x.dtype == y.dtype and x.device == y.device):
                return _meshgrid_2d_medium_fast(x, y, indexing)
    
    if not _is_tensor_list(tensors):
        return _meshgrid_general(tensors, indexing)
    
    if ndim == 2:
        nx = tensors[0].numel()
        ny = tensors[1].numel()
        if nx <= 100 and ny <= 100:
            return _meshgrid_2d_fast(tensors, indexing)
    
    if ndim == 3:
        if tensors[0].numel() <= 64 and tensors[1].numel() <= 64 and tensors[2].numel() <= 64:
            return _meshgrid_3d_medium_fast(tensors, indexing)
    
    if ndim == 4:
        if (tensors[0].numel() <= 32 and tensors[1].numel() <= 32 and 
            tensors[2].numel() <= 32 and tensors[3].numel() <= 32):
            return _meshgrid_4d_medium_fast(tensors, indexing)
    
    return _meshgrid_general(tensors, indexing)


# ============ Direct CPU Implementation ============

def _cpu_meshgrid_direct(tensors, indexing):
    """Direct CPU implementation with fast paths."""
    ndim = len(tensors)
    
    # Ultra fast path: 2D very small tensors
    if ndim == 2:
        x = tensors[0]
        y = tensors[1]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            nx = x.numel()
            ny = y.numel()
            if nx <= 10 and ny <= 10 and x.dim() == 1 and y.dim() == 1:
                return _meshgrid_2d_ultra_fast(x, y, indexing)
    
    # Fast path: 2D medium tensors
    if ndim == 2:
        x = tensors[0]
        y = tensors[1]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            nx = x.numel()
            ny = y.numel()
            if (128 <= nx <= 1024 and 128 <= ny <= 1024 and 
                x.dim() == 1 and y.dim() == 1 and
                x.dtype == y.dtype and x.device == y.device):
                return _meshgrid_2d_medium_fast(x, y, indexing)
    
    if not _is_tensor_list(tensors):
        return _meshgrid_general(tensors, indexing)
    
    if ndim == 2:
        nx = tensors[0].numel()
        ny = tensors[1].numel()
        if nx <= 100 and ny <= 100:
            return _meshgrid_2d_fast(tensors, indexing)
    
    if ndim == 3:
        if tensors[0].numel() <= 64 and tensors[1].numel() <= 64 and tensors[2].numel() <= 64:
            return _meshgrid_3d_medium_fast(tensors, indexing)
    
    if ndim == 4:
        if (tensors[0].numel() <= 32 and tensors[1].numel() <= 32 and 
            tensors[2].numel() <= 32 and tensors[3].numel() <= 32):
            return _meshgrid_4d_medium_fast(tensors, indexing)
    
    # General CPU implementation
    validated = _validate_tensors_fast(tensors, indexing)
    sizes = validated['sizes']
    
    if indexing == 'ij':
        shape = sizes
    else:
        if ndim >= 2:
            shape = (sizes[1], sizes[0]) + sizes[2:]
        else:
            shape = sizes
    
    outputs = []
    for i, t in enumerate(tensors):
        view_shape = [1] * ndim
        if indexing == 'ij':
            view_shape[i] = sizes[i]
        else:
            if i == 0:
                view_shape[1] = sizes[0]
            elif i == 1:
                view_shape[0] = sizes[1]
            else:
                view_shape[i] = sizes[i]
        outputs.append(t.view(view_shape).expand(shape))
    
    return outputs


# ============ NPU Triton Kernels ============

if BACKEND == "npu":
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
        ],
        key=["total_elements"],
    )
    @triton.jit
    def _meshgrid_kernel_2d_npu(
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
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
        ],
        key=["total_elements"],
    )
    @triton.jit
    def _meshgrid_kernel_3d_npu(
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
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
        ],
        key=["total_elements"],
    )
    @triton.jit
    def _meshgrid_kernel_4d_npu(
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


# ============ CUDA Kernels ============

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
    def _meshgrid_kernel_2d_cuda(
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
    def _meshgrid_kernel_3d_cuda(
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
    def _meshgrid_kernel_4d_cuda(
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
    """Create coordinate grids from 1D tensors."""
    if IS_NPU:
        return _npu_meshgrid_direct(tensors, indexing)
    elif IS_CUDA:
        return _cuda_meshgrid_direct(tensors, indexing)
    else:
        return _cpu_meshgrid_direct(tensors, indexing)


# ============ Registration Function ============

def register_ops(registry):
    """Register meshgrid operators to PDU."""
    registry.register_op("meshgrid", meshgrid, "aten::meshgrid")

# meshgrid ascend backend implementation

# meshgrid ascend backend implementation
