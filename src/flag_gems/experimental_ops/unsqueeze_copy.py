import torch
import triton
import triton.language as tl


@triton.jit
def _unsqueeze_copy_kernel(
    src_ptr,          # pointer to input tensor data
    dst_ptr,          # pointer to output tensor data
    sizes_ptr,        # pointer to int64 sizes of src tensor (NDIM)
    src_strides_ptr,  # pointer to int64 strides of src tensor (NDIM)
    dst_strides_ptr,  # pointer to int64 strides of dst tensor (NDIM + 1)
    n_elements,       # total number of elements to copy (src.numel() == dst.numel())
    NDIM: tl.constexpr,
    INSERT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # use int64 for index math
    offs = offs.to(tl.int64)

    # Compute source and destination element offsets using shape/strides
    src_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    dst_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    rem = offs
    # Decompose linear index into multi-dimensional indices (row-major order)
    for rev_d in range(NDIM - 1, -1, -1):
        sz_d = tl.load(sizes_ptr + rev_d)       # scalar int64
        idx_d = rem % sz_d
        rem = rem // sz_d

        sstride_d = tl.load(src_strides_ptr + rev_d)
        src_off += idx_d * sstride_d

        # Map source dim rev_d to destination dim (account for inserted dim)
        if rev_d < INSERT_DIM:
            dstride_d = tl.load(dst_strides_ptr + rev_d)
            dst_off += idx_d * dstride_d
        else:
            dstride_shift = tl.load(dst_strides_ptr + (rev_d + 1))
            dst_off += idx_d * dstride_shift

    vals = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + dst_off, vals, mask=mask)


def _launch_unsqueeze_copy(src: torch.Tensor, dim: int, out: torch.Tensor):
    assert src.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert src.dtype == out.dtype, "Dtype mismatch between src and out"

    n_elements = src.numel()
    if n_elements == 0:
        return  # nothing to copy

    # Build metadata arrays on device
    sizes = torch.tensor(list(src.shape), dtype=torch.int64, device=src.device)
    src_strides = torch.tensor(list(src.stride()), dtype=torch.int64, device=src.device)
    dst_strides = torch.tensor(list(out.stride()), dtype=torch.int64, device=out.device)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _unsqueeze_copy_kernel[grid](
        src,
        out,
        sizes,
        src_strides,
        dst_strides,
        n_elements,
        NDIM=src.dim(),
        INSERT_DIM=dim,
        BLOCK_SIZE=1024,
    )


def unsqueeze_copy(x: torch.Tensor, dim: int):
    # Normalize dim
    dim_normalized = dim if dim >= 0 else dim + x.dim() + 1
    if not (0 <= dim_normalized <= x.dim()):
        raise IndexError(f"dim {dim} out of range for tensor with {x.dim()} dims")

    new_shape = list(x.shape)
    new_shape.insert(dim_normalized, 1)
    out = torch.empty(new_shape, device=x.device, dtype=x.dtype)

    _launch_unsqueeze_copy(x, dim_normalized, out)
    return out


def unsqueeze_copy_out(x: torch.Tensor, dim: int, out: torch.Tensor):
    # Normalize dim
    dim_normalized = dim if dim >= 0 else dim + x.dim() + 1
    if not (0 <= dim_normalized <= x.dim()):
        raise IndexError(f"dim {dim} out of range for tensor with {x.dim()} dims")

    if out.device != x.device:
        raise ValueError("out tensor must be on the same device as input")
    if out.dtype != x.dtype:
        raise ValueError("out tensor must have the same dtype as input")

    # Ensure out has the correct shape (resize_ follows PyTorch out semantics)
    expected_shape = list(x.shape)
    expected_shape.insert(dim_normalized, 1)
    if list(out.shape) != expected_shape:
        out.resize_(expected_shape)

    _launch_unsqueeze_copy(x, dim_normalized, out)
    return out