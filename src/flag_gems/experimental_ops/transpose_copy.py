import torch
import triton
import triton.language as tl


@triton.jit
def transpose_copy_kernel(
    src_ptr,  # *Pointer* to source tensor data
    dst_ptr,  # *Pointer* to destination tensor data
    sizes_ptr,  # *Pointer* to int64 sizes of output tensor (after transpose)
    src_strides_mapped_ptr,  # *Pointer* to int64 source strides mapped to output dims
    dst_strides_ptr,  # *Pointer* to int64 destination strides
    n_elements,  # total number of elements
    RANK: tl.constexpr,  # tensor rank (ndim)
    BLOCK_SIZE: tl.constexpr,  # block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Work with int64 for safety
    linear = offs.to(tl.int64)

    # Compute per-element source and destination offsets using mixed-radix decomposition
    src_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    dst_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    tmp = linear

    for i in range(RANK):
        size_i = tl.load(sizes_ptr + i)
        idx_i = tmp % size_i
        tmp = tmp // size_i

        dst_stride_i = tl.load(dst_strides_ptr + i)
        src_stride_i = tl.load(src_strides_mapped_ptr + i)

        dst_off += idx_i * dst_stride_i
        src_off += idx_i * src_stride_i

    vals = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + dst_off, vals, mask=mask)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim - ndim if dim >= ndim else dim})"  # noqa: E501
        )
    return dim


def transpose_copy_int_out(x: torch.Tensor, dim0: int, dim1: int, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == out.dtype, "Input and output dtypes must match"

    ndim = x.dim()
    dim0 = _normalize_dim(dim0, ndim)
    dim1 = _normalize_dim(dim1, ndim)

    # Compute output sizes (swap dimensions)
    sizes_out = list(x.size())
    sizes_out[dim0], sizes_out[dim1] = sizes_out[dim1], sizes_out[dim0]

    # Validate output shape
    if list(out.size()) != sizes_out:
        raise ValueError(
            f"out tensor has incorrect shape. Expected {sizes_out}, got {list(out.size())}"
        )

    # Prepare strides
    strides_src = list(x.stride())
    # Map source strides to output dimensions (swap dim0 and dim1)
    src_strides_mapped = strides_src.copy()
    src_strides_mapped[dim0], src_strides_mapped[dim1] = (
        strides_src[dim1],
        strides_src[dim0],
    )

    # Destination strides from 'out' as-is
    dst_strides = list(out.stride())

    # Prepare metadata tensors on device
    device = x.device
    sizes_t = torch.tensor(sizes_out, dtype=torch.int64, device=device)
    src_strides_mapped_t = torch.tensor(
        src_strides_mapped, dtype=torch.int64, device=device
    )
    dst_strides_t = torch.tensor(dst_strides, dtype=torch.int64, device=device)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    transpose_copy_kernel[grid](
        x,
        out,
        sizes_t,
        src_strides_mapped_t,
        dst_strides_t,
        n_elements,
        RANK=ndim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def transpose_copy_int(x: torch.Tensor, dim0: int, dim1: int):
    assert x.is_cuda, "Input must be a CUDA tensor"

    ndim = x.dim()
    dim0 = _normalize_dim(dim0, ndim)
    dim1 = _normalize_dim(dim1, ndim)

    # Output sizes and strides (swap dims/strides to match transpose view semantics)
    sizes_out = list(x.size())
    strides_out = list(x.stride())
    sizes_out[dim0], sizes_out[dim1] = sizes_out[dim1], sizes_out[dim0]
    strides_out[dim0], strides_out[dim1] = strides_out[dim1], strides_out[dim0]

    # Allocate output tensor with transposed strides
    out = torch.empty_strided(sizes_out, strides_out, dtype=x.dtype, device=x.device)

    # Prepare strides mapping for source (match output dims)
    strides_src = list(x.stride())
    src_strides_mapped = strides_src.copy()
    src_strides_mapped[dim0], src_strides_mapped[dim1] = (
        strides_src[dim1],
        strides_src[dim0],
    )

    # Metadata tensors
    device = x.device
    sizes_t = torch.tensor(sizes_out, dtype=torch.int64, device=device)
    src_strides_mapped_t = torch.tensor(
        src_strides_mapped, dtype=torch.int64, device=device
    )
    dst_strides_t = torch.tensor(out.stride(), dtype=torch.int64, device=device)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    transpose_copy_kernel[grid](
        x,
        out,
        sizes_t,
        src_strides_mapped_t,
        dst_strides_t,
        n_elements,
        RANK=ndim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
