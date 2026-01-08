import torch
import triton
import triton.language as tl


@triton.jit
def square_(
    x_ptr,  # Pointer to tensor storage
    n_elements,  # Number of elements to process
    shape_ptr,  # Pointer to int64 shape array of length NDIM
    strides_ptr,  # Pointer to int64 stride array of length NDIM (in elements)
    storage_offset,  # Base storage offset (in elements)
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Compute strided offsets for arbitrary views
    tmp = idx.to(tl.int64)
    off = tmp * 0 + storage_offset  # vector initialized with storage_offset

    # Map linear indices to strided offsets (iterate from last dim to first)
    for d in range(NDIM - 1, -1, -1):
        size_d = tl.load(shape_ptr + d)
        stride_d = tl.load(strides_ptr + d)
        idx_d = tmp % size_d
        off += idx_d * stride_d
        tmp = tmp // size_d

    ptrs = x_ptr + off
    vals = tl.load(ptrs, mask=mask)
    squared = vals * vals
    tl.store(ptrs, squared, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper of the same name
square___triton_kernel = square_


def square_(*args, **kwargs):
    x = None
    if len(args) > 0:
        x = args[0]
    elif "input" in kwargs:
        x = kwargs["input"]
    elif "self" in kwargs:
        x = kwargs["self"]
    if x is None:
        raise ValueError(
            "square_ expects a tensor as the first positional argument or as 'input'/'self' keyword."
        )

    # Fallback for non-CUDA tensors or unsupported dtypes
    if not x.is_cuda or x.numel() == 0:
        return torch.ops.aten.square_(x)

    # Triton kernel launch
    n_elements = x.numel()
    shape = torch.tensor(x.shape, dtype=torch.int64, device=x.device)
    strides = torch.tensor(x.stride(), dtype=torch.int64, device=x.device)
    storage_offset = x.storage_offset()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    square___triton_kernel[grid](
        x,  # in-place
        n_elements,
        shape,
        strides,
        storage_offset,
        BLOCK_SIZE=BLOCK_SIZE,
        NDIM=x.ndim,
    )
    return x
