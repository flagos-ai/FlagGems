import torch
import triton
import triton.language as tl


@triton.jit
def as_strided(
    in_ptr,  # pointer to input tensor data
    out_ptr,  # pointer to output tensor data
    sizes_ptr,  # pointer to int64 sizes array (length = dims)
    strides_ptr,  # pointer to int64 strides array (length = dims)
    storage_offset,  # int64: storage offset in elements
    n_elements,  # total number of elements in the output
    in_numel,  # total number of elements in the input (for safety masking)
    dims,  # number of dimensions
    BLOCK_SIZE: tl.constexpr,
    MAX_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    linear_idx = block_start + tl.arange(0, BLOCK_SIZE)
    valid_o = linear_idx < n_elements

    # Cast to int64 for index arithmetic
    rem = tl.cast(linear_idx, dtype=tl.int64)
    pos = tl.zeros([BLOCK_SIZE], dtype=tl.int64) + tl.cast(storage_offset, tl.int64)

    # Compute position in input using output linear index and provided sizes/strides
    # Iterate from the last dimension to the first dimension
    for d in range(MAX_D):
        j = dims - 1 - d
        has_dim = j >= 0  # scalar predicate
        size_j = tl.load(sizes_ptr + j, mask=has_dim, other=1)
        stride_j = tl.load(strides_ptr + j, mask=has_dim, other=0)

        idx_j = rem % size_j
        pos = pos + idx_j * stride_j
        rem = rem // size_j

    # Safety mask for input bounds (avoid OOB loads)
    pos_valid = (pos >= 0) & (pos < tl.cast(in_numel, tl.int64))
    m = valid_o & pos_valid

    # Load and store
    vals = tl.load(in_ptr + pos, mask=m, other=0)
    tl.store(out_ptr + linear_idx, vals, mask=valid_o)


# Keep a reference to the kernel before defining the wrapper with the same name
as_strided_kernel = as_strided


def as_strided(*args, **kwargs):
    # Parse arguments similar to torch.as_strided: (tensor, size, stride, storage_offset=0)
    if len(args) >= 3:
        x = args[0]
        size = args[1]
        stride = args[2]
        storage_offset = (
            args[3]
            if len(args) >= 4
            else kwargs.get("storage_offset", kwargs.get("offset", 0))
        )
    else:
        x = kwargs.get("tensor", kwargs.get("input", kwargs.get("x")))
        size = kwargs.get("size")
        stride = kwargs.get("stride")
        storage_offset = kwargs.get("storage_offset", kwargs.get("offset", 0))

    if x is None or size is None or stride is None:
        raise ValueError(
            "as_strided requires arguments: tensor, size, stride, and optional storage_offset."
        )

    # Prepare output tensor
    out = torch.empty(size, device=x.device, dtype=x.dtype)

    # Handle degenerate case
    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Prepare sizes and strides tensors on device
    size_list = list(size)
    stride_list = list(stride)
    dims = len(size_list)
    MAX_D = 8
    if dims > MAX_D:
        raise ValueError(
            f"as_strided Triton kernel supports up to {MAX_D} dimensions, got {dims}."
        )

    sizes_t = torch.tensor(size_list, dtype=torch.int64, device=x.device)
    strides_t = torch.tensor(stride_list, dtype=torch.int64, device=x.device)
    in_numel = x.numel()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    as_strided_kernel[grid](
        x,
        out,
        sizes_t,
        strides_t,
        int(storage_offset),
        n_elements,
        in_numel,
        dims,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_D=MAX_D,
    )
    return out
