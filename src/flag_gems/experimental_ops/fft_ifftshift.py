import torch
import triton
import triton.language as tl


@triton.jit
def fft_ifftshift(
    in_ptr_u8,  # pointer to input tensor as bytes
    out_ptr_u8,  # pointer to output tensor as bytes
    sizes_ptr,  # int64[NDIMS]
    in_strides_ptr,  # int64[NDIMS], in elements
    out_strides_ptr,  # int64[NDIMS], in elements
    adds_ptr,  # int64[NDIMS], per-dim add = floor(size/2) if shifted else 0
    n_elements,  # total number of elements
    ELEMENT_SIZE: tl.constexpr,  # number of bytes per element
    NDIMS: tl.constexpr,  # number of dimensions
    BLOCK_SIZE: tl.constexpr,  # tile size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    offs64 = offs.to(tl.int64)

    # Compute multi-dimensional indices from linear index (row-major)
    tmp = offs64
    in_off_elems = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    out_off_elems = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Iterate from last dim to first to compute indices
    for d in range(NDIMS - 1, -1, -1):
        size_d = tl.load(sizes_ptr + d)  # scalar int64
        idx_d = tmp % size_d
        tmp = tmp // size_d

        add_d = tl.load(adds_ptr + d)  # scalar int64
        idx_in_d = idx_d + add_d
        # modulo to wrap within size_d
        idx_in_d = idx_in_d - (idx_in_d // size_d) * size_d

        in_stride_d = tl.load(in_strides_ptr + d)
        out_stride_d = tl.load(out_strides_ptr + d)

        in_off_elems += idx_in_d * in_stride_d
        out_off_elems += idx_d * out_stride_d

    in_byte_base = in_off_elems * ELEMENT_SIZE
    out_byte_base = out_off_elems * ELEMENT_SIZE

    # Copy ELEMENT_SIZE bytes per element
    for b in range(ELEMENT_SIZE):
        src_addr = in_ptr_u8 + in_byte_base + b
        dst_addr = out_ptr_u8 + out_byte_base + b
        val = tl.load(src_addr, mask=mask, other=0)
        tl.store(dst_addr, val, mask=mask)


# Keep a handle to the kernel before defining the wrapper with the same name
fft_ifftshift_kernel = fft_ifftshift


def fft_ifftshift(*args, **kwargs):
    x = None
    dims = None

    # Parse input tensor
    if len(args) >= 1:
        x = args[0]
    else:
        # try kwargs
        x = (
            kwargs.get("input", None)
            or kwargs.get("self", None)
            or kwargs.get("tensor", None)
        )
    if x is None:
        raise ValueError("fft_ifftshift expects at least one tensor argument as input.")

    # Parse dims (can be in args[1], or kwargs 'dim'/'dims')
    if len(args) >= 2:
        dims = args[1]
    else:
        dims = kwargs.get("dim", kwargs.get("dims", None))

    # Normalize dims
    if dims is None:
        dims_list = list(range(x.ndim))
    else:
        if isinstance(dims, int):
            dims_list = [dims]
        else:
            dims_list = list(dims)
        # normalize negative dims
        dims_list = [(d + x.ndim) % x.ndim for d in dims_list]
        # remove duplicates while preserving order
        seen = set()
        tmp = []
        for d in dims_list:
            if d not in seen:
                tmp.append(d)
                seen.add(d)
        dims_list = tmp

    # Handle scalars or empty tensors quickly
    if x.ndim == 0 or x.numel() == 0:
        return x.clone()

    device = x.device
    out = torch.empty_like(x)

    # Prepare metadata
    sizes = torch.tensor(list(x.shape), device=device, dtype=torch.int64)
    in_strides = torch.tensor(list(x.stride()), device=device, dtype=torch.int64)
    out_strides = torch.tensor(list(out.stride()), device=device, dtype=torch.int64)

    # Per-dimension add amount = floor(size/2) if dimension is included, else 0
    add_list = [
        (sizes[d].item() // 2) if d in set(dims_list) else 0 for d in range(x.ndim)
    ]
    adds = torch.tensor(add_list, device=device, dtype=torch.int64)

    n_elements = x.numel()
    NDIMS = x.ndim
    ELEMENT_SIZE = x.element_size()

    # Use byte pointers by viewing as uint8 without changing storage
    x_u8 = x.view(torch.uint8)
    out_u8 = out.view(torch.uint8)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = 1024

    fft_ifftshift_kernel[grid](
        x_u8,
        out_u8,
        sizes,
        in_strides,
        out_strides,
        adds,
        n_elements,
        ELEMENT_SIZE=ELEMENT_SIZE,
        NDIMS=NDIMS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
