import torch
import triton
import triton.language as tl


@triton.jit
def as_strided_copy_kernel(
    x_ptr,
    out_ptr,
    size_ptr,
    in_stride_ptr,
    out_stride_ptr,
    storage_offset,
    n_elements,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Work in int64 for offsets to avoid overflow
    tmp = offsets.to(tl.int64)
    src_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    dst_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Compute multi-dimensional indices from linear index (row-major order)
    # and accumulate source and destination offsets using respective strides.
    # Iterate from last dimension to first.
    for d in range(NDIM - 1, -1, -1):
        size_d = tl.load(size_ptr + d).to(tl.int64)
        coord_d = tl.where(size_d != 0, tmp % size_d, tl.zeros_like(tmp))
        tmp = tl.where(size_d != 0, tmp // size_d, tmp)

        in_stride_d = tl.load(in_stride_ptr + d).to(tl.int64)
        out_stride_d = tl.load(out_stride_ptr + d).to(tl.int64)

        src_offset += coord_d * in_stride_d
        dst_offset += coord_d * out_stride_d

    # Apply storage offset for the source
    src_offset += tl.full([BLOCK_SIZE], storage_offset, dtype=tl.int64)

    # Load and store
    val = tl.load(x_ptr + src_offset, mask=mask)
    tl.store(out_ptr + dst_offset, val, mask=mask)


def _prod(iterable):
    p = 1
    for v in iterable:
        p *= int(v)
    return p


def _to_int64_device_tensor(seq, device):
    return torch.as_tensor(list(seq), dtype=torch.int64, device=device)


def _run_as_strided_copy_kernel(x, size, in_stride, storage_offset, out):
    device = x.device
    assert out.device == device, "Input and output must be on the same device."
    assert out.dtype == x.dtype, "Output dtype must match input dtype."

    ndim = len(size)
    n_elements = _prod(size)

    if n_elements == 0:
        return out

    # Prepare metadata tensors on device
    size_t = _to_int64_device_tensor(size, device)
    in_stride_t = _to_int64_device_tensor(in_stride, device)
    out_stride_t = _to_int64_device_tensor(out.stride(), device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    as_strided_copy_kernel[grid](
        x,  # source pointer (tensor)
        out,  # destination pointer (tensor with desired strides)
        size_t,  # sizes
        in_stride_t,  # source view strides (relative to x's storage)
        out_stride_t,  # actual output strides
        int(storage_offset),  # storage offset into x's storage
        n_elements,  # number of logical elements to copy
        NDIM=ndim,
        BLOCK_SIZE=1024,
    )
    return out


def as_strided_copy(*args, **kwargs):
    # Expected signature: (input, size, stride, storage_offset=0)
    # Allow keyword variants as well.
    if len(args) >= 3:
        x = args[0]
        size = args[1]
        in_stride = args[2]
        storage_offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
    else:
        x = kwargs.get("input", kwargs.get("x"))
        size = kwargs.get("size")
        in_stride = kwargs.get("stride")
        storage_offset = kwargs.get("storage_offset", 0)

    assert (
        x is not None and size is not None and in_stride is not None
    ), "Missing required arguments: input, size, stride"
    device = x.device
    out = torch.empty_strided(size, in_stride, dtype=x.dtype, device=device)
    return _run_as_strided_copy_kernel(x, size, in_stride, storage_offset, out)


def as_strided_copy_out(*args, **kwargs):
    # Expected signature for out variant: (input, size, stride, storage_offset=0, out=...)
    # Accept positional and keyword forms.
    x = None
    size = None
    in_stride = None
    storage_offset = kwargs.get("storage_offset", 0)
    out = kwargs.get("out", None)

    # Positional handling
    if len(args) >= 3:
        x = args[0]
        size = args[1]
        in_stride = args[2]
        if len(args) >= 4:
            storage_offset = args[3]
        if len(args) >= 5:
            out = args[4]

    # Keyword fallbacks
    if x is None:
        x = kwargs.get("input", kwargs.get("x"))
    if size is None:
        size = kwargs.get("size")
    if in_stride is None:
        in_stride = kwargs.get("stride")
    if out is None:
        out = kwargs.get("out")

    assert (
        x is not None and size is not None and in_stride is not None and out is not None
    ), "Missing required arguments for out variant."
    # Validate output shape and strides
    assert tuple(out.size()) == tuple(size), "Output size must match requested size."
    # Note: PyTorch out variant expects provided out tensor; we use its actual strides for storage.
    return _run_as_strided_copy_kernel(x, size, in_stride, storage_offset, out)
