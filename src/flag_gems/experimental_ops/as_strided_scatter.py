import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _as_strided_scatter_kernel(
    src_ptr,
    out_ptr,
    sizes_ptr,
    strides_ptr,
    n_src_elements,
    ndim,
    storage_offset,
    base_numel,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_src_elements

    # Load source values
    src_vals = tl.load(src_ptr + offsets, mask=mask)

    # Compute linear target offsets in the base (out) tensor storage
    tmp = offsets.to(tl.int64)
    lin = tmp * 0 + tl.full(offsets.shape, storage_offset, tl.int64)

    # Convert linear indices to multi-dimensional indices and map with provided strides
    for k in tl.static_range(NDIM):
        d = NDIM - 1 - k
        size_d = tl.load(sizes_ptr + d)
        stride_d = tl.load(strides_ptr + d)
        idx_d = tmp % size_d
        tmp = tmp // size_d
        lin += idx_d * stride_d

    # Bounds mask to avoid invalid writes
    mlin = (lin >= 0) & (lin < base_numel)
    final_mask = mask & mlin

    tl.store(out_ptr + lin, src_vals, mask=final_mask)


def as_strided_scatter(
    self: torch.Tensor, src: torch.Tensor, size, stride, storage_offset: int = 0
):
    assert self.is_cuda and src.is_cuda, "Tensors must be CUDA for Triton kernels."
    assert self.dtype == src.dtype, "self and src must have the same dtype."
    device = self.device

    # Validate size/stride and src size
    ndim = len(size)
    n_src = 1
    for s in size:
        n_src *= int(s)
    assert src.numel() == n_src, "src.numel() must equal product of 'size'."

    # Prepare output
    out = torch.empty_like(self)

    # Copy self into out
    n_out = out.numel()
    grid_copy = lambda meta: (triton.cdiv(n_out, meta["BLOCK_SIZE"]),)
    _copy_kernel[grid_copy](self, out, n_out, BLOCK_SIZE=1024)

    # Prepare size/stride tensors
    sizes_t = torch.tensor(list(size), dtype=torch.int64, device=device)
    strides_t = torch.tensor(list(stride), dtype=torch.int64, device=device)

    # Scatter src into out according to as_strided parameters
    grid_scatter = lambda meta: (triton.cdiv(n_src, meta["BLOCK_SIZE"]),)
    _as_strided_scatter_kernel[grid_scatter](
        src,
        out,
        sizes_t,
        strides_t,
        n_src,
        ndim,
        int(storage_offset),
        out.numel(),
        BLOCK_SIZE=1024,
        NDIM=ndim,
    )

    return out


def as_strided_scatter_out(
    self: torch.Tensor,
    src: torch.Tensor,
    size,
    stride,
    storage_offset: int = 0,
    out: torch.Tensor = None,
):
    assert self.is_cuda and src.is_cuda, "Tensors must be CUDA for Triton kernels."
    assert self.dtype == src.dtype, "self and src must have the same dtype."
    device = self.device

    # Validate size/stride and src size
    ndim = len(size)
    n_src = 1
    for s in size:
        n_src *= int(s)
    assert src.numel() == n_src, "src.numel() must equal product of 'size'."

    # Prepare output
    if out is None:
        out = torch.empty_like(self)
    else:
        assert out.is_cuda, "out must be a CUDA tensor."
        assert (
            out.shape == self.shape and out.dtype == self.dtype
        ), "out must match self in shape and dtype."

    # Copy self into out
    n_out = out.numel()
    grid_copy = lambda meta: (triton.cdiv(n_out, meta["BLOCK_SIZE"]),)
    _copy_kernel[grid_copy](self, out, n_out, BLOCK_SIZE=1024)

    # Prepare size/stride tensors
    sizes_t = torch.tensor(list(size), dtype=torch.int64, device=device)
    strides_t = torch.tensor(list(stride), dtype=torch.int64, device=device)

    # Scatter src into out according to as_strided parameters
    grid_scatter = lambda meta: (triton.cdiv(n_src, meta["BLOCK_SIZE"]),)
    _as_strided_scatter_kernel[grid_scatter](
        src,
        out,
        sizes_t,
        strides_t,
        n_src,
        ndim,
        int(storage_offset),
        out.numel(),
        BLOCK_SIZE=1024,
        NDIM=ndim,
    )

    return out
