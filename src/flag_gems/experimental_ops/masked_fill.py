import torch
import triton
import triton.language as tl


@triton.jit
def masked_fill_kernel(
    x_ptr,
    mask_ptr,
    fill_ptr,
    out_ptr,
    n_elements,
    shape_ptr,
    x_stride_ptr,
    mask_stride_ptr,
    fill_stride_ptr,
    out_stride_ptr,
    NDIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    # Use int64 for address computations
    rem = offsets.to(tl.int64)
    out_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    x_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    mask_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    fill_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Compute per-dimension indices and offsets
    for d in range(NDIMS - 1, -1, -1):
        size_d = tl.load(shape_ptr + d)
        idx_d = rem % size_d
        rem = rem // size_d

        sx_d = tl.load(x_stride_ptr + d)
        sm_d = tl.load(mask_stride_ptr + d)
        sf_d = tl.load(fill_stride_ptr + d)
        so_d = tl.load(out_stride_ptr + d)

        x_off += idx_d * sx_d
        mask_off += idx_d * sm_d
        fill_off += idx_d * sf_d
        out_off += idx_d * so_d

    # Load values
    x_val = tl.load(x_ptr + x_off, mask=valid)
    m_val = tl.load(mask_ptr + mask_off, mask=valid)
    f_val = tl.load(fill_ptr + fill_off, mask=valid)

    # Convert mask to boolean
    m_bool = m_val != 0

    # Compute output
    out_val = tl.where(m_bool, f_val, x_val)

    # Store
    tl.store(out_ptr + out_off, out_val, mask=valid)


def _launch_masked_fill(
    x: torch.Tensor, mask: torch.Tensor, fill: torch.Tensor, out: torch.Tensor
):
    assert x.is_cuda and mask.is_cuda and fill.is_cuda and out.is_cuda
    assert x.device == mask.device == fill.device == out.device

    # Output shape is the shape of x
    shape = tuple(x.shape)
    NDIMS = len(shape)

    # Ensure dtype matches
    fill = fill.to(dtype=x.dtype)

    # Make sure mask is boolean then cast to int8 for robust kernel comparison
    mask = mask.to(torch.bool).to(torch.int8)

    # Expand mask and fill to match x's shape for proper broadcasting strides
    mask_exp = mask.expand(shape)
    fill_exp = fill.expand(shape)

    # Prepare stride and shape tensors (int64) on device
    shape_t = torch.tensor(shape, dtype=torch.int64, device=x.device)
    x_stride_t = torch.tensor(x.stride(), dtype=torch.int64, device=x.device)
    mask_stride_t = torch.tensor(mask_exp.stride(), dtype=torch.int64, device=x.device)
    fill_stride_t = torch.tensor(fill_exp.stride(), dtype=torch.int64, device=x.device)
    out_stride_t = torch.tensor(out.stride(), dtype=torch.int64, device=x.device)

    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    masked_fill_kernel[grid](
        x,
        mask_exp,
        fill_exp,
        out,
        n_elements,
        shape_t,
        x_stride_t,
        mask_stride_t,
        fill_stride_t,
        out_stride_t,
        NDIMS=NDIMS,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def masked_fill_Scalar(self: torch.Tensor, mask: torch.Tensor, value):
    # Create a 1-element tensor of the scalar value on device with matching dtype
    fill = torch.tensor(value, dtype=self.dtype, device=self.device)
    out = torch.empty_like(self)
    _launch_masked_fill(self, mask, fill, out)
    return out


def masked_fill_Tensor(self: torch.Tensor, mask: torch.Tensor, value: torch.Tensor):
    fill = value.to(device=self.device, dtype=self.dtype)
    out = torch.empty_like(self)
    _launch_masked_fill(self, mask, fill, out)
    return out


def masked_fill_Scalar_out(
    self: torch.Tensor, mask: torch.Tensor, value, out: torch.Tensor
):
    assert out.is_cuda and out.device == self.device
    assert out.shape == self.shape
    # Create a 1-element tensor of the scalar value on device with matching dtype
    fill = torch.tensor(value, dtype=self.dtype, device=self.device)
    _launch_masked_fill(self, mask, fill, out)
    return out


def masked_fill_Tensor_out(
    self: torch.Tensor, mask: torch.Tensor, value: torch.Tensor, out: torch.Tensor
):
    assert out.is_cuda and out.device == self.device
    assert out.shape == self.shape
    fill = value.to(device=self.device, dtype=self.dtype)
    _launch_masked_fill(self, mask, fill, out)
    return out
