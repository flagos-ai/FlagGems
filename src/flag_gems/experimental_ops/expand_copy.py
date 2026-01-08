import torch
import triton
import triton.language as tl


@triton.jit
def expand_copy_kernel(
    in_ptr,  # pointer to input tensor data
    out_ptr,  # pointer to output tensor data (contiguous)
    in_strides_ptr,  # pointer to int64 array of aligned input strides (elements)
    out_shape_ptr,  # pointer to int64 array of output shape
    out_strides_c_ptr,  # pointer to int64 array of contiguous output strides (elements)
    n_elements,  # total number of elements in output
    BLOCK_SIZE: tl.constexpr,  # elements per program
    MAX_DIMS: tl.constexpr,  # maximum number of dimensions supported
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    offsets_i64 = offsets.to(tl.int64)
    # compute source linear offsets respecting broadcasting
    src_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # For each dimension, compute the coordinate and contribute to src offset
    # Using contiguous strides for output to decode index
    for d in tl.static_range(MAX_DIMS):
        out_stride_c = tl.load(out_strides_c_ptr + d)  # int64 scalar
        dim_size = tl.load(out_shape_ptr + d)  # int64 scalar
        in_stride = tl.load(
            in_strides_ptr + d
        )  # int64 scalar (0 if broadcast on this dim)

        # idx at this dimension: (offset // out_stride_c) % dim_size
        q = offsets_i64 // out_stride_c
        idx_d = q % dim_size
        contrib = idx_d * in_stride
        src_offsets = src_offsets + contrib

    # Load from input and store to output
    vals = tl.load(in_ptr + src_offsets, mask=mask)
    tl.store(out_ptr + offsets_i64, vals, mask=mask)


# Helper to resolve expand sizes and aligned strides
def _resolve_expand_shape_and_strides(
    inp: torch.Tensor, size, implicit=False, max_dims=8
):
    # Normalize size to list of ints
    if isinstance(size, torch.Size):
        size_list = list(size)
    elif isinstance(size, (list, tuple)):
        size_list = list(size)
    else:
        raise TypeError("size must be a list/tuple/torch.Size of integers")
    out_dims = len(size_list)
    in_shape = list(inp.shape)
    in_dims = len(in_shape)

    # Resolve -1 and verify broadcastability
    shift = out_dims - in_dims
    out_shape = [1] * out_dims
    for k in range(out_dims):
        sidx = k - shift
        src_dim = in_shape[sidx] if sidx >= 0 else 1
        desired = size_list[k]
        if desired == -1:
            out_size = src_dim
        else:
            out_size = int(desired)
            if out_size < 0:
                raise ValueError("expand_copy: negative size not allowed (except -1)")
            if src_dim != 1 and sidx >= 0 and out_size != src_dim:
                raise ValueError(
                    f"expand_copy: the expanded size of the tensor ({out_size}) must match "
                    f"the existing size ({src_dim}) at non-singleton dimension {sidx}."
                )
        if out_size <= 0:
            raise ValueError("expand_copy: expanded size must be positive")
        out_shape[k] = out_size

    if len(out_shape) > max_dims:
        raise ValueError(
            f"expand_copy: rank {len(out_shape)} exceeds MAX_DIMS={max_dims}"
        )

    # Compute contiguous output strides (in elements)
    out_strides_c = [1] * out_dims
    for i in range(out_dims - 2, -1, -1):
        out_strides_c[i] = out_strides_c[i + 1] * out_shape[i + 1]
    # Pad to max_dims
    out_strides_c += [1] * (max_dims - out_dims)
    out_shape_padded = out_shape + [1] * (max_dims - out_dims)

    # Align input strides to output dims and handle broadcasting by setting stride=0 where broadcast occurs
    in_strides = list(inp.stride())
    aligned_in_strides = [0] * out_dims
    for k in range(out_dims):
        sidx = k - shift
        if sidx < 0:
            aligned_in_strides[k] = 0
        else:
            src_dim = in_shape[sidx]
            if src_dim == 1 and out_shape[k] != 1:
                aligned_in_strides[k] = 0
            else:
                # src_dim == out_shape[k] ensured by earlier check
                aligned_in_strides[k] = in_strides[sidx]
    aligned_in_strides += [0] * (max_dims - out_dims)

    return out_shape, out_shape_padded, out_strides_c, aligned_in_strides


def _launch_expand_copy_kernel(
    inp: torch.Tensor, out: torch.Tensor, max_dims=8, block_size=1024
):
    device = inp.device
    assert (
        out.is_contiguous()
    ), "expand_copy: output must be contiguous when launching kernel"
    # Compute aligned strides arrays
    (
        out_shape,
        out_shape_padded,
        out_strides_c,
        aligned_in_strides,
    ) = _resolve_expand_shape_and_strides(inp, out.shape, max_dims=max_dims)

    n_elements = 1
    for s in out_shape:
        n_elements *= int(s)
    if n_elements == 0:
        return

    # Prepare metadata tensors on device
    out_shape_t = torch.tensor(out_shape_padded, dtype=torch.long, device=device)
    out_strides_c_t = torch.tensor(out_strides_c, dtype=torch.long, device=device)
    in_strides_t = torch.tensor(aligned_in_strides, dtype=torch.long, device=device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    expand_copy_kernel[grid](
        inp,
        out,
        in_strides_t,
        out_shape_t,
        out_strides_c_t,
        n_elements,
        BLOCK_SIZE=block_size,
        MAX_DIMS=max_dims,
    )


def expand_copy(self: torch.Tensor, size, implicit: bool = False):
    """
    Wrapper for aten::expand_copy(Tensor self, SymInt[] size, bool implicit) -> Tensor
    """
    if not isinstance(self, torch.Tensor):
        raise TypeError("expand_copy: 'self' must be a torch.Tensor")
    device = self.device
    dtype = self.dtype

    max_dims = 8
    out_shape, _, _, _ = _resolve_expand_shape_and_strides(
        self, size, implicit=implicit, max_dims=max_dims
    )
    # Allocate contiguous output
    out = torch.empty(out_shape, device=device, dtype=dtype)
    if out.numel() == 0:
        return out
    _launch_expand_copy_kernel(self, out, max_dims=max_dims, block_size=1024)
    return out


def expand_copy_out(
    self: torch.Tensor, size, implicit: bool = False, out: torch.Tensor = None
):
    """
    Wrapper for aten::expand_copy.out(Tensor self, SymInt[] size, bool implicit, *, Tensor(a!) out) -> Tensor(a!)
    """
    if not isinstance(self, torch.Tensor):
        raise TypeError("expand_copy_out: 'self' must be a torch.Tensor")
    max_dims = 8
    out_shape, _, _, _ = _resolve_expand_shape_and_strides(
        self, size, implicit=implicit, max_dims=max_dims
    )

    if out is None:
        out = torch.empty(out_shape, device=self.device, dtype=self.dtype)
    else:
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(
                f"expand_copy_out: provided 'out' has shape {tuple(out.shape)} but expected {tuple(out_shape)}"
            )
        if out.device != self.device:
            raise ValueError(
                "expand_copy_out: 'out' must be on the same device as 'self'"
            )
        if out.dtype != self.dtype:
            raise ValueError(
                "expand_copy_out: 'out' must have the same dtype as 'self'"
            )

    if out.numel() == 0:
        return out

    if out.is_contiguous():
        _launch_expand_copy_kernel(self, out, max_dims=max_dims, block_size=1024)
    else:
        # Compute into a contiguous temp and copy into non-contiguous 'out'
        temp = torch.empty(out_shape, device=self.device, dtype=self.dtype)
        _launch_expand_copy_kernel(self, temp, max_dims=max_dims, block_size=1024)
        out.copy_(temp)
    return out
