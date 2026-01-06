import torch
import triton
import triton.language as tl


@triton.jit
def expand(
    x_ptr,
    out_ptr,
    n_elements,
    ndims,
    out_shape_ptr,
    out_cumprod_ptr,
    in_stride_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute input offsets corresponding to each output linear index
    in_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Accumulate contributions per dimension
    for d in range(MAX_DIMS):
        # Load scalars defining the output decomposition and input strides
        s = tl.load(out_shape_ptr + d)
        stride_right = tl.load(out_cumprod_ptr + d)
        in_stride = tl.load(in_stride_ptr + d)
        # idx along dimension d for each linear offset
        idx_d = (offsets // stride_right) % s
        # contribution to input linear offset
        in_offsets += idx_d * in_stride

    # Load from input using computed offsets and store to output
    x = tl.load(x_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


_expand_kernel = expand


def expand(*args, **kwargs):
    x = args[0]
    size = args[1]
    implicit = kwargs.get(  # noqa: F841
        "implicit", False
    )  # not used but accepted for signature compatibility

    if not isinstance(size, (list, tuple, torch.Size)):
        raise TypeError("expand size must be a list/tuple/torch.Size of ints")

    size = list(size)
    in_shape = list(x.shape)
    in_strides = list(x.stride())

    out_ndim = len(size)
    in_ndim = len(in_shape)

    if in_ndim > out_ndim:
        raise RuntimeError(
            f"expand: requested size has fewer dimensions ({out_ndim}) than input ({in_ndim})"
        )

    # Pad input shape/strides on the left to match output ndim
    if in_ndim < out_ndim:
        pad = out_ndim - in_ndim
        in_shape = [1] * pad + in_shape
        # For padded (new) leading dims, stride effectively is 0 since they will be broadcast
        in_strides = [0] * pad + in_strides

    # Resolve -1 and validate broadcastability
    out_shape = []
    for d in range(out_ndim):
        req = size[d]
        src = in_shape[d]
        if req == -1:
            target = src
        else:
            target = req
        if src != target and src != 1:
            raise RuntimeError(
                f"The expanded size of the tensor ({target}) must match the existing size ({src}) at non-singleton "
                f"dimension {d}. Target sizes must be the same, or -1, or the size of dimension in the original tensor must be 1."  # noqa: E501
            )
        out_shape.append(int(target))

    # Effective input strides: 0 for broadcasted dims, original stride otherwise
    in_stride_eff = [
        int(in_strides[d]) if in_shape[d] != 1 else 0 for d in range(out_ndim)
    ]

    # Prepare decomposition multipliers: product of sizes to the right for each dim
    out_cumprod_right = [0] * out_ndim
    prod = 1
    for d in range(out_ndim - 1, -1, -1):
        out_cumprod_right[d] = prod
        prod *= out_shape[d]

    # Allocate output
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Triton kernel parameters
    BLOCK_SIZE = 1024
    MAX_DIMS = max(out_ndim, 1)  # at least 1
    # Round up MAX_DIMS to a reasonable static upper bound for compilation (e.g., 16)
    # but ensure arrays we pass match MAX_DIMS in kernel
    STATIC_MAX = 16
    if MAX_DIMS > STATIC_MAX:
        STATIC_MAX = MAX_DIMS

    # Create device arrays for shapes/strides with padding for MAX_DIMS
    pad_len = STATIC_MAX - out_ndim
    out_shape_arr = torch.tensor(
        out_shape + [1] * pad_len, dtype=torch.int64, device=x.device
    )
    out_cumprod_arr = torch.tensor(
        out_cumprod_right + [1] * pad_len, dtype=torch.int64, device=x.device
    )
    in_stride_arr = torch.tensor(
        in_stride_eff + [0] * pad_len, dtype=torch.int64, device=x.device
    )

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _expand_kernel[grid](
        x,
        out,
        n_elements,
        out_ndim,
        out_shape_arr,
        out_cumprod_arr,
        in_stride_arr,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_DIMS=STATIC_MAX,
    )
    return out
