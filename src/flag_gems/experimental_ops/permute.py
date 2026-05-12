import torch
import triton
import triton.language as tl


@triton.jit
def permute_kernel(
    x_ptr,  # *Pointer* to input tensor
    y_ptr,  # *Pointer* to output tensor
    n_elements,  # total number of elements
    ndim,  # number of dimensions
    in_strides_perm_ptr,  # int64[ndim]: input strides permuted by dims
    out_shape_ptr,  # int64[ndim]: output shape
    out_postfix_ptr,  # int64[ndim]: product of sizes after each axis in output
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    offs64 = offs.to(tl.int64)
    in_index = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    for k in range(MAX_DIMS):
        cond = k < ndim
        step_k = tl.load(out_postfix_ptr + k, mask=cond, other=1).to(tl.int64)
        size_k = tl.load(out_shape_ptr + k, mask=cond, other=1).to(tl.int64)
        stride_k = tl.load(in_strides_perm_ptr + k, mask=cond, other=0).to(tl.int64)
        coord_k = (offs64 // step_k) % size_k
        in_index += coord_k * stride_k

    vals = tl.load(x_ptr + in_index, mask=mask)
    tl.store(y_ptr + offs64, vals, mask=mask)


def permute(*args, **kwargs):
    # Parse arguments to support common PyTorch calling patterns
    if len(args) == 0:
        raise TypeError("permute() missing required argument: 'input'")

    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("First argument to permute must be a torch.Tensor")

    # Determine dims from args/kwargs
    dims = kwargs.get("dims", None)
    if dims is None:
        # If two args and second is sequence, treat as dims
        if len(args) == 2 and isinstance(args[1], (list, tuple)):
            dims = args[1]
        else:
            # Treat remaining positional args as dims varargs
            dims = args[1:]
    dims = tuple(int(d) for d in dims)

    ndim = x.dim()
    if len(dims) != ndim:
        raise ValueError(
            f"permute(): dims length {len(dims)} does not match tensor ndim {ndim}"
        )

    # Normalize negative dims and validate permutation
    dims = tuple([d % ndim for d in dims])
    if len(set(dims)) != ndim:
        raise ValueError(
            "permute(): dims must be a permutation of [0..ndim-1] with no repeats"
        )

    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device")

    device = x.device
    dtype = x.dtype

    in_shape = tuple(x.shape)
    out_shape = tuple(in_shape[d] for d in dims)

    # Prepare strides in elements
    in_strides = tuple(x.stride())
    in_strides_perm = tuple(in_strides[d] for d in dims)

    # Compute postfix products for output shape: prod(out_shape[k+1:])
    out_postfix = []
    p = 1
    for size in reversed(out_shape):
        out_postfix.append(p)
        p *= int(size)
    out_postfix = list(reversed(out_postfix))

    # Create output tensor (contiguous layout)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Move metadata to device (int64)
    in_strides_perm_t = torch.tensor(in_strides_perm, dtype=torch.int64, device=device)
    out_shape_t = torch.tensor(out_shape, dtype=torch.int64, device=device)
    out_postfix_t = torch.tensor(out_postfix, dtype=torch.int64, device=device)

    BLOCK_SIZE = 1024
    MAX_DIMS = max(1, min(16, ndim))  # noqa: F841 cap unrolling to 16

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    permute_kernel[grid](
        x,
        out,
        n_elements,
        ndim,
        in_strides_perm_t,
        out_shape_t,
        out_postfix_t,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_DIMS=16,
    )
    return out
