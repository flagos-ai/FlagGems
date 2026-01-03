import torch
import triton
import triton.language as tl


@triton.jit
def unsqueeze_kernel(
    src_ptr,            # *Pointer* to input tensor data.
    dst_ptr,            # *Pointer* to output tensor data.
    out_numel,          # Total number of elements in output (same as input).
    in_strides_ptr,     # *Pointer* to input strides (length = RANK-1).
    out_shape_ptr,      # *Pointer* to output shape (length = RANK).
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed by each program.
    RANK: tl.constexpr,        # Rank of the output tensor.
    UNSQ_DIM: tl.constexpr,    # The dimension at which to unsqueeze (compile-time constant).
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = offsets.to(tl.int64)
    mask = offsets < out_numel

    tmp = offsets
    src_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    # Decompose linear index into multi-dimensional coordinates and map to input offset
    for k in range(RANK - 1, -1, -1):
        s_k = tl.load(out_shape_ptr + k)
        c_k = tmp % s_k
        tmp = tmp // s_k
        if k != UNSQ_DIM:
            in_k = k if k < UNSQ_DIM else k - 1
            stride_in_k = tl.load(in_strides_ptr + in_k)
            src_offset += c_k * stride_in_k

    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def unsqueeze(*args, **kwargs):
    # Expect signature: unsqueeze(x, dim)
    if len(args) >= 2:
        x, dim = args[0], args[1]
    else:
        x = kwargs.get('self', kwargs.get('input', None))
        dim = kwargs.get('dim', None)
    assert isinstance(x, torch.Tensor), "unsqueeze expects a torch.Tensor as the first argument."
    assert isinstance(dim, int), "unsqueeze expects an integer 'dim' argument."
    assert x.is_cuda, "Input tensor must be on CUDA device."

    ndims = x.dim()
    if dim < 0:
        dim += ndims + 1
    assert 0 <= dim <= ndims, f"dim must be in range [0, {ndims}] after normalization."

    out_shape = list(x.shape)
    out_shape.insert(dim, 1)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    numel = out.numel()
    # Prepare strides and shape tensors on device
    in_strides = torch.tensor(x.stride(), dtype=torch.int64, device=x.device)
    out_shape_t = torch.tensor(out_shape, dtype=torch.int64, device=x.device)

    RANK = len(out_shape)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    unsqueeze_kernel[grid](
        x, out, numel,
        in_strides, out_shape_t,
        BLOCK_SIZE=BLOCK_SIZE, RANK=RANK, UNSQ_DIM=dim
    )
    return out