import torch
import triton
import triton.language as tl


@triton.jit
def _slice_backward_scatter_kernel(
    grad_ptr,  # *Pointer* to grad (input) vector
    out_ptr,  # *Pointer* to output (full grad) vector
    n_elements,  # numel of grad
    inner,  # product of sizes after 'dim'
    gdim,  # size of grad along 'dim'
    odim,  # size of output along 'dim'
    start,  # normalized start index along 'dim'
    step,  # step along 'dim'
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    offs = offs.to(tl.int64)
    mask = offs < n_elements

    inner_i64 = tl.full([1], inner, tl.int64)
    gdim_i64 = tl.full([1], gdim, tl.int64)
    odim_i64 = tl.full([1], odim, tl.int64)
    start_i64 = tl.full([1], start, tl.int64)
    step_i64 = tl.full([1], step, tl.int64)

    # Decompose linear index into (outer, g_idx_dim, inner_idx)
    outer = offs // (gdim_i64 * inner_i64)
    inner_idx = offs % inner_i64
    g_idx_dim = (offs // inner_i64) % gdim_i64

    out_dim_index = start_i64 + g_idx_dim * step_i64
    valid_o = (out_dim_index >= 0) & (out_dim_index < odim_i64)
    o = outer * (odim_i64 * inner_i64) + out_dim_index * inner_i64 + inner_idx

    m = mask & valid_o
    val = tl.load(grad_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + o, val, mask=m)


def _normalize_slice_params(input_sizes, dim, start, end, step):
    D = len(input_sizes)
    if dim < 0:
        dim += D
    size_dim = int(input_sizes[dim])

    if step is None:
        step = 1
    if step == 0:
        raise ValueError("slice step cannot be zero")

    if start is None:
        start = 0 if step > 0 else size_dim - 1

    # Normalize start into valid index range
    if start < 0:
        start += size_dim

    if step > 0:
        # Clamp into [0, size_dim]
        if start < 0:
            start = 0
        if start > size_dim:
            start = size_dim
    else:
        # Clamp into [0, size_dim-1]
        if start < 0:
            start = 0
        if start >= size_dim:
            start = size_dim - 1

    return dim, int(start), int(step)


def _launch_slice_backward_kernel(grad, input_sizes, dim, start, end, step, out):
    # Ensure contiguous
    grad_c = grad.contiguous()
    out_c = out.contiguous()

    # Normalize parameters
    dim, start_n, step_n = _normalize_slice_params(
        list(input_sizes), int(dim), start, end, step
    )

    # Compute inner, gdim, odim
    sizes = list(input_sizes)
    odim = int(sizes[dim])
    gdim = int(grad_c.shape[dim])
    inner = 1
    for s in sizes[dim + 1 :]:
        inner *= int(s)

    n_elements = grad_c.numel()
    if n_elements == 0:
        return out  # nothing to do

    # Zero the output tensor
    out_c.zero_()

    # Launch Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _slice_backward_scatter_kernel[grid](
        grad_c,
        out_c,
        n_elements,
        inner,
        gdim,
        odim,
        start_n,
        step_n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_c


def slice_backward(grad, input_sizes, dim, start, end, step):
    """
    Python wrapper for aten::slice_backward
    """
    out = torch.empty(tuple(input_sizes), device=grad.device, dtype=grad.dtype)
    out = _launch_slice_backward_kernel(grad, input_sizes, dim, start, end, step, out)
    return out


def slice_backward_out(grad, input_sizes, dim, start, end, step, out):
    """
    Python wrapper for aten::slice_backward.out
    """
    if tuple(out.shape) != tuple(input_sizes):
        raise ValueError("Output tensor shape must match input_sizes")
    if out.device != grad.device or out.dtype != grad.dtype:
        raise ValueError("Output tensor must have same device and dtype as grad")
    _launch_slice_backward_kernel(grad, input_sizes, dim, start, end, step, out)
    return out
