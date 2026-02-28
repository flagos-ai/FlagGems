import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x, mask=mask)


@triton.jit
def _slice_scatter_kernel(
    src_ptr,  # pointer to src tensor (flattened)
    out_ptr,  # pointer to output tensor (flattened)
    outer,  # number of chunks before the sliced dimension
    dim_size,  # size of the sliced dimension in the output
    inner,  # number of elements after the sliced dimension
    start,  # start index along the sliced dimension
    step,  # step along the sliced dimension
    m_size,  # number of indices along the sliced dimension to scatter (len of slice)
    n_src_elements,  # total number of elements in src (outer * m_size * inner)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_src_elements

    # Promote to int64 for intermediate index math
    offs_i64 = offs.to(tl.int64)

    inner64 = tl.full([BLOCK_SIZE], inner, tl.int64)
    m_size64 = tl.full([BLOCK_SIZE], m_size, tl.int64)
    dim_size64 = tl.full([BLOCK_SIZE], dim_size, tl.int64)
    start64 = tl.full([BLOCK_SIZE], start, tl.int64)
    step64 = tl.full([BLOCK_SIZE], step, tl.int64)

    chunk64 = m_size64 * inner64
    o = offs_i64 // chunk64
    rem = offs_i64 - o * chunk64
    m = rem // inner64
    i = rem - m * inner64

    dest_d = start64 + m * step64  # index along sliced dimension
    dest_linear = o * (dim_size64 * inner64) + dest_d * inner64 + i

    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(out_ptr + dest_linear.to(tl.int32), val, mask=mask)


def _normalize_slice_params(size, start, end, step):
    assert step is not None and step != 0, "step must be non-zero"
    # This implementation supports only positive step for simplicity
    assert step > 0, "Only positive step is supported in this Triton implementation"
    if start is None:
        start = 0
    if end is None:
        end = size
    if start < 0:
        start += size
    if end < 0:
        end += size
    # Clamp to [0, size]
    start = max(0, min(start, size))
    end = max(0, min(end, size))
    if end <= start:
        m = 0
    else:
        m = (end - start + step - 1) // step
    return start, end, step, m


def _slice_scatter_impl(input, src, out, dim=0, start=None, end=None, step=1):
    assert (
        input.is_cuda and src.is_cuda and out.is_cuda
    ), "All tensors must be CUDA tensors"
    assert (
        input.is_contiguous() and src.is_contiguous() and out.is_contiguous()
    ), "Tensors must be contiguous"
    assert input.dtype == src.dtype == out.dtype, "All tensors must have the same dtype"
    assert input.shape == out.shape, "Output must have same shape as input"

    ndim = input.dim()
    if ndim == 0:
        # Scalar case: slice along dim doesn't apply, just copy input to out (and no scatter)
        if input.numel() > 0:
            n = input.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            _copy_kernel[grid](input, out, n, BLOCK_SIZE=1024)
        return out

    dim = dim if dim >= 0 else dim + ndim
    assert 0 <= dim < ndim, "dim out of range"

    size_d = input.size(dim)
    s, e, st, m = _normalize_slice_params(size_d, start, end, step)

    # Compute outer and inner sizes
    outer = 1
    for k in range(0, dim):
        outer *= input.size(k)
    inner = 1
    for k in range(dim + 1, ndim):
        inner *= input.size(k)

    # Validate src shape/numel
    expected_src_numel = outer * m * inner
    assert src.numel() == expected_src_numel, (
        f"src numel mismatch: got {src.numel()}, expected {expected_src_numel} "
        f"(outer={outer}, m={m}, inner={inner})"
    )

    # 1) Copy input -> out
    n = input.numel()
    if n > 0:
        grid_copy = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _copy_kernel[grid_copy](input, out, n, BLOCK_SIZE=1024)

    # 2) Scatter src into the sliced region
    if expected_src_numel > 0:
        grid_scatter = lambda meta: (
            triton.cdiv(expected_src_numel, meta["BLOCK_SIZE"]),
        )
        _slice_scatter_kernel[grid_scatter](
            src,
            out,
            outer,
            size_d,
            inner,
            s,
            st,
            m,
            expected_src_numel,
            BLOCK_SIZE=1024,
        )

    return out


def slice_scatter(input, src, dim=0, start=None, end=None, step=1):
    out = torch.empty_like(input)
    return _slice_scatter_impl(
        input, src, out, dim=dim, start=start, end=end, step=step
    )


def slice_scatter_out(input, src, dim=0, start=None, end=None, step=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    return _slice_scatter_impl(
        input, src, out, dim=dim, start=start, end=end, step=step
    )
