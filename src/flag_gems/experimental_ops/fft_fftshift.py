import torch
import triton
import triton.language as tl


@triton.jit
def fft_fftshift(
    in_ptr,
    out_ptr,
    shape_ptr,
    in_stride_ptr,
    out_stride_ptr,
    shift_ptr,
    n_dims,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Work with 64-bit indices to avoid overflow on large tensors/strides
    q64 = offs.to(tl.int64)
    dst_off = tl.zeros_like(q64)
    src_off = tl.zeros_like(q64)

    # Unrolled loop up to MAX_DIMS, with masking when t >= n_dims
    for t in range(MAX_DIMS):
        active = t < n_dims
        dd = tl.where(active, n_dims - 1 - t, 0)

        size_d = tl.where(active, tl.load(shape_ptr + dd), 1).to(tl.int64)
        i_d = q64 % size_d
        q64 = q64 // size_d

        shift_d = tl.where(active, tl.load(shift_ptr + dd), 0).to(tl.int64)
        in_stride_d = tl.where(active, tl.load(in_stride_ptr + dd), 0).to(tl.int64)
        out_stride_d = tl.where(active, tl.load(out_stride_ptr + dd), 0).to(tl.int64)

        src_i_d = i_d - shift_d
        src_i_d = src_i_d + size_d
        src_i_d = src_i_d % size_d

        src_off += src_i_d * in_stride_d
        dst_off += i_d * out_stride_d

    val = tl.load(in_ptr + src_off, mask=mask)
    tl.store(out_ptr + dst_off, val, mask=mask)


_fft_fftshift_kernel = fft_fftshift


def fft_fftshift(*args, **kwargs):
    # Resolve input tensor and dim(s) similar to torch.fft.fftshift
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", kwargs.get("x", None)))
    dim = kwargs.get("dim", None)
    if dim is None and len(args) > 1:
        dim = args[1]

    if not isinstance(x, torch.Tensor):
        raise TypeError("fft_fftshift expects a torch.Tensor as the first argument")

    n_dims = x.ndim
    if dim is None:
        dims = list(range(n_dims))
    else:
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        # Normalize negative dims
        dims = [(d + n_dims) % n_dims for d in dims]

    # Prepare output tensor
    out = torch.empty_like(x)

    # Prepare shape and strides
    device = x.device
    shape_t = torch.tensor(x.shape, dtype=torch.int64, device=device)
    in_stride_t = torch.tensor(x.stride(), dtype=torch.int64, device=device)
    out_stride_t = torch.tensor(out.stride(), dtype=torch.int64, device=device)

    # Prepare per-dim shifts: size//2 for dims, else 0
    shifts = torch.zeros(n_dims, dtype=torch.int64, device=device)
    for d in dims:
        shifts[d] = x.shape[d] // 2

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fft_fftshift_kernel[grid](
        x,
        out,
        shape_t,
        in_stride_t,
        out_stride_t,
        shifts,
        n_dims,
        n_elements,
        BLOCK_SIZE=1024,
        MAX_DIMS=16,
    )
    return out
