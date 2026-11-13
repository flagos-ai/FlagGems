import torch
import triton
import triton.language as tl


@triton.jit
def _vecdot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_batch,
    vdim,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduction kernel: each program handles all batches via grid-strided loop."""
    pid = tl.program_id(0)
    n_pids = tl.num_programs(0)

    for b in range(pid, n_batch, n_pids):
        base = b * vdim
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, vdim, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < vdim
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            acc += x * y
        tl.store(out_ptr + b, tl.sum(acc, axis=0))


def linalg_vecdot(x, y, dim=-1):
    # Fast path: dim=-1, same shape, already contiguous, no half-precision
    # The benchmark (and most real use cases) hit this path
    if (
        dim == -1
        and x.is_contiguous()
        and y.is_contiguous()
        and x.dtype not in (torch.float16, torch.bfloat16)
    ):
        if x.shape != y.shape:
            raise ValueError("Input shapes must match")
        vdim = x.shape[-1]
        n_batch = x.numel() // vdim
        bs = 32
        while bs < vdim and bs < 1024:
            bs <<= 1
        out_shape = x.shape[:-1]
        out = torch.empty(
            out_shape if out_shape else (), dtype=x.dtype, device=x.device
        )
        _vecdot_kernel[(min(n_batch, 8),)](x, y, out, n_batch, vdim, bs)
        return out

    # General path: non-standard dims, non-contiguous, or half-precision inputs
    if x.shape != y.shape:
        raise ValueError("Input shapes must match")

    ndim = x.dim()

    if dim < 0:
        dim += ndim

    if dim == ndim - 1:
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
    else:
        x = x.movedim(dim, -1).contiguous()
        y = y.movedim(dim, -1).contiguous()

    vdim = x.shape[-1]
    n_batch = x.numel() // vdim

    bs = 32
    while bs < vdim and bs < 1024:
        bs <<= 1

    out_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    out_shape = x.shape[:-1]
    out = torch.empty(out_shape if out_shape else (), dtype=out_dtype, device=x.device)

    _vecdot_kernel[(min(n_batch, 8),)](x, y, out, n_batch, vdim, bs)

    if out_dtype != x.dtype:
        return out.to(x.dtype)
    return out


def linalg_vecdot_out(x, y, dim=-1, out=None):
    if out is None:
        return linalg_vecdot(x, y, dim=dim)
    res = linalg_vecdot(x, y, dim=dim)
    out.copy_(res)
    return out
