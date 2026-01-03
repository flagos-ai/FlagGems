import torch
import triton
import triton.language as tl
from functools import reduce
from operator import mul


@triton.jit
def amin_reduce_last_kernel(
    x_ptr,
    out_ptr,
    M,  # number of rows (outer size)
    K,  # reduction length (last-axis size)
    stride_xm,
    stride_xk,
    init,  # identity value for min (same dtype as x)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    mask_m = pid < M
    acc = init
    k = 0
    while k < K:
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = mask_m & (offs < K)
        vals = tl.load(x_ptr + pid * stride_xm + offs * stride_xk, mask=mask, other=init)
        block_min = tl.min(vals, axis=0)
        acc = tl.minimum(acc, block_min)
        k += BLOCK_SIZE
    tl.store(out_ptr + pid, acc, mask=mask_m)


def _prod(seq):
    return int(reduce(mul, seq, 1))


def _parse_dims(dim, ndim):
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, (list, tuple)):
        dims = [int(d) for d in dim]
    else:
        dims = [int(dim)]
    # normalize negatives and remove duplicates preserving order
    seen = set()
    norm = []
    for d in dims:
        dd = d if d >= 0 else d + ndim
        if dd < 0 or dd >= ndim:
            raise IndexError("Dimension out of range in amin")
        if dd not in seen:
            norm.append(dd)
            seen.add(dd)
    return norm


def _amin_impl(x: torch.Tensor, dim=None, keepdim: bool = False, out: torch.Tensor = None):
    if not x.is_cuda:
        raise RuntimeError("Triton amin kernel requires CUDA tensors")
    ndim = x.ndim
    reduce_dims = _parse_dims(dim, ndim)
    if len(reduce_dims) == 0:
        # No reduction dims specified, return input (or copy into out)
        if out is None:
            return x.clone()
        if out.numel() != x.numel():
            raise RuntimeError("out tensor has incorrect number of elements for amin with empty dims")
        out.copy_(x)
        return out

    # Determine output shape
    input_sizes = list(x.size())
    keep_sizes = input_sizes.copy()
    for d in reduce_dims:
        keep_sizes[d] = 1
    non_reduce_dims = [i for i in range(ndim) if i not in reduce_dims]
    non_reduce_sizes = [input_sizes[i] for i in non_reduce_dims]

    final_shape = keep_sizes if keepdim else non_reduce_sizes

    # Prepare permutation: move non-reduced dims first, reduced dims last
    perm = non_reduce_dims + reduce_dims
    x_perm = x.permute(perm)
    x_perm = x_perm.contiguous()

    # Flatten into [M, K]
    M = _prod(non_reduce_sizes) if len(non_reduce_sizes) > 0 else 1
    K = _prod([input_sizes[i] for i in reduce_dims]) if len(reduce_dims) > 0 else 1

    if K == 0:
        raise RuntimeError("amin reduction has an empty dimension (no identity for min)")

    x_2d = x_perm.view(M, K)

    # Identity/initial value for min based on dtype
    dt = x.dtype
    if dt.is_floating_point:
        init_val = float("inf")
    elif dt == torch.bool:
        init_val = True
    else:
        # integer types
        info = torch.iinfo(dt)
        init_val = int(info.max)

    # Prepare output row vector of length M
    if out is None:
        out_row = torch.empty((M,), dtype=x.dtype, device=x.device)
        out_target = None
    else:
        # Ensure out shape matches final_shape
        expected_numel = _prod(final_shape) if len(final_shape) > 0 else 1
        if out.numel() != expected_numel:
            raise RuntimeError("out tensor has incorrect number of elements")
        # We will write into a contiguous view; if out isn't contiguous, use a temp and then reshape/copy back
        if out.is_contiguous():
            out_row = out.view(M)
            out_target = out
        else:
            out_row = torch.empty((M,), dtype=out.dtype, device=out.device)
            out_target = out

    # Strides for x_2d (contiguous row-major)
    stride_xm = x_2d.stride(0)
    stride_xk = x_2d.stride(1)

    # Launch kernel
    grid = lambda meta: (M,)
    BLOCK_SIZE = 1024
    amin_reduce_last_kernel[grid](
        x_2d,
        out_row,
        M,
        K,
        stride_xm,
        stride_xk,
        init_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape to target final shape
    if len(final_shape) == 0:
        result = out_row.view(())
    else:
        result = out_row.view(final_shape)

    if out_target is not None:
        # If original 'out' was non-contiguous, copy result into it respecting shape
        if not out_target.is_contiguous():
            # Copy into the provided 'out' tensor
            out_target.copy_(result)
            return out_target
        return out_target
    return result


def amin(*args, **kwargs):
    # Parse args to match aten.amin
    if len(args) == 0:
        raise RuntimeError("amin requires at least one tensor argument")
    x = args[0]
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)

    # Positional handling: amin(x, dim), amin(x, dim, keepdim)
    if len(args) >= 2:
        if isinstance(args[1], (int, list, tuple)):
            dim = args[1]
        elif isinstance(args[1], bool):
            keepdim = args[1]
    if len(args) >= 3:
        if isinstance(args[2], bool):
            keepdim = args[2]

    return _amin_impl(x, dim=dim, keepdim=keepdim, out=None)


def amin_out(*args, **kwargs):
    # Expected signature: amin_out(x, dim, keepdim, out) or with out as kwarg
    if len(args) == 0:
        raise RuntimeError("amin_out requires at least one tensor argument")
    x = args[0]

    # Extract out
    out = kwargs.get("out", None)
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)

    # Positional arguments
    # Try to detect out as last positional if provided
    if len(args) >= 2:
        if isinstance(args[1], (int, list, tuple)):
            dim = args[1]
        elif isinstance(args[1], bool):
            keepdim = args[1]
        elif isinstance(args[1], torch.Tensor):
            out = args[1]
    if len(args) >= 3:
        if isinstance(args[2], bool):
            keepdim = args[2]
        elif isinstance(args[2], torch.Tensor) and out is None:
            out = args[2]
    if len(args) >= 4 and out is None and isinstance(args[3], torch.Tensor):
        out = args[3]

    if out is None:
        raise RuntimeError("amin_out requires an 'out' tensor argument")

    return _amin_impl(x, dim=dim, keepdim=keepdim, out=out)