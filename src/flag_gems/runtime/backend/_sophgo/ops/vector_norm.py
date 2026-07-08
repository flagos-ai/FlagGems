import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


logger = logging.getLogger(__name__)


def _generic_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    from flag_gems.ops.vector_norm import vector_norm as generic_vector_norm

    return generic_vector_norm(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)


def _is_full_reduction(x, dim):
    if dim is None:
        return True
    if isinstance(dim, int):
        dims = [dim % x.ndim]
    else:
        dims = [d % x.ndim for d in dim]
    return len(set(dims)) == x.ndim


def _normalize_reduce_dim(x, dim):
    if dim is None:
        return tuple(range(x.ndim))
    if isinstance(dim, int):
        return (dim % x.ndim,)
    return tuple(sorted({d % x.ndim for d in dim}))


def _vector_norm_decomposed(x, ord, dim, keepdim, dtype):
    work = x.to(dtype) if dtype is not None and x.dtype != dtype else x
    compute_dtype = (
        torch.float32
        if work.dtype in (torch.float16, torch.bfloat16)
        else work.dtype
    )
    reduce_dim = _normalize_reduce_dim(work, dim)
    keep_axes = [i for i in range(work.ndim) if i not in reduce_dim]
    perm = keep_axes + list(reduce_dim)
    transposed = work.permute(perm).contiguous()
    outer = 1
    for i in keep_axes:
        outer *= work.shape[i]
    reduce_size = 1
    for i in reduce_dim:
        reduce_size *= work.shape[i]
    flat = transposed.reshape(outer, reduce_size)
    if compute_dtype != flat.dtype:
        flat = flat.to(compute_dtype)
    abs_flat = torch.abs(flat)

    if ord in (2, 2.0):
        if len(reduce_dim) > 1 and flat.dtype == torch.float32 and reduce_size > 4096:
            chunk = 1024
            pad = (-reduce_size) % chunk
            if pad:
                flat_sq = flat * flat
                flat_sq = torch.cat(
                    [
                        flat_sq,
                        torch.zeros(
                            (outer, pad),
                            dtype=flat_sq.dtype,
                            device=flat_sq.device,
                        ),
                    ],
                    dim=-1,
                )
            else:
                flat_sq = flat * flat
            partial = flat_sq.reshape(outer, -1, chunk).sum(dim=-1)
            reduced = torch.sqrt(partial.sum(dim=-1))
        else:
            reduced = torch.sqrt(torch.sum(abs_flat * abs_flat, dim=-1))
    elif ord == 1:
        reduced = torch.sum(abs_flat, dim=-1)
    elif ord == 0:
        reduced = torch.sum((flat != 0).to(flat.dtype), dim=-1)
    elif ord == float("inf") or ord == math.inf:
        reduced = torch.amax(abs_flat, dim=-1)
    elif ord == -float("inf") or ord == -math.inf:
        reduced = -torch.amax(-abs_flat, dim=-1)
    else:
        return _generic_vector_norm(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

    out_dtype = dtype if dtype is not None else work.dtype
    if reduced.dtype != out_dtype:
        reduced = reduced.to(out_dtype)

    kept_shape = [work.shape[i] for i in keep_axes]
    if not kept_shape:
        reduced = reduced.reshape(())
        if keepdim:
            return reduced.reshape([1] * work.ndim)
        return reduced

    reduced = reduced.reshape(kept_shape)
    if not keepdim:
        return reduced

    out_shape = [1] * work.ndim
    for axis, size in zip(keep_axes, kept_shape):
        out_shape[axis] = size
    return reduced.reshape(out_shape)


@libentry()
@triton.jit
def vector_norm_l2_full_small_kernel(X, Out, NUMEL, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(x * x))
    tl.store(Out, out)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS SOPHGO VECTOR_NORM")

    if ord not in (2, 2.0, 1, 0, float("inf"), -float("inf"), math.inf, -math.inf):
        return _generic_vector_norm(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)
    if not _is_full_reduction(x, dim):
        return _vector_norm_decomposed(x, ord, dim, keepdim, dtype)

    out_dtype = dtype if dtype is not None else x.dtype
    if out_dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return _generic_vector_norm(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

    numel = x.numel()
    if numel == 0:
        return _vector_norm_decomposed(x, ord, dim, keepdim, dtype)
    if ord not in (2, 2.0) or numel > 256:
        return _vector_norm_decomposed(x, ord, dim, keepdim, dtype)

    original_ndim = x.ndim
    x = x.contiguous().reshape(-1)
    out = torch.empty([1], dtype=out_dtype, device=x.device)
    block_size = max(32, triton.next_power_of_2(numel))

    with torch_device_fn.device(x.device):
        vector_norm_l2_full_small_kernel[(1,)](
            x,
            out,
            numel,
            BLOCK_SIZE=block_size,
        )

    if keepdim:
        return out.reshape([1] * original_ndim)
    return out[0]
