import logging
import torch
import triton
import triton.language as tl
from ..utils.pointwise_dynamic import pointwise_dynamic


logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


@triton.jit
def _mul_row_broadcast_kernel_2d(
    full_ptr,
    scalar_ptr,
    out_ptr,
    stride_full0,
    stride_full1,
    stride_scalar0,
    stride_scalar1,
    stride_out0,
    stride_out1,
    n_cols,
    num_blocks,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(scalar_ptr + row * stride_scalar0 + 0 * stride_scalar1)
    for blk in range(num_blocks):
        cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        v = tl.load(
            full_ptr + row * stride_full0 + cols * stride_full1,
            mask=mask,
            other=0.0,
        )
        tl.store(
            out_ptr + row * stride_out0 + cols * stride_out1,
            s * v,
            mask=mask,
        )


@triton.jit
def _mul_row_broadcast_kernel_3d(
    full_ptr,
    scalar_ptr,
    out_ptr,
    stride_f0,
    stride_f1,
    stride_f2,
    stride_s0,
    stride_s1,
    stride_s2,
    stride_o0,
    stride_o1,
    stride_o2,
    d1,
    n_cols,
    num_blocks,
    BLOCK_N: tl.constexpr,
):
    """One program per (d0, d1) slice; multiply full[..., :] by scalar[..., 0]."""
    row = tl.program_id(0)
    i0 = row // d1
    i1 = row % d1

    s = tl.load(
        scalar_ptr + i0 * stride_s0 + i1 * stride_s1 + 0 * stride_s2,
    )

    base_f = i0 * stride_f0 + i1 * stride_f1
    base_o = i0 * stride_o0 + i1 * stride_o1

    for blk in range(num_blocks):
        cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        v = tl.load(
            full_ptr + base_f + cols * stride_f2,
            mask=mask,
            other=0.0,
        )
        tl.store(
            out_ptr + base_o + cols * stride_o2,
            s * v,
            mask=mask,
        )


def _pick_full_scalar_2d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]] | None:
    if a.ndim != 2 or b.ndim != 2:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0]:
        return None
    if a.shape[1] == 1 and b.shape[1] > 1:
        return b, a, tuple(b.shape)
    if b.shape[1] == 1 and a.shape[1] > 1:
        return a, b, tuple(a.shape)
    return None


def _pick_full_scalar_3d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]] | None:
    if a.ndim != 3 or b.ndim != 3:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        return None
    if a.shape[2] == 1 and b.shape[2] > 1:
        return b, a, tuple(b.shape)
    if b.shape[2] == 1 and a.shape[2] > 1:
        return a, b, tuple(a.shape)
    return None


def _mul_fast_row_broadcast_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    picked = _pick_full_scalar_2d(a, b)
    if picked is None:
        return None
    full, scalar, out_shape = picked
    m, n = out_shape
    out = torch.empty(out_shape, device=full.device, dtype=full.dtype)
    block_n = 1024
    num_blocks = triton.cdiv(n, block_n)
    grid = (m,)
    _mul_row_broadcast_kernel_2d[grid](
        full,
        scalar,
        out,
        full.stride(0),
        full.stride(1),
        scalar.stride(0),
        scalar.stride(1),
        out.stride(0),
        out.stride(1),
        n,
        num_blocks,
        BLOCK_N=block_n,
    )
    return out


# It's estimated that the final implementation will be changed to perform line broadcasting when b.dim=1.
def _mul_fast_row_broadcast_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    picked = _pick_full_scalar_3d(a, b)
    if picked is None:
        return None
    full, scalar, out_shape = picked
    d0, d1, n = out_shape
    m = d0 * d1
    out = torch.empty(out_shape, device=full.device, dtype=full.dtype)
    block_n = 1024*4
    num_blocks = triton.cdiv(n, block_n)
    grid = (m,)
    _mul_row_broadcast_kernel_3d[grid](
        full,
        scalar,
        out,
        full.stride(0),
        full.stride(1),
        full.stride(2),
        scalar.stride(0),
        scalar.stride(1),
        scalar.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        d1,
        n,
        num_blocks,
        BLOCK_N=block_n,
    )
    return out


def _mul_fast_row_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    if a.ndim == 3 and b.ndim == 3:
        return _mul_fast_row_broadcast_3d(a, b)
    if a.ndim == 2 and b.ndim == 2:
        return _mul_fast_row_broadcast_2d(a, b)
    return None


def mul(A, B):
    logger.debug("GEMS_TSINGMICRO MUL")    
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        fast = _mul_fast_row_broadcast(A, B)
        if fast is not None:
            return fast
        return mul_func(A, B)
    if isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B)
    if isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A)
    return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_TSINGMICRO MUL_")
    if isinstance(B, torch.Tensor):
        fast = _mul_fast_row_broadcast(A, B)
        if fast is not None:
            A.copy_(fast)
            return A
        return mul_func(A, B, out0=A)
    return mul_func_scalar(A, B, out0=A)
