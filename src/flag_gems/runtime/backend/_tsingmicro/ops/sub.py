import logging
import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_BLOCK_N = 1024 * 4  # 4096 — 2D/3D fast path


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


@triton.jit
def _sub_row_broadcast_kernel_2d(
    full_ptr,
    brc_ptr,
    out_ptr,
    alpha,
    stride_full0,
    stride_full1,
    stride_brc0,
    stride_brc1,
    stride_out0,
    stride_out1,
    n_cols,
    num_blocks,
    FULL_IS_MINUEND: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    brc = tl.load(brc_ptr + row * stride_brc0 + 0 * stride_brc1)
    for blk in range(num_blocks):
        cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        v = tl.load(
            full_ptr + row * stride_full0 + cols * stride_full1,
            mask=mask,
            other=0.0,
        )
        if FULL_IS_MINUEND:
            out = v - alpha * brc
        else:
            out = brc - alpha * v
        tl.store(
            out_ptr + row * stride_out0 + cols * stride_out1,
            out,
            mask=mask,
        )


@triton.jit
def _sub_row_broadcast_kernel_3d(
    full_ptr,
    brc_ptr,
    out_ptr,
    alpha,
    stride_f0,
    stride_f1,
    stride_f2,
    stride_b0,
    stride_b1,
    stride_b2,
    stride_o0,
    stride_o1,
    stride_o2,
    d1,
    n_cols,
    num_blocks,
    FULL_IS_MINUEND: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    i0 = row // d1
    i1 = row % d1

    brc = tl.load(
        brc_ptr + i0 * stride_b0 + i1 * stride_b1 + 0 * stride_b2,
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
        if FULL_IS_MINUEND:
            out = v - alpha * brc
        else:
            out = brc - alpha * v
        tl.store(
            out_ptr + base_o + cols * stride_o2,
            out,
            mask=mask,
        )


def _pick_sub_broadcast_2d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[bool, torch.Tensor, torch.Tensor, tuple[int, int]] | None:
    """Return ``(full_is_minuend, full, brc, out_shape)`` for ``a - alpha * b``."""
    if a.ndim != 2 or b.ndim != 2:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0]:
        return None
    if a.shape[1] > 1 and b.shape[1] == 1:
        return True, a, b, tuple(a.shape)
    if a.shape[1] == 1 and b.shape[1] > 1:
        return False, b, a, tuple(b.shape)
    return None


def _pick_sub_broadcast_3d(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[bool, torch.Tensor, torch.Tensor, tuple[int, int, int]] | None:
    if a.ndim != 3 or b.ndim != 3:
        return None
    if a.device != b.device or a.dtype != b.dtype:
        return None
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        return None
    if a.shape[2] > 1 and b.shape[2] == 1:
        return True, a, b, tuple(a.shape)
    if a.shape[2] == 1 and b.shape[2] > 1:
        return False, b, a, tuple(b.shape)
    return None


def _sub_fast_broadcast_2d(
    a: torch.Tensor, b: torch.Tensor, alpha: float
) -> torch.Tensor | None:
    picked = _pick_sub_broadcast_2d(a, b)
    if picked is None:
        return None
    full_is_minuend, full, brc, out_shape = picked
    m, n = out_shape
    out = torch.empty(out_shape, device=full.device, dtype=full.dtype)
    block_n = _BLOCK_N
    num_blocks = triton.cdiv(n, block_n)
    _sub_row_broadcast_kernel_2d[(m,)](
        full,
        brc,
        out,
        alpha,
        full.stride(0),
        full.stride(1),
        brc.stride(0),
        brc.stride(1),
        out.stride(0),
        out.stride(1),
        n,
        num_blocks,
        FULL_IS_MINUEND=full_is_minuend,
        BLOCK_N=block_n,
    )
    return out


def _sub_fast_broadcast_3d(
    a: torch.Tensor, b: torch.Tensor, alpha: float
) -> torch.Tensor | None:
    picked = _pick_sub_broadcast_3d(a, b)
    if picked is None:
        return None
    full_is_minuend, full, brc, out_shape = picked
    d0, d1, n = out_shape
    m = d0 * d1
    out = torch.empty(out_shape, device=full.device, dtype=full.dtype)
    block_n = _BLOCK_N
    num_blocks = triton.cdiv(n, block_n)
    _sub_row_broadcast_kernel_3d[(m,)](
        full,
        brc,
        out,
        alpha,
        full.stride(0),
        full.stride(1),
        full.stride(2),
        brc.stride(0),
        brc.stride(1),
        brc.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        d1,
        n,
        num_blocks,
        FULL_IS_MINUEND=full_is_minuend,
        BLOCK_N=block_n,
    )
    return out


def _sub_fast_broadcast(
    a: torch.Tensor, b: torch.Tensor, alpha: float
) -> torch.Tensor | None:
    if a.ndim == 3 and b.ndim == 3:
        return _sub_fast_broadcast_3d(a, b, alpha)
    if a.ndim == 2 and b.ndim == 2:
        return _sub_fast_broadcast_2d(a, b, alpha)
    return None


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_TSINGMICRO SUB ")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        fast = _sub_fast_broadcast(A, B, alpha)
        if fast is not None:
            return fast
        return sub_func(A, B, alpha)
    if isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
    if isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_TSINGMICRO SUB_ ")
    if isinstance(B, torch.Tensor):
        fast = _sub_fast_broadcast(A, B, alpha)
        if fast is not None:
            A.copy_(fast)
            return A
        return sub_func(A, B, alpha, out0=A)
    return sub_func_tensor_scalar(A, B, alpha, out0=A)
