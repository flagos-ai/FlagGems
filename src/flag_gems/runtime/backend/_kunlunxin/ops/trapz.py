"""Kunlunxin override for aten::trapezoid.dx / aten::trapz.dx.

The generic ``trapezoid_kernel_dx`` fails to compile on XPU3
("'tt.addptr' op requires the same encoding for all operands"), so every
benchmark case errors out. The trapezoidal rule with constant spacing is

    dx * (sum(y) - 0.5 * y[..., 0] - 0.5 * y[..., -1]),

so the override reuses the vendor TLE row reduce (exact and 3-15x faster than
aten's ``trapezoid`` here) and folds the two endpoint corrections in a tiny
epilogue kernel. The backward (``dx * [0.5, 1, ..., 1, 0.5]`` broadcast) also
runs as a Triton kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.sum import sum_dim as _vendor_sum_dim
from flag_gems.utils.shape_utils import dim_compress

logger = logging.getLogger(__name__)

_BLOCK = 1024

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _trapz_epilogue_kernel(
    tmp_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    dx,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Masked memory access is unreliable on this backend at small M (the tail
    # lanes of a single under-full program trip an on-device kernel
    # exception, status 719/714), observed on both the masked load and the
    # masked store. Every address is therefore clamped into bounds; the
    # out-of-range tail lanes redundantly rewrite the same (last) element,
    # which is harmless since the value is value-identical.
    raw_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.minimum(raw_idx, M - 1)
    t = tl.load(tmp_ptr + idx)
    first = tl.load(y_ptr + idx * N).to(tl.float32)
    last = tl.load(y_ptr + idx * N + (N - 1)).to(tl.float32)
    r = dx * (t - 0.5 * (first + last))
    tl.store(out_ptr + idx, r.to(OUT_DTYPE))


@triton.jit
def _trapz_bwd_kernel(
    go_ptr,
    out_ptr,
    M,
    N,
    dx,
    OUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row: a tiny [BLOCK_M, BLOCK_N] tile store is miscompiled
    # on XPU3 (sporadic wrong rows), so keep the store one-dimensional.
    r = tl.program_id(0)
    g = tl.load(go_ptr + r).to(tl.float32)
    k = tl.arange(0, BLOCK_N)
    mask = k < N
    w = 1.0 - 0.5 * (k == 0).to(tl.float32) - 0.5 * (k == (N - 1)).to(tl.float32)
    tl.store(out_ptr + r * N + k, (g * (w * dx)).to(OUT_DTYPE), mask=mask)


def _normalize_dim(dim, ndim):
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _normalize_dx(dx):
    # Mirror ATen: dx must be a real number; bool/complex (incl. 0-d tensors)
    # are rejected before any float() coercion.
    dx_val = dx.item() if isinstance(dx, torch.Tensor) else dx
    if isinstance(dx_val, (bool, complex)) or not isinstance(dx_val, (int, float)):
        raise RuntimeError("trapezoid: Currently, we only support dx as a real number.")
    return float(dx_val)


def _compressed_order(y, dim):
    # Same ordering rule as flag_gems.utils.dim_compress (sorted by stride).
    batch_dim = [i for i in range(y.ndim) if i != dim]
    return batch_dim + [dim]


class _TrapzFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, dx, dim):
        shape = y.shape
        N = shape[dim]
        out_shape = shape[:dim] + shape[dim + 1 :]
        order = _compressed_order(y, dim)
        ctx.dim = dim
        ctx.dx = dx
        ctx.N = N
        ctx.y_shape = shape
        ctx.y_dtype = y.dtype
        ctx.order = order
        if N <= 1:
            ctx.yc_shape = None
            return torch.zeros(out_shape, dtype=y.dtype, device=y.device)

        yc = dim_compress(y, dim)
        M = yc.numel() // N
        ctx.yc_shape = yc.shape
        y2 = yc.reshape(-1, N)

        tmp = _vendor_sum_dim(y2, [1], dtype=torch.float32)
        out = torch.empty((M,), dtype=y.dtype, device=y.device)
        with torch_device_fn.device(y.device):
            _trapz_epilogue_kernel[(triton.cdiv(M, _BLOCK),)](
                tmp,
                y2,
                out,
                M,
                N,
                dx,
                OUT_DTYPE=_TL_DTYPE[y.dtype],
                BLOCK=_BLOCK,
                buffer_size_limit=2048,
            )
        return out.reshape(out_shape)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        N = ctx.N
        y_shape = ctx.y_shape
        y_dtype = ctx.y_dtype
        order = ctx.order
        yc_shape = ctx.yc_shape
        if N <= 1:
            return (
                torch.zeros(y_shape, dtype=y_dtype, device=grad_output.device),
                None,
                None,
            )

        # Align the upstream grad with the integration dim and compress it the
        # same way as y (pure device-side data movement).
        go_u = grad_output.unsqueeze(dim)
        go_c = go_u.movedim(dim, -1).contiguous().reshape(-1)
        M = go_c.numel()

        grad_c = torch.empty(yc_shape, dtype=y_dtype, device=grad_output.device)
        bn = min(4096, triton.next_power_of_2(N))
        with torch_device_fn.device(grad_output.device):
            _trapz_bwd_kernel[(M,)](
                go_c,
                grad_c,
                M,
                N,
                ctx.dx,
                OUT_DTYPE=_TL_DTYPE[y_dtype],
                BLOCK_N=bn,
                num_warps=1,
                buffer_size_limit=2048,
            )
        inv = [order.index(i) for i in range(len(order))]
        grad_y = grad_c.permute(inv).contiguous().view(y_shape)
        return grad_y, None, None


def trapz(y, dx=1.0, dim=-1):
    logger.debug("GEMS_KUNLUNXIN TRAPEZOID_DX")
    dim = _normalize_dim(dim, y.ndim)
    if y.shape[dim] == 0:
        out_shape = y.shape[:dim] + y.shape[dim + 1 :]
        return torch.zeros(out_shape, dtype=y.dtype, device=y.device)

    if y.dtype == torch.bool:
        raise RuntimeError(
            "trapezoid: received a bool input for `y`, but bool is not supported"
        )
    if y.is_complex():
        raise RuntimeError(
            "trapz: complex inputs are not supported by the FlagGems Triton kernel"
        )

    dx = _normalize_dx(dx)
    if not y.is_floating_point() or y.dtype == torch.float64:
        # Integer inputs are promoted to fp32 (aten); fp64 is silently
        # downcast by this backend, so normalize it here as well.
        y = y.to(torch.float32)

    return _TrapzFn.apply(y, dx, dim)
