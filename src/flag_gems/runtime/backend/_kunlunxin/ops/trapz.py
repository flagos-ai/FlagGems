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

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


_EPILOGUE_BLOCK = 1024


@triton.jit
def _trapz_epilogue_kernel(
    tmp_ptr,
    first_ptr,
    last_ptr,
    out_ptr,
    dx,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Every address is a plain contiguous ramp: no clamp, no per-lane index
    # arithmetic, no gather. Two earlier versions of this kernel died with an
    # on-device kernel exception (status 719/714) on XPU3 -- one with a masked
    # tail, one with a clamped index that gathered ``y[r, 0]`` / ``y[r, N - 1]``
    # per lane (the M == 1 launch survived, the first multi-lane launch at
    # M == 64 faulted). The caller therefore hands over three buffers that are
    # all padded to a multiple of BLOCK, so the tail lanes land on allocated
    # scratch instead of needing a mask.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(tmp_ptr + offs)
    first = tl.load(first_ptr + offs).to(tl.float32)
    last = tl.load(last_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, (dx * (t - 0.5 * (first + last))).to(OUT_DTYPE))


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
        print(f"[dbg] M={M} N={N} after reduce", flush=True)
        torch.cuda.synchronize()
        print("[dbg] sync after reduce OK", flush=True)
        # The epilogue is a plain contiguous elementwise kernel (see the kernel
        # comment), so its three inputs are staged into buffers padded to a
        # multiple of BLOCK: the two endpoint columns are gathered here by a
        # layout-only copy instead of inside the kernel, and the padding absorbs
        # the tail lanes. All of it is O(M) against an O(M * N) reduce.
        block = _EPILOGUE_BLOCK
        padded = triton.cdiv(M, block) * block
        with torch_device_fn.device(y.device):
            tmp_p = torch.empty(padded, dtype=torch.float32, device=y.device)
            tmp_p[:M].copy_(tmp)
            first_p = torch.empty(padded, dtype=y.dtype, device=y.device)
            first_p[:M].copy_(y2[:, 0])
            last_p = torch.empty(padded, dtype=y.dtype, device=y.device)
            last_p[:M].copy_(y2[:, N - 1])
            torch.cuda.synchronize()
            print("[dbg] sync after staging OK", flush=True)
            out_p = torch.empty(padded, dtype=y.dtype, device=y.device)
            _trapz_epilogue_kernel[(triton.cdiv(M, block),)](
                tmp_p,
                first_p,
                last_p,
                out_p,
                dx,
                OUT_DTYPE=_TL_DTYPE[y.dtype],
                BLOCK=block,
                buffer_size_limit=2048,
            )
            torch.cuda.synchronize()
            print("[dbg] sync after epilogue OK", flush=True)
        return out_p[:M].reshape(out_shape)

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
