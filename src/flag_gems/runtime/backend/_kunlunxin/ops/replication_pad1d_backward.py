import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _gems_copy(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy ``src`` into ``dst`` via the generic gems Triton copy kernel.

    ``flag_gems.ops.copy.copy_`` is the ``pointwise_dynamic`` ``_copy_kernel``;
    calling it by function reference bypasses the dispatcher, so it never re-enters
    the vendor ``copy_`` (whose strided branch is a known wedge). It handles
    broadcasting and arbitrary strides. Here it only ever materialises a contiguous
    ``dst`` from a strided real ``src``, which lands squarely on its Triton path
    (real dtype, strided layout, non-empty).
    """
    _gems_copy_(dst, src)
    return dst


def _ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Row-major contiguous ``t`` with no torch data-movement fallback.

    Already-contiguous tensors (the common case: native forward emits a contiguous
    grad_output) are returned unchanged -- a true no-op, no extra kernel launch.
    A strided grad_output is materialised into a fresh contiguous buffer via the
    gems Triton copy. Zero-element tensors are returned as-is: the caller
    early-returns before the kernel reads them, and gems ``copy_`` would redispatch
    to aten on an empty dst (which we must avoid).
    """
    if t.numel() == 0 or t.is_contiguous():
        return t
    return _gems_copy(torch.empty(t.shape, dtype=t.dtype, device=t.device), t)


_SCALAR_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _scalar_name(dtype):
    return _SCALAR_NAMES.get(dtype, str(dtype).replace("torch.", ""))


@triton.jit
def _replication_pad1d_backward_fold_kernel(
    go_ptr,
    gi_ptr,
    W_out,
    W_in,
    pl,
    total,
    go_row_stride,
    go_col_stride,
    gi_row_stride,
    gi_col_stride,
    MAXG: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIG: tl.constexpr,
):
    """Fold grad_output columns back onto grad_input columns.

    Each lane owns one flattened (N*C, W_in) input column ``o``. The set of
    grad_output columns that replicate onto it is a contiguous run ``[lo,
    lo+cnt)``; edges gather ``pad+1`` columns, the interior a single column.
    ``cnt`` is data-dependent, so the run is summed with a compile-time
    ``tl.static_range(MAXG)`` bound plus a ``c < cnt`` mask -- this avoids the
    runtime-bound ``scf.for`` loops the TritonXPU legalize pass rejects with
    "operand does not dominate this use".

    ``CONTIG`` selects between two address modes. When both ``grad_output`` and
    ``grad_input`` are row-major contiguous (the plain real path) the store lands
    at ``gi_ptr + o`` -- a compiler-provable stride-1 write that lowers to a block
    DMA. When the caller passes a strided view (the complex path folds directly
    into ``view_as_real(grad_input)[..., comp]``, whose element stride is 2) the
    store is addressed through the explicit row/column strides, letting the kernel
    write straight into ``grad_input`` with no scratch buffer and no ``copy_``.
    """
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    iw = o % W_in
    nc = o // W_in

    lo_raw = tl.where(iw == 0, 0, tl.where(iw == W_in - 1, pl + W_in - 1, pl + iw))
    lo = tl.minimum(tl.maximum(lo_raw, 0), W_out - 1)

    cnt = tl.where(
        iw == 0,
        tl.where(W_in == 1, W_out, tl.minimum(tl.maximum(pl + 1, 0), W_out)),
        tl.where(
            iw == W_in - 1,
            tl.where(lo_raw >= W_out, 0, W_out - tl.maximum(lo_raw, 0)),
            tl.where((lo_raw >= 0) & (lo_raw < W_out), 1, 0),
        ),
    )

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if CONTIG:
        out_base = nc * W_out
        for c in tl.static_range(MAXG):
            cc = tl.minimum(c, tl.maximum(cnt - 1, 0))
            v = tl.load(go_ptr + out_base + lo + cc, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += tl.where(c < cnt, v, 0.0)
        tl.store(gi_ptr + o, acc.to(gi_ptr.dtype.element_ty), mask=mask)
    else:
        go_base = nc * go_row_stride
        for c in tl.static_range(MAXG):
            cc = tl.minimum(c, tl.maximum(cnt - 1, 0))
            v = tl.load(
                go_ptr + go_base + (lo + cc) * go_col_stride, mask=mask, other=0.0
            ).to(tl.float32)
            acc += tl.where(c < cnt, v, 0.0)
        gi_off = nc * gi_row_stride + iw * gi_col_stride
        tl.store(gi_ptr + gi_off, acc.to(gi_ptr.dtype.element_ty), mask=mask)


def _run_fold(
    grad_output_2d: torch.Tensor,
    grad_input: torch.Tensor,
    W_out,
    W_in,
    pl,
    contiguous: bool = True,
):
    """Fold grad_output_2d (NC, W_out) onto grad_input (NC, W_in).

    Both are 2-D. When ``contiguous`` is True they are row-major contiguous and
    the kernel uses the stride-1 fast path; otherwise the actual element strides
    are passed and the kernel writes straight into the (possibly strided) view.
    """
    total = grad_input.numel()
    if total == 0:
        return grad_input
    maxg = max(int(pl) + 1, W_out - W_in - int(pl) + 1, 1)
    if W_in == 1:
        maxg = max(maxg, W_out)
    maxg = max(maxg, 1)
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    go_row_stride, go_col_stride = grad_output_2d.stride()
    gi_row_stride, gi_col_stride = grad_input.stride()
    with torch_device_fn.device(grad_input.device):
        _replication_pad1d_backward_fold_kernel[grid](
            grad_output_2d,
            grad_input,
            W_out,
            W_in,
            int(pl),
            total,
            go_row_stride,
            go_col_stride,
            gi_row_stride,
            gi_col_stride,
            MAXG=maxg,
            BLOCK=BLOCK,
            CONTIG=contiguous,
        )
    return grad_input


def replication_pad1d_backward(
    grad_output: torch.Tensor, self_tensor: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D_BACKWARD")
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    dim = self_tensor.dim()
    if dim not in (2, 3):
        raise ValueError(
            "replication_pad1d_backward expects 2D (C, W) or 3D (N, C, W) input"
        )

    if grad_output.dtype != self_tensor.dtype:
        raise RuntimeError(
            f"expected scalar type {_scalar_name(self_tensor.dtype)} but found "
            f"{_scalar_name(grad_output.dtype)}"
        )
    if grad_output.device != self_tensor.device:
        raise RuntimeError(
            f"expected grad_output to be on device {self_tensor.device} but found "
            f"{grad_output.device}"
        )

    grad_input = torch.empty(
        self_tensor.shape,
        device=self_tensor.device,
        dtype=self_tensor.dtype,
    )

    if dim == 3:
        N, C, W_in = self_tensor.shape
    else:
        C, W_in = self_tensor.shape
        N = 1
    W_out = W_in + left + right

    expected_grad_output_shape = (N, C, W_out) if dim == 3 else (C, W_out)
    if tuple(grad_output.shape) != expected_grad_output_shape:
        raise ValueError(
            f"grad_output has incorrect shape. Expected {expected_grad_output_shape}, "
            f"got {tuple(grad_output.shape)}"
        )

    if self_tensor.is_complex():
        go_r = torch.view_as_real(grad_output)
        gi_r = torch.view_as_real(grad_input)
        # Real and imaginary parts fold independently, so run the real kernel once
        # per component. view_as_real of a contiguous complex tensor is contiguous,
        # so slicing the trailing axis yields a strided (element stride 2) view whose
        # (N, C) axes collapse cleanly; the kernel folds straight into it, so there is
        # no scratch buffer and no copy_ write-back.
        for comp in (0, 1):
            go_c = go_r[..., comp].reshape(N * C, W_out)
            gi_c = gi_r[..., comp].reshape(N * C, W_in)
            _run_fold(go_c, gi_c, W_out, W_in, left, contiguous=False)
        return grad_input

    go_flat = _ensure_contiguous(grad_output).reshape(N * C, W_out)
    _run_fold(go_flat, grad_input.reshape(N * C, W_in), W_out, W_in, left)
    return grad_input
