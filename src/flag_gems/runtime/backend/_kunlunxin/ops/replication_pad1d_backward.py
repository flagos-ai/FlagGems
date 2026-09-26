import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

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
    MAXG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold grad_output columns back onto grad_input columns.

    Each lane owns one flattened (N*C, W_in) input column ``o``. The set of
    grad_output columns that replicate onto it is a contiguous run ``[lo,
    lo+cnt)``; edges gather ``pad+1`` columns, the interior a single column.
    ``cnt`` is data-dependent, so the run is summed with a compile-time
    ``tl.static_range(MAXG)`` bound plus a ``c < cnt`` mask -- this avoids the
    runtime-bound ``scf.for`` loops the TritonXPU legalize pass rejects with
    "operand does not dominate this use".
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

    out_base = nc * W_out
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in tl.static_range(MAXG):
        cc = tl.minimum(c, tl.maximum(cnt - 1, 0))
        v = tl.load(go_ptr + out_base + lo + cc, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(c < cnt, v, 0.0)

    tl.store(gi_ptr + o, acc.to(gi_ptr.dtype.element_ty), mask=mask)


def _run_fold(grad_output_2d: torch.Tensor, grad_input: torch.Tensor, W_out, W_in, pl):
    """grad_output_2d: contiguous (NC, W_out); grad_input: contiguous (NC, W_in)."""
    total = grad_input.numel()
    if total == 0:
        return grad_input
    maxg = max(int(pl) + 1, W_out - W_in - int(pl) + 1, 1)
    if W_in == 1:
        maxg = max(maxg, W_out)
    maxg = max(maxg, 1)
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    with torch_device_fn.device(grad_input.device):
        _replication_pad1d_backward_fold_kernel[grid](
            grad_output_2d,
            grad_input,
            W_out,
            W_in,
            int(pl),
            total,
            MAXG=maxg,
            BLOCK=BLOCK,
        )
    return grad_input


def _fold_component(go_real: torch.Tensor, W_out, W_in, pl, N, C):
    """Sum-fold a single real component; returns a contiguous (N*C, W_in)."""
    go_flat = go_real.contiguous().reshape(N * C, W_out)
    gi_flat = torch.empty((N * C, W_in), device=go_real.device, dtype=go_real.dtype)
    _run_fold(go_flat, gi_flat, W_out, W_in, pl)
    return gi_flat


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
        for comp in (0, 1):
            folded = _fold_component(go_r[..., comp], W_out, W_in, left, N, C)
            gi_r[..., comp].copy_(folded.reshape(gi_r[..., comp].shape))
        return grad_input

    go_flat = grad_output.contiguous().reshape(N * C, W_out)
    _run_fold(go_flat, grad_input.reshape(N * C, W_in), W_out, W_in, left)
    return grad_input
