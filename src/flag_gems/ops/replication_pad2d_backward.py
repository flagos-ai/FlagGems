"""Triton implementation of ``aten::replication_pad2d_backward``.

For a 2-D input with shape ``(*, H_in, W_in)`` and padding
``(pad_left, pad_right, pad_top, pad_bottom)`` the forward maps each
output position ``(y, x)`` to the input position

    (h, w) = (clamp(y - pad_top, 0, H_in - 1),
              clamp(x - pad_left, 0, W_in - 1))

The backward therefore accumulates ``grad_output[y, x]`` into
``grad_input[clamp(y - pad_top, 0, H_in - 1),
             clamp(x - pad_left, 0, W_in - 1)]`` -- a scatter-add over the
spatial dimensions.  We do this with a single Triton kernel using
``tl.atomic_add`` (the same pattern ``replication_pad1d_backward`` and
``reflection_pad1d_backward`` use).

aten schema::

    replication_pad2d_backward(Tensor grad_output, Tensor self,
                               SymInt[4] padding) -> Tensor
    replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self,
                                          SymInt[4] padding, *,
                                          Tensor(a!) grad_input)
                                          -> Tensor(a!)
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _replication_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    B,
    H_in,
    W_in,
    pad_left,
    pad_top,
    H_out,
    W_out,
    BLOCK_W: tl.constexpr,
):
    """One program handles a row ``pid_b`` (a flattened (N, C) index) and a
    tile of ``BLOCK_W`` output spatial positions on a particular output row
    ``pid_h``.  Each loaded ``grad_output`` element is atomically added to
    its clamped input position."""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_out = offs_w < W_out

    base_out = pid_b * H_out * W_out + pid_h * W_out
    base_in = pid_b * H_in * W_in

    grad = tl.load(grad_output_ptr + base_out + offs_w, mask=mask_out, other=0.0)
    grad_f32 = grad.to(tl.float32)

    # Map output (y, x) to input (h, w) via clamp.
    y = pid_h - pad_top
    h = tl.where(y < 0, 0, tl.where(y > H_in - 1, H_in - 1, y))

    x = offs_w.to(tl.int32) - pad_left
    w = tl.where(x < 0, 0, tl.where(x > W_in - 1, W_in - 1, x))

    tl.atomic_add(
        grad_input_ptr + base_in + h * W_in + w,
        grad_f32,
        mask=mask_out,
    )


@triton.jit
def _copy_rows_kernel(in_ptr, out_ptr, B, N, BLOCK_W: tl.constexpr):
    """Generic ``BLOCK_W``-tile row copy used both for the ``pad == 0`` fast
    path and for the final f32 -> input-dtype cast."""
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < N) & (pid_b < B)

    base = pid_b * N
    vals = tl.load(in_ptr + base + offs_w, mask=mask, other=0)
    tl.store(out_ptr + base + offs_w, vals, mask=mask)


def _launch(grad_output: torch.Tensor, self: torch.Tensor, padding, grad_input=None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: "
            "(pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left = int(padding[0])
    pad_right = int(padding[1])
    pad_top = int(padding[2])
    pad_bottom = int(padding[3])
    if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
        raise ValueError("padding values must be >= 0")
    if self.dim() < 2:
        raise ValueError("self must have at least 2 dimensions (H, W)")

    grad_output_c = grad_output.contiguous()
    x = self.contiguous()

    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    if H_in <= 0 or W_in <= 0:
        raise ValueError("input spatial dims (H, W) must both be > 0")

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right
    if int(grad_output_c.shape[-2]) != H_out or int(grad_output_c.shape[-1]) != W_out:
        raise ValueError(
            f"grad_output spatial shape ({int(grad_output_c.shape[-2])}, "
            f"{int(grad_output_c.shape[-1])}) does not match expected "
            f"({H_out}, {W_out})."
        )

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Accumulate in float32 for safe atomic adds across dtypes.
    grad_input_f32 = torch.zeros_like(x, dtype=torch.float32)

    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        # Degenerate case: backward is a plain copy.
        n_per_row = H_in * W_in
        grid = (B, triton.cdiv(n_per_row, 256))
        with torch_device_fn.device(x.device):
            _copy_rows_kernel[grid](
                grad_output_c, grad_input_f32, B, n_per_row, BLOCK_W=256
            )
    else:
        grid = (B, H_out, triton.cdiv(W_out, 256))
        with torch_device_fn.device(x.device):
            _replication_pad2d_backward_kernel[grid](
                grad_output_c,
                grad_input_f32,
                B,
                H_in,
                W_in,
                pad_left,
                pad_top,
                H_out,
                W_out,
                BLOCK_W=256,
            )

    target_dtype = x.dtype if grad_input is None else grad_input.dtype
    if target_dtype == torch.float32:
        casted = grad_input_f32
    else:
        casted = torch.empty(x.shape, device=x.device, dtype=target_dtype)
        n_per_row = H_in * W_in
        grid = (B, triton.cdiv(n_per_row, 256))
        with torch_device_fn.device(x.device):
            _copy_rows_kernel[grid](grad_input_f32, casted, B, n_per_row, BLOCK_W=256)

    if grad_input is None:
        return casted

    if tuple(grad_input.shape) != tuple(x.shape):
        grad_input.resize_(x.shape)
    grad_input.copy_(casted)
    return grad_input


def replication_pad2d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD")
    return _launch(grad_output, self, padding, grad_input=None)


def replication_pad2d_backward_grad_input(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    padding,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    """``replication_pad2d_backward.grad_input`` writes into ``grad_input``
    and returns it (preserves the ``grad_input=`` kwarg identity)."""
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD GRAD_INPUT")
    return _launch(grad_output, self, padding, grad_input=grad_input)
