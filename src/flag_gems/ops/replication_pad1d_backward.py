"""Triton implementation of ``aten::replication_pad1d_backward``.

Replication padding fills the padded region by *replicating* the boundary
input values.  For a 1-D input with shape ``(*, L_in)`` and padding
``(pad_left, pad_right)`` the forward maps each output position ``j``
to the input position ``i = clamp(j - pad_left, 0, L_in - 1)``.  The
backward therefore needs to accumulate ``grad_output[j]`` into
``grad_input[clamp(j - pad_left, 0, L_in - 1)]`` -- a scatter-add along
the spatial dimension.  We do this with a single Triton kernel using
``tl.atomic_add`` (the same pattern ``reflection_pad1d_backward`` uses).

aten schema::

    replication_pad1d_backward(Tensor grad_output, Tensor self,
                               SymInt[2] padding) -> Tensor
    replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self,
                                          SymInt[2] padding, *,
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
def _replication_pad1d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    B,
    L_in,
    pad_left,
    L_out,
    BLOCK_W: tl.constexpr,
):
    """One program handles a row (``pid_b``) and a tile of ``BLOCK_W`` output
    positions.  Each loaded grad_output element is atomically added to its
    corresponding clamped input index."""
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_out = offs_w < L_out

    base_out = pid_b * L_out
    base_in = pid_b * L_in

    grad = tl.load(grad_output_ptr + base_out + offs_w, mask=mask_out, other=0.0)
    grad_f32 = grad.to(tl.float32)

    # Map each output column to the replicated input column.
    # input_index = clamp(out_index - pad_left, 0, L_in - 1)
    x = offs_w.to(tl.int32) - pad_left
    iw = tl.where(x < 0, 0, tl.where(x > L_in - 1, L_in - 1, x))

    tl.atomic_add(grad_input_ptr + base_in + iw, grad_f32, mask=mask_out)


@triton.jit
def _copy_rows_kernel(in_ptr, out_ptr, B, W, BLOCK_W: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < W) & (pid_b < B)

    base = pid_b * W
    vals = tl.load(in_ptr + base + offs_w, mask=mask, other=0)
    tl.store(out_ptr + base + offs_w, vals, mask=mask)


def _launch(grad_output: torch.Tensor, self: torch.Tensor, padding, grad_input=None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if self.dim() < 1:
        raise ValueError("self must have at least 1 dimension")

    grad_output_c = grad_output.contiguous()
    x = self.contiguous()

    L_in = int(x.shape[-1])
    if L_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    L_out = L_in + pad_left + pad_right
    if int(grad_output_c.shape[-1]) != L_out:
        raise ValueError(
            f"grad_output last-dim ({int(grad_output_c.shape[-1])}) does not "
            f"match expected L_in + pad_left + pad_right ({L_out})."
        )

    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Accumulate in float32 so atomic adds are safe across dtypes.
    grad_input_f32 = torch.zeros_like(x, dtype=torch.float32)

    if pad_left == 0 and pad_right == 0:
        # Degenerate case: just copy grad_output as the gradient.
        grid = (B, triton.cdiv(L_in, 256))
        with torch_device_fn.device(x.device):
            _copy_rows_kernel[grid](grad_output_c, grad_input_f32, B, L_in, BLOCK_W=256)
    else:
        grid = (B, triton.cdiv(L_out, 256))
        with torch_device_fn.device(x.device):
            _replication_pad1d_backward_kernel[grid](
                grad_output_c, grad_input_f32, B, L_in, pad_left, L_out, BLOCK_W=256
            )

    # Cast to the requested output dtype.
    target_dtype = x.dtype if grad_input is None else grad_input.dtype
    if target_dtype == torch.float32:
        casted = grad_input_f32
    else:
        casted = torch.empty(x.shape, device=x.device, dtype=target_dtype)
        grid = (B, triton.cdiv(L_in, 256))
        with torch_device_fn.device(x.device):
            _copy_rows_kernel[grid](grad_input_f32, casted, B, L_in, BLOCK_W=256)

    if grad_input is None:
        return casted

    if tuple(grad_input.shape) != tuple(x.shape):
        grad_input.resize_(x.shape)
    grad_input.copy_(casted)
    return grad_input


def replication_pad1d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS REPLICATION_PAD1D_BACKWARD")
    return _launch(grad_output, self, padding, grad_input=None)


def replication_pad1d_backward_grad_input(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    padding,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    """``replication_pad1d_backward.grad_input`` writes into the supplied
    tensor and returns it (PyTorch's ``out=``-style convention)."""
    logger.debug("GEMS REPLICATION_PAD1D_BACKWARD GRAD_INPUT")
    return _launch(grad_output, self, padding, grad_input=grad_input)
