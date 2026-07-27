import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _replication_pad3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    D_in,
    H_in,
    W_in,
    pad_left,
    pad_top,
    pad_front,
    D_out,
    H_out,
    W_out,
    BLOCK: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_w = tl.program_id(1)
    offsets = pid_w * BLOCK + tl.arange(0, BLOCK)
    total = D_out * H_out * W_out
    mask = offsets < total

    d_out = offsets // (H_out * W_out)
    h_out = (offsets // W_out) % H_out
    w_out = offsets % W_out

    d = tl.maximum(0, tl.minimum(D_in - 1, d_out - pad_front))
    h = tl.maximum(0, tl.minimum(H_in - 1, h_out - pad_top))
    w = tl.maximum(0, tl.minimum(W_in - 1, w_out - pad_left))

    output_base = pid_batch * total
    input_base = pid_batch * D_in * H_in * W_in
    grad = tl.load(
        grad_output_ptr + output_base + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    input_offset = input_base + d * H_in * W_in + h * W_in + w
    tl.atomic_add(grad_input_ptr + input_offset, grad, mask=mask)


@triton.jit
def _copy_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


def replication_pad3d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    if not isinstance(padding, (list, tuple)) or len(padding) != 6:
        raise ValueError("padding must contain six values")
    if self.dim() < 3:
        raise ValueError("self must have at least three dimensions")
    if grad_output.device != self.device or grad_output.dtype != self.dtype:
        raise ValueError("grad_output and self must have the same device and dtype")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("replication_pad3d_backward supports floating point dtypes")

    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = map(int, padding)
    x = self.contiguous()
    grad_output = grad_output.contiguous()
    d_in, h_in, w_in = (int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1]))
    d_out = d_in + pad_front + pad_back
    h_out = h_in + pad_top + pad_bottom
    w_out = w_in + pad_left + pad_right
    expected_spatial = (d_out, h_out, w_out)
    if tuple(grad_output.shape[-3:]) != expected_spatial:
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(grad_output.shape[-3:])} does not match {expected_spatial}"
        )
    if tuple(grad_output.shape[:-3]) != tuple(x.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")

    batch = math.prod(x.shape[:-3]) if x.dim() > 3 else 1
    input_voxels = d_in * h_in * w_in
    output_voxels = d_out * h_out * w_out
    grad_output = grad_output.reshape(-1)

    if all(value == 0 for value in (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)):
        return grad_output.reshape(x.shape)

    grad_input = torch.zeros_like(x, dtype=torch.float32).reshape(-1)
    with torch_device_fn.device(x.device):
        _replication_pad3d_backward_kernel[(batch, (output_voxels + 255) // 256)](
            grad_output,
            grad_input,
            d_in,
            h_in,
            w_in,
            pad_left,
            pad_top,
            pad_front,
            d_out,
            h_out,
            w_out,
            BLOCK=256,
        )

    if x.dtype == torch.float32:
        return grad_input.reshape(x.shape)
    result = torch.empty_like(x)
    with torch_device_fn.device(x.device):
        _copy_kernel[((batch * input_voxels + 255) // 256,)](
            grad_input, result.reshape(-1), batch * input_voxels, BLOCK=256
        )
    return result
