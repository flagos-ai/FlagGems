import torch
import triton
import triton.language as tl


@triton.jit
def xlogy_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    # result = where(x == 0, 0, x * log(y))
    res = tl.where(x_f32 == 0.0, 0.0, x_f32 * tl.log(y_f32))

    tl.store(out_ptr + offsets, res, mask=mask)


def _ensure_tensor_on_device(obj, device, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype)
    else:
        return torch.as_tensor(obj, device=device, dtype=dtype)


def _prepare_tensors(self, other, out=None):
    # Determine device
    if isinstance(self, torch.Tensor):
        device = self.device
    elif isinstance(other, torch.Tensor):
        device = other.device
    else:
        raise ValueError("At least one of the inputs must be a Tensor.")

    if device.type != "cuda":
        raise ValueError("Triton kernels require CUDA tensors.")

    # Type promotion following PyTorch semantics
    if isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor):
        result_dtype = torch.result_type(self, other)
    elif isinstance(self, torch.Tensor):
        other_tmp = torch.as_tensor(other)
        result_dtype = torch.result_type(self, other_tmp)
    else:
        self_tmp = torch.as_tensor(self)
        result_dtype = torch.result_type(self_tmp, other)

    t_self = _ensure_tensor_on_device(self, device, result_dtype)
    t_other = _ensure_tensor_on_device(other, device, result_dtype)

    # Broadcast to a common shape
    b_self, b_other = torch.broadcast_tensors(t_self, t_other)

    # Prepare output
    if out is None:
        out_tensor = torch.empty(b_self.shape, device=device, dtype=result_dtype)
        return b_self.contiguous(), b_other.contiguous(), out_tensor, out_tensor
    else:
        if out.device != device:
            raise ValueError("Output tensor must be on the same device as inputs.")
        # Out dtype/shape should be able to hold result
        expected_shape = b_self.shape
        if out.shape != expected_shape:
            raise ValueError(
                f"Output tensor has shape {out.shape}, expected {expected_shape}."
            )
        if out.dtype != result_dtype:
            raise ValueError(
                f"Output tensor has dtype {out.dtype}, expected {result_dtype}."
            )
        # If out is contiguous, write directly; otherwise use a temporary
        if out.is_contiguous():
            return b_self.contiguous(), b_other.contiguous(), out, out
        else:
            tmp = torch.empty(expected_shape, device=device, dtype=result_dtype)
            return b_self.contiguous(), b_other.contiguous(), tmp, out


def _launch_xlogy(self, other, out=None):
    x, y, dst, final_out = _prepare_tensors(self, other, out)
    n_elements = dst.numel()
    if n_elements == 0:
        if final_out is not dst:
            final_out.copy_(dst)
        return final_out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    xlogy_kernel[grid](x, y, dst, n_elements, BLOCK_SIZE=1024)

    if final_out is not dst:
        final_out.copy_(dst)
    return final_out


# Wrappers corresponding to ATen operator interfaces
def xlogy_Tensor(self: torch.Tensor, other: torch.Tensor):
    return _launch_xlogy(self, other, out=None)


def xlogy_Scalar_Other(self: torch.Tensor, other):
    return _launch_xlogy(self, other, out=None)


def xlogy_Scalar_Self(self, other: torch.Tensor):
    return _launch_xlogy(self, other, out=None)


def xlogy_OutTensor(self: torch.Tensor, other: torch.Tensor, out: torch.Tensor):
    return _launch_xlogy(self, other, out=out)


def xlogy_OutScalar_Self(self, other: torch.Tensor, out: torch.Tensor):
    return _launch_xlogy(self, other, out=out)


def xlogy_OutScalar_Other(self: torch.Tensor, other, out: torch.Tensor):
    return _launch_xlogy(self, other, out=out)
