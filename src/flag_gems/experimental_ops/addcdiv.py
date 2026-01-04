import torch
import triton
import triton.language as tl


@triton.jit
def addcdiv_kernel(
    self_ptr, t1_ptr, t2_ptr, out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(self_ptr + offsets, mask=mask)
    b = tl.load(t1_ptr + offsets, mask=mask)
    c = tl.load(t2_ptr + offsets, mask=mask)

    val_vec = tl.full(offsets.shape, value, a.dtype)
    result = a + (b / c) * val_vec
    tl.store(out_ptr + offsets, result, mask=mask)


def _prepare_addcdiv_tensors(self, tensor1, tensor2):
    if not (self.is_cuda and tensor1.is_cuda and tensor2.is_cuda):
        raise NotImplementedError(
            "addcdiv Triton implementation requires CUDA tensors."
        )
    if not (self.device == tensor1.device == tensor2.device):
        raise ValueError("All tensors must be on the same CUDA device.")
    a, b, c = torch.broadcast_tensors(self, tensor1, tensor2)
    # Determine common dtype for computation
    common_dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    a = a.to(dtype=common_dtype).contiguous()
    b = b.to(dtype=common_dtype).contiguous()
    c = c.to(dtype=common_dtype).contiguous()
    return a, b, c, common_dtype


def _launch_addcdiv(a, b, c, out, value):
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # value can be Python number or 0-d tensor; convert to float
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("value must be a scalar.")
        # move to same device if needed, then to host scalar
        if value.device.type == "cuda" and value.device != a.device:
            raise ValueError(
                "Scalar tensor 'value' must be on the same device as inputs."
            )
        value = float(value.to(dtype=out.dtype).item())
    else:
        value = float(value)
    addcdiv_kernel[grid](a, b, c, out, n_elements, value, BLOCK_SIZE=BLOCK_SIZE)


def addcdiv(self, tensor1, tensor2, *, value=1):
    """
    Returns self + value * tensor1 / tensor2 (element-wise).
    """
    a, b, c, common_dtype = _prepare_addcdiv_tensors(self, tensor1, tensor2)
    out = torch.empty_like(a, dtype=common_dtype, device=a.device)
    _launch_addcdiv(a, b, c, out, value)
    return out


def addcdiv_out(self, tensor1, tensor2, *, value=1, out=None):
    """
    Writes self + value * tensor1 / tensor2 (element-wise) into out.
    """
    if out is None:
        raise ValueError("out tensor must be provided for addcdiv_out.")
    a, b, c, common_dtype = _prepare_addcdiv_tensors(self, tensor1, tensor2)

    # Ensure out has correct device, dtype, and shape
    if not out.is_cuda:
        raise NotImplementedError("out tensor must be a CUDA tensor.")
    if out.device != a.device:
        raise ValueError("out tensor must be on the same device as inputs.")
    if out.dtype != common_dtype:
        raise TypeError(f"out tensor has dtype {out.dtype}, expected {common_dtype}.")
    if out.shape != a.shape:
        out.resize_(a.shape)

    if out.is_contiguous():
        _launch_addcdiv(a, b, c, out, value)
    else:
        tmp = torch.empty_like(a, dtype=common_dtype, device=a.device)
        _launch_addcdiv(a, b, c, tmp, value)
        out.copy_(tmp)
    return out
