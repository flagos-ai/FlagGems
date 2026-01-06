import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp2_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load inputs and upcast to fp32 for numerics
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)

    # Numerically-stable logaddexp2:
    # logaddexp2(x, y) = m + log2(1 + 2^(-|x - y|)), where m = max(x, y)
    ln2 = 0.6931471805599453
    inv_ln2 = 1.4426950408889634

    d = tl.abs(x - y)
    m = tl.maximum(x, y)
    t = tl.exp(-d * ln2)  # 2^(-|x-y|) = exp(-(abs(x-y)) * ln(2))
    res = m + tl.log(1.0 + t) * inv_ln2  # log2(1 + t) = ln(1+t) / ln(2)

    # Store; Triton will cast to the dtype of out_ptr as needed
    tl.store(out_ptr + offs, res, mask=mask)


def _broadcast_and_check(x, y):
    # Convert scalars to tensors
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    # Broadcast
    bx, by = torch.broadcast_tensors(x, y)
    return bx, by


def _choose_out_dtype(x, y, out=None):
    if out is not None:
        return out.dtype
    # Prefer highest precision floating dtype present; else default dtype
    float_priority = [torch.float64, torch.float32, torch.bfloat16, torch.float16]
    for dt in float_priority:
        if x.dtype == dt or y.dtype == dt:
            return dt
    # If none are floating, use default dtype
    return torch.get_default_dtype()


def _launch_kernel(xc, yc, outc):
    n_elements = outc.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    logaddexp2_kernel[grid](xc, yc, outc, n_elements, BLOCK_SIZE=1024)


def logaddexp2(x, y):
    bx, by = _broadcast_and_check(x, y)

    # Fallback for unsupported devices or complex dtype
    if (
        bx.device.type != "cuda"
        or by.device.type != "cuda"
        or bx.device != by.device
        or bx.is_complex()
        or by.is_complex()
    ):
        return torch.ops.aten.logaddexp2(bx, by)

    out_dtype = _choose_out_dtype(bx, by, out=None)
    out = torch.empty(bx.shape, device=bx.device, dtype=out_dtype)

    # Ensure contiguous 1D buffers for the kernel
    xc = bx.contiguous().view(-1)
    yc = by.contiguous().view(-1)
    outc = out.contiguous().view(-1)

    _launch_kernel(xc, yc, outc)
    return out


def logaddexp2_out(x, y, out):
    if out is None:
        raise ValueError("out tensor must be provided for logaddexp2_out")

    bx, by = _broadcast_and_check(x, y)

    # Fallback for unsupported devices or complex dtype
    if (
        out.device.type != "cuda"
        or bx.device.type != "cuda"
        or by.device.type != "cuda"
        or not (bx.device == by.device == out.device)
        or bx.is_complex()
        or by.is_complex()
        or out.is_complex()
    ):
        # Use PyTorch implementation for unsupported cases
        return torch.ops.aten.logaddexp2.out(bx, by, out=out)

    # Shape and dtype checks
    if out.shape != bx.shape:
        raise ValueError(
            f"out tensor has shape {out.shape}, expected {bx.shape} from broadcast"
        )
    # We allow dtype differences; computation will write to out's dtype

    # Prepare contiguous buffers
    xc = bx.contiguous().view(-1)
    yc = by.contiguous().view(-1)

    if out.is_contiguous():
        outc = out.view(-1)
        _launch_kernel(xc, yc, outc)
        return out
    else:
        # Compute into a temporary contiguous buffer then copy back
        tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        outc = tmp.view(-1)
        _launch_kernel(xc, yc, outc)
        out.copy_(tmp)
        return out
