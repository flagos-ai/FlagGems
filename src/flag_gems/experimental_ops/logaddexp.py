import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)

    # logaddexp = log(e^x+e^y) = log(e^m*(e^(x-m)+e^(y-m)))
    #           = m + log(e^0 + e^(-|x-y|))
    #           = m + log(1 + e^(-|x-y|))
    m = tl.maximum(x, y)
    d = tl.abs(x-y)
    t = tl.exp(-d)
    res = m + tl.log(1+t)

    tl.store(out_ptr + offs, res, mask=mask)

def _broadcast_and_check(x, y):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    bx, by = torch.broadcast_tensors(x, y)
    return bx, by

def _choose_out_dtype(x, y, out=None):
    if out is not None:
        return out.dtype
    
    float_priority = [torch.float64, torch.float32, torch.bfloat16, torch.float16]
    
    for dt in float_priority:
        if x.dtype == dt or y.dtype == dt:
            return dt

    return torch.get_default_dtype()

def _launch_kernel(xc, yc, outc):
    n_elements = outc.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    logaddexp_kernel[grid](xc, yc, outc, n_elements, BLOCK_SIZE=1024)

def logaddexp(x, y):
    bx, by = _broadcast_and_check(x, y)

    if (
        bx.device.type != "cuda"
        or by.device.type != "cuda"
        or bx.device != by.device
        or bx.is_complex()
        or by.is_complex()
    ):
        return torch.os.aten.logaddexp(bx, by)

    out_dtype = _choose_out_dtype(bx, by, out=None)
    out = torch.empty(bx.shape, device=bx.device, dtype=out_dtype)
    
    xc = bx.contiguous().view(-1)
    yc = by.contiguous().view(-1)
    outc = out.contiguous().view(-1)

    _launch_kernel(xc, yc, outc)    
    #python warpper
    return out

def logaddexp_out(x, y, out):
    if out is None:
        raise ValueError("out tensor must be provided for logaddexp_out")
    
    bx, by = _broadcast_and_check(x, y)

    if (
       out.device.type != "cuda"
       or bx.device.type != "cuda"
       or by.device.type != "cuda"
       or not (bx.device == by.device== out.device)
       or bx.is_complex()
       or by.is_complex()
       or out.is_complex()
    ):
        return torch.ops.aten.logaddexp2.out(bx, by, out=out)

    if out.shape != bx.shape:
        raise ValueError(f"out tesnsor has shape {out.shape}, expected {bx.shape} from broadcast")

    xc = bx.contiguous().view(-1)
    yc = by.contiguous().view(-1)

    if out.is_contiguous():
        outc = out.view(-1)
        _launch_kernel(xc, yc, outc)
        return out
    else:
        tmp = torch.empty_like(out, memory_formate=torch.contingous_format)
        outc = tmp.view(-1)
        _launch_kernel(xc, yc, outc)
        out.copy_(tmp)
        return out
