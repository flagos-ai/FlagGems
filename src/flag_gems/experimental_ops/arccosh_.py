import torch
import triton
import triton.language as tl


@triton.jit
def arccosh_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)
    one = 1.0
    y32 = tl.log(x32 + tl.sqrt((x32 - one) * (x32 + one)))
    y = y32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


_kernel_ref_arccosh_ = arccosh_


def arccosh_(*args, **kwargs):
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None or not isinstance(x, torch.Tensor):
        raise TypeError("arccosh_ expects a single torch.Tensor input")

    if not x.is_cuda:
        return torch.ops.aten.arccosh_(x)

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.ops.aten.arccosh_(x)

    BLOCK_SIZE = 1024

    if x.is_contiguous():
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _kernel_ref_arccosh_[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return x
    else:
        tmp = x.contiguous()
        n_elements = tmp.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _kernel_ref_arccosh_[grid](tmp, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(tmp)
        return x
