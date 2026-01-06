import torch
import triton
import triton.language as tl


@triton.jit
def asinh_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, COMPUTE_FP32: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if COMPUTE_FP32:
        x32 = x.to(tl.float32)
        y32 = tl.log(x32 + tl.sqrt(x32 * x32 + 1.0))
        y = y32.to(x.dtype)
    else:
        y = tl.log(x + tl.sqrt(x * x + 1.0))

    tl.store(x_ptr + offsets, y, mask=mask)


asinh__kernel = asinh_


def asinh_(*args, **kwargs):
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        for key in ("input", "self", "x"):
            val = kwargs.get(key, None)
            if isinstance(val, torch.Tensor):
                x = val
                break
    if x is None:
        raise ValueError("asinh_: expected a Tensor as the first argument")

    if not x.is_cuda:
        return torch.ops.aten.asinh_(x)

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.ops.aten.asinh_(x)

    BLOCK_SIZE = 1024
    COMPUTE_FP32 = x.dtype in (torch.float16, torch.bfloat16)

    if x.is_contiguous():
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        asinh__kernel[grid](
            x, n_elements, BLOCK_SIZE=BLOCK_SIZE, COMPUTE_FP32=COMPUTE_FP32
        )
        return x
    else:
        y = x.contiguous()
        n_elements = y.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        asinh__kernel[grid](
            y, n_elements, BLOCK_SIZE=BLOCK_SIZE, COMPUTE_FP32=COMPUTE_FP32
        )
        x.copy_(y)
        return x
