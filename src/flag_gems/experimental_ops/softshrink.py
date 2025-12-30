import torch
import triton
import triton.language as tl


@triton.jit
def softshrink_kernel(x_ptr, out_ptr, n_elements, lambd, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x32 = x.to(tl.float32)

    l = lambd  # scalar float32

    gt = x32 > l
    lt = x32 < -l
    res32 = tl.where(gt, x32 - l, tl.where(lt, x32 + l, 0.0))

    # Propagate NaN: if x is NaN, keep it
    res32 = tl.where(x32 != x32, x32, res32)

    res = res32.to(x.dtype)
    tl.store(out_ptr + offsets, res, mask=mask)


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype {t.dtype}. Supported dtypes are float16, bfloat16, and float32.")


def _launch_softshrink_kernel(x: torch.Tensor, out: torch.Tensor, lambd: float):
    n_elements = x.numel()
    if n_elements == 0:
        return
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    softshrink_kernel[grid](
        x, out, n_elements, float(lambd),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )


def softshrink(input: torch.Tensor, lambd: float = 0.5):
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    _check_supported_dtype(input)
    x = input.contiguous()
    out = torch.empty_like(x)
    _launch_softshrink_kernel(x, out, lambd)
    return out.reshape_as(input)


def softshrink_out(input: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None):
    if out is None:
        raise ValueError("Argument 'out' must be provided for softshrink_out.")
    if not input.is_cuda or not out.is_cuda:
        raise ValueError("Input and out tensors must be on CUDA device.")
    if input.shape != out.shape:
        raise ValueError(f"Shape mismatch: input.shape={input.shape}, out.shape={out.shape}")
    if input.dtype != out.dtype:
        raise TypeError(f"Dtype mismatch: input.dtype={input.dtype}, out.dtype={out.dtype}")
    _check_supported_dtype(input)

    x = input.contiguous()
    if out.is_contiguous():
        out_buf = out
    else:
        out_buf = torch.empty_like(out, memory_format=torch.contiguous_format)

    _launch_softshrink_kernel(x, out_buf, lambd)

    if out_buf.data_ptr() != out.data_ptr():
        out.copy_(out_buf)
    return out