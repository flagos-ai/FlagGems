import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def softshrink_kernel(x, lambd):
    x32 = x.to(tl.float32)
    gt = x32 > lambd
    lt = x32 < -lambd
    res32 = tl.where(gt, x32 - lambd, tl.where(lt, x32 + lambd, 0.0))
    res32 = tl.where(x32 != x32, x32, res32)
    return res32.to(x.dtype)


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {t.dtype}. Supported dtypes are float16, bfloat16, and float32."
        )


def softshrink(input: torch.Tensor, lambd: float = 0.5):
    _check_supported_dtype(input)
    return softshrink_kernel(input, float(lambd))


def softshrink_out(input: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None):
    if out is None:
        raise ValueError("Argument 'out' must be provided for softshrink_out.")
    _check_supported_dtype(input)
    result = softshrink_kernel(input, float(lambd))
    out.copy_(result)
    return out
