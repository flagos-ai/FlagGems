import torch
import triton
import triton.language as tl


@triton.jit
def elu_kernel(
    x_ptr, out_ptr, n_elements, alpha, scale, input_scale, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    pos = x32 > 0.0
    neg = alpha * (tl.exp(input_scale * x32) - 1.0)
    y32 = tl.where(pos, x32, neg)
    y32 = scale * y32

    y = y32.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _parse_elu_args(args, kwargs, expect_out: bool = False):
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
        arg_idx = 1
    else:
        x = kwargs.get("input", kwargs.get("self", kwargs.get("x", None)))
        arg_idx = 0

    if x is None:
        raise ValueError("elu expects a Tensor as the first argument (input/self/x).")

    def _get_scalar(name, default, idx):
        if name in kwargs:
            return float(kwargs[name])
        elif len(args) > idx:
            return float(args[idx])
        else:
            return float(default)

    alpha = _get_scalar("alpha", 1.0, arg_idx + 0)
    scale = _get_scalar("scale", 1.0, arg_idx + 1)
    input_scale = _get_scalar("input_scale", 1.0, arg_idx + 2)

    out = None
    if expect_out:
        if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
            out = kwargs["out"]
        elif len(args) > arg_idx + 3 and isinstance(args[arg_idx + 3], torch.Tensor):
            out = args[arg_idx + 3]
        elif len(args) > arg_idx + 4 and isinstance(args[arg_idx + 4], torch.Tensor):
            out = args[arg_idx + 4]
        else:
            raise ValueError("elu_out expects an 'out' tensor argument.")

    return x, alpha, scale, input_scale, out


def _launch_elu_kernel(
    x: torch.Tensor, out: torch.Tensor, alpha: float, scale: float, input_scale: float
):
    if not x.is_cuda or not out.is_cuda:
        raise RuntimeError("elu Triton kernel requires CUDA tensors.")
    if x.numel() != out.numel():
        raise ValueError("Input and output must have the same number of elements.")
    if x.dtype != out.dtype:
        raise ValueError("Input and output must have the same dtype.")
    if not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("Input and output must be contiguous tensors.")

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    elu_kernel[grid](
        x,
        out,
        n_elements,
        float(alpha),
        float(scale),
        float(input_scale),
        BLOCK_SIZE=BLOCK_SIZE,
    )


def elu(*args, **kwargs):
    x, alpha, scale, input_scale, _ = _parse_elu_args(args, kwargs, expect_out=False)
    out = torch.empty_like(x)
    _launch_elu_kernel(x.contiguous(), out, alpha, scale, input_scale)
    return out


def elu_out(*args, **kwargs):
    x, alpha, scale, input_scale, out = _parse_elu_args(args, kwargs, expect_out=True)
    if out is None:
        raise ValueError("elu_out requires an 'out' tensor.")
    _launch_elu_kernel(x.contiguous(), out, alpha, scale, input_scale)
    return out
