import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # *Pointer* to input tensor
    out_ptr,  # *Pointer* to output tensor
    n_elements,  # Number of elements
    beta,  # beta scalar (float32)
    threshold,  # threshold scalar (float32)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)

    z = x_fp32 * beta
    # compute softplus in a numerically stable way:
    # if z > threshold => x
    # else => log(1 + exp(z)) / beta
    exp_z = tl.exp(z)
    sp = tl.log(1.0 + exp_z) / beta
    y_fp32 = tl.where(z > threshold, x_fp32, sp)

    y = y_fp32.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _softplus_launch(x: torch.Tensor, beta: float, threshold: float, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "Supported dtypes: float16, bfloat16, float32"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    softplus_kernel[grid](
        x, out, n_elements, float(beta), float(threshold), BLOCK_SIZE=1024
    )
    return out


def _parse_softplus_args(args, kwargs, expect_out: bool = False):
    # ATen signature: softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
    # ATen signature: softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise ValueError("softplus expects 'self' tensor as the first argument")

    beta = kwargs.get("beta", 1.0)
    if len(args) >= 2:
        beta = args[1]
    threshold = kwargs.get("threshold", 20.0)
    if len(args) >= 3:
        threshold = args[2]

    out = None
    if expect_out:
        if "out" in kwargs:
            out = kwargs["out"]
        elif len(args) >= 4:
            out = args[3]

    return x, float(beta), float(threshold), out


def softplus(*args, **kwargs):
    x, beta, threshold, _ = _parse_softplus_args(args, kwargs, expect_out=False)
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    _softplus_launch(x.view(-1), beta, threshold, out.view(-1))
    return out


def softplus_out(*args, **kwargs):
    x, beta, threshold, out = _parse_softplus_args(args, kwargs, expect_out=True)
    if out is None:
        out = torch.empty_like(x)
    # Ensure contiguous buffers for kernel execution
    x_c = x.contiguous() if not x.is_contiguous() else x
    out_c_needs_copyback = False
    if (
        (not out.is_contiguous())
        or (out.shape != x.shape)
        or (out.dtype != x.dtype)
        or (out.device != x.device)
    ):
        out_c = torch.empty_like(x_c)
        out_c_needs_copyback = True
    else:
        out_c = out

    _softplus_launch(x_c.view(-1), beta, threshold, out_c.view(-1))

    if out_c_needs_copyback:
        out.copy_(out_c)
    return out
