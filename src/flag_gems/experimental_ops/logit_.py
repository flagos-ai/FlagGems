import torch
import triton
import triton.language as tl


@triton.jit
def logit_(
    x_ptr,
    n_elements,
    eps,
    has_eps: tl.constexpr,
    COMPUTE_FP32: tl.constexpr,
    COMPUTE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Promote to higher precision for computation if needed
    if COMPUTE_FP32:
        xc = x.to(tl.float32)
        if has_eps:
            xc = tl.maximum(xc, eps)
            xc = tl.minimum(xc, 1.0 - eps)
        y = tl.log(xc / (1.0 - xc))
        out = y.to(x.dtype)
    elif COMPUTE_FP64:
        xc = x  # already float64
        if has_eps:
            xc = tl.maximum(xc, eps)
            xc = tl.minimum(xc, 1.0 - eps)
        out = tl.log(xc / (1.0 - xc))
    else:
        # float32 compute
        xc = x
        if has_eps:
            xc = tl.maximum(xc, eps)
            xc = tl.minimum(xc, 1.0 - eps)
        out = tl.log(xc / (1.0 - xc))

    tl.store(x_ptr + offsets, out, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper with the same name
logit___kernel = logit_


def logit_(*args, **kwargs):
    # Parse arguments similar to torch.logit_(input, eps=None)
    if len(args) == 0:
        raise TypeError("logit_ expected at least 1 argument (got 0)")
    x = args[0]
    eps = None
    if len(args) > 1:
        eps = args[1]
    if "eps" in kwargs:
        eps = kwargs["eps"]

    if not isinstance(x, torch.Tensor):
        raise TypeError("logit_ expects a torch.Tensor as the first argument")
    if not x.is_cuda:
        raise ValueError("logit_ Triton implementation requires a CUDA tensor")
    if not x.is_floating_point():
        raise TypeError("logit_ expects a floating point tensor")

    has_eps = eps is not None
    eps_value = float(eps) if has_eps else 0.0

    # Work on a contiguous buffer; copy back if needed to preserve in-place semantics
    needs_copy_back = not x.is_contiguous()
    buf = x if not needs_copy_back else x.contiguous()

    n_elements = buf.numel()
    if n_elements == 0:
        return x

    dtype = buf.dtype
    compute_in_fp32 = dtype in (torch.float16, torch.bfloat16)
    compute_in_fp64 = dtype == torch.float64

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    logit___kernel[grid](
        buf,
        n_elements,
        eps_value,
        has_eps=has_eps,
        COMPUTE_FP32=compute_in_fp32,
        COMPUTE_FP64=compute_in_fp64,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if needs_copy_back:
        x.copy_(buf)

    return x
