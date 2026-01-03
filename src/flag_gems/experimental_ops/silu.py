import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(
    x_ptr,           # *Pointer* to input tensor
    y_ptr,           # *Pointer* to output tensor
    n_elements,      # Number of elements
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_IN_FP32: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    if COMPUTE_IN_FP32:
        xf = x.to(tl.float32)
        yf = xf / (1.0 + tl.exp(-xf))
        y = yf.to(x.dtype)
    else:
        y = x / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


def _silu_impl(x: torch.Tensor, out: torch.Tensor = None):
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if not torch.is_floating_point(x):
        raise TypeError("silu expects a floating point tensor.")
    if out is None:
        out = torch.empty_like(x)
    else:
        if not out.is_cuda:
            raise ValueError("Output tensor must be on CUDA device.")
        if out.shape != x.shape:
            raise ValueError(f"Output shape {out.shape} does not match input shape {x.shape}.")
        if out.dtype != x.dtype:
            raise TypeError(f"Output dtype {out.dtype} does not match input dtype {x.dtype}.")

    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    if n_elements == 0:
        if out_contig is not out:
            out.copy_(out_contig)
        return out

    compute_in_fp32 = x_contig.dtype in (torch.float16, torch.bfloat16)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](
        x_contig,
        out_contig,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        COMPUTE_IN_FP32=compute_in_fp32,
    )

    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def silu(*args, **kwargs):
    # Expecting signature similar to aten.silu(self)
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise TypeError("silu expects a tensor as the first argument.")
    return _silu_impl(x)


def silu_out(*args, **kwargs):
    # Expecting signature similar to aten.silu.out(self, out)
    x = None
    out = None

    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))

    if len(args) >= 2:
        out = args[1]
    else:
        out = kwargs.get("out", None)

    if x is None or out is None:
        raise TypeError("silu_out expects input and out tensors.")

    _silu_impl(x, out)
    return out