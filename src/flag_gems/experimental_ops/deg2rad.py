import math
import torch
import triton
import triton.language as tl


@triton.jit
def deg2rad_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * scale
    tl.store(y_ptr + offsets, y, mask=mask)


def _launch_deg2rad_kernel(x_contig: torch.Tensor, out_contig: torch.Tensor):
    assert x_contig.is_cuda and out_contig.is_cuda, "Tensors must be on CUDA device"
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scale = math.pi / 180.0
    deg2rad_kernel[grid](
        x_contig, out_contig, n_elements, scale, BLOCK_SIZE=1024
    )


def deg2rad(*args, **kwargs):
    # Expecting a single input tensor
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", None)
        if x is None:
            x = kwargs.get("self", None)
    if x is None:
        raise TypeError("deg2rad expected a single input tensor")

    if not isinstance(x, torch.Tensor):
        raise TypeError("deg2rad input must be a torch.Tensor")

    if not x.is_cuda:
        raise AssertionError("deg2rad expects CUDA tensors")

    if x.is_complex():
        raise NotImplementedError("Complex tensors are not supported in this Triton implementation")

    # Determine result dtype: keep floating dtype; promote integer/bool to float32
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_cast = x.to(result_dtype)
    x_contig = x_cast.contiguous()
    out = torch.empty_like(x_contig, dtype=result_dtype, device=x.device)

    _launch_deg2rad_kernel(x_contig.view(-1), out.view(-1))

    return out.view_as(x)


def deg2rad_out(*args, **kwargs):
    # Expecting input tensor and out tensor, either as positional or keyword args
    x = None
    out = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", None) or kwargs.get("self", None)
    if len(args) >= 2:
        out = args[1]
    else:
        out = kwargs.get("out", None)

    if x is None or out is None:
        raise TypeError("deg2rad_out expected (input, out) tensors")

    if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError("deg2rad_out arguments must be torch.Tensor")

    if not x.is_cuda or not out.is_cuda:
        raise AssertionError("deg2rad_out expects CUDA tensors")

    if x.is_complex() or out.is_complex():
        raise NotImplementedError("Complex tensors are not supported in this Triton implementation")

    if out.numel() != x.numel():
        raise RuntimeError("deg2rad_out: 'out' must have the same number of elements as 'input'")

    if out.device != x.device:
        raise RuntimeError("deg2rad_out: 'out' must be on the same device as 'input'")

    # Compute in the dtype of 'out' to match out-variant semantics
    x_cast = x.to(out.dtype)
    x_contig = x_cast.contiguous()

    if out.is_contiguous():
        out_contig = out
        need_copy_back = False
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        need_copy_back = True

    _launch_deg2rad_kernel(x_contig.view(-1), out_contig.view(-1))

    if need_copy_back:
        out.copy_(out_contig)

    return out