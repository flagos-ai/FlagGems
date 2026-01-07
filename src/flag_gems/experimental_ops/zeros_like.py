import torch
import triton
import triton.language as tl


@triton.jit
def _fill_zero_kernel(
    out_ptr,  # *Pointer* to output vector.
    n_elements,  # Number of elements to write.
    BLOCK_SIZE: tl.constexpr,  # Number of elements per program.
    OUT_DTYPE: tl.constexpr,  # Triton dtype for the output.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    zeros = tl.full([BLOCK_SIZE], 0, dtype=OUT_DTYPE)
    tl.store(out_ptr + offsets, zeros, mask=mask)


def _torch_dtype_to_triton_dtype(dtype: torch.dtype):
    # Map torch dtypes to Triton dtypes
    if dtype is torch.float32:
        return tl.float32
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.float64:
        return tl.float64
    if dtype is torch.int8:
        return tl.int8
    if dtype is torch.uint8:
        return tl.uint8
    if dtype is torch.int16:
        return tl.int16
    if dtype is torch.int32:
        return tl.int32
    if dtype is torch.int64:
        return tl.int64
    if dtype is torch.bool:
        # Triton bool storage is not directly exposed; use int8 for 0/1 storage
        return tl.int8
    raise NotImplementedError(f"Unsupported dtype for Triton zeros_like: {dtype}")


def _launch_fill_zero(out: torch.Tensor, block_size: int = 4096):
    # Fallback for non-CUDA or empty tensors
    n_elements = out.numel()
    if n_elements == 0:
        return
    if not out.is_cuda:
        out.zero_()
        return
    # For simplicity, only handle contiguous tensors with the Triton kernel.
    # Fallback to PyTorch for non-contiguous outputs.
    if not out.is_contiguous():
        out.zero_()
        return
    out_dtype = _torch_dtype_to_triton_dtype(out.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fill_zero_kernel[grid](out, n_elements, BLOCK_SIZE=block_size, OUT_DTYPE=out_dtype)


def zeros_like(*args, **kwargs):
    # Extract input tensor (first positional or 'input'/'self' kw)
    inp = None
    if len(args) >= 1:
        inp = args[0]
    else:
        inp = kwargs.get("input", kwargs.get("self", None))
    if inp is None:
        raise ValueError("zeros_like expects an input tensor as the first argument.")

    dtype = kwargs.get("dtype", None)
    layout = kwargs.get("layout", None)
    device = kwargs.get("device", None)
    pin_memory = kwargs.get("pin_memory", None)
    memory_format = kwargs.get("memory_format", torch.preserve_format)

    # Allocate output tensor with requested properties
    out = torch.empty_like(
        inp,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory if pin_memory is not None else False,
        memory_format=memory_format,
    )
    _launch_fill_zero(out)
    return out


def zeros_like_out(*args, **kwargs):
    # Expected signature: zeros_like.out(input, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None, out) # noqa: E501
    # Extract input and out tensors
    inp = None
    if len(args) >= 1:
        inp = args[0]
    else:
        inp = kwargs.get("input", kwargs.get("self", None))
    out = kwargs.get("out", None)
    if out is None and len(args) >= 2:
        out = args[-1]
    if inp is None or out is None:
        raise ValueError("zeros_like_out expects 'input' and 'out' tensors.")

    # Optional consistency checks per .out semantics (if provided)
    dtype = kwargs.get("dtype", None)
    device = kwargs.get("device", None)
    if dtype is not None and out.dtype != dtype:
        raise ValueError(f"Provided dtype {dtype} does not match out.dtype {out.dtype}")
    if device is not None and str(out.device) != str(device):
        raise ValueError(
            f"Provided device {device} does not match out.device {out.device}"
        )
    # Shape/layout checks could be added; we keep minimal checks for generality.

    _launch_fill_zero(out)
    return out
