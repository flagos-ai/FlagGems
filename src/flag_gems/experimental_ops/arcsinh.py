import torch
import triton
import triton.language as tl


@triton.jit
def arcsinh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Compute asinh using: asinh(x) = log(x + sqrt(x*x + 1))
    x_f32 = x.to(tl.float32)
    tmp = x_f32 * x_f32 + 1.0
    sqrt_term = tl.sqrt(tmp)
    y_f32 = tl.log(x_f32 + sqrt_term)

    # Store result; will cast to out dtype as needed
    tl.store(out_ptr + offsets, y_f32, mask=mask)


def _ensure_cuda_tensor(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if not t.is_cuda:
        raise ValueError("Input tensors must be on CUDA device")
    if t.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported by this Triton kernel"
        )


def _arcsinh_impl(input_tensor: torch.Tensor, out_tensor: torch.Tensor = None):
    _ensure_cuda_tensor(input_tensor)

    # Determine result dtype following basic promotion: float -> same, otherwise float32
    if input_tensor.is_floating_point():
        result_dtype = input_tensor.dtype
    else:
        result_dtype = torch.float32

    x = input_tensor
    n_elements = x.numel()

    if out_tensor is None:
        out = torch.empty_like(x, dtype=result_dtype, device=x.device)
    else:
        _ensure_cuda_tensor(out_tensor)
        if out_tensor.numel() != n_elements:
            raise ValueError(
                "Output tensor must have the same number of elements as input"
            )
        # Enforce dtype consistent with promotion
        if out_tensor.dtype != (result_dtype):
            raise TypeError(
                f"Output tensor has dtype {out_tensor.dtype}, expected {result_dtype}"
            )
        out = out_tensor

    # Work with contiguous buffers for the kernel
    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else out.contiguous()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    arcsinh_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # If out was non-contiguous, copy back
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)

    return out


def arcsinh(input_tensor: torch.Tensor):
    return _arcsinh_impl(input_tensor)


def arcsinh_out(input_tensor: torch.Tensor, out: torch.Tensor):
    return _arcsinh_impl(input_tensor, out)
