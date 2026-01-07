import torch
import triton
import triton.language as tl


@triton.jit
def view_as_real(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


# Preserve a reference to the Triton kernel before defining the Python wrapper
view_as_real_kernel = view_as_real


def view_as_real(*args, **kwargs):
    if len(args) < 1 and "input" not in kwargs:
        raise ValueError("view_as_real expects a single tensor argument")
    x = args[0] if len(args) >= 1 else kwargs["input"]

    # If not on CUDA, fall back to PyTorch implementation
    if not x.is_cuda:
        return torch.ops.aten.view_as_real(x)

    if not torch.is_complex(x):
        raise TypeError("view_as_real expects a complex tensor input")

    # Map complex dtype to its corresponding real component dtype
    dtype_map = {
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }
    # Optional support for complex32 if available
    if hasattr(torch, "complex32"):
        dtype_map[torch.complex32] = torch.float16

    base_dtype = dtype_map.get(x.dtype, None)
    if base_dtype is None:
        raise TypeError(f"Unsupported complex dtype: {x.dtype}")

    x_contig = x.contiguous()
    num_complex = x_contig.numel()
    # Each complex element has two base components (real, imag)
    total_base_elems = num_complex * 2

    # Reinterpret complex storage as a flat base-dtype tensor [r0,i0,r1,i1,...]
    x_base_flat = x_contig.view(base_dtype).view(-1)

    # Allocate output with shape (..., 2) and flatten
    out_shape = (*x_contig.shape, 2)
    out = torch.empty(out_shape, dtype=base_dtype, device=x_contig.device)
    out_flat = out.view(-1)

    if total_base_elems == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_base_elems, meta["BLOCK_SIZE"]),)
    view_as_real_kernel[grid](
        x_base_flat, out_flat, total_base_elems, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
