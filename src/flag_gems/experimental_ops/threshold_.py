import torch
import triton
import triton.language as tl


@triton.jit
def threshold_(x_ptr,  # Pointer to input/output tensor (in-place)
               n_elements,  # Number of elements
               threshold_ptr,  # Pointer to scalar threshold (0-d tensor)
               value_ptr,  # Pointer to scalar value (0-d tensor)
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Load scalars (dtype matches x because we pass 0-d tensors of x.dtype)
    thr = tl.load(threshold_ptr)
    val = tl.load(value_ptr)

    # Apply threshold in-place: if x <= thr, set to val, else keep x
    out = tl.where(x <= thr, val, x)

    # Store back
    tl.store(x_ptr + offsets, out, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper of the same name
threshold__triton_kernel = threshold_


def threshold_(*args, **kwargs):
    # Extract arguments similar to aten.threshold_ signature: (self, threshold, value=0)
    x = kwargs.get('input', args[0] if len(args) > 0 else None)
    threshold = kwargs.get('threshold', args[1] if len(args) > 1 else None)
    value = kwargs.get('value', args[2] if len(args) > 2 else 0)

    if x is None or threshold is None:
        raise ValueError("threshold_ requires at least (input, threshold) arguments")

    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernel execution")
    if x.is_complex():
        raise ValueError("Complex dtypes are not supported by this kernel")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous for this Triton kernel")

    n_elements = x.numel()

    # Prepare scalar tensors for threshold and value with matching dtype/device
    thr_t = torch.tensor(threshold, dtype=x.dtype, device=x.device)
    val_t = torch.tensor(value, dtype=x.dtype, device=x.device)

    # Launch configuration
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel (in-place)
    threshold__triton_kernel[grid](x, n_elements, thr_t, val_t, BLOCK_SIZE=BLOCK_SIZE)

    return x