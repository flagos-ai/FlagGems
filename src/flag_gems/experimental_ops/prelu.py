import torch
import triton
import triton.language as tl


@triton.jit
def prelu(x_ptr,  # *Pointer* to input tensor.
          w_ptr,  # *Pointer* to weight tensor (scalar or per-channel vector).
          out_ptr,  # *Pointer* to output tensor.
          n_elements,  # Total number of elements in input.
          S,  # Spatial size = product of dims after channel dim (or 1 if none).
          C,  # Number of channels (or 1).
          w_is_scalar: tl.constexpr,  # Whether weight is a single scalar.
          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if w_is_scalar:
        alpha = tl.load(w_ptr)  # scalar
        y = tl.where(x >= 0, x, alpha * x)
    else:
        c = (offsets // S) % C
        alpha = tl.load(w_ptr + c, mask=mask)
        y = tl.where(x >= 0, x, alpha * x)

    tl.store(out_ptr + offsets, y, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name.
prelu_kernel = prelu


def prelu(*args, **kwargs):
    # Extract inputs
    if len(args) >= 2:
        x, weight = args[0], args[1]
    else:
        x = kwargs.get('input', kwargs.get('self'))
        weight = kwargs.get('weight')
    if x is None or weight is None:
        raise ValueError("prelu expects (input, weight) as arguments.")

    if not (x.is_cuda and weight.is_cuda):
        raise AssertionError("Tensors must be CUDA tensors.")

    # Ensure dtype match
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    # Determine channel count C and spatial size S
    ndim = x.dim()
    if weight.numel() == 1:
        C = 1
        S = 1
        w_is_scalar = True
    else:
        if ndim == 0:
            raise AssertionError("Non-scalar weight provided for a 0-dim input.")
        if ndim == 1:
            C = x.shape[0]
            S = 1
        else:
            C = x.shape[1]
            S = 1
            if ndim > 2:
                for d in x.shape[2:]:
                    S *= d
        if weight.numel() != C:
            raise AssertionError(f"Weight numel ({weight.numel()}) must equal channel dimension size ({C}).")
        w_is_scalar = False

    # Make sure S and C are at least 1 to avoid div/mod by zero in kernel math
    C = max(int(C), 1)
    S = max(int(S), 1)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    prelu_kernel[grid](
        x, weight, out,
        n_elements, S, C,
        w_is_scalar=w_is_scalar,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out