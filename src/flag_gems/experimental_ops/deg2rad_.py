import torch
import triton
import triton.language as tl


@triton.jit
def deg2rad_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    factor = 0.017453292519943295  # pi / 180
    y = x * factor
    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name
deg2rad__kernel = deg2rad_


def deg2rad_(*args, **kwargs):
    # Extract the input tensor
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        # Try common keyword names
        x = kwargs.get("input", kwargs.get("self", None))
    if x is None:
        raise ValueError("deg2rad_ expects a tensor as its first argument.")

    if not isinstance(x, torch.Tensor):
        raise TypeError("deg2rad_ expects a torch.Tensor as input.")

    # Handle empty tensor quickly
    n_elements = x.numel()
    if n_elements == 0:
        return x

    # If not CUDA or not contiguous or unsupported dtype, fallback to PyTorch scalar multiply in-place
    factor = 0.017453292519943295  # pi / 180
    if (
        (x.device.type != "cuda")
        or (not x.is_contiguous())
        or x.is_complex()
        or (not x.is_floating_point())
    ):
        x.mul_(factor)
        return x

    # Launch Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    deg2rad__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
