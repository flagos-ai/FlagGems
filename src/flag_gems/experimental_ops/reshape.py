import torch
import triton
import triton.language as tl


@triton.jit
def reshape(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


_reshape_kernel = reshape


def reshape(*args, **kwargs):
    # Parse inputs
    if len(args) >= 2:
        x = args[0]
        target_shape = args[1]
    else:
        x = None
        target_shape = None
        for key in ("x", "input", "a", "tensor", "self"):
            if key in kwargs:
                x = kwargs[key]
                break
        for key in ("shape", "new_shape", "size"):
            if key in kwargs:
                target_shape = kwargs[key]
                break

    assert x is not None, "Input tensor is required."
    assert target_shape is not None, "Target shape is required."
    assert x.is_cuda, "Input tensor must be on CUDA device."

    # Normalize target shape
    if isinstance(target_shape, torch.Size):
        new_shape = tuple(target_shape)
    else:
        new_shape = tuple(int(s) for s in target_shape)

    # Infer -1 if present
    if any(s == -1 for s in new_shape):
        negs = sum(1 for s in new_shape if s == -1)
        assert negs == 1, "Only one -1 is allowed in the target shape."
        known_prod = 1
        for s in new_shape:
            if s != -1:
                known_prod *= s
        total = x.numel()
        assert (
            known_prod != 0 and total % known_prod == 0
        ), "Invalid shape for inference."
        inferred = total // known_prod
        new_shape = tuple(inferred if s == -1 else s for s in new_shape)

    # Allocate output and launch kernel to copy data in row-major order
    out = torch.empty(new_shape, device=x.device, dtype=x.dtype)
    n_elements = out.numel()
    assert x.numel() == n_elements, "Reshape must not change the number of elements."

    src = x.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _reshape_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out
