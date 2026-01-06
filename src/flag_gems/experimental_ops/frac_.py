import torch
import triton
import triton.language as tl


@triton.jit
def frac_(
    x_ptr,  # *Pointer* to input/output tensor (in-place).
    n_elements,  # Number of elements.
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_fp32 = x.to(tl.float32)
    t = tl.where(x_fp32 >= 0, tl.floor(x_fp32), tl.ceil(x_fp32))
    f = x_fp32 - t
    f = f.to(x.dtype)
    tl.store(x_ptr + offsets, f, mask=mask)


# Preserve a handle to the Triton kernel before redefining the Python wrapper with the same name.
frac__triton_kernel = frac_


def frac_(*args, **kwargs):
    # Expect a single tensor as the first positional arg (in-place operation).
    x = args[0] if len(args) > 0 else kwargs.get("input", None)
    if x is None:
        raise ValueError("frac_ expects a tensor as the first argument")

    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device")

    if torch.is_complex(x):
        raise NotImplementedError(
            "Complex dtypes are not supported by this Triton frac_ kernel"
        )

    if not torch.is_floating_point(x):
        raise TypeError("frac_ expects a floating point tensor")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    # Ensure we operate on contiguous memory; if not, use a temporary contiguous buffer and copy back.
    need_copy_back = not x.is_contiguous()
    target = x.contiguous() if need_copy_back else x

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    frac__triton_kernel[grid](target, n_elements, BLOCK_SIZE=1024)

    if need_copy_back:
        x.copy_(target)

    return x
