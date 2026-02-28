import torch
import triton
import triton.language as tl


@triton.jit
def digamma_(
    x_ptr,  # Pointer to input/output tensor (in-place)
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)

    pi = 3.1415926535897932384626433832795028841971

    # Reflection for x < 0.5: psi(x) = psi(1 - x) - pi * cot(pi * x)
    reflect_mask = x_f32 < 0.5
    xr = tl.where(reflect_mask, 1.0 - x_f32, x_f32)

    # Use recurrence to shift xr to >= 8 for better asymptotic precision
    s = tl.zeros_like(x_f32)
    y = xr
    for _ in range(8):
        m = y < 8.0
        s = s - tl.where(m, 1.0 / y, 0.0)
        y = tl.where(m, y + 1.0, y)

    # Asymptotic expansion for digamma at large y
    r = 1.0 / y
    r2 = r * r
    t2 = r2
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    series = (
        (-0.5 * r)
        + (-1.0 / 12.0) * t2
        + (1.0 / 120.0) * t4
        + (-1.0 / 252.0) * t6
        + (1.0 / 240.0) * t8
    )
    psi_y = tl.log(y) + s + series

    # Apply reflection if needed
    cot_term = tl.cos(pi * x_f32) / tl.sin(pi * x_f32)
    result = tl.where(reflect_mask, psi_y - pi * cot_term, psi_y)

    result = result.to(x.dtype)
    tl.store(x_ptr + offsets, result, mask=mask)


_KERNEL_DIGAMMA_INPLACE = digamma_


def digamma_(*args, **kwargs):
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("digamma_ expects a torch.Tensor as the first argument")
    if not x.is_cuda:
        return torch.ops.aten.digamma_(x)

    # Handle non-contiguous tensors by operating on a contiguous copy and copying back
    if not x.is_contiguous():
        y = x.contiguous()
        n_elements = y.numel()
        if n_elements == 0:
            return x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _KERNEL_DIGAMMA_INPLACE[grid](y, n_elements, BLOCK_SIZE=1024)
        x.copy_(y)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _KERNEL_DIGAMMA_INPLACE[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
