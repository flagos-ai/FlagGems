import torch
import triton
import triton.language as tl


@triton.jit
def threshold_kernel(
    x_ptr,  # *Pointer* to input tensor
    y_ptr,  # *Pointer* to output tensor
    n_elements,  # Number of elements
    threshold,  # Scalar threshold
    value,  # Scalar value to use when x <= threshold
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > threshold, x, value)
    tl.store(y_ptr + offsets, y, mask=mask)


def _coerce_scalars_for_dtype(dtype, threshold, value):
    if dtype.is_complex:
        raise TypeError("aten.threshold does not support complex dtypes.")
    if dtype == torch.bool:
        raise TypeError("aten.threshold does not support bool dtype.")
    if dtype.is_floating_point:
        thr = float(threshold)
        val = float(value)
    else:
        thr = int(threshold)
        val = int(value)
    return thr, val


def threshold(input: torch.Tensor, threshold, value):
    if input.device.type != "cuda":
        raise RuntimeError("This Triton implementation requires CUDA tensors.")
    x = input.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)

    if n_elements == 0:
        return out

    thr_scalar, val_scalar = _coerce_scalars_for_dtype(x.dtype, threshold, value)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    threshold_kernel[grid](
        x,
        out,
        n_elements,
        thr_scalar,
        val_scalar,
        BLOCK_SIZE=1024,
    )
    return out


def threshold_out(input: torch.Tensor, threshold, value, out: torch.Tensor):
    if input.device.type != "cuda" or out.device.type != "cuda":
        raise RuntimeError("This Triton implementation requires CUDA tensors.")
    if out.shape != input.shape:
        raise RuntimeError(
            f"out shape {out.shape} must match input shape {input.shape}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"out dtype {out.dtype} must match input dtype {input.dtype}"
        )

    x = input.contiguous()
    n_elements = x.numel()

    # Prepare output (contiguous temp if needed)
    y = out if out.is_contiguous() else torch.empty_like(x)

    if n_elements == 0:
        if y is not out:
            out.copy_(y)
        return out

    thr_scalar, val_scalar = _coerce_scalars_for_dtype(x.dtype, threshold, value)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    threshold_kernel[grid](
        x,
        y,
        n_elements,
        thr_scalar,
        val_scalar,
        BLOCK_SIZE=1024,
    )

    if y is not out:
        out.copy_(y)
    return out
