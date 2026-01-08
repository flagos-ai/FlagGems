import torch
import triton
import triton.language as tl


@triton.jit
def huber_loss_element_kernel(
    x_ptr,  # pointer to input tensor (broadcasted, contiguous, flattened)
    y_ptr,  # pointer to target tensor (broadcasted, contiguous, flattened)
    out_ptr,  # pointer to output tensor (contiguous, flattened)
    n_elements,  # number of elements
    delta,  # huber delta (scalar)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    diff = x - y
    absdiff = tl.abs(diff)

    # loss = 0.5 * diff^2 if absdiff <= delta else delta * (absdiff - 0.5 * delta)
    loss_quad = 0.5 * diff * diff
    loss_linear = delta * (absdiff - 0.5 * delta)
    loss = tl.where(absdiff <= delta, loss_quad, loss_linear)

    tl.store(out_ptr + offsets, loss, mask=mask)


@triton.jit
def reduce_sum_kernel(
    x_ptr,  # pointer to input tensor (contiguous, flattened)
    out_ptr,  # pointer to single scalar (float32) to accumulate sum into
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    vals_f32 = vals.to(tl.float32)
    partial_sum = tl.sum(vals_f32, axis=0)
    tl.atomic_add(out_ptr, partial_sum)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        elif r == "mean":
            return 1
        elif r == "sum":
            return 2
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    elif isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    else:
        raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def huber_loss(input, target, reduction=1, delta=1.0):
    reduction = _normalize_reduction(reduction)
    if not (input.is_cuda and target.is_cuda):
        raise AssertionError("Triton kernels require CUDA tensors")
    device = input.device
    # Promote dtype similar to PyTorch type promotion rules
    result_dtype = torch.result_type(input, target)

    # Broadcast tensors to a common shape
    x_b, y_b = torch.broadcast_tensors(input.to(result_dtype), target.to(result_dtype))
    x_b = x_b.contiguous()
    y_b = y_b.contiguous()
    numel = x_b.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    if reduction == 0:  # 'none'
        out = torch.empty_like(x_b, dtype=result_dtype, device=device)
        huber_loss_element_kernel[grid](
            x_b, y_b, out, numel, float(delta), BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        # Compute elementwise loss into a temporary buffer
        tmp = torch.empty_like(x_b, dtype=result_dtype, device=device)
        huber_loss_element_kernel[grid](
            x_b, y_b, tmp, numel, float(delta), BLOCK_SIZE=BLOCK_SIZE
        )
        # Reduce to scalar using float32 accumulator
        acc = torch.zeros((), dtype=torch.float32, device=device)
        reduce_sum_kernel[grid](tmp, acc, numel, BLOCK_SIZE=BLOCK_SIZE)
        if reduction == 1:  # mean
            val = (acc / numel).to(result_dtype)
        else:  # sum
            val = acc.to(result_dtype)
        return val


def huber_loss_out(input, target, reduction=1, delta=1.0, out=None):
    if out is None:
        raise ValueError("huber_loss_out requires an 'out' tensor")
    reduction = _normalize_reduction(reduction)
    if not (input.is_cuda and target.is_cuda and out.is_cuda):
        raise AssertionError("Triton kernels require CUDA tensors")

    device = input.device
    # Determine result dtype; use out.dtype if provided to match .out behavior
    # but ensure it's compatible with promoted dtype
    promoted_dtype = torch.result_type(input, target)
    result_dtype = out.dtype

    # Broadcast tensors
    x_b, y_b = torch.broadcast_tensors(
        input.to(promoted_dtype), target.to(promoted_dtype)
    )
    x_b = x_b.contiguous()
    y_b = y_b.contiguous()
    numel = x_b.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    if reduction == 0:  # 'none'
        # Ensure out has correct shape
        if out.numel() != numel or out.shape != x_b.shape:
            raise ValueError(
                f"'out' tensor must have shape {tuple(x_b.shape)} for reduction='none'"
            )
        # Compute into a temporary if out is not contiguous or dtype mismatches
        needs_tmp = (not out.is_contiguous()) or (out.dtype != result_dtype)
        if needs_tmp:
            tmp = torch.empty_like(x_b, dtype=result_dtype, device=device)
            huber_loss_element_kernel[grid](
                x_b.to(result_dtype),
                y_b.to(result_dtype),
                tmp,
                numel,
                float(delta),
                BLOCK_SIZE=BLOCK_SIZE,
            )
            out.copy_(tmp)
        else:
            huber_loss_element_kernel[grid](
                x_b.to(result_dtype),
                y_b.to(result_dtype),
                out,
                numel,
                float(delta),
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return out
    else:
        # Compute elementwise loss into temporary (in promoted dtype), then reduce to scalar
        tmp = torch.empty_like(x_b, dtype=promoted_dtype, device=device)
        huber_loss_element_kernel[grid](
            x_b, y_b, tmp, numel, float(delta), BLOCK_SIZE=BLOCK_SIZE
        )
        acc = torch.zeros((), dtype=torch.float32, device=device)
        reduce_sum_kernel[grid](tmp, acc, numel, BLOCK_SIZE=BLOCK_SIZE)
        if reduction == 1:  # mean
            val = (acc / numel).to(result_dtype)
        else:  # sum
            val = acc.to(result_dtype)
        # Ensure out is scalar/0-d
        if out.numel() != 1 or out.dim() > 1:
            raise ValueError(
                "For reduction='mean' or 'sum', 'out' must be a scalar (0-d or 1-element) tensor"
            )
        # Copy the scalar value into out
        out.copy_(val)
        return out
