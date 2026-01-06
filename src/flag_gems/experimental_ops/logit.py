import torch
import triton
import triton.language as tl


@triton.jit
def logit_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    eps,
    HAS_EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)

    if HAS_EPS:
        lo = eps
        hi = 1.0 - eps
        x_f32 = tl.minimum(tl.maximum(x_f32, lo), hi)

    y = tl.log(x_f32 / (1.0 - x_f32))
    tl.store(y_ptr + offsets, y.to(OUT_DTYPE), mask=mask)


def _to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return None


def _logit_impl(input: torch.Tensor, eps=None, out: torch.Tensor = None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device for Triton kernel.")
    if not input.is_floating_point():
        raise TypeError("logit expected a floating point tensor as input")
    if eps is not None:
        eps = float(eps)
        if not (0.0 <= eps <= 0.5):
            raise ValueError("eps must be in the range [0.0, 0.5].")

    in_contig = input.contiguous()
    in_supported = _to_triton_dtype(in_contig.dtype) is not None
    in_kernel = in_contig if in_supported else in_contig.to(torch.float32)

    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if not out.is_cuda:
            raise AssertionError("Out tensor must be on CUDA device for Triton kernel.")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != input.dtype:
            raise TypeError("For logit.out, out.dtype must match input.dtype")
        # Decide working output (contiguous and with supported dtype)
        out_supported = _to_triton_dtype(out.dtype) is not None
        need_copy_back = (not out.is_contiguous()) or (not out_supported)

        if need_copy_back:
            work_dtype = out.dtype if out_supported else torch.float32
            work_out = torch.empty_like(out, dtype=work_dtype)
        else:
            work_out = out

        n_elements = in_kernel.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        triton_dtype = _to_triton_dtype(work_out.dtype)
        logit_kernel[grid](
            in_kernel,
            work_out,
            n_elements,
            eps if eps is not None else 0.0,
            HAS_EPS=(eps is not None),
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DTYPE=triton_dtype,
        )

        if need_copy_back:
            out.copy_(work_out.to(out.dtype))
        return out

    # out is None -> produce and return a new tensor
    desired_dtype = input.dtype
    desired_supported = _to_triton_dtype(desired_dtype) is not None
    if desired_supported:
        result = torch.empty_like(input, dtype=desired_dtype)
        work_out = result
    else:
        # compute in fp32, cast back to desired
        work_out = torch.empty_like(input, dtype=torch.float32)

    n_elements = in_kernel.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    triton_dtype = _to_triton_dtype(work_out.dtype)
    logit_kernel[grid](
        in_kernel,
        work_out,
        n_elements,
        eps if eps is not None else 0.0,
        HAS_EPS=(eps is not None),
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_DTYPE=triton_dtype,
    )

    if desired_supported:
        return work_out
    else:
        return work_out.to(desired_dtype)


def logit(input, eps=None):
    return _logit_impl(input, eps=eps, out=None)


def logit_out(input, eps=None, out=None):
    if out is None:
        raise TypeError("logit_out requires an 'out' tensor.")
    return _logit_impl(input, eps=eps, out=out)
