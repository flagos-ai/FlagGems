import torch
import triton
import triton.language as tl


@triton.jit
def ceil_(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    CAST_TO_FP32: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if IS_FLOAT:
        if CAST_TO_FP32:
            x_fp32 = tl.cast(x, tl.float32)
            y_fp32 = tl.ceil(x_fp32)
            y = tl.cast(y_fp32, OUT_DTYPE)
        else:
            y = tl.ceil(x)
    else:
        y = x

    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a reference to the Triton kernel before defining the wrapper with the same name.
ceil__triton_kernel = ceil_


def ceil_(*args, **kwargs):
    # Extract the input tensor; support typical in-place API signatures.
    x = None
    if len(args) == 1:
        x = args[0]
    else:
        # Common keyword names for in-place unary ops
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None:
        raise ValueError("ceil_ expects a single tensor argument.")

    if not x.is_cuda:
        raise ValueError("ceil_ Triton implementation requires a CUDA tensor.")

    if torch.is_complex(x):
        raise NotImplementedError(
            "ceil_ for complex tensors is not supported in this Triton implementation."
        )

    # Prepare a contiguous tensor to operate on; preserve semantics by copying back if needed.
    needs_copy_back = False
    y = x
    if not x.is_contiguous():
        y = x.contiguous()
        needs_copy_back = True

    n_elements = y.numel()
    if n_elements == 0:
        return x

    float_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    is_float = x.dtype in float_dtypes
    cast_to_fp32 = x.dtype in {torch.float16, torch.bfloat16}

    # Map torch dtype to Triton dtype for casting back when needed.
    triton_dtype = None
    if x.dtype == torch.float16:
        triton_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    elif x.dtype == torch.float32:
        triton_dtype = tl.float32
    elif x.dtype == torch.float64:
        triton_dtype = tl.float64
    else:
        # For non-float types, OUT_DTYPE is unused; default to tl.int32 to satisfy signature.
        triton_dtype = tl.int32

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    ceil__triton_kernel[grid](
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_FLOAT=is_float,
        CAST_TO_FP32=cast_to_fp32,
        OUT_DTYPE=triton_dtype,
    )

    if needs_copy_back:
        x.copy_(y)
    return x
