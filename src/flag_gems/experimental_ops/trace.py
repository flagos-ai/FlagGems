import torch
import triton
import triton.language as tl


@triton.jit
def trace_kernel(
    x_ptr,
    stride0,
    stride1,
    diag_len,
    out_ptr,
    OUT_TYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Accumulate in float32 for numerical stability across input dtypes
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    sdiag = stride0 + stride1

    i = 0
    while i < diag_len:
        idx = i + tl.arange(0, BLOCK)
        mask = idx < diag_len
        ptrs = x_ptr + idx * sdiag
        vals = tl.load(ptrs, mask=mask, other=0)
        acc += tl.cast(vals, tl.float32)
        i += BLOCK

    total = tl.sum(acc, axis=0)

    # Cast to desired output dtype based on OUT_TYPE code
    if OUT_TYPE == 0:
        val = tl.cast(total, tl.float16)
    elif OUT_TYPE == 1:
        val = tl.cast(total, tl.bfloat16)
    elif OUT_TYPE == 2:
        val = tl.cast(total, tl.float32)
    elif OUT_TYPE == 3:
        val = tl.cast(total, tl.float64)
    elif OUT_TYPE == 4:
        val = tl.cast(total, tl.int32)
    elif OUT_TYPE == 5:
        val = tl.cast(total, tl.int64)
    elif OUT_TYPE == 6:
        val = tl.cast(total, tl.int16)
    elif OUT_TYPE == 7:
        val = tl.cast(total, tl.int8)
    elif OUT_TYPE == 8:
        val = tl.cast(total, tl.uint8)
    else:
        val = tl.cast(total, tl.float32)

    tl.store(out_ptr, val)


def _dtype_to_code(dtype: torch.dtype) -> int:
    mapping = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        torch.float64: 3,
        torch.int32: 4,
        torch.int64: 5,
        torch.int16: 6,
        torch.int8: 7,
        torch.uint8: 8,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for trace kernel: {dtype}")
    return mapping[dtype]


def _launch_trace_kernel(input: torch.Tensor, out: torch.Tensor):
    if not input.is_cuda or not out.is_cuda:
        raise ValueError("trace kernel requires CUDA tensors")
    if input.dim() != 2:
        raise ValueError(f"trace expects a 2D tensor, got {input.dim()}D")
    if out.numel() != 1:
        raise ValueError("out tensor must have a single element (0-dim/scalar)")

    n0, n1 = input.shape
    diag_len = min(n0, n1)
    s0, s1 = input.stride()

    out_code = _dtype_to_code(out.dtype)

    grid = lambda meta: (1,)
    trace_kernel[grid](
        input,
        s0,
        s1,
        diag_len,
        out,
        OUT_TYPE=out_code,
        BLOCK=1024,
    )


def trace(input: torch.Tensor):
    out = torch.empty((), device=input.device, dtype=input.dtype)
    _launch_trace_kernel(input, out)
    return out


def trace_out(input: torch.Tensor, out: torch.Tensor):
    _launch_trace_kernel(input, out)
    return out
