import torch
import triton
import triton.language as tl


@triton.jit
def trunc_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, DTYPE_CODE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # DTYPE_CODE:
    # 0 -> integer types (copy)
    # 1 -> float16
    # 2 -> bfloat16
    # 3 -> float32
    # 4 -> float64
    if DTYPE_CODE == 0:
        y = x
    elif DTYPE_CODE == 1:
        xf = x.to(tl.float32)
        y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(tl.float16)
    elif DTYPE_CODE == 2:
        xf = x.to(tl.float32)
        y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(tl.bfloat16)
    elif DTYPE_CODE == 3:
        xf = x
        y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))
    elif DTYPE_CODE == 4:
        xf = x
        y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))
    else:
        # Fallback: copy
        y = x

    tl.store(out_ptr + offsets, y, mask=mask)


def _dtype_code(t: torch.Tensor) -> int:
    if t.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        return 0
    if t.dtype == torch.float16:
        return 1
    if t.dtype == torch.bfloat16:
        return 2
    if t.dtype == torch.float32:
        return 3
    if t.dtype == torch.float64:
        return 4
    raise NotImplementedError(f"Unsupported dtype: {t.dtype}")


def _launch_trunc(inp: torch.Tensor, out: torch.Tensor):
    assert inp.numel() == out.numel()
    assert inp.device.type == "cuda" and out.device.type == "cuda"
    n_elements = inp.numel()
    if n_elements == 0:
        return

    code = _dtype_code(inp)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    trunc_kernel[grid](inp, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, DTYPE_CODE=code)


def trunc(input: torch.Tensor):
    # Allocate output
    out = torch.empty_like(input)

    if input.is_complex():
        # Work on real view
        in_r = torch.view_as_real(input)
        out_r = torch.view_as_real(out)
        if not in_r.is_contiguous() or not out_r.is_contiguous():
            in_r_c = in_r.contiguous()
            out_r_c = out_r.contiguous()
            _launch_trunc(in_r_c.view(-1), out_r_c.view(-1))
            out_r.copy_(out_r_c)
        else:
            _launch_trunc(in_r.view(-1), out_r.view(-1))
    else:
        inp_c = input if input.is_contiguous() else input.contiguous()
        out_c = out if out.is_contiguous() else out.contiguous()
        _launch_trunc(inp_c.view(-1), out_c.view(-1))
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)

    return out


def trunc_out(input: torch.Tensor, out: torch.Tensor):
    assert input.shape == out.shape, "input and out must have the same shape"
    assert input.dtype == out.dtype, "input and out must have the same dtype"
    assert (
        input.device.type == "cuda" and out.device.type == "cuda"
    ), "Tensors must be on CUDA device"

    if input.is_complex():
        in_r = torch.view_as_real(input)
        out_r = torch.view_as_real(out)
        if not in_r.is_contiguous() or not out_r.is_contiguous():
            in_r_c = in_r.contiguous()
            out_r_c = out_r.contiguous()
            _launch_trunc(in_r_c.view(-1), out_r_c.view(-1))
            out_r.copy_(out_r_c)
        else:
            _launch_trunc(in_r.view(-1), out_r.view(-1))
    else:
        inp_c = input if input.is_contiguous() else input.contiguous()
        if out.is_contiguous():
            _launch_trunc(inp_c.view(-1), out.view(-1))
        else:
            out_c = out.contiguous()
            _launch_trunc(inp_c.view(-1), out_c.view(-1))
            out.copy_(out_c)

    return out
